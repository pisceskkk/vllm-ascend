# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send producer/consumer requests and retain each leg's raw HTTP response."""

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def save_record(path, record):
    path.joinpath("record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def request(leg, base_url, payload, record, path, timeout):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    record[leg] = {"url": url, "request": payload}
    save_record(path, record)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": record["request_id"]},
    )
    # The two node URLs must be reached directly, regardless of download proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    path.joinpath(f"{leg}_response.json").write_bytes(body)
    record[leg].update(status=status, response_file=f"{leg}_response.json")
    save_record(path, record)
    if status != 200:
        raise RuntimeError(f"{leg} returned HTTP {status}; raw response preserved")
    return json.loads(body)


def check_response(response):
    choice = response["choices"][0]
    text = (choice.get("message") or {}).get("content")
    tokens = choice.get("token_ids")
    issues = []
    if not isinstance(text, str) or not text.strip():
        issues.append("empty output text")
    if not tokens:
        issues.append("missing or empty token_ids")
    elif all(token == 0 for token in tokens):
        issues.append("output consists only of token ID 0")
    if response.get("usage", {}).get("completion_tokens", 0) <= 0:
        issues.append("no completion tokens")
    if choice.get("stop_reason") == "recomputed":
        issues.append("request was preempted: a PD proxy must reissue prefill")
    if choice.get("finish_reason") not in ("stop", "length"):
        issues.append(f"unexpected finish_reason: {choice.get('finish_reason')}")
    selected_logprobs = (choice.get("logprobs") or {}).get("content") or []
    if any(not math.isfinite(item["logprob"]) for item in selected_logprobs):
        issues.append("non-finite output token logprob")
    return {
        "text": text,
        "token_ids": tokens,
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "usage": response.get("usage"),
        "issues": issues,
    }


def run_case(index, prompt, args):
    path = args.output / f"{index:04d}"
    path.mkdir()
    record = {"index": index, "request_id": f"pd-smoke-{uuid.uuid4().hex}"}
    started = time.monotonic()
    max_tokens = args.max_tokens[index % len(args.max_tokens)]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
        "return_token_ids": True,
        "logprobs": not args.no_logprobs,
    }
    if not args.no_logprobs:
        payload["top_logprobs"] = 3
    try:
        prefill = dict(payload, max_tokens=1, min_tokens=1)
        prefill["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        producer = request("producer", args.prefill, prefill, record, path, args.timeout)
        params = producer.get("kv_transfer_params")
        if not params or not params.get("remote_block_ids"):
            raise RuntimeError("Producer did not return transferable KV block IDs")
        # Preserve all producer response fields and all KV groups verbatim.
        payload["kv_transfer_params"] = params
        consumer = request("consumer", args.decode, payload, record, path, args.timeout)
        record["summary"] = check_response(consumer)
        record["ok"] = not record["summary"]["issues"]
    except Exception as error:
        record.update(ok=False, error=repr(error))
    record["seconds"] = round(time.monotonic() - started, 3)
    save_record(path, record)
    print(json.dumps({key: record[key] for key in ("index", "ok", "seconds")}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True, help="Producer base URL")
    parser.add_argument("--decode", required=True, help="Consumer base URL")
    parser.add_argument("--model", default="dsv41-recompute-audit")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[192], help="Generation budgets, cycled per request"
    )
    parser.add_argument("--suite", choices=("short", "burst", "mixed"), default="short")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--no-logprobs", action="store_true")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.timeout, *args.max_tokens) <= 0:
        parser.error("requests, concurrency, timeout and max-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    short = [
        "What is 12 minus 7? Reply with only the number.",
        "What is 3 plus 5? Reply with only the number.",
        "What is 6 times 7? Reply with only the number.",
        "Write the word BLUE in lowercase. Reply with only that word.",
    ]
    prompts = {
        "short": short,
        "burst": ["Count 1 to 50."],
        "mixed": ["Count 1 to 10.", "Count 1 to 50.", short[0], short[-1]],
    }[args.suite]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_case, index, prompts[index % len(prompts)], args) for index in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    summary = {
        "requests": len(records),
        "passed": sum(record["ok"] for record in records),
        "failed_indices": sorted(record["index"] for record in records if not record["ok"]),
        "length_limited_indices": sorted(
            record["index"] for record in records if record.get("summary", {}).get("finish_reason") == "length"
        ),
    }
    args.output.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return int(bool(summary["failed_indices"]))


if __name__ == "__main__":
    raise SystemExit(main())
