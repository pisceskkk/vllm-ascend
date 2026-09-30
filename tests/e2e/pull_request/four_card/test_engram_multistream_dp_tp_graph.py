# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined DP2/TP2 head reconstruction with isolated producer groups."""

import socket

import torch.multiprocessing as mp

from tests.e2e.pull_request.two_card.test_engram_multistream_dp_graph import _worker


def test_engram_dp_tp_graph_overlap_with_unequal_and_empty_dp_batches():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, "dp_tp"), nprocs=4, join=True)


if __name__ == "__main__":
    test_engram_dp_tp_graph_overlap_with_unequal_and_empty_dp_batches()
