# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Keep HiSparse KV preprocessing numerically identical to native SFA."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

# One fixed BF16 activation row with unit gamma and epsilon 1e-5. Its RMS
# normalization lands within 8e-8 of three BF16 rounding midpoints. Repeating
# this row reproduces the standalone-vs-fused difference; no model is loaded.
_ROUNDING_ROW_HEX = (
    "bc3df03d173e0d3c1bbe9cbcefbdb73822bde4ba14bd2ebda73dff3d003d1e3d1e3d053eec3da83d21be63bd8d3d40bd1cbda03bb1bd1abe04bb583c"
    "563ec63bfdbdd1bd6b3d70bde73d103e343e0e3de13d1ebc323e03bd77bdec3c3abd2abca6bd043d4ebdf5bd1e3ce13da73dbebd083da73d063ec03d"
    "1cbe71bced3c92bd3cbdd13d71bdc83dc13ca2bb85bcb1bd533d4a3dbabd57bdb43cc33c0bbb163e683d953dbeba65bdcb3ac73c56bd23bd86bd12bd"
    "9ebd2abeb93def3ce33d7f3d783dca3cc3bc9b3d6b3d513d043d1d3dd33da03c9c3d023d59bd943dc9bd29bd2dbd8f3dcc3c0c3cba3d863d1f3ea23d"
    "3b3db73d083df7bc9d3cc8bd43be19bef5bd91bd4ebe00bd65b9403e7f3dc5bcf6bb1b3d82bc883d25bd81bbe2bd553dc7bc673c0d3ce3bd74bc36bd"
    "ba3d213c9d3b2dbbee3d6a3d823d8abb1dbd4abd883d383d85bdd73dacbdc5bae2bd57bdd3bd32bec9bd05be06bd6a3cd93d333dbf3c8ebcb73dc73a"
    "11bd89bd213b0abc09bd663b5abccdbd0b3c623d753d603eb3bdc6bd813d9d3b2c3cb6bb9c3d3c3d853c96bb903d3fbdf33cd2ba0ebd25be05bd353d"
    "edbdd3bb5b3e843d0e3dd53ca93cefbc9cbc633d463b57bd033cc8bda13dbe3dc4bd2f3e41bdeabd0f3dc9bd7a3cc13df8bd5b3c263d223ed83dcd3d"
    "353d8ebdc53d673d0dbeccbda1bd1bbd4abdfbbca7bd3bbd00bd2bbc4d3c70bc74bdf13dd7bc2b3e31bd4abcd0bdb8bd9fbce4bba13b95bd2d3e51bd"
    "1c3e9cbddfbae73d37bdddbce5bcaebae7bd39bda3bd0ebc0f3c83bd85bc34bbcf3cb53cc93c27bbabbdd6bd923c013ea43d3a3d2ebd643da13de33d"
    "2e3c8dbd943c9ebdc63d92bb9ebd673d14bd283d8fbce43d323d883c493c5d3d66b85cbdafbdb1bd2d3c373d873d4abda5bdb9bcb8bd71bd76bd14be"
    "443b083daf3cdf3d743d91bd913ddf3d3cbd2d3d53bd1cbe95bd15be88bdf53b8c3c763b9e3de33d2e3e82bc3e3d9ebd8dbd073ce3bc483d353d023e"
    "4cbbb83d0b3e3c3e0abe74bd04be0fbdaabdd6bd3a3bf93a573c5f3e0c3d283dcc3d2b3d043eae3ca8bd90bd0abcb33dbebdc9bdee3cfbbdbd397dbd"
    "243dbe3df13d203d84bd043dc0bdb0bca6bd06bd313dacbdf7bc05be263ec93dc03dd73c05bdad3d32bbcfbd31be5cbd84bda7bdd43d1d3d143e6a3d"
    "053ead3b57bd173dd0bb83bdf73ce3bd773e323e99bbea3d793d5bbd90bce53d07bed2bc79bd98bd81bc713d41bd2dbc153d443c493e2b3e29be53bd"
    "40be02bed83d13bdba3da2bc7f3d2bbded3dde3dba3cdcbd9dbc533e18beacbd54bd8fbb003db2bdc63df5bc033d763e1f3e13bcd6bc47bd33bd27be"
    "ae3d323d183dc0bdbfbd50bd89bcc43cfebdacbdbabcf8bcea3cce3d22be963d093e153e2c3d2d3b2abcdc3d0cbc48bdb5bd1d3d40bdddbdb7bce23d"
    "063e123b"
)


def _setup():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith(("Ascend950", "Ascend910_93")):
        pytest.skip("HiSparse integration requires Ascend 950 or 910_93")
    torch.npu.set_device(0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("tokens", [17, 131])
def test_fused_preprocess_writes_split_planes(monkeypatch, capture, tokens):
    _setup()
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.distributed.kv_transfer.hisparse.layout import cache_components, split_paged_cache

    width = 576
    row = torch.frombuffer(bytearray.fromhex(_ROUNDING_ROW_HEX), dtype=torch.bfloat16).clone()
    values = torch.empty(tokens, 2624, dtype=torch.bfloat16, device="npu")[:, 2048:]
    source = torch.cat((row, torch.arange(64).to(torch.bfloat16) / 64)).expand(tokens, -1).clone()
    values.copy_(source.to("npu"))
    cos = torch.ones(tokens, 1, 1, 64, dtype=values.dtype, device="npu")
    sin = torch.zeros_like(cos)
    gamma = torch.ones(512, dtype=values.dtype, device="npu")
    local_slots = torch.arange(tokens, dtype=torch.int64, device="npu")
    reference_k = torch.empty((tokens + 127) // 128, 128, 1, 512, dtype=values.dtype, device="npu")
    reference_r = torch.empty((tokens + 127) // 128, 128, 1, 64, dtype=values.dtype, device="npu")

    def native():
        torch_npu.npu_kv_rmsnorm_rope_cache(
            values.view(tokens, 1, 1, width),
            gamma,
            cos,
            sin,
            local_slots,
            reference_r,
            reference_k,
            epsilon=1e-5,
            cache_mode="PA",
        )
        return torch.cat((reference_k.view(-1, 512)[:tokens], reference_r.view(-1, 64)[:tokens]), -1).cpu()

    reference = native()
    old_split, _ = torch_npu.npu_rms_norm(values[:, :512], gamma, epsilon=1e-5)
    if torch.npu.get_device_name(0).startswith("Ascend950"):
        assert not torch.equal(old_split.cpu(), reference[:, :512])
    raw = torch.empty(4 * 6 * 128 * width * 2, dtype=torch.int8, device="npu")
    backing = raw.view(values.dtype)
    view = split_paged_cache(
        raw,
        dtype=values.dtype,
        row_width=width,
        byte_offset=2 * 128 * width * 2,
        block_stride=6 * 128 * width * 2,
        num_blocks=4,
        block_size=128,
    )
    slots_cpu = 256 + torch.arange(tokens, dtype=torch.int64)
    slots_cpu[::4] = -1
    slots = slots_cpu.to("npu")
    ends = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    mirrored = []
    handle = SimpleNamespace(
        view=view,
        write_target=lambda count, slot_count: (view.cache, slots, count),
        finish_kv_update=lambda: mirrored.append(True),
    )
    impl = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_kv_heads=1,
        kv_a_layernorm=SimpleNamespace(weight=gamma, variance_epsilon=1e-5),
        hisparse_cache=handle,
    )

    def forward():
        return AscendHiSparseImpl.exec_kv(impl, values, cos, sin, None, slots, SimpleNamespace(query_start_loc=ends))

    def verify(valid):
        expected_backing = torch.full_like(backing, -11).cpu()
        offset = 0
        for component in cache_components(view.cache):
            expected = expected_backing.as_strided(component.shape, component.stride(), component.storage_offset())
            size = component.shape[-1]
            for row_id, slot in enumerate(slots_cpu.tolist()):
                if row_id < valid and slot >= 0:
                    expected[slot // 128, slot % 128].copy_(reference[row_id, offset : offset + size])
            offset += size
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)

    graph = None
    try:
        backing.fill_(-11)
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                forward()
                forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                result = forward()
        else:
            result = forward()
        assert result == (None, None) and mirrored
        verify(tokens)
        for iteration in range(3):
            values.copy_((source * (iteration + 2)).to("npu"))
            reference = native()
            slots_cpu = 128 + torch.arange(tokens, dtype=torch.int64)
            slots_cpu[1::4] = -1
            slots.copy_(slots_cpu)
            valid = tokens - iteration - 1
            ends[1].fill_(valid)
            backing.fill_(-11)
            if capture:
                graph.replay()
            else:
                forward()
            verify(valid)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
