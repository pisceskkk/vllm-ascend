from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DSV4_MTP_SOURCE = (
    _REPO_ROOT / "vllm_ascend/models/deepseek_v4_mtp.py"
).read_text()


def test_dsv4_mtp_projection_layers_are_quantization_aware():
    assert "self.e_proj = ReplicatedLinear" in _DSV4_MTP_SOURCE
    assert "self.h_proj = ReplicatedLinear" in _DSV4_MTP_SOURCE


def test_dsv4_mtp_projection_scales_are_remapped_for_loading():
    assert '".e_proj."' in _DSV4_MTP_SOURCE
    assert '".h_proj."' in _DSV4_MTP_SOURCE
    assert '".weight_scale_inv"' in _DSV4_MTP_SOURCE
