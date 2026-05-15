from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from vllm_ascend.spec_decode.eagle_proposer import AscendSpecDecodeBaseProposer


def _mtp_proposer(use_compress: bool = True):
    proposer = AscendSpecDecodeBaseProposer.__new__(
        AscendSpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.use_compress = use_compress
    proposer.use_cuda_graph = False
    proposer.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(
            has_full_cudagraphs=lambda: False)))
    return proposer


def test_mtp_shares_target_embeddings_when_compress_enabled():
    proposer = _mtp_proposer(use_compress=True)
    target_embed_tokens = nn.Embedding(8, 4)
    draft_embed_tokens = nn.Embedding(8, 4)
    target_model = SimpleNamespace(model=SimpleNamespace(
        embed_tokens=target_embed_tokens))
    proposer.model = SimpleNamespace(model=SimpleNamespace(
        embed_tokens=draft_embed_tokens))

    with patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group",
               return_value=SimpleNamespace(world_size=1)):
        proposer._maybe_share_embeddings(target_model)

    assert proposer.model.model.embed_tokens is target_embed_tokens


def test_mtp_shares_target_lm_head_with_layer_shared_heads():
    proposer = _mtp_proposer()
    target_model = nn.Module()
    target_model.lm_head = nn.Linear(4, 8, bias=False)

    class _DraftLayer(nn.Module):

        def __init__(self):
            super().__init__()
            self.shared_head = nn.Module()
            self.shared_head.head = nn.Linear(4, 8, bias=False)

    draft_model = nn.Module()
    draft_model.model = nn.Module()
    draft_model.model.layers = nn.ModuleDict({
        "0": _DraftLayer(),
        "1": _DraftLayer(),
    })
    proposer.model = draft_model

    proposer._maybe_share_lm_head(target_model)

    assert proposer.model.lm_head is target_model.lm_head
    for layer in proposer.model.model.layers.values():
        assert layer.shared_head.head is target_model.lm_head
