from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _li_c8_spec() -> AscendSFAIndexerCacheSpec:
    return AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        cache_dtype_str="auto",
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        sfa_dcp_replicated_indexer_size=1,
    )


def test_get_kv_cache_spec_includes_glm_li_c8_indexer():
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    indexer = MagicMock(spec=DeepseekV32IndexerCache)
    indexer.cache_config.block_size = 128
    config = MagicMock()
    config.model_config.hf_text_config.index_head_dim = 128
    config.model_config.dtype = torch.bfloat16
    config.cache_config.cache_dtype = "auto"
    config.parallel_config.decode_context_parallel_size = 1
    ascend_config = MagicMock()
    ascend_config.is_sparse_li_c8_layer.return_value = True

    with (
        patch(
            "vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config",
            return_value={"model.layers.0.self_attn.indexer.k_cache": indexer},
        ),
        patch(
            "vllm_ascend.worker.v2.attn_utils.get_ascend_config",
            return_value=ascend_config,
        ),
        patch(
            "vllm_ascend.worker.v2.attn_utils.enable_sfa_dcp_replicated_indexer",
            return_value=False,
        ),
    ):
        specs = get_kv_cache_spec(config)

    spec = specs["model.layers.0.self_attn.indexer.k_cache"]
    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.dtype == torch.int8
    assert spec.scale_dim == 1
    assert spec.scale_dtype == torch.float16


def test_li_c8_indexer_allocate_and_reshape_keeps_key_and_scale_handles():
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    spec = _li_c8_spec()
    group_spec = MagicMock()
    group_spec.kv_cache_spec = spec
    group_spec.layer_names = [layer_name]
    tensor_config = MagicMock()
    tensor_config.shared_by = [layer_name]
    tensor_config.size = 2 * spec.page_size_bytes
    cache_config = MagicMock()
    cache_config.kv_cache_groups = [group_spec]
    cache_config.kv_cache_tensors = [tensor_config]
    cache_config.num_blocks = 2
    vllm_config = MagicMock()
    vllm_config.kv_transfer_config = None

    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=vllm_config,
    ):
        raw = _allocate_kv_cache(cache_config, {}, torch.device("cpu"))

    assert len(raw[layer_name]) == 2
    assert raw[layer_name][0].numel() == 2 * 128 * 128
    assert raw[layer_name][1].numel() == 2 * 128 * 2

    attn_group = MagicMock()
    attn_group.kv_cache_group_id = 0
    attn_group.kv_cache_spec = spec
    attn_group.layer_names = [layer_name]
    attn_group.backend = AscendSFAIndexerBackend
    vllm_config.kv_transfer_config = None
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=vllm_config,
    ):
        caches = _reshape_kv_cache_v2(
            [attn_group], raw, "auto", [128], {}, cache_config
        )

    indexer_k, indexer_scale = caches[layer_name]
    assert indexer_k.shape == (2, 128, 1, 128)
    assert indexer_k.dtype == torch.int8
    assert indexer_scale.shape == (2, 128, 1, 1)
    assert indexer_scale.dtype == torch.float16
