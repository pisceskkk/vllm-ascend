# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component planes over the existing shared HiSparse allocation.

Logical block IDs still own one slab across all cache groups. The slab's
latent and RoPE portions move into two dense planes of the same backing;
neither component acquires an independent block allocator or HBM budget.
Runtime views retain gaps between slabs, while attention sees dense planes
and scales block IDs by the common number of MLA pages per slab.
"""

import torch
from vllm.v1.hisparse.runtime import PagedCacheView

LATENT_WIDTH = 512
ROPE_WIDTH = 64
PACKED_WIDTH = LATENT_WIDTH + ROPE_WIDTH


def cache_components(cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rope = getattr(cache, "_hisparse_rope", None)
    return (cache,) if rope is None else (cache, rope)


def combine_components(components: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(components) == 2:
        components[0]._hisparse_rope = components[1]
    elif len(components) != 1:
        raise ValueError("Ascend HiSparse requires one packed or two split components")
    return components[0]


def split_host_cache(cache: torch.Tensor) -> torch.Tensor:
    """Bind empty host storage before its first write; do not convert payload."""
    if cache.shape[-1] != PACKED_WIDTH:
        return cache.view(-1, cache.shape[-1])
    rows = cache.numel() // PACKED_WIDTH
    flat = cache.view(-1)
    latent = flat[: rows * LATENT_WIDTH].view(rows, LATENT_WIDTH)
    rope = flat[rows * LATENT_WIDTH :].view(rows, ROPE_WIDTH)
    return combine_components((latent, rope))


def split_paged_cache(raw_tensor, *, dtype, row_width, byte_offset, block_stride, num_blocks, block_size):
    if row_width != PACKED_WIDTH:
        return PagedCacheView.bind(
            raw_tensor,
            dtype=dtype,
            row_width=row_width,
            byte_offset=byte_offset,
            block_stride=block_stride,
            num_blocks=num_blocks,
            block_size=block_size,
        )
    itemsize = dtype.itemsize
    page_bytes = block_size * PACKED_WIDTH * itemsize
    if block_stride % page_bytes or byte_offset % page_bytes:
        raise ValueError("Split HiSparse requires whole MLA pages in each shared slab")
    if byte_offset < 0 or byte_offset + page_bytes > block_stride:
        raise ValueError("Split HiSparse layer page must fit in its logical slab")
    flat = raw_tensor.view(dtype).view(-1)
    if flat.numel() * itemsize != num_blocks * block_stride:
        raise ValueError("Split HiSparse must retain the complete shared device backing")
    components, attention_components = [], []
    plane_start = 0
    for width in (LATENT_WIDTH, ROPE_WIDTH):
        plane_stride = block_stride * width // PACKED_WIDTH // itemsize
        offset = byte_offset * width // PACKED_WIDTH // itemsize
        plane_elements = num_blocks * plane_stride
        plane = flat[plane_start : plane_start + plane_elements]
        cache = torch.as_strided(
            plane,
            (num_blocks, block_size, width),
            (plane_stride, width, 1),
            storage_offset=plane.storage_offset() + offset,
        )
        attention = plane[offset:].view(-1, block_size, width)
        components.append(cache)
        attention_components.append(attention)
        plane_start += plane_elements
    return PagedCacheView(
        combine_components(tuple(components)),
        combine_components(tuple(attention_components)),
        block_size,
        block_stride // (PACKED_WIDTH * itemsize),
    )
