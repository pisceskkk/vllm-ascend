/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"

#if __has_include("smem/device/smem_shm_aicore_base_api.h")
#include "smem/device/smem_shm_aicore_base_api.h"

namespace {
constexpr uint32_t KVPP_MTE_TILE_BYTES = 64 * 1024;
constexpr int32_t KVPP_MTE_EVENT_ID = 0;
} // namespace

extern "C" __global__ __aicore__ void kvpp_mte_copy_bytes(
    __gm__ uint8_t* source, __gm__ uint8_t* destination, uint64_t length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, KVPP_MTE_TILE_BYTES);
    AscendC::LocalTensor<uint8_t> local = buffer.Get<uint8_t>();
    __ubuf__ uint8_t* local_address =
        (__ubuf__ uint8_t*)local.GetPhyAddr();

    uint64_t offset = 0;
    while (offset < length) {
        const uint32_t bytes = static_cast<uint32_t>(
            (length - offset) > KVPP_MTE_TILE_BYTES
                ? KVPP_MTE_TILE_BYTES
                : (length - offset));
        smem_shm_copy_gm2ub<uint8_t>(local_address, source + offset,
                                     bytes, false);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(KVPP_MTE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(KVPP_MTE_EVENT_ID);
        smem_shm_copy_ub2gm<uint8_t>(destination + offset,
                                     local_address, bytes, false);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(KVPP_MTE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(KVPP_MTE_EVENT_ID);
        offset += bytes;
    }
}

namespace vllm_ascend {
void kvpp_mte_copy_impl(void* stream, void* source, void* destination,
                        uint64_t length)
{
    kvpp_mte_copy_bytes<<<1, nullptr, stream>>>(source, destination, length);
}
} // namespace vllm_ascend
#endif
