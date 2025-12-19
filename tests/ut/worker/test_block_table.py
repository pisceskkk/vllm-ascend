#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for BlockTable.compute_slot_mapping function."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.block_table import BlockTable


def compute_expected_slot_mapping(req_indices: np.ndarray,
                                   positions: np.ndarray,
                                   block_table: np.ndarray,
                                   block_size: int,
                                   max_num_blocks_per_req: int,
                                   blocks_per_phys_block: int,
                                   dcp_world_size: int,
                                   pcp_world_size: int,
                                   dcp_rank: int,
                                   pcp_rank: int,
                                   cp_kv_cache_interleave_size: int) -> np.ndarray:
    """
    Compute expected slot mapping based on the algorithm in BlockTable.compute_slot_mapping.
    
    This function replicates the logic of compute_slot_mapping to generate
    expected values for test verification.
    """
    if dcp_world_size * pcp_world_size > 1:
        # Multi-GPU scenario
        virtual_block_size = block_size * dcp_world_size * pcp_world_size
        logical_block_idx = positions // virtual_block_size
        block_table_indices = (req_indices * max_num_blocks_per_req *
                               blocks_per_phys_block + logical_block_idx)
        
        # Flatten block_table and get block numbers
        block_numbers = block_table.ravel()[block_table_indices]
        
        virtual_block_offsets = positions % virtual_block_size
        current_rank = dcp_world_size * pcp_rank + dcp_rank
        
        # Calculate mask for local tokens
        mask = (virtual_block_offsets // cp_kv_cache_interleave_size %
                (dcp_world_size * pcp_world_size) == current_rank)
        
        # Calculate local block_offsets
        block_offsets = (virtual_block_offsets //
                        (dcp_world_size * pcp_world_size * cp_kv_cache_interleave_size) *
                        cp_kv_cache_interleave_size +
                        virtual_block_offsets % cp_kv_cache_interleave_size)
        
        # Calculate slot_mapping
        slot_mapping = block_numbers * block_size + block_offsets
        
        # Use -1 for non-local tokens
        return np.where(mask, slot_mapping, -1)
    else:
        # Single GPU scenario
        logical_block_idx = positions // block_size
        block_table_indices = (req_indices * max_num_blocks_per_req *
                               blocks_per_phys_block + logical_block_idx)
        
        block_numbers = block_table.ravel()[block_table_indices]
        block_offsets = positions % block_size
        
        return block_numbers * block_size + block_offsets


@pytest.mark.parametrize(
    "dcp_size, pcp_size, interleave_size",
    [
        # Single GPU scenarios
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 4),
        # Data parallel scenarios (dcp_size > 1)
        (2, 1, 1),
        (2, 1, 2),
        (2, 1, 4),
        (4, 1, 1),
        (4, 1, 2),
        # Pipeline parallel scenarios (pcp_size > 1)
        (1, 2, 1),
        (1, 2, 2),
        (1, 2, 4),
        (1, 4, 1),
        (1, 4, 2),
        # Combined parallel scenarios
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 4),
        (4, 2, 1),
        (2, 4, 2),
    ]
)
def test_compute_slot_mapping_various_configs(dcp_size: int, pcp_size: int,
                                               interleave_size: int):
    """
    Test compute_slot_mapping with various dcp_size, pcp_size, and interleave_size.
    
    This test verifies that the slot mapping is correctly computed for different
    parallel configurations and interleave sizes.
    """
    # Test parameters
    block_size = 16
    max_num_reqs = 8
    max_num_blocks_per_req = 32
    max_num_batched_tokens = 256
    device = torch.device("cpu")
    pin_memory = False
    
    # Test with multiple DCP/PCP ranks when world_size > 1
    ranks_to_test = []
    if dcp_size * pcp_size > 1:
        # Test each rank in the world
        for pcp_rank in range(pcp_size):
            for dcp_rank in range(dcp_size):
                ranks_to_test.append((dcp_rank, pcp_rank))
    else:
        # Single GPU case
        ranks_to_test = [(0, 0)]
    
    for dcp_rank, pcp_rank in ranks_to_test:
        # Mock the DCP/PCP group functions
        with patch('vllm_ascend.worker.block_table.get_dcp_group') as mock_dcp, \
             patch('vllm_ascend.worker.block_table.get_pcp_group') as mock_pcp:
            
            # Setup mock DCP group
            mock_dcp_group = MagicMock()
            mock_dcp_group.world_size = dcp_size
            mock_dcp_group.rank_in_group = dcp_rank
            mock_dcp.return_value = mock_dcp_group
            
            # Setup mock PCP group
            mock_pcp_group = MagicMock()
            mock_pcp_group.world_size = pcp_size
            mock_pcp_group.rank_in_group = pcp_rank
            mock_pcp.return_value = mock_pcp_group
            
            # Create BlockTable instance
            block_table = BlockTable(
                block_size=block_size,
                max_num_reqs=max_num_reqs,
                max_num_blocks_per_req=max_num_blocks_per_req,
                max_num_batched_tokens=max_num_batched_tokens,
                pin_memory=pin_memory,
                device=device,
                kernel_sizes=[block_size],
                cp_kv_cache_interleave_size=interleave_size,
            )
            
            # Setup block table with some block IDs
            # Simulate a scenario with 3 requests
            num_test_reqs = 3
            for req_idx in range(num_test_reqs):
                # Each request has some blocks allocated
                # For simplicity, assign sequential block IDs
                block_ids = list(range(req_idx * 5, req_idx * 5 + 5))
                block_table.add_row(block_ids, req_idx)
            
            # Test case: Multiple tokens from different requests at various positions
            req_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2])
            positions = np.array([0, 5, 15, 2, 20, 8, 16, 31])
            
            # Compute slot mapping using the function
            block_table.compute_slot_mapping(req_indices, positions)
            actual_slot_mapping = block_table.slot_mapping.np[:len(req_indices)].copy()
            
            # Compute expected slot mapping
            expected_slot_mapping = compute_expected_slot_mapping(
                req_indices=req_indices,
                positions=positions,
                block_table=block_table.block_table.np,
                block_size=block_size,
                max_num_blocks_per_req=max_num_blocks_per_req,
                blocks_per_phys_block=block_table.blocks_per_phys_block,
                dcp_world_size=dcp_size,
                pcp_world_size=pcp_size,
                dcp_rank=dcp_rank,
                pcp_rank=pcp_rank,
                cp_kv_cache_interleave_size=interleave_size,
            )
            
            # Verify the results
            np.testing.assert_array_equal(
                actual_slot_mapping,
                expected_slot_mapping,
                err_msg=f"Slot mapping mismatch for dcp_size={dcp_size}, "
                        f"pcp_size={pcp_size}, interleave_size={interleave_size}, "
                        f"dcp_rank={dcp_rank}, pcp_rank={pcp_rank}"
            )


@pytest.mark.parametrize(
    "block_size, interleave_size",
    [
        (8, 1),
        (8, 2),
        (16, 1),
        (16, 2),
        (16, 4),
        (32, 1),
        (32, 4),
        (32, 8),
    ]
)
def test_compute_slot_mapping_different_block_sizes(block_size: int,
                                                     interleave_size: int):
    """
    Test compute_slot_mapping with different block sizes and interleave sizes.
    
    This test ensures the function works correctly with various block size
    configurations.
    """
    # Fixed parameters
    dcp_size = 2
    pcp_size = 2
    dcp_rank = 0
    pcp_rank = 0
    max_num_reqs = 4
    max_num_blocks_per_req = 16
    max_num_batched_tokens = 128
    device = torch.device("cpu")
    pin_memory = False
    
    # Mock the DCP/PCP group functions
    with patch('vllm_ascend.worker.block_table.get_dcp_group') as mock_dcp, \
         patch('vllm_ascend.worker.block_table.get_pcp_group') as mock_pcp:
        
        # Setup mock DCP group
        mock_dcp_group = MagicMock()
        mock_dcp_group.world_size = dcp_size
        mock_dcp_group.rank_in_group = dcp_rank
        mock_dcp.return_value = mock_dcp_group
        
        # Setup mock PCP group
        mock_pcp_group = MagicMock()
        mock_pcp_group.world_size = pcp_size
        mock_pcp_group.rank_in_group = pcp_rank
        mock_pcp.return_value = mock_pcp_group
        
        # Create BlockTable instance
        block_table = BlockTable(
            block_size=block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            kernel_sizes=[block_size],
            cp_kv_cache_interleave_size=interleave_size,
        )
        
        # Setup block table with block IDs
        num_test_reqs = 2
        for req_idx in range(num_test_reqs):
            block_ids = list(range(req_idx * 4, req_idx * 4 + 4))
            block_table.add_row(block_ids, req_idx)
        
        # Test case
        req_indices = np.array([0, 0, 1, 1])
        positions = np.array([0, block_size // 2, block_size, block_size * 2])
        
        # Compute slot mapping
        block_table.compute_slot_mapping(req_indices, positions)
        actual_slot_mapping = block_table.slot_mapping.np[:len(req_indices)].copy()
        
        # Compute expected slot mapping
        expected_slot_mapping = compute_expected_slot_mapping(
            req_indices=req_indices,
            positions=positions,
            block_table=block_table.block_table.np,
            block_size=block_size,
            max_num_blocks_per_req=max_num_blocks_per_req,
            blocks_per_phys_block=block_table.blocks_per_phys_block,
            dcp_world_size=dcp_size,
            pcp_world_size=pcp_size,
            dcp_rank=dcp_rank,
            pcp_rank=pcp_rank,
            cp_kv_cache_interleave_size=interleave_size,
        )
        
        # Verify the results
        np.testing.assert_array_equal(
            actual_slot_mapping,
            expected_slot_mapping,
            err_msg=f"Slot mapping mismatch for block_size={block_size}, "
                    f"interleave_size={interleave_size}"
        )


def test_compute_slot_mapping_edge_cases():
    """
    Test compute_slot_mapping with edge cases such as empty requests,
    single token, and boundary positions.
    """
    block_size = 16
    max_num_reqs = 4
    max_num_blocks_per_req = 8
    max_num_batched_tokens = 64
    device = torch.device("cpu")
    pin_memory = False
    dcp_size = 1
    pcp_size = 1
    interleave_size = 1
    
    # Mock the DCP/PCP group functions
    with patch('vllm_ascend.worker.block_table.get_dcp_group') as mock_dcp, \
         patch('vllm_ascend.worker.block_table.get_pcp_group') as mock_pcp:
        
        # Setup mocks for single GPU
        mock_dcp_group = MagicMock()
        mock_dcp_group.world_size = dcp_size
        mock_dcp_group.rank_in_group = 0
        mock_dcp.return_value = mock_dcp_group
        
        mock_pcp_group = MagicMock()
        mock_pcp_group.world_size = pcp_size
        mock_pcp_group.rank_in_group = 0
        mock_pcp.return_value = mock_pcp_group
        
        # Create BlockTable instance
        block_table = BlockTable(
            block_size=block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            kernel_sizes=[block_size],
            cp_kv_cache_interleave_size=interleave_size,
        )
        
        # Test case 1: Single token at position 0
        block_table.add_row([10], 0)
        req_indices = np.array([0])
        positions = np.array([0])
        
        block_table.compute_slot_mapping(req_indices, positions)
        actual = block_table.slot_mapping.np[:1]
        expected = compute_expected_slot_mapping(
            req_indices, positions, block_table.block_table.np,
            block_size, max_num_blocks_per_req, block_table.blocks_per_phys_block,
            dcp_size, pcp_size, 0, 0, interleave_size
        )
        np.testing.assert_array_equal(actual, expected)
        
        # Test case 2: Token at block boundary
        req_indices = np.array([0])
        positions = np.array([block_size - 1])  # Last position in first block
        
        block_table.compute_slot_mapping(req_indices, positions)
        actual = block_table.slot_mapping.np[:1]
        expected = compute_expected_slot_mapping(
            req_indices, positions, block_table.block_table.np,
            block_size, max_num_blocks_per_req, block_table.blocks_per_phys_block,
            dcp_size, pcp_size, 0, 0, interleave_size
        )
        np.testing.assert_array_equal(actual, expected)
        
        # Test case 3: Multiple tokens spanning multiple blocks
        block_table.add_row([20, 21, 22], 1)
        req_indices = np.array([1, 1, 1, 1])
        positions = np.array([0, block_size - 1, block_size, block_size * 2])
        
        block_table.compute_slot_mapping(req_indices, positions)
        actual = block_table.slot_mapping.np[:4]
        expected = compute_expected_slot_mapping(
            req_indices, positions, block_table.block_table.np,
            block_size, max_num_blocks_per_req, block_table.blocks_per_phys_block,
            dcp_size, pcp_size, 0, 0, interleave_size
        )
        np.testing.assert_array_equal(actual, expected)


def test_compute_slot_mapping_interleave_patterns():
    """
    Test slot mapping with different interleave patterns to ensure correct
    token distribution across ranks.
    """
    block_size = 16
    max_num_reqs = 4
    max_num_blocks_per_req = 16
    max_num_batched_tokens = 128
    device = torch.device("cpu")
    pin_memory = False
    
    # Test with dcp_size=2, pcp_size=2, and interleave_size=2
    dcp_size = 2
    pcp_size = 2
    interleave_size = 2
    
    # Test each rank to verify correct masking
    for pcp_rank in range(pcp_size):
        for dcp_rank in range(dcp_size):
            with patch('vllm_ascend.worker.block_table.get_dcp_group') as mock_dcp, \
                 patch('vllm_ascend.worker.block_table.get_pcp_group') as mock_pcp:
                
                mock_dcp_group = MagicMock()
                mock_dcp_group.world_size = dcp_size
                mock_dcp_group.rank_in_group = dcp_rank
                mock_dcp.return_value = mock_dcp_group
                
                mock_pcp_group = MagicMock()
                mock_pcp_group.world_size = pcp_size
                mock_pcp_group.rank_in_group = pcp_rank
                mock_pcp.return_value = mock_pcp_group
                
                block_table = BlockTable(
                    block_size=block_size,
                    max_num_reqs=max_num_reqs,
                    max_num_blocks_per_req=max_num_blocks_per_req,
                    max_num_batched_tokens=max_num_batched_tokens,
                    pin_memory=pin_memory,
                    device=device,
                    kernel_sizes=[block_size],
                    cp_kv_cache_interleave_size=interleave_size,
                )
                
                # Setup blocks
                block_table.add_row([100, 101, 102], 0)
                
                # Test with consecutive positions to verify interleave pattern
                # With interleave_size=2 and world_size=4, the pattern repeats every 8 positions
                num_positions = 32
                req_indices = np.zeros(num_positions, dtype=np.int32)
                positions = np.arange(num_positions, dtype=np.int32)
                
                block_table.compute_slot_mapping(req_indices, positions)
                actual = block_table.slot_mapping.np[:num_positions].copy()
                
                expected = compute_expected_slot_mapping(
                    req_indices, positions, block_table.block_table.np,
                    block_size, max_num_blocks_per_req, block_table.blocks_per_phys_block,
                    dcp_size, pcp_size, dcp_rank, pcp_rank, interleave_size
                )
                
                np.testing.assert_array_equal(
                    actual, expected,
                    err_msg=f"Interleave pattern mismatch for dcp_rank={dcp_rank}, "
                            f"pcp_rank={pcp_rank}"
                )
                
                # Verify that non-local tokens are marked as -1
                current_rank = dcp_size * pcp_rank + dcp_rank
                for pos in range(num_positions):
                    virtual_block_offset = pos % (block_size * dcp_size * pcp_size)
                    expected_rank = (virtual_block_offset // interleave_size) % (dcp_size * pcp_size)
                    
                    if expected_rank != current_rank:
                        assert actual[pos] == -1, \
                            f"Position {pos} should be -1 for rank {current_rank}, got {actual[pos]}"
