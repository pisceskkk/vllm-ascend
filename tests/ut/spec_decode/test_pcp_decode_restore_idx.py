"""
Tests for the enter_fa_decode_restore_idx computation in PCP utils.

Verifies that the hybrid attention decode restore index correctly handles
MTP speculative decoding where each decode request may have multiple tokens
(decode_threshold > 1).
"""

import numpy as np
import pytest


def compute_enter_fa_decode_restore_idx(
    num_decode_reqs,
    num_scheduled_tokens,
    pcp_world_size,
    max_scheduled_tokens,
):
    """
    Replicates the enter_fa_decode_restore_idx computation from
    update_tokens_for_pcp in pcp_utils.py.
    """
    decode_tokens_per_req = num_scheduled_tokens[:num_decode_reqs]
    num_entries_per_group = np.repeat(decode_tokens_per_req, pcp_world_size)
    req_starts = np.zeros(num_decode_reqs, dtype=np.int64)
    if num_decode_reqs > 1:
        req_starts[1:] = np.cumsum(decode_tokens_per_req[:-1])
    rank_offsets = np.tile(
        np.arange(pcp_world_size, dtype=np.int64) * max_scheduled_tokens,
        num_decode_reqs,
    )
    req_offsets = np.repeat(req_starts, pcp_world_size)
    group_base = req_offsets + rank_offsets
    cumsum = np.cumsum(num_entries_per_group)
    total = cumsum[-1] if len(cumsum) > 0 else 0
    arange = np.arange(total, dtype=np.int64)
    offsets = np.zeros(total, dtype=np.int64)
    offsets[cumsum[:-1]] = num_entries_per_group[:-1]
    token_arange = arange - np.cumsum(offsets)
    group_starts = np.repeat(group_base, num_entries_per_group)
    return group_starts + token_arange


class TestEnterFaDecodeRestoreIdx:

    def test_single_req_single_token_pcp2(self):
        """Non-MTP case: 1 decode req, 1 token, pcp_size=2."""
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs=1,
            num_scheduled_tokens=np.array([1], dtype=np.int64),
            pcp_world_size=2,
            max_scheduled_tokens=1,
        )
        # After all_gather: [rank0_t0, rank1_t0]
        # Restore: [rank0_t0, rank1_t0]
        np.testing.assert_array_equal(idx, [0, 1])

    def test_two_reqs_single_token_pcp2(self):
        """Non-MTP case: 2 decode reqs, 1 token each, pcp_size=2."""
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs=2,
            num_scheduled_tokens=np.array([1, 1], dtype=np.int64),
            pcp_world_size=2,
            max_scheduled_tokens=2,
        )
        # After all_gather (2 per rank × 2 ranks = 4):
        #   rank0: [req0_t0, req1_t0] at indices [0, 1]
        #   rank1: [req0_t0, req1_t0] at indices [2, 3]
        # Restore: [req0_t0_r0, req0_t0_r1, req1_t0_r0, req1_t0_r1]
        np.testing.assert_array_equal(idx, [0, 2, 1, 3])

    def test_single_req_mtp_two_tokens_pcp2(self):
        """MTP case: 1 decode req, 2 tokens (decode_threshold=2), pcp_size=2."""
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs=1,
            num_scheduled_tokens=np.array([2], dtype=np.int64),
            pcp_world_size=2,
            max_scheduled_tokens=2,
        )
        # After all_gather (2 per rank × 2 ranks = 4):
        #   rank0: [req0_t0, req0_t1] at indices [0, 1]
        #   rank1: [req0_t0, req0_t1] at indices [2, 3]
        # Restore: [req0_t0_r0, req0_t1_r0, req0_t0_r1, req0_t1_r1]
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    def test_two_reqs_mtp_two_tokens_pcp2(self):
        """MTP case: 2 decode reqs, 2 tokens each, pcp_size=2."""
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs=2,
            num_scheduled_tokens=np.array([2, 2], dtype=np.int64),
            pcp_world_size=2,
            max_scheduled_tokens=4,
        )
        # After all_gather (4 per rank × 2 ranks = 8):
        #   rank0: [req0_t0, req0_t1, req1_t0, req1_t1] at indices [0-3]
        #   rank1: [req0_t0, req0_t1, req1_t0, req1_t1] at indices [4-7]
        # Restore: [req0_t0_r0, req0_t1_r0, req0_t0_r1, req0_t1_r1,
        #           req1_t0_r0, req1_t1_r0, req1_t0_r1, req1_t1_r1]
        np.testing.assert_array_equal(idx, [0, 1, 4, 5, 2, 3, 6, 7])

    def test_two_reqs_mtp_two_tokens_pcp4(self):
        """MTP case: 2 decode reqs, 2 tokens each, pcp_size=4."""
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs=2,
            num_scheduled_tokens=np.array([2, 2], dtype=np.int64),
            pcp_world_size=4,
            max_scheduled_tokens=4,
        )
        # Each request: 2 tokens × 4 ranks = 8 entries
        # Total: 16 entries
        assert len(idx) == 16
        # Check req0 tokens from each rank are consecutive
        # req0 from rank0: tokens at [0, 1]
        # req0 from rank1: tokens at [4, 5]
        # req0 from rank2: tokens at [8, 9]
        # req0 from rank3: tokens at [12, 13]
        np.testing.assert_array_equal(
            idx[:8], [0, 1, 4, 5, 8, 9, 12, 13]
        )
        # req1 from rank0: tokens at [2, 3]
        # req1 from rank1: tokens at [6, 7]
        # req1 from rank2: tokens at [10, 11]
        # req1 from rank3: tokens at [14, 15]
        np.testing.assert_array_equal(
            idx[8:], [2, 3, 6, 7, 10, 11, 14, 15]
        )

    def test_logits_indices_consistency_mtp(self):
        """
        Verify that logits_indices_pcp correctly maps to the right tokens
        after restoration for MTP with 2 decode reqs and 2 tokens each.
        """
        pcp_size = 2
        num_decode_reqs = 2
        num_scheduled_tokens = np.array([2, 2], dtype=np.int64)
        max_scheduled_tokens = 4

        # Compute restore idx
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs, num_scheduled_tokens, pcp_size, max_scheduled_tokens
        )

        # Simulate all_gather output
        # rank0: [req0_t0_r0, req0_t1_r0, req1_t0_r0, req1_t1_r0]
        # rank1: [req0_t0_r1, req0_t1_r1, req1_t0_r1, req1_t1_r1]
        all_gather = np.array([
            0, 1, 2, 3,  # rank 0
            10, 11, 12, 13,  # rank 1 (using 10+ to distinguish)
        ])

        # Apply restore
        restored = all_gather[idx]

        # Compute logits_indices_pcp (same as _calc_spec_decode_metadata)
        cu_tokens = np.cumsum(num_scheduled_tokens)  # [2, 4]
        num_pcp_pads = num_scheduled_tokens * pcp_size - num_scheduled_tokens  # [2, 2]
        cu_tokens_pcp = cu_tokens * pcp_size - num_pcp_pads  # [2, 6]
        num_draft_tokens = np.array([1, 1])
        num_sampled_tokens = num_draft_tokens + 1  # [2, 2]
        cu_num_sampled = np.cumsum(num_sampled_tokens)  # [2, 4]
        cumsums_offsets = np.repeat(cu_num_sampled - num_sampled_tokens, num_sampled_tokens)
        arange = np.arange(sum(num_sampled_tokens)) - cumsums_offsets
        logits_indices_pcp = np.repeat(cu_tokens_pcp - num_sampled_tokens, num_sampled_tokens) + arange

        # Get sampled hidden states
        sampled = restored[logits_indices_pcp]

        # req0: should get both tokens from rank0
        # logits_indices_pcp[0:2] should map to req0_t0_r0 and req0_t1_r0
        assert sampled[0] == 0  # req0_t0_r0
        assert sampled[1] == 1  # req0_t1_r0
        # req1: should get both tokens from rank0
        assert sampled[2] == 2  # req1_t0_r0
        assert sampled[3] == 3  # req1_t1_r0

    def test_logits_indices_hybrid_attn_no_spec_decode(self):
        """
        Verify the hybrid attention logits_indices computation (spec_decode
        is None case) correctly picks the last token for each decode request.
        """
        pcp_size = 2
        num_decode_reqs = 2
        num_scheduled_tokens = np.array([2, 2], dtype=np.int64)
        max_scheduled_tokens = 4

        # Compute restore idx
        idx = compute_enter_fa_decode_restore_idx(
            num_decode_reqs, num_scheduled_tokens, pcp_size, max_scheduled_tokens
        )

        # Simulate all_gather output
        all_gather = np.arange(8)  # [0, 1, 2, 3, 4, 5, 6, 7]

        # Apply restore
        restored = all_gather[idx]

        # Compute logits_indices for hybrid attn (spec_decode is None)
        tokens_original = np.array([2, 2])
        decode_pads = tokens_original[:num_decode_reqs] * (pcp_size - 1)
        tokens_logits = tokens_original + np.pad(decode_pads, (0, 0))
        logits_indices = np.cumsum(tokens_logits) - 1

        # Should pick the last token for each request
        # For req0: last token is req0_t1_r1 (rank1's copy of token 1)
        # For req1: last token is req1_t1_r1
        # In restored layout: [req0_t0_r0, req0_t1_r0, req0_t0_r1, req0_t1_r1,
        #                       req1_t0_r0, req1_t1_r0, req1_t0_r1, req1_t1_r1]
        assert logits_indices[0] == 3  # index 3 = req0_t1_r1
        assert logits_indices[1] == 7  # index 7 = req1_t1_r1
