import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType, get_mrv2_in_profile_run
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class TestNPUModelRunnerV2(unittest.TestCase):
    @staticmethod
    def _make_runner(max_num_tokens: int = 16):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.max_num_tokens = max_num_tokens
        runner.vllm_config = MagicMock()
        return runner

    def test_profile_run_marks_only_mc2_warmup_dummy_run(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=8),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(8, True), (16, True)])
        self.assertFalse(get_mrv2_in_profile_run())

    def test_profile_run_keeps_normal_dummy_run_outside_profile_override(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=32),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(16, True)])

    def test_prepare_inputs_rejects_cp_spec_decode(self):
        runner = self._make_runner(max_num_tokens=16)
        runner.pcp_size = 2
        runner.dcp_size = 1
        runner._update_seq_lens_cpu = MagicMock()
        scheduler_output = MagicMock(
            total_num_scheduled_tokens=4,
            num_scheduled_tokens={"req-0": 4},
            scheduled_spec_decode_tokens={"req-0": [1]},
        )
        batch_desc = MagicMock(num_tokens=4, num_reqs=1, cg_mode=CUDAGraphMode.NONE)

        with self.assertRaisesRegex(NotImplementedError, "spec decode"):
            runner.prepare_inputs(scheduler_output, batch_desc)

    @patch("vllm_ascend.worker.v2.model_states.default.build_attn_metadata", return_value={"ok": True})
    def test_prepare_attn_populates_cp_metadata(self, mock_build_attn_metadata):
        state = AscendModelState.__new__(AscendModelState)
        state.max_model_len = 128

        pcp_manager = MagicMock()
        pcp_manager.pcp_padded_slot_mapping_list = []
        pcp_manager.initialize_slot_mapping.side_effect = (
            lambda: pcp_manager.pcp_padded_slot_mapping_list.append(torch.full((4,), -1, dtype=torch.int32))
        )
        pcp_manager.generate_pcp_metadata.return_value = ("pcp-meta", torch.ones((2, 4), dtype=torch.int32))
        pcp_manager.get_padded_slot_mapping.return_value = torch.full((4,), -1, dtype=torch.int32)
        state.model_runner = MagicMock(use_cp=True, pcp_size=2, pcp_manager=pcp_manager)

        input_batch = MagicMock(
            num_reqs=2,
            num_tokens=4,
            num_reqs_after_padding=2,
            num_tokens_after_padding=4,
            query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
            query_lens=torch.tensor([2, 2], dtype=torch.int32),
            num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
            dcp_local_seq_lens=torch.ones((2, 2, 1), dtype=torch.int32),
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            seq_lens_np=np.array([2, 2], dtype=np.int32),
            num_computed_tokens_cpu=torch.zeros((2,), dtype=torch.int32),
            positions=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            attn_state="decode",
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        block_tables = (torch.ones((2, 4), dtype=torch.int32),)
        slot_mappings = torch.ones((1, 4), dtype=torch.int32)

        ret = state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_mappings,
            attn_groups=[],
            kv_cache_config=MagicMock(),
        )

        self.assertEqual(ret, {"ok": True})
        pcp_manager.generate_pcp_metadata.assert_called_once()
        pcp_manager.get_padded_slot_mapping.assert_called_once()
        mock_build_attn_metadata.assert_called_once()
        self.assertEqual(
            mock_build_attn_metadata.call_args.kwargs["prefill_context_parallel_metadata"],
            "pcp-meta",
        )
