"""Startup resolution and validation of DeepSeek-V4.1 asymmetric PD."""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.deepseek_v4_hook import (
    resolve_deepseek_v41_asymmetric_pd,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _v41_config(**overrides):
    values = dict(
        model_type="deepseek_v41",
        num_hidden_layers=40,
        hidden_size=5120,
        n_routed_experts=384,
        hc_pre_from_prev_sublayer=True,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 23, 27, 31, 35],
        engram_layer_ids=[1, 13],
        window_size=128,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41EncoderOnlyPrefillArgs(CustomTestCase):
    def make_args(self, role, config=None, **fields):
        fields.setdefault("dsv41_encoder_only_prefill", True)
        args = ServerArgs(
            model_path="dummy",
            disaggregation_mode=role,
            **fields,
        )
        # Checkpoint metadata without loading weights or downloading a model.
        args._model_config = SimpleNamespace(
            hf_config=config or _v41_config(), context_len=1048576
        )
        parse_cuda_graph_config(args)
        return args

    def test_roles_accept_the_official_layout(self):
        args = self.make_args("prefill", enable_decoder_swa_bounded_replay=True)
        resolve_deepseek_v41_asymmetric_pd(args)
        self.assertTrue(resolving_view(args).dsv41_encoder_only_prefill)
        # Prefill never runs the decoder layers, so it must not split at them.
        self.assertFalse(resolving_view(args).enable_decoder_swa_bounded_replay)
        args = self.make_args("decode")
        resolve_deepseek_v41_asymmetric_pd(args)
        self.assertTrue(resolving_view(args).enable_decoder_swa_bounded_replay)
        resolve_deepseek_v41_asymmetric_pd(
            self.make_args("prefill", dp_size=2, tp_size=2, enable_dp_attention=True)
        )
        # Decode drafts from its own tail extend.
        resolve_deepseek_v41_asymmetric_pd(
            self.make_args("decode", speculative_algorithm="DSPARK")
        )

    def test_prefill_phase_runs_eager_on_both_roles(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                args = self.make_args(
                    role,
                    cuda_graph_backend_prefill=Backend.BREAKABLE,
                    cuda_graph_backend_decode=Backend.FULL,
                )
                resolve_deepseek_v41_asymmetric_pd(args)
                config = resolving_view(args).cuda_graph_config
                self.assertEqual(config.prefill.backend, Backend.DISABLED)
                self.assertEqual(config.decode.backend, Backend.FULL)

    def test_asymmetric_pd_is_opt_in(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        common = ["--model-path", "dummy"]
        self.assertFalse(parser.parse_args(common).dsv41_encoder_only_prefill)
        enabled = parser.parse_args(common + ["--dsv41-encoder-only-prefill"])
        self.assertTrue(enabled.dsv41_encoder_only_prefill)
        args = self.make_args(
            "prefill", dsv41_encoder_only_prefill=False, speculative_algorithm="DSPARK"
        )
        before = resolving_view(args).cuda_graph_config
        resolve_deepseek_v41_asymmetric_pd(args)
        self.assertIs(resolving_view(args).cuda_graph_config, before)

    def test_opt_in_requires_v41_flash_and_mooncake_pd(self):
        for args in (
            self.make_args("null"),
            self.make_args("prefill", _v41_config(model_type="deepseek_v4")),
            self.make_args("prefill", _v41_config(num_hidden_layers=48)),
            self.make_args("decode", _v41_config(hidden_size=7168)),
            self.make_args("decode", disaggregation_transfer_backend="nixl"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "--dsv41-encoder-only-prefill"):
                    resolve_deepseek_v41_asymmetric_pd(args)

    def test_rejects_unsupported_features(self):
        for role, fields, message in (
            ("prefill", {"speculative_algorithm": "DSPARK"}, "speculative"),
            ("decode", {"disaggregation_decode_enable_radix_cache": True}, "radix"),
            ("prefill", {"enable_pd_role_switch": True}, "role switching"),
        ):
            with self.subTest(role=role, **fields):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_deepseek_v41_asymmetric_pd(
                        self.make_args(role, **fields)
                    )

    def test_rejects_other_layouts(self):
        for config in (
            _v41_config(kv_source_layer_ids=[]),
            _v41_config(engram_layer_ids=[1, 20]),
            _v41_config(index_source_layer_ids=[2, 8, 14]),
            # A c4 layer would hold a pending block at the split.
            _v41_config(compress_ratios=[0, 0, 4] + [2] * 17 + [1] * 20 + [0] * 3),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    resolve_deepseek_v41_asymmetric_pd(
                        self.make_args("prefill", config)
                    )


if __name__ == "__main__":
    unittest.main()
