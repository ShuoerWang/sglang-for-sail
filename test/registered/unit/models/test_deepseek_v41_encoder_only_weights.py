import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.models.deepseek_v4 import (
    DeepseekV41EncoderBoundaryLayer,
    dsv41_encoder_only_prefill_weight_needed,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDsv41EncoderOnlyWeights(CustomTestCase):
    def test_weight_filter_keeps_only_boundary_producers(self):
        boundary = 20
        kept = (
            "model.embed_tokens.weight",
            "model.layers.19.mlp.experts.w2_weight",
            "model.layers.20.input_layernorm.weight",
            "model.layers.20.self_attn.compressor.wkv.weight",
            "model.layers.20.self_attn.indexer.wk.weight",
            "model.layers.20.self_attn.indexer.k_norm.weight",
        )
        skipped = (
            "model.layers.20.post_attention_layernorm.weight",
            "model.layers.20.hc_attn_fn",
            "model.layers.20.self_attn.wq_a.weight",
            "model.layers.20.self_attn.indexer.wq_b.weight",
            "model.layers.20.self_attn.indexer.weights_proj.weight",
            "model.layers.20.mlp.experts.w2_weight",
            "model.layers.21.self_attn.wkv.weight",
            "model.norm.weight",
            "model.hc_head_fn",
            "lm_head.weight",
            "mtp.0.ffn.shared_experts.w1.weight",
            "model.mtp.layers.0.ffn.shared_experts.w1.weight",
        )
        for name in kept:
            self.assertTrue(dsv41_encoder_only_prefill_weight_needed(name, boundary))
        for name in skipped:
            self.assertFalse(dsv41_encoder_only_prefill_weight_needed(name, boundary))

    @patch("sglang.srt.models.deepseek_v4.get_attn_backend")
    def test_boundary_writes_kv_from_the_combined_input(self, backend):
        layer = object.__new__(DeepseekV41EncoderBoundaryLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = nn.Identity()
        layer.self_attn = SimpleNamespace(compress_ratio=1)
        residual = torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)
        normalized = torch.full((1, 16), 7.0)
        layer._hc_combine = Mock(return_value=normalized)
        positions = torch.zeros(1, dtype=torch.long)
        pre = torch.ones(1, 4)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        layer.write_global_cache_only(positions, residual, pre, batch)
        layer._hc_combine.assert_called_once_with(
            residual, apply_pre=pre, norm=layer.input_layernorm
        )
        backend.return_value.forward_low_ratio_sources.assert_called_once_with(
            layer=layer.self_attn,
            x=normalized,
            q_lora=None,
            positions=positions,
            forward_batch=batch,
            run_compressor=True,
            run_indexer=False,
        )


if __name__ == "__main__":
    unittest.main()
