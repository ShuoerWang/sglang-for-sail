"""Regressions for the PPU complete-call ABI and routing semantics."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

from sglang.srt.layers.moe import mega_moe_ppu as ppu


@pytest.fixture
def runtime(monkeypatch):
    dg = ModuleType("deep_gemm")
    dg.get_symm_buffer_for_mega_moe = Mock(return_value=object())
    dg.transform_weights_for_mega_moe = Mock(side_effect=lambda a, b, **kw: (a, b))
    dg.mega_moe_pre_dispatch = Mock()
    dg.fp4_fp4_mega_moe = Mock(side_effect=lambda y, *a, **kw: y.fill_(2))
    monkeypatch.setitem(sys.modules, "deep_gemm", dg)
    group = object()
    monkeypatch.setattr(ppu, "get_parallel", lambda: SimpleNamespace(
        moe_ep_group=SimpleNamespace(device_group=group)))
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK", "64")
    ppu._SYMM_BUFFERS.clear()
    yield dg
    ppu._SYMM_BUFFERS.clear()


@pytest.fixture
def experts():
    layer = torch.nn.Module()
    for name, shape in (
        ("w13_weight", (2, 256, 256)),
        ("w13_weight_scale", (2, 256, 16)),
        ("w2_weight", (2, 512, 64)),
        ("w2_weight_scale", (2, 512, 4)),
    ):
        layer.register_parameter(name, torch.nn.Parameter(
            torch.zeros(shape, dtype=torch.uint8), requires_grad=False))
    layer.num_experts, layer.hidden_size = 8, 512
    layer.intermediate_size_per_partition = 128
    layer.with_bias, layer.num_fused_shared_experts = False, 0
    layer._mega_moe_weights_built = False
    layer.moe_runner_config = SimpleNamespace(
        params_dtype=torch.bfloat16, is_gated=True, activation="silu",
        apply_router_weight_on_input=False, gemm1_alpha=None, gemm1_clamp_limit=None)
    return layer


def forward(experts, n, **kwargs):
    ids = torch.zeros((n, 2), dtype=torch.int32) if n else None
    weights = torch.full((n, 2), 0.5, dtype=torch.bfloat16) if n else None
    return ppu.run_mega_routed_experts(
        experts, torch.zeros((n, 512), dtype=torch.bfloat16), ids, weights,
        num_tokens=n, top_k=2, **kwargs)


def test_transform_uses_raw_scales_once(runtime, experts):
    raw_scale = experts.w13_weight_scale.data
    ppu.prepare_mega_moe_weights(experts)
    ppu.prepare_mega_moe_weights(experts)
    runtime.transform_weights_for_mega_moe.assert_called_once()
    call = runtime.transform_weights_for_mega_moe.call_args
    assert call.args[0][1].dtype == torch.uint8
    assert call.args[0][1].data_ptr() == raw_scale.data_ptr()
    assert call.kwargs == {"activation": "swiglu"}


@pytest.mark.parametrize("n", [0, 1, 63, 64])
@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_complete_call_includes_empty_ranks_and_output_scaling(runtime, experts, n, scale):
    ppu.prepare_mega_moe_weights(experts)
    y = forward(
        experts,
        n,
        routed_scaling_factor=scale,
        activation_clamp=10.0,
        global_num_tokens=64,
    )
    pre = runtime.mega_moe_pre_dispatch.call_args
    assert len(pre.args) == 4
    assert pre.kwargs == {"num_tokens": n}
    assert pre.args[1].dtype == torch.int64
    assert pre.args[2].dtype == torch.float32
    assert pre.args[1].shape == (n, 2)
    compute = runtime.fp4_fp4_mega_moe.call_args
    assert compute.args[0].shape == (n, 512)
    assert compute.kwargs["recipe"] == (1, 1, 32)
    assert compute.kwargs["activation_clamp"] == 10.0
    assert "global_num_tokens" not in compute.kwargs
    torch.testing.assert_close(y, torch.full_like(y, 2 * scale))


@pytest.mark.parametrize("n", [0, 8])
def test_global_num_tokens_and_workspace_views(runtime, experts, n):
    def pre_dispatch(
        x, ids, weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights, num_tokens
    ):
        pass

    def compute(y, *args, global_num_tokens=None, **kwargs):
        y.fill_(2)

    views = SimpleNamespace(
        x=object(),
        x_sf=object(),
        topk_idx=object(),
        topk_weights=object(),
        num_ranks=2,
        num_max_tokens_per_rank=64,
    )
    runtime.get_symm_buffer_for_mega_moe.return_value = views
    runtime.mega_moe_pre_dispatch = create_autospec(pre_dispatch)
    runtime.fp4_fp4_mega_moe = create_autospec(compute, side_effect=compute)
    ppu.prepare_mega_moe_weights(experts)

    y = forward(experts, n, global_num_tokens=8)

    assert runtime.get_symm_buffer_for_mega_moe.call_args.kwargs["mma_type"] == "mxf4xmxf4"
    pre = runtime.mega_moe_pre_dispatch.call_args
    assert pre.args[3:] == (views.x, views.x_sf, views.topk_idx, views.topk_weights)
    assert pre.kwargs == {"num_tokens": n}
    assert runtime.fp4_fp4_mega_moe.call_args.kwargs["global_num_tokens"] == 8
    torch.testing.assert_close(y, torch.full_like(y, 2))


def test_workspace_reused_and_activation_separated(runtime, experts):
    ppu.prepare_mega_moe_weights(experts)
    forward(experts, 1)
    forward(experts, 0)
    assert runtime.get_symm_buffer_for_mega_moe.call_count == 1
    experts.moe_runner_config.activation = "situ"
    experts.moe_runner_config.gemm1_alpha = 3.0
    experts.moe_runner_config.gemm1_clamp_limit = 20.0
    forward(experts, 1)
    assert runtime.get_symm_buffer_for_mega_moe.call_count == 2
    assert runtime.get_symm_buffer_for_mega_moe.call_args.kwargs["mma_type"] == "fp4xfp4"
    kwargs = runtime.fp4_fp4_mega_moe.call_args.kwargs
    assert (kwargs["activation"], kwargs["situ_beta"], kwargs["situ_linear_beta"]) == (
        "situ", 3.0, 20.0)


def test_capacity_failure_precedes_collectives(runtime, experts):
    ppu.prepare_mega_moe_weights(experts)
    with pytest.raises(ValueError, match="capacity"):
        forward(experts, 65)
    runtime.get_symm_buffer_for_mega_moe.assert_not_called()


def test_old_api_and_fused_shared_experts_rejected(runtime, experts):
    del runtime.fp4_fp4_mega_moe
    with pytest.raises(RuntimeError, match="complete-call"):
        ppu.prepare_mega_moe_weights(experts)
    runtime.fp4_fp4_mega_moe = Mock()
    experts.num_fused_shared_experts = 1
    with pytest.raises(ValueError, match="shared experts"):
        ppu.prepare_mega_moe_weights(experts)
    runtime.transform_weights_for_mega_moe.assert_not_called()
