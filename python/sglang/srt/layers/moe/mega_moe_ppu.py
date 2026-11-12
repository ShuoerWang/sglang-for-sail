# SPDX-License-Identifier: Apache-2.0
"""PPU MXFP4 MegaMoE using DeepGEMM 2.1's complete-call interface."""

from __future__ import annotations

from inspect import signature

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel


# Equal geometries execute sequentially; overlap is rejected by argument validation.
_SYMM_BUFFERS: dict = {}


def _activation(experts) -> str:
    activation = experts.moe_runner_config.activation
    if activation not in ("silu", "situ"):
        raise ValueError("PPU MegaMoE supports only SiLU and SiTU gated experts.")
    return "swiglu" if activation == "silu" else "situ"


def prepare_mega_moe_weights(experts) -> None:
    import deep_gemm

    if getattr(experts, "_mega_moe_weights_built", False):
        return
    names = (
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_mega_moe",
        "mega_moe_pre_dispatch",
        "fp4_fp4_mega_moe",
    )
    if not all(callable(getattr(deep_gemm, name, None)) for name in names):
        raise RuntimeError("PPU MegaMoE requires DeepGEMM's MXFP4 complete-call API.")
    config = experts.moe_runner_config
    if config.params_dtype != torch.bfloat16:
        raise ValueError("PPU MegaMoE requires BF16 activations.")
    if (
        experts.with_bias
        or experts.num_fused_shared_experts
        or not config.is_gated
        or config.apply_router_weight_on_input
    ):
        raise ValueError(
            "PPU MegaMoE requires bias-free gated routed experts, output router "
            "weights, and separately computed shared experts."
        )
    # Pass raw uint8 checkpoint scales directly, before generic preprocessing.
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        (experts.w13_weight.data, experts.w13_weight_scale.data),
        (experts.w2_weight.data, experts.w2_weight_scale.data),
        activation=_activation(experts),
    )
    experts.w13_weight.data, experts.w13_weight_scale.data = l1
    experts.w2_weight.data, experts.w2_weight_scale.data = l2
    experts.mega_l1_weights = l1
    experts.mega_l2_weights = l2
    experts._mega_moe_uses_workspace_views = (
        "buf_x" in signature(deep_gemm.mega_moe_pre_dispatch).parameters
    )
    experts._mega_moe_accepts_global_num_tokens = (
        "global_num_tokens" in signature(deep_gemm.fp4_fp4_mega_moe).parameters
    )
    experts._mega_moe_weights_built = True


def _get_symm_buffer(experts, device, capacity, top_k):
    import deep_gemm

    group = get_parallel().moe_ep_group.device_group
    activation = _activation(experts)
    key = (
        group,
        device,
        capacity,
        experts.num_experts,
        top_k,
        experts.hidden_size,
        experts.intermediate_size_per_partition,
        activation,
    )
    if key not in _SYMM_BUFFERS:
        _SYMM_BUFFERS[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts=experts.num_experts,
            num_max_tokens_per_rank=capacity,
            num_topk=top_k,
            hidden=experts.hidden_size,
            intermediate_hidden=experts.intermediate_size_per_partition,
            mma_type=(
                "mxf4xmxf4" if experts._mega_moe_uses_workspace_views else "fp4xfp4"
            ),
            activation=activation,
        )
    return _SYMM_BUFFERS[key]


def run_mega_routed_experts(
    experts,
    hidden_states,
    topk_ids,
    topk_weights,
    *,
    num_tokens,
    top_k,
    global_num_tokens: int | None = None,
    activation_clamp=None,
    routed_scaling_factor=1.0,
):
    import deep_gemm

    if not experts._mega_moe_weights_built:
        raise RuntimeError("PPU MegaMoE weights must be prepared at model load.")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("PPU MegaMoE requires BF16 hidden states.")
    capacity = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if capacity <= 0 or num_tokens > capacity:
        raise ValueError(f"PPU MegaMoE received {num_tokens} tokens; capacity={capacity}.")
    if num_tokens != hidden_states.shape[0]:
        raise ValueError("PPU MegaMoE token count must match the input rows.")
    buf = _get_symm_buffer(experts, hidden_states.device, capacity, top_k)
    if num_tokens == 0:
        topk_ids = torch.empty((0, top_k), device=hidden_states.device, dtype=torch.int64)
        topk_weights = torch.empty_like(topk_ids, dtype=torch.float32)
    dispatch_buffers = (
        (buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights)
        if experts._mega_moe_uses_workspace_views
        else (buf,)
    )
    deep_gemm.mega_moe_pre_dispatch(
        hidden_states.contiguous(),
        topk_ids.to(torch.int64).contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        *dispatch_buffers,
        num_tokens=num_tokens,
    )
    # Every EP rank participates, including ranks with zero input tokens.
    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    config = experts.moe_runner_config
    kwargs = {}
    if global_num_tokens is not None and experts._mega_moe_accepts_global_num_tokens:
        kwargs["global_num_tokens"] = min(
            global_num_tokens, buf.num_ranks * buf.num_max_tokens_per_rank
        )
    if _activation(experts) == "situ":
        kwargs["situ_beta"] = config.gemm1_alpha if config.gemm1_alpha is not None else 4.0
        kwargs["situ_linear_beta"] = (
            config.gemm1_clamp_limit if config.gemm1_clamp_limit is not None else 25.0
        )
    deep_gemm.fp4_fp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation=_activation(experts),
        activation_clamp=activation_clamp,
        **kwargs,
    )
    if routed_scaling_factor != 1.0:
        y.mul_(routed_scaling_factor)
    return y
