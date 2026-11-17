from __future__ import annotations

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


@cache_once
def _jit_exact_topk_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "dsa_exact_topk",
        *args,
        cuda_files=["dsa/exact_topk.cuh"],
        cuda_wrappers=[("exact_topk", f"dsa::ExactTopKKernel<{args}>::run")],
    )


def exact_topk(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    *,
    k: int,
    force_last: int = 128,
    offsets: torch.Tensor | None = None,
    order_mode: str | None = None,
    out_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select logical positions from each row's valid score prefix.

    Reserve the last valid positions within the output width with force_last.
    Select the remaining positions by score, favoring the earlier position on
    ties. Write the positions in ascending order, add the per-row offsets if
    supplied, and fill unused entries with -1.

    FP32 treats signed zeros as equal. BF16 ranks sign-ordered raw bits.
    order_mode, when given, must be "fp32" or "bf16_bits" to match the input
    dtype. Scores have unit inner stride. Lengths and offsets are contiguous
    int32 vectors. Every length is between zero and the score width.
    """
    if not (0 < k < (1 << 31) and 0 <= force_last <= k and k - force_last <= 8192):
        raise ValueError(
            "require positive int32 k, 0 <= force_last <= k, and k - force_last <= 8192"
        )
    if order_mode is not None and scores.dtype != {
        "fp32": torch.float32,
        "bf16_bits": torch.bfloat16,
    }.get(order_mode):
        raise ValueError("order_mode must match the FP32 or BF16 score dtype")
    output_shape = (scores.shape[0], k)
    if out_indices is None:
        out_indices = scores.new_empty(output_shape, dtype=torch.int32)
    elif out_indices.shape != output_shape:
        raise ValueError("out_indices must have shape (scores.shape[0], k)")
    _jit_exact_topk_module().exact_topk(scores, lengths, out_indices, offsets, force_last)
    return out_indices
