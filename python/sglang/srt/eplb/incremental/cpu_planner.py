# Copyright 2026 SGLang Team. Licensed under the Apache License, Version 2.0.
"""Background CPU planning from authoritative GPU placement snapshots."""

import logging
import threading
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def load_cpu_planner():
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_incremental_cpu_planner",
        sources=[str(Path(__file__).parent / "csrc" / "cpu_planner.cpp")],
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        with_cuda=False,
        verbose=False,
    )


class CpuPlanner:
    def __init__(self, *, threads):
        if type(threads) is not int or not 1 <= threads <= 1024:
            raise ValueError("CPU planner threads must be in [1, 1024]")
        self.library = load_cpu_planner()
        self.threads = threads
        self.shape = None
        logging.getLogger(__name__).info("CPU incremental planner: %d threads", threads)

    def plan(
        self,
        tokens,
        resident,
        owners,
        ranks,
        budget,
        moves,
        ranks_per_node,
        split_numerator,
        split_denominator,
        merge_numerator,
        merge_denominator,
        gain_divisor,
    ):
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("CPU planner must execute in the background worker")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU planner cannot execute inside CUDA graph capture")
        tensors = (tokens, resident, owners)
        if tokens.ndim != 2 or resident.ndim != 2 or owners.ndim != 3:
            raise ValueError("Invalid planner dimensions")
        layers, experts = tokens.shape
        physical = resident.shape[1]
        if not (
            layers > 0
            and 0 < experts <= 1024
            and 0 < ranks <= 64
            and 0 < physical <= 1152
            and physical % ranks == 0
            and physical // ranks <= 64
        ):
            raise ValueError("Unsupported planner geometry")
        if resident.shape != (layers, physical) or owners.shape != (
            layers,
            experts,
            ranks,
        ):
            raise ValueError("Inconsistent planner geometry")
        if not all(t.is_cuda and t.device == tokens.device for t in tensors):
            raise ValueError("Planner inputs must share one CUDA device")
        if (
            tokens.dtype != torch.int64
            or resident.dtype != torch.int32
            or owners.dtype != torch.int32
        ):
            raise ValueError("Expected int64 counts and int32 layouts")
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("Contiguous planner inputs required")
        shape = (layers, experts, physical, ranks, tokens.device)
        if self.shape is None:
            self.inputs = [
                torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors
            ]
            self.outputs = [
                torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                for t in (resident, owners, resident)
            ]
            self.device_outputs = [
                torch.empty_like(t) for t in (resident, owners, resident)
            ]
            self.ready = torch.cuda.Event()
            self.shape = shape
        elif shape != self.shape:
            raise RuntimeError("CPU planner geometry changed")
        # Snapshots belong to the runtime's generation, not a second CPU state.
        for dst, src in zip(self.inputs, tensors):
            dst.copy_(src, non_blocking=True)
        self.ready.record(torch.cuda.current_stream())
        self.ready.synchronize()
        self.library.plan_into(
            *self.inputs,
            *self.outputs,
            ranks,
            budget,
            moves,
            ranks_per_node,
            split_numerator,
            split_denominator,
            merge_numerator,
            merge_denominator,
            gain_divisor,
            self.threads,
        )
        for dst, src in zip(self.device_outputs, self.outputs):
            dst.copy_(src, non_blocking=True)
        return self.device_outputs
