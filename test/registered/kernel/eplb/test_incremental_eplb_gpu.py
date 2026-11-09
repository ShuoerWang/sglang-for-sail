"""Device correctness for copies, load collection and replay diagnostics."""

import importlib.util
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.eplb.incremental import copy_engine
from sglang.srt.eplb.incremental import planner as planner_module
from sglang.srt.eplb.incremental.puller import ExpertPuller
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/PPU required"
)


def load_diagnostics(name):
    source = (
        Path(__file__).resolve().parents[4]
        / f"python/sglang/srt/eplb/incremental/{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


debug = load_diagnostics("balance_debug")


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "CUDA/PPU required"
)
class TestBalanceGraphReplay(unittest.TestCase):
    def test_external_events_and_counts_update_each_replay(self):
        planner = SimpleNamespace(
            ep_rank=0,
            npe=3,
            published_count=0,
            resident_p2l=torch.tensor([[0, 1, 1]], dtype=torch.int32, device="cuda"),
        )
        dispatch = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        with tempfile.TemporaryDirectory() as directory:
            collector = debug.BalanceDebug(directory, planner, dispatch)
            collector.layers = [0]
            collector.capture_key = "test"
            collector.batch_size = 1
            work = torch.zeros(1024, device="cuda")
            counts = torch.tensor([3, 0, 0], dtype=torch.int32, device="cuda")
            collector.counts["test"] = {0: torch.empty_like(counts)}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            names = (
                "dispatch_send_ms",
                "dispatch_receive_ms",
                "expert_compute_ms",
                "gemm0_ms",
                "gemm1_ms",
                "combine_send_ms",
                "combine_receive_ms",
            )
            with torch.cuda.stream(stream):
                for name in names:
                    collector.measure(0, name, work.add_, 1)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    collector.counts["test"][0].copy_(counts)
                    for name in names:
                        collector.measure(0, name, work.add_, 1)
            torch.cuda.current_stream().wait_stream(stream)
            collector.capture_key = None
            origin = torch.cuda.Event(enable_timing=True)
            puller = SimpleNamespace(
                _nstep=1,
                _active_generation=0,
                _debug_round=SimpleNamespace(events={"target_start": origin}),
            )
            for step, tokens in ((1, 3), (2, 7), (3, 11)):
                counts[0] = tokens
                origin.record()
                graph.replay()
                puller._nstep = step
                collector.after_replay("test", puller)
            rows = [
                json.loads(line) for line in collector.path.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [r["layers"][0]["total_tokens"] for r in rows[1:]], [3, 7, 11]
            )
            for row in rows[1:]:
                for name in names:
                    self.assertGreaterEqual(row["layers"][0][name], 0)
                    start = row["layers"][0][name[:-3] + "_start_ms"]
                    end = row["layers"][0][name[:-3] + "_end_ms"]
                    self.assertGreaterEqual(start, 0)
                    self.assertGreaterEqual(end, start)


def test_worker_copy_preserves_all_regions():
    assert copy_engine.is_available(), copy_engine.extension_error()
    device = torch.cuda.current_device()
    source = torch.arange(64, dtype=torch.uint8, device="cuda")
    target = torch.zeros_like(source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def copy():
        with torch.cuda.device(device):
            return copy_engine.issue_batch(
                [source.data_ptr(), source.data_ptr() + 32],
                [target.data_ptr(), target.data_ptr() + 32],
                [32, 32],
                stream=stream,
            )

    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(copy).result(timeout=30) == 2
    stream.synchronize()
    torch.testing.assert_close(target, source, rtol=0, atol=0)


def test_load_collection_and_generation_ring():
    portable = planner_module._load_ext(peer_memory=False)
    peer = planner_module._load_ext()
    assert portable is not None and peer is not None
    assert not hasattr(portable, "peer_gather")
    counts = torch.tensor([[[3, 4]], [[5, 6]]], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([[0, 1, 0, 1]], device="cuda", dtype=torch.int32)
    output = torch.empty((1, 2), device="cuda", dtype=torch.float32)
    portable.phys_to_logical(counts, mapping, output, 2)
    assert output.tolist() == [[8, 10]]
    plane = 37
    stride = ((32 + plane + 31) // 32) * 32
    region = torch.zeros(4 * stride, device="cuda", dtype=torch.int32)
    pointers = torch.tensor([region.data_ptr()], device="cuda", dtype=torch.int64)
    gathered = torch.empty((1, plane), device="cuda", dtype=torch.int32)
    for generation in range(1, 11):
        counts = torch.arange(plane, device="cuda", dtype=torch.int32) + generation
        peer.publish_counts(region, counts, generation)
        peer.peer_gather(pointers, gathered, plane, 1, generation)
        torch.testing.assert_close(gathered[0], counts, rtol=0, atol=0)


def test_publication_wait_runs_on_worker():
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    sequence = torch.zeros(3, dtype=torch.int32, pin_memory=True)
    stream = torch.cuda.Stream()
    planner = SimpleNamespace(
        L=1,
        npe=3,
        E=1,
        R=1,
        pull_width=1,
        spares_total=2,
        pull_list=torch.full((1, 1, 4), -1, dtype=torch.int32, pin_memory=True),
        pull_experts=torch.zeros((1, 1), dtype=torch.int32),
        pull_seq=sequence,
        plan_stream=stream,
        published_count=0,
        prepare_step=lambda: None,
        launch_plan=Mock(side_effect=lambda: assert_worker()),
    )
    tables = SimpleNamespace(
        num_local=3,
        num_homes=1,
        moe_layers=[0],
        peer_base=[[0]],
        local_base=[0],
        slot_offset=[((), (), ())],
        region_bytes=[()],
        bytes_per_expert=[0],
        ep_size=1,
    )
    puller = ExpertPuller(planner, tables)

    def assert_worker():
        assert threading.get_ident() != main_thread

    def publish():
        assert_worker()
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test publication not released")
        sequence.copy_(torch.tensor([1, -1, 0], dtype=torch.int32))

    try:
        puller.begin_step()
        planner.launch_plan.assert_not_called()
        planner.published_count = 1
        planner.plan_stream = Mock(synchronize=publish)
        puller.after_replay()
        assert entered.wait(5)
        assert not puller._pending.done()
    finally:
        release.set()
        puller._finish_pending()
        planner.plan_stream = stream
        puller.close()
    planner.launch_plan.assert_called_once()
    assert puller.stats["empty"] == 1


@pytest.mark.parametrize("published,expected", [(5, 1), (0, 1)])
def test_peer_generation_failure_is_bounded(published, expected):
    # Device assertions poison their CUDA context, so exercise them in a child.
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import torch
        from sglang.srt.eplb.incremental import planner
        ext = planner._load_ext()
        assert ext is not None
        plane, stride = 1, 64
        region = torch.zeros(4 * stride, device='cuda', dtype=torch.int32)
        if {published}:
            ext.publish_counts(region, torch.ones(1, device='cuda', dtype=torch.int32), {published})
        pointers = torch.tensor([region.data_ptr()], device='cuda', dtype=torch.int64)
        output = torch.empty((1, plane), device='cuda', dtype=torch.int32)
        ext.peer_gather(pointers, output, plane, 1, {expected})
        torch.cuda.synchronize()
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode != 0
    assert "Incremental EPLB peer generation" in result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
