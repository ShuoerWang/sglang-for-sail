import pytest
import torch

from sglang.kernels.ops.attention.dsa.exact_topk import exact_topk
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


def _reference_indices(scores, lengths, offsets, k, force_last):
    if scores.dtype == torch.bfloat16:
        bits = scores.contiguous().view(torch.int16).int() & 0xFFFF
        keys = torch.where(bits & 0x8000 != 0, bits ^ 0xFFFF, bits | 0x8000)
    else:
        keys = scores
    result = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for row, (length, offset) in enumerate(zip(lengths, offsets)):
        prefix = max(0, length - force_last)
        keep = min(k, length) - (length - prefix)
        ranked = keys[row, :prefix].argsort(descending=True, stable=True)
        picked = torch.cat((ranked[:keep].sort().values, torch.arange(prefix, length)))
        result[row, : picked.numel()] = picked.to(torch.int32) + offset
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k,force_last", [(2176, 128), (8320, 128), (8192, 0)])
@pytest.mark.parametrize("width", [12291, 131075])
@torch.inference_mode()
def test_exact_topk(dtype, k, force_last, width):
    rng = torch.Generator().manual_seed(371)
    lengths = [0, 127, k - 1, k + 1, width - 5, width - 6, width - 7, width]
    offsets = [37 * i for i in range(len(lengths))] if force_last else [0] * len(lengths)
    scores = torch.randn((len(lengths), width), generator=rng).to(dtype)
    # Signed zeros compete for one score-selected position on row 3.
    scores[3].fill_(1)
    scores[3, 0], scores[3, 1] = -0.0, 0.0
    # Large cutoff groups must fill every position, even at K=8192.
    scores[4].fill_(0.5)
    scores[5].fill_(float("-inf"))
    if dtype == torch.bfloat16:
        scores[6].view(torch.int16).copy_(torch.arange(width).to(torch.int16))
    padded = torch.empty((len(lengths), width + 8), dtype=dtype, device="cuda")
    device_scores = padded[:, 1 : width + 1]
    device_scores.copy_(scores)
    out = torch.full((len(lengths), k), -2, dtype=torch.int32, device="cuda")
    actual = exact_topk(
        device_scores,
        torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        k=k,
        offsets=(
            torch.tensor(offsets, dtype=torch.int32, device="cuda") if force_last else None
        ),
        out_indices=out,
        force_last=force_last,
    )
    assert actual is out
    expected = _reference_indices(scores, lengths, offsets, k, force_last)
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_topk_graph_replay(dtype):
    k, width, force_last = 8320, 32768, 128
    rng = torch.Generator().manual_seed(74)
    scores = torch.randint(-3, 4, (3, width), generator=rng).to(dtype)
    device_scores = scores.cuda()
    lengths = torch.tensor([0, 129, width], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([11, 17, 23], dtype=torch.int32, device="cuda")
    out = torch.empty((3, k), dtype=torch.int32, device="cuda")

    def launch():
        exact_topk(
            device_scores,
            lengths,
            k=k,
            offsets=offsets,
            out_indices=out,
            force_last=force_last,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for lens, bias in (
        ([width - 1, k + 1, 33], [100, 200, 300]),
        ([0, 0, width], [0, 37, 71]),
    ):
        lengths.copy_(torch.tensor(lens, dtype=torch.int32))
        offsets.copy_(torch.tensor(bias, dtype=torch.int32))
        out.fill_(-2)
        graph.replay()
        expected = _reference_indices(scores, lens, bias, k, force_last)
        assert torch.equal(out.cpu(), expected)


def test_exact_topk_recent_window_only():
    scores = torch.zeros((3, 256), dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([0, 73, 239], dtype=torch.int32, device="cuda")
    output = exact_topk(scores, lengths, k=128, order_mode="bf16_bits")
    expected = _reference_indices(scores.cpu(), [0, 73, 239], [0, 0, 0], 128, 128)
    assert torch.equal(output.cpu(), expected)
