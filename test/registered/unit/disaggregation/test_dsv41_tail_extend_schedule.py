"""Decode-side scheduling of the DeepSeek-V4.1 asymmetric PD tail extend."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import (
    dsv41_encoder_only_unsupported,
    dsv41_encoder_prefix_len,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Mixin = SchedulerDisaggregationDecodeMixin


def _req(prompt_len):
    tail_start = dsv41_encoder_prefix_len(prompt_len, 128)
    req = SimpleNamespace(
        full_untruncated_fill_ids=list(range(prompt_len)),
        dsv41_tail_start=tail_start,
        kv=SimpleNamespace(kv_allocated_len=tail_start, req_pool_idx=0),
        extend_range=SimpleNamespace(start=0, end=tail_start),
        already_computed=0,
        _cache_breakdown_computed=False,
        logprob_start_len=prompt_len,
    )
    req.set_extend_range = partial(Req.set_extend_range, req)
    return req


def _batch(mode, size):
    batch = MagicMock()
    batch.forward_mode = mode
    batch.is_extend_in_batch = mode.is_extend()
    batch.batch_size.return_value = size
    batch.is_empty.return_value = size == 0
    return batch


def _scheduler(
    waiting, running, last_batch=None, max_prefill_tokens=16384, free_tokens=1 << 20
):
    return SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=list(waiting),
        last_batch=last_batch,
        req_to_token_pool=MagicMock(size=64),
        max_running_requests=64,
        max_prefill_tokens=max_prefill_tokens,
        token_to_kv_pool_allocator=MagicMock(
            page_size=256, available_size=MagicMock(return_value=free_tokens)
        ),
        tree_cache=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        is_mixed_chunk=False,
        running_batch=running,
    )


@patch(
    "sglang.srt.disaggregation.decode.get_parallel",
    new=lambda: SimpleNamespace(attn_dp_size=1),
)
@patch("sglang.srt.disaggregation.decode.set_time_batch")
@patch("sglang.srt.disaggregation.decode.ScheduleBatch")
class TestDsv41TailExtendSchedule(CustomTestCase):
    def test_split_keeps_one_window_for_decode(self, *_):
        self.assertEqual(dsv41_encoder_prefix_len(1000, 128), 872)
        # Even, so no ratio-2 half-pair is pending at the split.
        self.assertEqual(dsv41_encoder_prefix_len(1001, 128), 872)
        self.assertEqual(dsv41_encoder_prefix_len(129, 128), 0)
        self.assertEqual(dsv41_encoder_prefix_len(128, 128), 0)
        self.assertEqual(dsv41_encoder_prefix_len(5, 128), 0)

    def test_tail_extends_on_top_of_the_transferred_prefix(self, schedule_batch, _):
        short, long = _req(40), _req(1000)
        sched = _scheduler([short, long], _batch(ForwardMode.DECODE, 0))
        batch = Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)

        self.assertIs(batch, schedule_batch.init_new.return_value)
        batch.prepare_for_extend.assert_called_once()
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [short, long])
        self.assertEqual(sched.waiting_queue, [])
        self.assertEqual((long.extend_range.start, long.extend_range.end), (872, 1000))
        self.assertEqual(long.already_computed, 872)
        self.assertIsNone(long.dsv41_tail_start)
        # Nothing was transferred for the short prompt: its row is released so
        # the extend allocates it whole.
        sched.req_to_token_pool.free.assert_called_once_with(short)
        self.assertEqual((short.extend_range.start, short.extend_range.end), (0, 40))

    def test_tail_admission_leaves_host_restored_requests_queued(
        self, schedule_batch, _
    ):
        restored, fresh = _req(1000), _req(40)
        restored.dsv41_tail_start = None
        sched = _scheduler([restored, fresh], _batch(ForwardMode.DECODE, 0))

        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)

        self.assertEqual(schedule_batch.init_new.call_args.args[0], [fresh])
        self.assertEqual(sched.waiting_queue, [restored])

    def test_host_restored_request_resumes_decode_before_a_pending_tail(
        self, schedule_batch, _
    ):
        restored, fresh = _req(1000), _req(40)
        restored.dsv41_tail_start = None
        restored.init_next_round_input = MagicMock()
        restored.kv.kv_committed_len = None
        sched = _scheduler([fresh, restored], _batch(ForwardMode.DECODE, 0))
        sched.max_running_requests = 1
        sched.scheduler_stage_metrics = None
        sched.enable_priority_scheduling = False
        sched.enable_hisparse = False
        sched.kv_checksum_computer = None
        sched.ngram_embedding_manager = MagicMock()
        sched.future_map = MagicMock()
        sched.chunked_req = None
        sched.batch_result_processor = MagicMock()
        sched.require_mlp_sync = False
        sched.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        prebuilt = _batch(ForwardMode.PREBUILT, 1)
        schedule_batch.init_new.return_value = prebuilt

        def prepare_decode(batch):
            batch.forward_mode = ForwardMode.DECODE
            return batch

        sched.update_running_batch = prepare_decode
        for method in (
            "get_new_prebuilt_batch",
            "_get_new_prebuilt_batch",
            "_verify_kv_checksums",
            "_merge_dsv41_tail_extend",
            "_get_new_dsv41_tail_extend_batch",
        ):
            setattr(sched, method, partial(getattr(Mixin, method), sched))

        with (
            patch(
                "sglang.srt.disaggregation.decode.dsv41_encoder_only_prefill_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.disaggregation.decode.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_radix_cache=False,
                    disaggregation_decode_host_receive_threshold=0,
                ),
            ),
            patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"),
        ):
            plan = Mixin.get_next_disagg_decode_batch_to_run(sched, sched.running_batch)

        self.assertIs(plan.batch_to_run, prebuilt)
        self.assertIs(plan.running_batch, prebuilt)
        self.assertEqual(plan.batch_to_run.forward_mode, ForwardMode.DECODE)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [restored])
        self.assertEqual(sched.waiting_queue, [fresh])
        prebuilt.prepare_for_prebuilt.assert_called_once()
        prebuilt.process_prebuilt.assert_called_once_with(sched.future_map)
        prebuilt.prepare_for_extend.assert_not_called()

    def test_running_decode_joins_local_and_peer_tail_steps(self, schedule_batch, _):
        for mode in (ForwardMode.EXTEND, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                running = _batch(ForwardMode.DECODE, 1)
                running.reqs = [SimpleNamespace(beam_group=None)]
                running.return_logprob = False
                tail = _batch(mode, int(mode.is_extend()))
                tail.return_logprob = False
                sched = _scheduler([], running)
                sched.update_running_batch = MagicMock(return_value=running)
                sched.spec_algorithm = SimpleNamespace(is_none=lambda: True)

                combined, next_running = Mixin._mix_dsv41_tail_with_decode(
                    sched, tail, running
                )

                self.assertIs(combined, tail)
                self.assertIs(next_running, schedule_batch.return_value)
                self.assertIs(combined.decoding_reqs, running.reqs)
                sched.update_running_batch.assert_called_once_with(running)
                tail.mix_with_running.assert_called_once_with(running)
                if mode.is_idle():
                    self.assertEqual(combined.prefill_input_ids_cpu.numel(), 0)
                    self.assertEqual(combined.extend_lens, [])

    def test_admission_respects_the_prefill_token_budget(self, schedule_batch, _):
        reqs = [_req(1000) for _ in range(3)]
        sched = _scheduler(reqs, _batch(ForwardMode.DECODE, 0), max_prefill_tokens=300)
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs[:2])
        self.assertEqual(sched.waiting_queue, reqs[2:])

    def test_tail_admission_reserves_deepep_tokens_for_running_decode(
        self, schedule_batch, _
    ):
        reqs = [_req(1000) for _ in range(2)]
        sched = _scheduler(reqs, _batch(ForwardMode.DECODE, 1))
        Mixin._get_new_dsv41_tail_extend_batch(
            sched, sched.running_batch, max_dispatch_tokens=256, decode_token_reserve=1
        )
        self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs[:1])
        self.assertEqual(sched.waiting_queue, reqs[1:])

    def test_admission_waits_for_free_pages(self, schedule_batch, _):
        # [972, 1100) needs one page past the transferred prefix's last page.
        reqs = [_req(1100) for _ in range(3)]
        sched = _scheduler(reqs, _batch(ForwardMode.DECODE, 0), free_tokens=512)
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs[:2])

        sched = _scheduler(reqs[2:], _batch(ForwardMode.DECODE, 4), free_tokens=0)
        self.assertIsNone(
            Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        )
        self.assertEqual(sched.waiting_queue, reqs[2:])

    def test_rebootstrap_tail_scores_no_replayed_tokens(self, *_):
        # Retracted after 9 tokens: prefill covers 1000 + 8, the tail replays
        # the emitted boundary token and samples the tenth.
        req = _req(1009)
        req.dsv41_tail_start = dsv41_encoder_prefix_len(1008, 128)
        req.pd_rebootstrap_in_progress = True
        req.logprob_start_len = 1000
        sched = _scheduler([req], _batch(ForwardMode.DECODE, 0))
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (880, 1009))
        self.assertFalse(req.pd_rebootstrap_in_progress)
        self.assertEqual(req.logprob_start_len, 1009)

    def test_decode_runs_between_tail_extends(self, schedule_batch, _):
        running = _batch(ForwardMode.DECODE, 4)
        sched = _scheduler(
            [_req(1000)], running, last_batch=_batch(ForwardMode.EXTEND, 1)
        )
        self.assertIsNone(Mixin._get_new_dsv41_tail_extend_batch(sched, running))
        schedule_batch.init_new.assert_not_called()

    def test_idle_rank_also_alternates_after_a_peer_tail(self, schedule_batch, _):
        last = _batch(ForwardMode.IDLE, 0)
        last.is_extend_in_batch = True
        sched = _scheduler([_req(1000)], _batch(ForwardMode.DECODE, 0), last_batch=last)
        with patch(
            "sglang.srt.disaggregation.decode.get_parallel",
            return_value=SimpleNamespace(attn_dp_size=8),
        ):
            self.assertIsNone(
                Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
            )
        schedule_batch.init_new.assert_not_called()

    def test_finished_tail_extend_merges_into_decode(self, *_):
        running, tail = _batch(ForwardMode.DECODE, 4), _batch(ForwardMode.EXTEND, 2)
        sched = _scheduler([], running, last_batch=tail)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, running), running)
        tail.filter_batch.assert_called_once()
        running.merge_batch.assert_called_once_with(tail)

        empty_running = _batch(ForwardMode.DECODE, 0)
        sched = _scheduler([], empty_running, last_batch=tail)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, empty_running), tail)

        sched = _scheduler([], running, last_batch=running)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, running), running)
        self.assertEqual(running.merge_batch.call_count, 1)


class TestDsv41EncoderOnlyRequests(CustomTestCase):
    @staticmethod
    def _req(**fields):
        values = dict(
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            return_logprob=False,
            logprob_start_len=-1,
            origin_input_ids=list(range(300)),
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            return_sampling_mask=False,
        )
        values.update(fields)
        return SimpleNamespace(**values)

    def test_peers_reject_what_the_prefix_cannot_serve(self):
        self.assertIsNone(dsv41_encoder_only_unsupported(self._req()))
        # Decode computes every output token's logprob.
        self.assertIsNone(
            dsv41_encoder_only_unsupported(
                self._req(return_logprob=True, logprob_start_len=300)
            )
        )
        for fields in (
            {"return_logprob": True, "logprob_start_len": 0},
            {"input_embeds": [[0.0]]},
            {"return_hidden_states": True},
            {"return_routed_experts": True},
            {"return_sampling_mask": True},
        ):
            with self.subTest(**fields):
                self.assertIsNotNone(
                    dsv41_encoder_only_unsupported(self._req(**fields))
                )


if __name__ == "__main__":
    unittest.main()
