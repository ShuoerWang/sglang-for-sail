"""E2E coverage for V4.1 DSpark prefill CP with prefill CUDA graphs on PPU.

The server validates the prefill-CP topology before CP size resolution and
keeps prefill CUDA graphs enabled, so the BREAKABLE graph must cover the live
context budget instead of the default 16k cap, and a CP-captured graph must
replay only for CP batches. GSM8K accuracy runs through the same server so a
mis-sized capture or a wrong CP shard surfaces as an accuracy drop.

Attention-TP verify padding needs attention TP > 1 and is covered outside this
configuration. Image preparation before CP sharding is exercised by the
text-only path here and by the CP policy unit tests.
"""

import os
import unittest

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "T-HEAD/DeepSeek-V4.1-Flash-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken",
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


def _sm89_or_newer() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and get_device_sm() >= 89
    except Exception:
        return False


@unittest.skipUnless(_sm89_or_newer(), "requires a device with SM89 or newer")
class TestDeepseekV4DsparkPrefillCpGraph(GSM8KMixin, CustomTestCase):
    """DSpark + interleave prefill CP with prefill CUDA graphs enabled."""

    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--served-model-name",
            "DeepSeek-V4-Flash",
            "--tp-size",
            "8",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--max-running-requests",
            "64",
            "--cuda-graph-max-bs-decode",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--disable-overlap-schedule",
            "--disable-shared-experts-fusion",
            # Prefill CUDA graphs stay enabled: the CP path must capture the
            # live context budget instead of the default 16k cap.
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-dp-lm-head",
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
        ]
        env = dict(os.environ)
        env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
        env.setdefault("SGLANG_DG_CACHE_DIR", "/tmp/dg_cache0")

        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
