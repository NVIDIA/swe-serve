# Copyright 2023-2024 SGLang Team
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Upstream SGLang DFLASH server E2E, packaged for the task checkpoint cache."""

import os
import unittest
from types import SimpleNamespace

import openai
import requests
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin, _check_accept_length
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.radix_cache_server_kit import gen_radix_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class OfflineGSM8KMixin(GSM8KMixin):
    """Source mixin with the same workload bound to verifier-pinned data."""

    def test_gsm8k(self):
        assert self.gsm8k_accuracy_thres == self.gsm8k_accuracy_thres
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=self.gsm8k_num_questions,
            num_threads=self.gsm8k_num_threads,
            num_shots=5,
            gsm8k_data_path=os.environ["SGLANG_UPSTREAM_E2E_GSM8K"],
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        if is_in_ci():
            write_github_step_summary(f"### test_gsm8k\n{metrics['score']=:.4f}\n")
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)
        _check_accept_length(self, self.base_url, self.gsm8k_accept_length_thres)


class TestDFlashServerBase(CustomTestCase, MatchedStopMixin, OfflineGSM8KMixin):
    max_running_requests = 64
    attention_backend = "flashinfer"
    page_size = 1
    other_launch_args = []
    model = os.environ["SGLANG_UPSTREAM_E2E_TARGET_MODEL"]
    draft_model = os.environ["SGLANG_UPSTREAM_E2E_DRAFT_MODEL"]
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.8

    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{find_available_port(21000)}"
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--page-size",
            str(cls.page_size),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--cuda-graph-bs",
            *[str(i) for i in range(1, cls.max_running_requests + 1)],
        ]
        launch_args.extend(cls.other_launch_args)
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
                1
            ), envs.SGLANG_SPEC_NAN_DETECTION.override(
                True
            ), envs.SGLANG_SPEC_OOB_DETECTION.override(
                True
            ):
                cls.process = popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=launch_args,
                )
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_early_stop(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        for i in range(8):
            max_tokens = (i % 3) + 1
            response = client.completions.create(
                model=self.model,
                prompt=f"There are {i} apples on the table. How to divide them equally?",
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].text
            print(f"early_stop: max_tokens={max_tokens}, text={text!r}")
        assert self.process.poll() is None

    def test_eos_handling(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Today is a sunny day and I like"}],
            max_tokens=256,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        print(f"eos_handling: text={text!r}")
        self.assertNotIn("<|eot_id|>", text)
        self.assertNotIn("<|end_of_text|>", text)
        assert self.process.poll() is None

    def test_greedy_determinism(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        prompt = "The capital of France is"
        outputs = []
        for _ in range(2):
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=32,
                temperature=0,
            )
            outputs.append(response.choices[0].text)
        print(f"determinism: {outputs=}")
        self.assertEqual(outputs[0], outputs[1])
        assert self.process.poll() is None


class TestDFlashServerPage256(TestDFlashServerBase):
    page_size = 256

    def test_radix_attention(self):
        nodes = gen_radix_tree(num_nodes=50)
        data = {
            "input_ids": [node["input_ids"] for node in nodes],
            "sampling_params": [
                {"max_new_tokens": node["decode_len"], "temperature": 0}
                for node in nodes
            ],
        }
        res = requests.post(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        assert self.process.poll() is None


class TestDFlashServerChunkedPrefill(TestDFlashServerBase):
    other_launch_args = ["--chunked-prefill-size", "4"]


class TestDFlashServerNoCudaGraph(TestDFlashServerBase):
    other_launch_args = ["--disable-cuda-graph"]


if __name__ == "__main__":
    unittest.main()
