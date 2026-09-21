# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-H100 public-server coverage for page-one EAGLE V2 tree runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from subprocess import Popen as launch_process

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import find_available_port

TARGET_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
TARGET_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
DRAFT_REVISION = "28a53ce8911434c031d7c78392abb26d898ec293"
DRAFT_SNAPSHOT = (
    "/hf-cache/hub/models--lmsys--sglang-EAGLE3-LLaMA3.1-Instruct-8B/"
    f"snapshots/{DRAFT_REVISION}"
)
SERVER_TIMEOUT = 900

PARITY_PROMPTS = (
    "The capital of France is",
    "Write the numbers from one to ten in order:",
    "A short explanation of gravity is",
    "In Python, a context manager is useful because",
    "The three primary colors of light are",
    "A careful recipe for vegetable soup begins with",
)
RAGGED_PROMPTS = (
    "Complete this phrase: once upon",
    "Name two ocean mammals:",
    "The square root of sixteen is",
    "Explain photosynthesis in one paragraph:",
)
RAGGED_LENGTHS = (1, 2, 3, 17)


def _wait_for_server(process: subprocess.Popen, base_url: str) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited with code {return_code}")
        try:
            response = requests.get(base_url + "/health_generate", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("server failed to start before the bounded timeout")


@contextmanager
def _server(other_args: list[str]):
    port = find_available_port(20000)
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["SGLANG_ENABLE_SPEC_V2"] = "1"
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("TRANSFORMERS_OFFLINE") == "1"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_MODEL,
        "--revision",
        TARGET_REVISION,
        *other_args,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = launch_process(command, env=env, start_new_session=True)
    try:
        _wait_for_server(process, base_url)
        assert process.poll() is None
        yield base_url, process
    finally:
        kill_process_tree(process.pid)
        try:
            process.wait(timeout=30)
        except Exception:
            pass
        time.sleep(3)


def _target_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
        "--page-size",
        "1",
        "--dtype",
        "float16",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.75",
    ]


def _eagle_args(*, topk: int, disable_overlap: bool) -> list[str]:
    args = [
        *_target_args(),
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DRAFT_SNAPSHOT,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        str(topk),
        "--speculative-num-draft-tokens",
        "4" if topk == 1 else "32",
    ]
    if disable_overlap:
        args.append("--disable-overlap-schedule")
    return args


def _post_generate(base_url: str, payload: dict, *, timeout: int = 300):
    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _tokens(payload: dict) -> list[int]:
    values = payload.get("meta_info", {}).get("output_token_logprobs")
    assert isinstance(values, list), payload
    return [int(value[1]) for value in values]


def _logprobs(payload: dict) -> list[float]:
    values = payload.get("meta_info", {}).get("output_token_logprobs")
    assert isinstance(values, list), payload
    return [float(value[0]) for value in values]


def _batch(
    base_url: str,
    prompts: tuple[str, ...],
    lengths: tuple[int, ...],
    *,
    penalties: bool,
) -> list[dict]:
    params = []
    for index, max_new_tokens in enumerate(lengths):
        item = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        }
        if penalties:
            item["frequency_penalty"] = 0.15 * (index + 1)
            item["presence_penalty"] = 0.1 * (index % 2)
        params.append(item)
    payload = _post_generate(
        base_url,
        {
            "text": list(prompts),
            "sampling_params": params,
            "return_logprob": [True] * len(prompts),
            "logprob_start_len": [-1] * len(prompts),
        },
    )
    assert isinstance(payload, list) and len(payload) == len(prompts), payload
    for item, expected_length in zip(payload, lengths, strict=True):
        assert len(_tokens(item)) == expected_length, item
        assert isinstance(item.get("text"), str), item
    return payload


def _tokenize(base_url: str, prompt: str) -> list[int]:
    response = requests.post(
        base_url + "/tokenize",
        json={"model": TARGET_MODEL, "prompt": prompt},
        timeout=60,
    )
    response.raise_for_status()
    values = response.json().get("tokens")
    assert isinstance(values, list) and values, response.text
    return [int(value) for value in values]


def _prefix_continuation(base_url: str) -> tuple[list[int], int]:
    prompt = "Name the planets of the solar system in order and briefly describe each:"
    prompt_ids = _tokenize(base_url, prompt)
    first = _post_generate(
        base_url,
        {
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 24,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
    )
    first_ids = _tokens(first)
    second = _post_generate(
        base_url,
        {
            "input_ids": prompt_ids + first_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
    )
    cached_tokens = int(second.get("meta_info", {}).get("cached_tokens", 0))
    assert cached_tokens > 0, second
    return first_ids + _tokens(second), cached_tokens


def _server_state(base_url: str) -> dict:
    response = requests.get(base_url + "/server_info", timeout=60)
    response.raise_for_status()
    states = response.json().get("internal_states")
    assert isinstance(states, list) and states, response.text
    return states[0]


def _assert_rescore_matches(base_url: str, prompt: str, generated: dict) -> None:
    prompt_ids = _tokenize(base_url, prompt)
    generated_ids = _tokens(generated)
    score = _post_generate(
        base_url,
        {
            "input_ids": prompt_ids + generated_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    scored = score["meta_info"]["input_token_logprobs"][len(prompt_ids) :]
    scored_values = [float(value[0]) for value in scored]
    decoded_values = _logprobs(generated)
    assert len(scored_values) == len(decoded_values)
    max_delta = max(
        abs(decoded - rescored)
        for decoded, rescored in zip(decoded_values, scored_values, strict=True)
    )
    print(f"public_eagle_rescore max_delta={max_delta:.6f}", flush=True)
    assert max_delta < 0.255


def _spec_metrics(payloads: list[dict]) -> tuple[int, int, int]:
    correct = sum(
        int(item.get("meta_info", {}).get("spec_num_correct_drafts", 0))
        for item in payloads
    )
    proposed = sum(
        int(item.get("meta_info", {}).get("spec_num_proposed_drafts", 0))
        for item in payloads
    )
    verify = sum(
        int(item.get("meta_info", {}).get("spec_verify_ct", 0))
        for item in payloads
    )
    assert correct > 0 and proposed > 0 and verify > 0, payloads
    return correct, proposed, verify


def _digest(values) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _run_mode(args: list[str], *, speculative: bool, disable_overlap: bool):
    with _server(args) as (base_url, process):
        normal = _batch(
            base_url,
            PARITY_PROMPTS,
            (32,) * len(PARITY_PROMPTS),
            penalties=False,
        )
        ragged = _batch(
            base_url,
            RAGGED_PROMPTS,
            RAGGED_LENGTHS,
            penalties=True,
        )
        continuation, cached_tokens = _prefix_continuation(base_url)
        evidence = {
            "normal_tokens": [_tokens(item) for item in normal],
            "normal_logprobs": [_logprobs(item) for item in normal],
            "ragged_tokens": [_tokens(item) for item in ragged],
            "continuation": continuation,
            "cached_tokens": cached_tokens,
        }
        if speculative:
            correct, proposed, verify = _spec_metrics(normal + ragged)
            state = _server_state(base_url)
            assert state.get("speculative_algorithm") == "EAGLE3", state
            assert int(state.get("page_size")) == 1, state
            assert int(state.get("speculative_eagle_topk")) == 8, state
            assert bool(state.get("disable_overlap_schedule")) is disable_overlap, state
            assert float(state.get("avg_spec_accept_length", 0.0)) > 1.0, state
            _assert_rescore_matches(base_url, PARITY_PROMPTS[0], normal[0])
            evidence.update(
                {
                    "correct": correct,
                    "proposed": proposed,
                    "verify": verify,
                    "accept_length": float(state["avg_spec_accept_length"]),
                }
            )
        assert process.poll() is None
        print(
            "public_eagle_mode "
            f"speculative={speculative} disable_overlap={disable_overlap} "
            f"token_sha256={_digest(evidence['normal_tokens'])} "
            f"cached_tokens={cached_tokens}",
            flush=True,
        )
        return evidence


def _assert_mode_matches_reference(reference: dict, candidate: dict) -> None:
    assert candidate["normal_tokens"] == reference["normal_tokens"]
    assert candidate["ragged_tokens"] == reference["ragged_tokens"]
    assert candidate["continuation"] == reference["continuation"]
    assert candidate["cached_tokens"] > 0
    for candidate_row, reference_row in zip(
        candidate["normal_logprobs"], reference["normal_logprobs"], strict=True
    ):
        assert len(candidate_row) == len(reference_row)
        assert max(
            abs(actual - expected)
            for actual, expected in zip(candidate_row, reference_row, strict=True)
        ) < 0.255


def test_public_page_one_tree_runtime_matches_target() -> None:
    reference = _run_mode(_target_args(), speculative=False, disable_overlap=False)
    overlap = _run_mode(
        _eagle_args(topk=8, disable_overlap=False),
        speculative=True,
        disable_overlap=False,
    )
    synchronous = _run_mode(
        _eagle_args(topk=8, disable_overlap=True),
        speculative=True,
        disable_overlap=True,
    )
    _assert_mode_matches_reference(reference, overlap)
    _assert_mode_matches_reference(reference, synchronous)


def test_public_topk1_chain_remains_healthy() -> None:
    with _server(_eagle_args(topk=1, disable_overlap=False)) as (
        base_url,
        process,
    ):
        payloads = _batch(
            base_url,
            PARITY_PROMPTS[:3],
            (24, 24, 24),
            penalties=False,
        )
        correct, proposed, verify = _spec_metrics(payloads)
        continuation, cached_tokens = _prefix_continuation(base_url)
        state = _server_state(base_url)
        assert int(state.get("page_size")) == 1, state
        assert int(state.get("speculative_eagle_topk")) == 1, state
        assert not bool(state.get("disable_overlap_schedule")), state
        assert float(state.get("avg_spec_accept_length", 0.0)) > 1.0, state
        assert continuation and cached_tokens > 0
        assert process.poll() is None
        print(
            "public_eagle_topk1 "
            f"correct={correct} proposed={proposed} verify={verify} "
            f"cached_tokens={cached_tokens}",
            flush=True,
        )
