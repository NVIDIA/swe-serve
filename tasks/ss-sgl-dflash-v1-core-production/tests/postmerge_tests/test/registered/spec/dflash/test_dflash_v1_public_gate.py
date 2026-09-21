# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded public-server acceptance for source-era non-overlap DFLASH V1."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager

import requests
import torch
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import find_available_port

SERVER_TIMEOUT = 900
TARGET_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
TARGET_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
DRAFT_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
DRAFT_REVISION = "d3af30def9601abdd10810aba220d692f0e803f0"
PROMPTS = (
    "The capital of France is",
    "Explain gravity in one short paragraph:",
)
PREFIX_TEXT = "Shared planets-and-orbits reference: " + (
    "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune. " * 48
)


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
    port = find_available_port(21000)
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
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
    process = subprocess.Popen(command, env=env, start_new_session=True)
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


def _request(base_url: str, prompt: str | list[int], max_new_tokens: int) -> dict:
    prompt_field = {"text": prompt} if isinstance(prompt, str) else {"input_ids": prompt}
    response = requests.post(
        base_url + "/generate",
        json={
            **prompt_field,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    output_ids = payload.get("output_ids")
    assert isinstance(output_ids, list) and output_ids, payload
    assert all(isinstance(token_id, int) for token_id in output_ids), payload
    return payload


def _tokenize(base_url: str, text: str) -> list[int]:
    response = requests.post(
        base_url + "/tokenize",
        json={"model": TARGET_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    token_ids = response.json().get("tokens")
    assert isinstance(token_ids, list) and token_ids
    return [int(token_id) for token_id in token_ids]


def _metric_value(base_url: str, mode: str) -> float:
    response = requests.get(base_url + "/metrics", timeout=30)
    response.raise_for_status()
    value = 0.0
    pattern = re.compile(
        r'^sglang:cuda_graph_passes_total\{(?=[^}]*mode="'
        + re.escape(mode)
        + r'")[^}]*\}\s+([0-9.eE+-]+)$'
    )
    for line in response.text.splitlines():
        match = pattern.match(line)
        if match:
            value += float(match.group(1))
    return value


def _spec_evidence(payloads: list[dict]) -> tuple[int, int]:
    verify_count = 0
    accepted_count = 0
    for payload in payloads:
        meta = payload.get("meta_info", {})
        verify_count += int(meta.get("spec_verify_ct", 0))
        accepted_count += int(meta.get("spec_accept_token_num", 0))
    assert verify_count > 0, payloads
    assert accepted_count > 0, payloads
    return verify_count, accepted_count


def _dflash_args(*, page_size: int, disable_graph: bool = False, chunked: bool = False):
    args = [
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--speculative-draft-model-revision",
        DRAFT_REVISION,
        "--page-size",
        str(page_size),
        "--max-running-requests",
        "8",
        "--cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "--enable-metrics",
    ]
    if disable_graph:
        args.append("--disable-cuda-graph")
    if chunked:
        args.extend(("--chunked-prefill-size", "4"))
    return args


def _target_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--page-size",
        "1",
        "--max-running-requests",
        "8",
    ]


def _exercise_dflash(
    args: list[str],
    prefix_ids: list[int] | None,
    *,
    graph_mode: str,
) -> dict:
    with _server(args) as (base_url, process):
        if prefix_ids is None:
            prefix_ids = _tokenize(base_url, PREFIX_TEXT)
        before_graph = _metric_value(base_url, "decode_cuda_graph")
        before_eager = _metric_value(base_url, "decode_none")
        prompt_payloads = [_request(base_url, prompt, 24) for prompt in PROMPTS]
        first = _request(base_url, prefix_ids, 24)
        first_ids = [int(token_id) for token_id in first["output_ids"]]
        second = _request(base_url, prefix_ids + first_ids, 8)
        second_ids = [int(token_id) for token_id in second["output_ids"]]
        payloads = [*prompt_payloads, first, second]
        verify_count, accepted_count = _spec_evidence(payloads)
        cached_tokens = int(second.get("meta_info", {}).get("cached_tokens", 0))
        assert cached_tokens > 0, second
        after_graph = _metric_value(base_url, "decode_cuda_graph")
        after_eager = _metric_value(base_url, "decode_none")
        assert process.poll() is None

        graph_delta = after_graph - before_graph
        eager_delta = after_eager - before_eager
        if graph_mode == "graph":
            assert graph_delta > 0, (before_graph, after_graph, after_eager)
        elif graph_mode == "eager":
            assert eager_delta > 0, (before_eager, after_eager, after_graph)
            assert graph_delta == 0, (before_graph, after_graph)
        else:
            raise AssertionError(graph_mode)

        return {
            "prefix_ids": prefix_ids,
            "prompt_outputs": [payload["output_ids"] for payload in prompt_payloads],
            "continuation": first_ids + second_ids,
            "cached_tokens": cached_tokens,
            "verify_count": verify_count,
            "accepted_count": accepted_count,
            "graph_delta": graph_delta,
            "eager_delta": eager_delta,
        }


def _target_reference(prefix_ids: list[int]) -> dict:
    with _server(_target_args()) as (base_url, process):
        prompt_outputs = [
            _request(base_url, prompt, 24)["output_ids"] for prompt in PROMPTS
        ]
        continuation = _request(base_url, prefix_ids, 32)["output_ids"]
        assert process.poll() is None
        return {
            "prompt_outputs": prompt_outputs,
            "continuation": continuation,
        }


def _assert_target_parity(result: dict, target: dict) -> None:
    assert result["prompt_outputs"] == target["prompt_outputs"]
    assert result["continuation"] == target["continuation"]


def test_public_dflash_v1_source_matrix() -> None:
    assert torch.cuda.is_available(), "CUDA is required for public DFLASH generation"
    assert torch.cuda.get_device_capability(0) == (9, 0)

    full_graph = _exercise_dflash(
        _dflash_args(page_size=1),
        None,
        graph_mode="graph",
    )
    target = _target_reference(full_graph["prefix_ids"])
    _assert_target_parity(full_graph, target)

    eager = _exercise_dflash(
        _dflash_args(page_size=1, disable_graph=True),
        full_graph["prefix_ids"],
        graph_mode="eager",
    )
    _assert_target_parity(eager, target)

    page256 = _exercise_dflash(
        _dflash_args(page_size=256),
        full_graph["prefix_ids"],
        graph_mode="graph",
    )
    assert page256["cached_tokens"] >= 256
    _assert_target_parity(page256, target)

    chunked = _exercise_dflash(
        _dflash_args(page_size=1, chunked=True),
        full_graph["prefix_ids"],
        graph_mode="graph",
    )
    _assert_target_parity(chunked, target)

    print(
        "child_a_public_matrix "
        f"graph_accept={full_graph['accepted_count']} "
        f"eager_accept={eager['accepted_count']} "
        f"page256_accept={page256['accepted_count']} "
        f"chunked_accept={chunked['accepted_count']} "
        f"page256_cached={page256['cached_tokens']}",
        flush=True,
    )
