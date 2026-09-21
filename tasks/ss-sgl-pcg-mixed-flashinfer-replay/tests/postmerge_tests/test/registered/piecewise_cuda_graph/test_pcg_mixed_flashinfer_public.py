# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public one-H100 serving coverage for mixed-chunk PCG replay."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import requests
from sglang.srt.utils import kill_process_tree

MODEL = os.environ["SGLANG_TEST_PCG_MODEL"]
SERVER_TIMEOUT = 900
REQUEST_TIMEOUT = 300
PORT_MIN = 20_000
PORT_COUNT = 25_000


def _free_loopback_port() -> int:
    job_text = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task_text = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    try:
        seed = int(job_text) if job_text is not None else os.getpid()
        seed = seed * 37 + int(task_text) * 101 + os.getpid()
    except ValueError:
        seed = os.getpid()
    start = PORT_MIN + seed % PORT_COUNT
    for offset in range(PORT_COUNT):
        port = PORT_MIN + (start - PORT_MIN + offset) % PORT_COUNT
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            try:
                reservation.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free collision-safe loopback port")


def _log_tail(path: Path, lines: int = 120) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except FileNotFoundError:
        return "<server log was not created>"


def _wait_for_server(process: subprocess.Popen, base_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited with code {return_code}\n{_log_tail(log_path)}")
        try:
            # Readiness must not execute generation: the scored workload, not a
            # one-token health probe, owns every PCG observation.
            response = requests.get(base_url + "/health", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"server startup timed out\n{_log_tail(log_path)}")


@contextmanager
def _server(*, piecewise_enabled: bool):
    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(prefix="pcg-mixed-flashinfer-", suffix=".log", delete=False)
    log_path = Path(log_file.name)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--attention-backend",
        "flashinfer",
        "--dtype",
        "float16",
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "64",
        "--piecewise-cuda-graph-tokens",
        "64",
        "128",
        "256",
        "--piecewise-cuda-graph-max-tokens",
        "256",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.72",
        "--disable-radix-cache",
        "--enable-metrics",
    ]
    if piecewise_enabled:
        command.append("--enforce-piecewise-cuda-graph")
    else:
        command.append("--disable-piecewise-cuda-graph")

    env = os.environ.copy()
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("TRANSFORMERS_OFFLINE") == "1"
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_server(process, base_url, log_path)
        assert process.poll() is None
        yield base_url, process, log_path
    finally:
        kill_process_tree(process.pid)
        try:
            process.wait(timeout=30)
        except Exception:
            pass
        log_file.close()
        time.sleep(3)


def _generate(base_url: str, prompt_or_ids, max_new_tokens: int) -> str:
    request = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if isinstance(prompt_or_ids, str):
        request["text"] = prompt_or_ids
    else:
        request["input_ids"] = prompt_or_ids
    response = requests.post(base_url + "/generate", json=request, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    text = body.get("text")
    assert isinstance(text, str) and text, body
    assert body.get("meta_info", {}).get("completion_tokens") == max_new_tokens, body
    return text


def _generate_streaming(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    first_token: Event,
) -> str:
    request = {
        "text": prompt,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    last_text = ""
    with requests.post(
        base_url + "/generate",
        json=request,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            payload = line.removeprefix(b"data:").strip()
            if payload == b"[DONE]":
                break
            body = json.loads(payload)
            text = body.get("text")
            if isinstance(text, str) and text:
                last_text = text
                first_token.set()
    assert first_token.is_set()
    assert last_text
    return last_text


def _prefill_passes(base_url: str, mode: str) -> float:
    response = requests.get(base_url + "/metrics", timeout=30)
    response.raise_for_status()
    total = 0.0
    for line in response.text.splitlines():
        if not line.startswith("sglang:cuda_graph_passes_total{"):
            continue
        if re.search(rf'(?:^|,)mode="{re.escape(mode)}"(?:,|}})', line) is None:
            continue
        total += float(line.rsplit(" ", 1)[1])
    return total


def _mixed_wave(
    base_url: str,
    wave: int,
    *,
    leader_tokens: int = 32,
    follower_delay: float = 0.08,
) -> dict[str, str]:
    prompts = {
        "leader": (
            f"Wave {wave}. Explain why careful software testing matters, using a "
            "concrete example and a short concluding checklist."
        ),
        # This request is necessarily split across many 64-token prefill chunks.
        # While the leader decodes, at least one of those chunks joins decode work.
        "follower-a": f"Wave {wave}. " + "reliable systems need careful testing. " * 160,
        "follower-b": f"Wave {wave}. Count from one to ten:",
        "follower-c": f"Wave {wave}. Complete this sentence: a reliable system",
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "leader": pool.submit(_generate, base_url, prompts["leader"], leader_tokens),
        }
        # The leader is decoding while fresh extend work joins the running batch.
        time.sleep(follower_delay)
        for name in ("follower-a", "follower-b", "follower-c"):
            futures[name] = pool.submit(_generate, base_url, prompts[name], 1)
        return {name: future.result() for name, future in futures.items()}


def _repeated_mixed_workload(base_url: str) -> list[dict[str, str]]:
    waves = [_mixed_wave(base_url, wave) for wave in (1, 2)]
    # The leader is background decode used only to open the mixed window. Its
    # later autoregressive tokens can diverge after harmless batch-dependent
    # floating-point differences. Compare the first token of every request
    # whose prefill actually joined decode work instead.
    return [
        {name: outputs[name] for name in ("follower-a", "follower-b", "follower-c")}
        for outputs in waves
    ]


def _graph_bound_mixed_wave(base_url: str, wave: int) -> dict[str, str]:
    first_token = Event()
    leader_prompt = f"Graph wave {wave}. Explain how to test a distributed system, with examples."
    follower_prompts = {
        "follower-a": f"Graph wave {wave}. " + "reliable systems need careful testing. " * 160,
        "follower-b": f"Graph wave {wave}. Count from one to ten:",
        "follower-c": f"Graph wave {wave}. Complete this sentence: a reliable system",
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        leader = pool.submit(
            _generate_streaming,
            base_url,
            leader_prompt,
            1024,
            first_token,
        )
        assert first_token.wait(timeout=60), "leader did not enter streaming decode"
        assert not leader.done(), "leader finished before the mixed window opened"

        # The leader's ordinary prefill is complete. Only follower prefill can
        # move these counters while the leader remains in decode.
        graph_before = _prefill_passes(base_url, "prefill_cuda_graph")
        eager_before = _prefill_passes(base_url, "prefill_none")
        followers = {
            name: pool.submit(_generate, base_url, prompt, 1) for name, prompt in follower_prompts.items()
        }
        outputs = {name: future.result() for name, future in followers.items()}
        assert not leader.done(), "leader left decode before follower prefill completed"
        graph_after = _prefill_passes(base_url, "prefill_cuda_graph")
        eager_after = _prefill_passes(base_url, "prefill_none")
        outputs["leader"] = leader.result()

    assert graph_after > graph_before
    assert eager_after == eager_before
    return outputs


def test_long_decode_mixed_chunk_reuses_extend_graph() -> None:
    with _server(piecewise_enabled=True) as (base_url, process, log_path):
        waves = [_graph_bound_mixed_wave(base_url, wave) for wave in (11, 12)]
        assert process.poll() is None, _log_tail(log_path)
    assert all(text for wave in waves for text in wave.values())


def test_mixed_chunk_replay_matches_disabled_reference() -> None:
    with _server(piecewise_enabled=False) as (base_url, process, log_path):
        reference = _repeated_mixed_workload(base_url)
        assert process.poll() is None, _log_tail(log_path)

    with _server(piecewise_enabled=True) as (base_url, process, log_path):
        replayed = _repeated_mixed_workload(base_url)
        assert process.poll() is None, _log_tail(log_path)

    assert replayed == reference


def test_ordinary_unpadded_flashinfer_remains_healthy() -> None:
    # The explicit non-PCG path does not pad query metadata to a capture bucket.
    prompt = "Explain in one sentence why deterministic regression tests are useful."
    with _server(piecewise_enabled=False) as (base_url, process, log_path):
        first = _generate(base_url, prompt, 8)
        second = _generate(base_url, prompt, 8)
        assert process.poll() is None, _log_tail(log_path)
    assert first == second
