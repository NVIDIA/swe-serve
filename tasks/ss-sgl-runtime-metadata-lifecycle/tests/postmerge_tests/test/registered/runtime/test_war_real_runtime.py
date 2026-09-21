# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-sided one-H100 event-order probes through initialized production servers."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import requests
from sglang.test.test_utils import kill_process_tree, popen_launch_server

TARGET = Path(
    "/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)
DRAFT = Path(
    "/hf-cache/hub/models--lmsys--sglang-EAGLE3-LLaMA3.1-Instruct-8B/"
    "snapshots/28a53ce8911434c031d7c78392abb26d898ec293"
)


def _collision_safe_url() -> str:
    job_text = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task_text = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    try:
        seed = int(job_text) if job_text is not None else os.getpid()
        seed = seed * 37 + int(task_text) * 101 + os.getpid()
    except ValueError:
        seed = os.getpid()
    for offset in range(25_000):
        port = 20_000 + (seed + offset) % 25_000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            try:
                reservation.bind(("127.0.0.1", port))
            except OSError:
                continue
            return f"http://127.0.0.1:{port}"
    raise RuntimeError("no free collision-safe verifier port")


def _run_probe(tmp_path, mode: str) -> dict:
    arm_file = tmp_path / f"{mode}.arm"
    result_file = tmp_path / f"{mode}.result.json"
    diagnostic_file = tmp_path / f"{mode}.diagnostic.json"
    stdout_path = Path(f"/logs/verifier/{mode}-server.stdout.log")
    stderr_path = Path(f"/logs/verifier/{mode}-server.stderr.log")
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        value for value in ("/tests/probe_dist", inherited_pythonpath, "/code/python") if value
    )
    env["SGLANG_PLUGINS"] = "event_order_probe"
    env["SGLANG_EVENT_ORDER_ARM_FILE"] = str(arm_file)
    env["SGLANG_EVENT_ORDER_RESULT_FILE"] = str(result_file)
    env["SGLANG_EVENT_ORDER_DIAGNOSTIC_FILE"] = str(diagnostic_file)
    env["SGLANG_EVENT_ORDER_MODE"] = mode

    base_url = _collision_safe_url()
    process = None
    other_args = [
        "--page-size",
        "1",
        "--attention-backend",
        "flashinfer",
        "--mem-fraction-static",
        "0.85",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "1024",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "2",
        "--max-total-tokens",
        "4500",
    ]
    if mode in {"eagle_graph", "eagle_graph_to_fallback"}:
        # Match the source-era EAGLE fixture: the target advertises a longer
        # context than this draft snapshot, which is valid for this runtime test.
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        other_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            str(DRAFT),
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "16",
            "--speculative-num-draft-tokens",
            "64",
            *other_args,
        ]
    elif mode == "plain_eager_fallback":
        other_args = ["--disable-cuda-graph", *other_args]
    elif mode == "ngram_fallback":
        other_args = [
            "--speculative-algorithm",
            "NGRAM",
            "--speculative-num-draft-tokens",
            "16",
            *other_args,
        ]
    else:
        raise ValueError(f"unknown verifier probe mode: {mode}")

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        try:
            process = popen_launch_server(
                str(TARGET),
                base_url,
                timeout=600,
                env=env,
                return_stdout_stderr=(stdout, stderr),
                other_args=other_args,
            )
            arm_file.write_text("armed\n")
            response = requests.post(
                base_url + "/generate",
                json={
                    "text": "Explain why synchronization dependencies form a partial order.",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 48},
                },
                timeout=300,
            )
            assert response.status_code == 200, response.text

            if mode == "eagle_graph_to_fallback" and not result_file.is_file():
                # The fallback boundary is consumed by the next real scheduling
                # write. Drive that write deterministically through the same
                # initialized server rather than assuming the first request
                # survives for another speculative iteration.
                followup = requests.post(
                    base_url + "/generate",
                    json={
                        "text": "Describe a second independent partial-order example.",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 24},
                    },
                    timeout=300,
                )
                assert followup.status_code == 200, followup.text

            deadline = time.monotonic() + (15 if mode == "eagle_graph_to_fallback" else 60)
            while not result_file.is_file() and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(
                        f"production server exited before probe result: {process.returncode}"
                    )
                time.sleep(0.1)
            diagnostic = (
                json.loads(diagnostic_file.read_text())
                if diagnostic_file.is_file()
                else {"stage": "no diagnostic emitted"}
            )
            assert result_file.is_file(), (
                f"production plugin did not emit an event-order result; last diagnostic={diagnostic}"
            )
            result = json.loads(result_file.read_text())
            assert result["used_initialized_runner"] is True
            assert result["probe_mode"] == mode
            return result
        finally:
            if process is not None:
                kill_process_tree(process.pid, wait_timeout=60)


def _assert_read_boundary_precedes_schedule(result: dict) -> None:
    assert result["read_boundary_complete_at_schedule_marker"] is True, result


def test_initialized_eagle_keeps_scheduling_behind_live_metadata_read(tmp_path):
    result = _run_probe(tmp_path, "eagle_graph")
    _assert_read_boundary_precedes_schedule(result)
    assert result["shared_write_marker_queued"] is True, result
    assert result["shared_write_progress_before_metadata_read"] == 0, result


def test_initialized_eagle_releases_then_falls_back_without_reusing_a_boundary(
    tmp_path,
):
    result = _run_probe(tmp_path, "eagle_graph_to_fallback")
    pairs = result["dependency_pairs"]
    assert len(pairs) == 2, result

    graph_iteration, fallback_iteration = pairs
    assert graph_iteration["kind"] == "graph", result
    assert graph_iteration["read_happens_before_write"] is True, result
    assert graph_iteration["tail_happens_before_write"] is False, result

    assert result["forced_graph_fallback"] is True, result
    assert result["fallback_iteration_recorded"] is True, result
    assert fallback_iteration["kind"] == "fallback", result
    assert fallback_iteration["read_happens_before_write"] is True, result
    assert fallback_iteration["tail_happens_before_write"] is True, result


def _assert_conservative_fallback(result: dict) -> None:
    _assert_read_boundary_precedes_schedule(result)
    assert result["tail_was_complete_at_schedule_marker"] is True, result


def test_initialized_plain_eager_decode_uses_conservative_fallback(tmp_path):
    _assert_conservative_fallback(_run_probe(tmp_path, "plain_eager_fallback"))


def test_initialized_ngram_decode_uses_conservative_fallback(tmp_path):
    _assert_conservative_fallback(_run_probe(tmp_path, "ngram_fallback"))
