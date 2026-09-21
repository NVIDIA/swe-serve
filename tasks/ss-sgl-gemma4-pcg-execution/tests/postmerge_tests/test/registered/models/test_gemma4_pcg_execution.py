# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import zlib
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import find_available_port, popen_launch_server
from transformers import AutoTokenizer

_SERVER_TIMEOUT_SECONDS = 1800
_REQUEST_TIMEOUT_SECONDS = 300
_CAPTURE_TOKENS = (64, 128, 256)


def _manifest() -> dict[str, Any]:
    value = json.loads(Path("/tests/model_assets.json").read_bytes())
    assert isinstance(value, dict)
    return value


def _profile_name() -> str:
    return os.environ.get("SWE_SERVE_HARDWARE_PROFILE", "h100_1")


def _profile() -> dict[str, Any]:
    profiles = _manifest()["profiles"]
    name = _profile_name()
    assert name in profiles, f"unknown hardware profile {name!r}"
    return profiles[name]


def _assert_hardware_profile() -> None:
    profile = _profile()
    required = int(profile["allocation_gpus"])
    visible = torch.cuda.device_count()
    assert visible == required, f"profile {_profile_name()} requires {required} GPUs, got {visible}"
    expected_name = str(profile["gpu_name_contains"]).lower()
    names = [torch.cuda.get_device_name(index) for index in range(required)]
    assert all(expected_name in name.lower() for name in names), (
        f"profile {_profile_name()} expected {profile['gpu_name_contains']}, got {names}"
    )


def _model_path() -> str:
    path = Path(os.environ["GEMMA4_MODEL_PATH"])
    assert path.is_dir(), f"missing pinned Gemma4 checkpoint: {path}"
    return str(path)


def _server_url(port_offset: int) -> str:
    return f"http://127.0.0.1:{find_available_port(32000 + port_offset)}"


@contextmanager
def _server(*, piecewise: bool, port_offset: int) -> Iterator[str]:
    profile = _profile()
    args = [
        "--tp-size",
        str(profile["tp_size"]),
        "--context-length",
        "2048",
        "--max-running-requests",
        "2",
        "--mem-fraction-static",
        "0.90",
    ]
    if piecewise:
        args.append("--disable-cuda-graph")
        args.extend(
            [
                "--enforce-piecewise-cuda-graph",
                "--piecewise-cuda-graph-tokens",
                *[str(value) for value in _CAPTURE_TOKENS],
            ]
        )
    else:
        args.extend(["--disable-piecewise-cuda-graph", "--disable-cuda-graph"])
    if profile["gpu_family"] == "blackwell":
        args.extend(["--attention-backend", "triton", "--disable-flashinfer-autotune"])

    base_url = _server_url(port_offset)
    process = popen_launch_server(
        _model_path(),
        base_url,
        timeout=_SERVER_TIMEOUT_SECONDS,
        other_args=args,
    )
    try:
        info = _get(base_url, "/server_info")
        assert int(info["tp_size"]) == int(profile["tp_size"]), info
        yield base_url
    finally:
        kill_process_tree(process.pid)


def _get(base_url: str, route: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}{route}", timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    body = response.json()
    assert isinstance(body, dict)
    return body


def _post(base_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}{route}", json=payload, timeout=_REQUEST_TIMEOUT_SECONDS)
    assert response.ok, f"POST {route} returned {response.status_code}: {response.text}"
    body = response.json()
    assert isinstance(body, dict)
    return body


def _served_model_id(base_url: str) -> str:
    models = _get(base_url, "/v1/models").get("data", [])
    assert models
    model_id = models[0].get("id")
    assert isinstance(model_id, str) and model_id
    return model_id


def _chat_prompt(content: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str) and prompt
    return prompt


def _generate(base_url: str, prompt: str, *, max_new_tokens: int = 8) -> dict[str, Any]:
    body = _post(
        base_url,
        "/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "return_text_in_logprobs": False,
        },
    )
    output_ids = body.get("output_ids")
    rows = body.get("meta_info", {}).get("output_token_logprobs")
    assert isinstance(body.get("text"), str) and body["text"]
    assert isinstance(output_ids, list) and output_ids
    assert isinstance(rows, list) and len(rows) == len(output_ids)
    for output_id, row in zip(output_ids, rows, strict=True):
        assert isinstance(row, list) and len(row) >= 2
        assert math.isfinite(float(row[0]))
        assert int(row[1]) == output_id
    return body


def _chat(base_url: str, content: str | list[dict[str, Any]]) -> str:
    body = _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": _served_model_id(base_url),
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    text = body["choices"][0]["message"]["content"]
    assert isinstance(text, str) and text
    return text


def _assert_choice(text: str, choices: Sequence[str], expected: str) -> None:
    matches = [choice for choice in choices if re.search(rf"\b{choice}\b", text.upper())]
    assert matches == [expected], f"expected {expected!r}, got {text!r}"


def _assert_generation_equivalent(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["output_ids"] == right["output_ids"]
    left_rows = left["meta_info"]["output_token_logprobs"]
    right_rows = right["meta_info"]["output_token_logprobs"]
    deltas = [
        abs(float(left_row[0]) - float(right_row[0]))
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    assert max(deltas, default=0.0) <= 3e-2, deltas


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _red_image_url() -> str:
    width = height = 112
    row = b"\x00" + bytes((255, 0, 0)) * width
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(row * height)),
            _png_chunk(b"IEND", b""),
        ]
    )
    fixture = _manifest()["generated_fixtures"]["solid_red_png"]
    assert len(payload) == fixture["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == fixture["sha256"]
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _image_content() -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": _red_image_url()}},
        {"type": "text", "text": "Is the image RED or BLUE? Answer only RED or BLUE."},
    ]


def _trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []
    return [json.loads(event_path.read_bytes()) for event_path in sorted(path.glob("*.json"))]


@pytest.fixture(scope="module")
def execution_evidence() -> dict[str, Any]:
    _assert_hardware_profile()
    trace_path = Path("/logs/verifier/gemma4-pcg-runtime")
    trace_path.mkdir(parents=True, exist_ok=True)
    for event_path in trace_path.glob("*.json"):
        event_path.unlink()
    os.environ["SGLANG_PCG_TRACE"] = str(trace_path)
    prompt = _chat_prompt("What is six multiplied by seven? Answer only 42.")

    with _server(piecewise=False, port_offset=41) as base_url:
        eager_text = _generate(base_url, prompt)
        eager_image = _chat(base_url, _image_content())

    with _server(piecewise=True, port_offset=42) as base_url:
        pcg_text = _generate(base_url, prompt)
        pcg_image = _chat(base_url, _image_content())
        repeated_text = [
            _chat(base_url, "What is six multiplied by seven? Answer only 42.") for _ in range(4)
        ]

    events = _trace_events(trace_path)
    counts = Counter(str(event["event"]) for event in events)
    print({"profile": _profile_name(), "trace_counts": dict(counts)})
    return {
        "eager_text": eager_text,
        "eager_image": eager_image,
        "pcg_text": pcg_text,
        "pcg_image": pcg_image,
        "repeated_text": repeated_text,
        "events": events,
        "counts": counts,
    }


def test_forced_pcg_captures_authentic_gemma4(execution_evidence: dict[str, Any]) -> None:
    counts = execution_evidence["counts"]
    assert counts["runner_init_end"] >= 1
    assert counts["capture_end"] >= 1
    init_events = [event for event in execution_evidence["events"] if event["event"] == "runner_init_end"]
    assert any(set(_CAPTURE_TOKENS).issubset(event["capture_num_tokens"]) for event in init_events)


def test_public_text_replays_with_logprob_parity(execution_evidence: dict[str, Any]) -> None:
    assert execution_evidence["counts"]["replay_end"] >= 1
    _assert_choice(execution_evidence["eager_text"]["text"], ("42", "36"), "42")
    _assert_choice(execution_evidence["pcg_text"]["text"], ("42", "36"), "42")
    _assert_generation_equivalent(execution_evidence["eager_text"], execution_evidence["pcg_text"])


def test_real_image_serving_preserves_active_pcg(execution_evidence: dict[str, Any]) -> None:
    assert execution_evidence["counts"]["runner_init_end"] >= 1
    assert execution_evidence["counts"]["replay_end"] >= 1
    _assert_choice(execution_evidence["eager_image"], ("RED", "BLUE"), "RED")
    _assert_choice(execution_evidence["pcg_image"], ("RED", "BLUE"), "RED")


def test_repeated_public_prefills_use_forced_pcg(execution_evidence: dict[str, Any]) -> None:
    assert execution_evidence["counts"]["replay_end"] >= 4
    for text in execution_evidence["repeated_text"]:
        _assert_choice(text, ("42", "36"), "42")


def test_eager_public_text_output_ids_and_logprobs_remain_compatible(
    execution_evidence: dict[str, Any],
) -> None:
    # The direct task-base MMMU P2P owns broad image-chat compatibility. Keep
    # this control focused on the distinct eager /generate response contract;
    # _generate already validates nonempty output IDs and aligned finite
    # token-logprob rows before this assertion runs.
    _assert_choice(execution_evidence["eager_text"]["text"], ("42", "36"), "42")
