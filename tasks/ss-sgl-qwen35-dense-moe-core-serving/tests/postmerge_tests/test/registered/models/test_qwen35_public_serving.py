# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from _qwen35_verifier_support import (
    find_available_port,
    kill_process_tree,
    popen_launch_server,
)
from transformers import AutoTokenizer

_SERVER_TIMEOUT_SECONDS = 1200
_REQUEST_TIMEOUT_SECONDS = 300


def _manifest() -> dict[str, Any]:
    value = json.loads(Path("/tests/model_assets.json").read_bytes())
    assert isinstance(value, dict)
    return value


def _profile_name() -> str:
    return os.environ.get("SWE_SERVE_HARDWARE_PROFILE", "h100_1")


def _profile() -> dict[str, Any]:
    profiles = _manifest()["profiles"]
    name = _profile_name()
    assert name in profiles, f"unknown Qwen3.5 hardware profile {name!r}"
    return profiles[name]


def _model_path() -> str:
    asset = _manifest()["assets"]["dense_native"]
    env_name = asset["model_path_env"]
    configured_path = os.environ.get(env_name)
    assert configured_path, f"{env_name} was not resolved by task prep"
    path = Path(configured_path)
    assert path.is_dir(), f"missing revision-pinned checkpoint: {path}"
    return str(path)


def _assert_hardware_profile() -> None:
    profile = _profile()
    required = int(profile["allocation_gpus"])
    visible = torch.cuda.device_count()
    assert visible >= required, f"profile {_profile_name()} requires {required} GPUs, got {visible}"
    expected_name = str(profile["gpu_name_contains"]).lower()
    names = [torch.cuda.get_device_name(index) for index in range(required)]
    assert all(expected_name in name.lower() for name in names), (
        f"profile {_profile_name()} expected {profile['gpu_name_contains']}, got {names}"
    )


def _common_server_args() -> list[str]:
    return [
        "--tp-size",
        str(_profile()["tp_size"]),
        "--context-length",
        "4096",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "8192",
        "--max-mamba-cache-size",
        "32",
        "--mem-fraction-static",
        "0.65",
        "--disable-cuda-graph",
        *list(_profile()["dense_server_args"]),
    ]


def _server_url() -> str:
    return f"http://127.0.0.1:{find_available_port(20000)}"


@contextmanager
def _server() -> Iterator[str]:
    base_url = _server_url()
    process = popen_launch_server(
        _model_path(),
        base_url,
        timeout=_SERVER_TIMEOUT_SECONDS,
        other_args=_common_server_args(),
        env={**os.environ, "SGLANG_QWEN35_GATE_COMPAT": "1"},
    )
    try:
        yield base_url
    finally:
        kill_process_tree(process.pid)


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    with _server() as base_url:
        yield base_url


def _get(base_url: str, route: str) -> Any:
    response = requests.get(f"{base_url}{route}", timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _post(base_url: str, route: str, payload: dict[str, Any]) -> Any:
    response = requests.post(
        f"{base_url}{route}",
        json=payload,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    assert response.ok, f"POST {route} returned HTTP {response.status_code}: {response.text}"
    return response.json()


def _served_model_id(base_url: str) -> str:
    body = _get(base_url, "/v1/models")
    models = body.get("data", [])
    assert models, f"OpenAI models endpoint returned no models: {body}"
    model_id = models[0].get("id")
    assert isinstance(model_id, str) and model_id
    return model_id


def _chat_text(base_url: str, content: str | list[dict[str, Any]]) -> str:
    body = _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": _served_model_id(base_url),
            "messages": [
                {"role": "system", "content": "Follow the requested answer format exactly."},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 16,
        },
    )
    text = body["choices"][0]["message"]["content"]
    assert isinstance(text, str) and text, body
    assert body["usage"]["completion_tokens"] > 0, body
    return text


def _assert_choice(text: str, choices: Sequence[str]) -> str:
    matches = [choice for choice in choices if re.search(rf"\b{re.escape(choice)}\b", text.upper())]
    assert len(matches) == 1, f"expected exactly one of {choices}, got {text!r}"
    return matches[0]


@lru_cache(maxsize=1)
def _offline_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        _model_path(),
        local_files_only=True,
        trust_remote_code=True,
    )
    assert tokenizer.chat_template, "pinned checkpoint has no chat template"
    return tokenizer


def _generate_prompt(question: str) -> str:
    prompt = _offline_tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "Follow the requested answer format exactly."},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(prompt, str) and prompt
    return prompt


def _fixture_data_url(name: str, payload: bytes) -> str:
    fixture = _manifest()["fixtures"][name]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    assert actual_sha256 == fixture["sha256"], (
        f"fixture {name} hash mismatch: {actual_sha256} != {fixture['sha256']}"
    )
    assert len(payload) == fixture["size_bytes"]
    return f"data:{fixture['media_type']};base64,{base64.b64encode(payload).decode('ascii')}"


def _generate_with_logprobs(base_url: str, prompt: str) -> dict[str, Any]:
    body = _post(
        base_url,
        "/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 12},
            "return_logprob": True,
            "return_text_in_logprobs": False,
        },
    )
    assert isinstance(body, dict), body
    assert isinstance(body.get("text"), str) and body["text"], body
    output_ids = body.get("output_ids")
    assert isinstance(output_ids, list) and output_ids, body
    assert all(isinstance(token_id, int) for token_id in output_ids), body
    rows = body.get("meta_info", {}).get("output_token_logprobs")
    assert isinstance(rows, list) and len(rows) == len(output_ids), body
    for row in rows:
        assert isinstance(row, list) and len(row) >= 2, body
        assert math.isfinite(float(row[0])), body
        assert int(row[1]) >= 0, body
    return body


def _generate_batch(base_url: str, prompts: Sequence[str]) -> list[dict[str, Any]]:
    body = _post(
        base_url,
        "/generate",
        {
            "text": list(prompts),
            "sampling_params": {"temperature": 0, "max_new_tokens": 12},
        },
    )
    assert isinstance(body, list) and len(body) == len(prompts), body
    for result in body:
        assert isinstance(result, dict), body
        assert isinstance(result.get("text"), str) and result["text"], result
        assert result.get("meta_info", {}).get("completion_tokens", 0) > 0, result
    return body


def test_authentic_checkpoint_loads_through_openai_models_api(server_url: str) -> None:
    _assert_hardware_profile()
    assert _served_model_id(server_url)


def test_openai_chat_known_answers_are_correct(server_url: str) -> None:
    _assert_hardware_profile()
    capital = _chat_text(server_url, "What is the capital of France? Answer only PARIS.")
    product = _chat_text(server_url, "What is six multiplied by seven? Answer only 42.")

    assert _assert_choice(capital, ("PARIS", "LONDON")) == "PARIS"
    assert _assert_choice(product, ("42", "36")) == "42"


def test_openai_video_request_identifies_semantic_color(server_url: str) -> None:
    _assert_hardware_profile()
    video_bytes = Path(os.environ["QWEN35_GREEN_VIDEO"]).read_bytes()
    video_url = _fixture_data_url("solid_green_mp4", video_bytes)
    content = [
        {"type": "video_url", "video_url": {"url": video_url}},
        {
            "type": "text",
            "text": "Is the video's dominant color GREEN or PURPLE? Answer only GREEN or PURPLE.",
        },
    ]

    answer = _chat_text(server_url, content)

    assert _assert_choice(answer, ("GREEN", "PURPLE")) == "GREEN"


def test_generate_known_answers_return_logprobs_at_declared_profile(server_url: str) -> None:
    _assert_hardware_profile()
    prompts = [
        _generate_prompt("What is the capital of Japan? Answer only TOKYO."),
        _generate_prompt("What is eight plus five? Answer only 13."),
    ]
    capital = _generate_with_logprobs(server_url, prompts[0])
    total = _generate_with_logprobs(server_url, prompts[1])

    assert _assert_choice(capital["text"], ("TOKYO", "OSAKA")) == "TOKYO"
    assert _assert_choice(total["text"], ("13", "15")) == "13"


def test_batch_generate_preserves_independent_known_answers_at_declared_profile(
    server_url: str,
) -> None:
    _assert_hardware_profile()
    prompts = [
        _generate_prompt("What is three plus four? Answer only 7."),
        _generate_prompt("At what temperature in degrees Celsius does water freeze? Answer only 0."),
    ]
    results = _generate_batch(server_url, prompts)

    assert _assert_choice(results[0]["text"], ("7", "9")) == "7"
    assert _assert_choice(results[1]["text"], ("0", "100")) == "0"
