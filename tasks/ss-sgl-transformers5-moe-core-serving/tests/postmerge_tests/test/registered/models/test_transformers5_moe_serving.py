# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import requests
from transformers5_moe_verifier_support import (
    find_available_port,
    kill_process_tree,
    popen_launch_server,
)

_MODEL_PATH = os.environ["TRANSFORMERS5_MOE_MODEL_PATH"]
_TP_SIZE = int(os.environ["TRANSFORMERS5_MOE_TP_SIZE"])
_REQUEST_TIMEOUT_SECONDS = 600


def _base_url() -> str:
    return f"http://127.0.0.1:{find_available_port(20000)}"


def _get(base_url: str, route: str) -> Any:
    response = requests.get(f"{base_url}{route}", timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


@contextmanager
def _running_server() -> Iterator[str]:
    base_url = _base_url()
    process = popen_launch_server(
        _MODEL_PATH,
        base_url,
        timeout=1800,
        other_args=[
            "--model-impl",
            "transformers",
            "--tp-size",
            str(_TP_SIZE),
            "--context-length",
            "1024",
            "--max-running-requests",
            "8",
            "--chunked-prefill-size",
            "512",
            "--max-prefill-tokens",
            "1024",
            "--mem-fraction-static",
            "0.9",
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--expert-distribution-recorder-mode",
            "stat",
            "--moe-runner-backend",
            "triton",
            "--disable-cuda-graph",
        ],
    )
    try:
        yield base_url
    finally:
        kill_process_tree(process.pid)


def test_layer_a_authentic_moe_model_loads_with_recorder() -> None:
    with _running_server() as base_url:
        models = _get(base_url, "/v1/models")
    entries = models["data"]
    assert len(entries) == 1
    assert entries[0]["id"] == _MODEL_PATH
