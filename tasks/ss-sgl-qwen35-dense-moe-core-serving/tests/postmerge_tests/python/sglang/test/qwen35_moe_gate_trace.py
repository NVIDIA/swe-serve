# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-owned observation of live Qwen3.5 sparse-expert dispatch."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

_SELECTED_ID_FIELDS = (
    "topk_ids",
    "topk_indices",
    "selected_experts",
    "expert_ids",
)
_ROUTER_LOGIT_FIELDS = ("router_logits", "gating_output")
_EXPERT_COUNT = 256
_EXPERTS_PER_TOKEN = 8


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dimension) for dimension in shape]


def _field(value: Any, names: Iterable[str]) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _route_payloads(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    payloads: list[Any] = []
    for name in (*_SELECTED_ID_FIELDS, *_ROUTER_LOGIT_FIELDS, "topk_output"):
        if name in kwargs:
            payloads.append(kwargs[name])
    payloads.extend(args[1:])
    return payloads


def _route_tensors(
    payloads: list[Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    selected_ids: torch.Tensor | None = None
    router_logits: torch.Tensor | None = None

    def is_selected_ids(value: Any) -> bool:
        shape = _shape(value)
        return (
            isinstance(value, torch.Tensor)
            and not torch.is_floating_point(value)
            and bool(shape)
            and shape[-1] == _EXPERTS_PER_TOKEN
        )

    def is_router_logits(value: Any) -> bool:
        shape = _shape(value)
        return (
            isinstance(value, torch.Tensor)
            and torch.is_floating_point(value)
            and bool(shape)
            and shape[-1] == _EXPERT_COUNT
        )

    for payload in payloads:
        candidate_ids = _field(payload, _SELECTED_ID_FIELDS)
        candidate_logits = _field(payload, _ROUTER_LOGIT_FIELDS)
        if selected_ids is None and is_selected_ids(candidate_ids):
            selected_ids = candidate_ids
        if router_logits is None and is_router_logits(candidate_logits):
            router_logits = candidate_logits

        if selected_ids is None and is_selected_ids(payload):
            selected_ids = payload
        if router_logits is None and is_router_logits(payload):
            router_logits = payload

    return selected_ids, router_logits


def _hidden_states(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    selected_ids: torch.Tensor | None,
    router_logits: torch.Tensor | None,
) -> torch.Tensor | None:
    candidate = kwargs.get("hidden_states")
    if isinstance(candidate, torch.Tensor):
        return candidate
    for value in args:
        if isinstance(value, torch.Tensor) and value is not selected_ids and value is not router_logits:
            return value
    return None


def _output_tensor(output: Any) -> torch.Tensor | None:
    candidate = output[0] if isinstance(output, tuple) and output else output
    return candidate if isinstance(candidate, torch.Tensor) else None


def _token_count(shape: list[int]) -> int:
    count = 1
    for dimension in shape[:-1]:
        count *= dimension
    return count


def _build_route_record(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> dict[str, Any] | None:
    payloads = _route_payloads(args, kwargs)
    selected_ids, router_logits = _route_tensors(payloads)
    hidden_states = _hidden_states(args, kwargs, selected_ids, router_logits)
    output_tensor = _output_tensor(output)

    if hidden_states is None or output_tensor is None or (selected_ids is None and router_logits is None):
        return None

    route_source = "selected_ids"
    if selected_ids is None:
        assert router_logits is not None
        selected_ids = torch.topk(
            router_logits,
            k=_EXPERTS_PER_TOKEN,
            dim=-1,
        ).indices
        route_source = "router_logits_topk"

    hidden_shape = _shape(hidden_states)
    topk_ids_shape = _shape(selected_ids)
    if not hidden_shape or not topk_ids_shape:
        return None
    topk_width = int(topk_ids_shape[-1])
    flat_topk_ids = selected_ids.detach().cpu().reshape(-1, topk_width)
    minimum_unique_experts_per_token = min(int(torch.unique(row).numel()) for row in flat_topk_ids)
    output_shape = _shape(output_tensor)
    observation_kind = (
        "route_selection" if output_shape == topk_ids_shape else "expert_forward"
    )

    return {
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown")),
        "backend": type(module).__name__,
        "backend_module": type(module).__module__,
        "observation_kind": observation_kind,
        "route_source": route_source,
        "topk_ids_shape": topk_ids_shape,
        "topk_width": topk_width,
        "routed_tokens": int(flat_topk_ids.shape[0]),
        "hidden_tokens": _token_count(hidden_shape),
        "minimum_unique_experts_per_token": minimum_unique_experts_per_token,
        "unique_experts": int(torch.unique(selected_ids).numel()),
        "minimum_expert": int(selected_ids.min().item()),
        "maximum_expert": int(selected_ids.max().item()),
        "router_logits_shape": _shape(router_logits),
        "hidden_shape": hidden_shape,
        "output_shape": output_shape,
        "output_finite": bool(torch.isfinite(output_tensor).all().item()),
    }


def _install_module_forward_trace(trace_root: str) -> Any:
    directory = Path(trace_root)
    capture_marker = directory / "capture-next"
    armed_marker = directory / "capture-ready"
    armed = threading.Event()
    stopped = threading.Event()

    def mark_armed() -> None:
        armed.set()
        armed_marker.write_text("ready\n", encoding="utf-8")

    if capture_marker.is_file():
        mark_armed()

    def watch_capture_marker() -> None:
        while not stopped.wait(0.01):
            if capture_marker.is_file() and not armed.is_set():
                mark_armed()

    watcher = threading.Thread(
        target=watch_capture_marker,
        name="qwen35-moe-trace-arm",
        daemon=True,
    )
    watcher.start()

    def observe(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not armed.is_set():
            return
        # The verifier waits for this acknowledgement before sending a routed
        # request. Do not consume a forward in the event/file publication gap.
        if not armed_marker.is_file():
            return
        if not capture_marker.is_file():
            armed.clear()
            armed_marker.unlink(missing_ok=True)
            return
        try:
            record = _build_route_record(module, args, kwargs, output)
        except Exception:  # preserve production; keep looking for a valid dispatch
            return
        if record is None:
            return

        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"route-{os.getpid()}.json").write_text(
            json.dumps(record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        capture_marker.unlink(missing_ok=True)
        armed_marker.unlink(missing_ok=True)
        armed.clear()

    hook_handle = torch.nn.modules.module.register_module_forward_hook(
        observe,
        with_kwargs=True,
    )

    class TraceHandle:
        def remove(self) -> None:
            stopped.set()
            hook_handle.remove()

    return TraceHandle()


def install_qwen35_moe_trace() -> None:
    trace_root = os.environ.get("SGLANG_QWEN35_MOE_TRACE_DIR")
    if trace_root:
        _install_module_forward_trace(trace_root)
