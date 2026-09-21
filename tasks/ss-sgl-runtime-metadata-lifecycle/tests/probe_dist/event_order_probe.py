# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only dependency tracing through SGLang's production plugin seams."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import torch
from dependency_trace import DependencyTrace
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_mode = os.environ.get("SGLANG_EVENT_ORDER_MODE", "")
_lock = threading.RLock()
_marker_context = threading.local()

_original_event_record = torch.cuda.Event.record
_original_event_wait = torch.cuda.Event.wait

_EAGLE_GRAPH_MODE = "eagle_graph"
_EAGLE_GRAPH_TO_FALLBACK_MODE = "eagle_graph_to_fallback"

_state = {
    "inside_eagle_generation": False,
    "inside_eagle_execute": False,
    "current_iteration": None,
    "next_iteration": 0,
    "pending_iterations": deque(),
    "paired_iterations": [],
    "trace": DependencyTrace(),
    "trace_active": False,
    "cuda_api_patched": False,
    "marker_events": [],
    "read_boundary_done": None,
    "tail_done": None,
    "injected": False,
    "observed": False,
    "graph_iterations_completed": 0,
    "forced_graph_fallback": False,
    "fallback_iteration": None,
    "fallback_batch_id": None,
    "fallback_query_count": 0,
    "fallback_iteration_recorded": False,
    "iteration_kinds": {},
}


def _armed() -> bool:
    value = os.environ.get("SGLANG_EVENT_ORDER_ARM_FILE")
    return bool(value) and Path(value).is_file()


def _write_result(payload: dict) -> None:
    destination = Path(os.environ["SGLANG_EVENT_ORDER_RESULT_FILE"])
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, destination)


def _write_diagnostic(stage: str) -> None:
    value = os.environ.get("SGLANG_EVENT_ORDER_DIAGNOSTIC_FILE")
    if not value:
        return
    trace = _state["trace"]
    with _lock:
        payload = {
            "stage": stage,
            "graph_iterations_completed": _state["graph_iterations_completed"],
            "forced_graph_fallback": _state["forced_graph_fallback"],
            "fallback_iteration": _state["fallback_iteration"],
            "fallback_query_count": _state["fallback_query_count"],
            "fallback_iteration_recorded": _state["fallback_iteration_recorded"],
            "pending_iterations": list(_state["pending_iterations"]),
            "paired_iterations": list(_state["paired_iterations"]),
            "trace_node_count": trace.node_count,
            "trace_edge_count": trace.edge_count,
            "unresolved_wait_count": len(trace.unresolved_waits),
        }
        destination = Path(value)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, destination)


def _stream_key(stream) -> str:
    if stream is None:
        stream = torch.cuda.current_stream()
    return f"{getattr(stream, 'device', None)}:{int(stream.cuda_stream)}"


def _tracing() -> bool:
    return bool(_state["trace_active"] and not _state["observed"])


def _is_eagle_mode() -> bool:
    return _mode in {_EAGLE_GRAPH_MODE, _EAGLE_GRAPH_TO_FALLBACK_MODE}


def _traced_event_record(event, stream=None):
    result = _original_event_record(event, stream)
    if _tracing():
        marker = getattr(_marker_context, "value", None)
        with _lock:
            _state["trace"].record_event(
                id(event),
                _stream_key(stream),
                marker=marker,
            )
    return result


def _traced_event_wait(event, stream=None):
    result = _original_event_wait(event, stream)
    if _tracing():
        with _lock:
            _state["trace"].wait_event(id(event), _stream_key(stream))
    return result


def _patch_cuda_dependency_apis() -> None:
    if _state["cuda_api_patched"]:
        return
    torch.cuda.Event.record = _traced_event_record
    torch.cuda.Event.wait = _traced_event_wait
    _state["cuda_api_patched"] = True


@contextmanager
def _marker(kind: str, iteration: int):
    previous = getattr(_marker_context, "value", None)
    _marker_context.value = (kind, iteration)
    try:
        yield
    finally:
        _marker_context.value = previous


def _record_marker(kind: str, iteration: int) -> int:
    with _marker(kind, iteration):
        event = torch.cuda.Event()
        event.record()
    _state["marker_events"].append(event)
    node = _state["trace"].marker(kind, iteration)
    if node is None:
        raise RuntimeError(f"dependency tracer missed {kind} marker for iteration {iteration}")
    return node


def _record_fallback_read_boundary() -> None:
    read_boundary_done = torch.cuda.Event()
    read_boundary_done.record()
    _state["read_boundary_done"] = read_boundary_done


def _append_unrelated_forward_tail(iteration: int | None = None) -> None:
    # This work is deliberately ordered after the live metadata reads. The
    # dependency graph—not its elapsed duration—determines whether scheduling
    # incorrectly depends on the unrelated tail.
    torch.cuda._sleep(6_000_000_000)
    if iteration is None:
        tail_done = torch.cuda.Event()
        tail_done.record()
        _state["tail_done"] = tail_done
    else:
        _record_marker("tail", iteration)


def _delay_before_fallback_boundary() -> None:
    torch.cuda._sleep(2_000_000_000)


def _around_eagle_metadata_boundary(original, backend, *args, **kwargs):
    result = original(backend, *args, **kwargs)
    iteration = _state["current_iteration"]
    if (
        _is_eagle_mode()
        and _state["inside_eagle_execute"]
        and _tracing()
        and iteration is not None
        and _state["trace"].marker("read", iteration) is None
    ):
        _record_marker("read", iteration)
    return result


def _around_eagle_generation(original, worker, *args, **kwargs):
    batch = args[0] if args else kwargs.get("batch")
    is_decode = bool(
        batch is not None
        and getattr(batch, "forward_mode", None) is not None
        and batch.forward_mode.is_decode()
    )
    should_probe = _is_eagle_mode() and is_decode and _armed() and not _state["observed"]
    if should_probe:
        _state["trace_active"] = True
        _state["inside_eagle_generation"] = True
    try:
        return original(worker, *args, **kwargs)
    finally:
        if should_probe:
            _state["inside_eagle_generation"] = False


def _around_eagle_execute(original, runner, *args, **kwargs):
    graph_iteration_limit = 1 if _mode == _EAGLE_GRAPH_TO_FALLBACK_MODE else 2
    should_probe = (
        _is_eagle_mode()
        and _state["inside_eagle_generation"]
        and _tracing()
        and _state["graph_iterations_completed"] < graph_iteration_limit
    )
    iteration = None
    if should_probe:
        iteration = _state["next_iteration"]
        _state["next_iteration"] += 1
        _state["current_iteration"] = iteration
        _state["inside_eagle_execute"] = True
    try:
        result = original(runner, *args, **kwargs)
    finally:
        if should_probe:
            _state["inside_eagle_execute"] = False
            _state["current_iteration"] = None
    if should_probe:
        _state["graph_iterations_completed"] += 1
        _state["iteration_kinds"][iteration] = "graph"
        if _state["trace"].marker("read", iteration) is None:
            raise RuntimeError(
                f"initialized production read boundary was not observed for iteration {iteration}"
            )
        _append_unrelated_forward_tail(iteration)
        _state["pending_iterations"].append(iteration)
        _write_diagnostic("graph_iteration_queued")
    return result


def _around_eagle_can_run_graph(original, runner, *args, **kwargs):
    can_run = original(runner, *args, **kwargs)
    forward_batch = args[0] if args else kwargs.get("forward_batch")
    is_selected_fallback_batch = (
        _state["forced_graph_fallback"]
        and not _state["fallback_iteration_recorded"]
        and forward_batch is not None
        and id(forward_batch) == _state["fallback_batch_id"]
    )
    if is_selected_fallback_batch:
        _state["fallback_query_count"] += 1
        _write_diagnostic("graph_fallback_reconfirmed")
        return False
    should_force_fallback = (
        _mode == _EAGLE_GRAPH_TO_FALLBACK_MODE
        and _state["inside_eagle_generation"]
        and _tracing()
        and _state["graph_iterations_completed"] >= 1
        and not _state["forced_graph_fallback"]
    )
    if not should_force_fallback:
        return can_run
    if not can_run:
        raise RuntimeError(
            "the transition probe requires an otherwise graph-eligible draft-extension iteration"
        )
    iteration = _state["next_iteration"]
    _state["next_iteration"] += 1
    _state["forced_graph_fallback"] = True
    _state["fallback_iteration"] = iteration
    _state["fallback_batch_id"] = id(forward_batch)
    _state["fallback_query_count"] = 1
    _state["iteration_kinds"][iteration] = "fallback"
    _write_diagnostic("graph_fallback_selected")
    return False


def _around_eagle_shared_write(original, *args, **kwargs):
    if _is_eagle_mode() and _tracing() and _state["pending_iterations"]:
        iteration = _state["pending_iterations"].popleft()
        _record_marker("write", iteration)
        _state["paired_iterations"].append(iteration)
        _write_diagnostic("shared_write_paired")
    return original(*args, **kwargs)


def _around_eager_forward(original, runner, *args, **kwargs):
    forward_batch = args[0] if args else kwargs.get("forward_batch")
    forward_mode = getattr(forward_batch, "forward_mode", None) if forward_batch is not None else None
    is_decode = bool(forward_mode is not None and forward_mode.is_decode())
    is_draft_extend = bool(forward_mode is not None and forward_mode.is_draft_extend_v2())
    is_forced_eagle_fallback = (
        _mode == _EAGLE_GRAPH_TO_FALLBACK_MODE
        and _state["inside_eagle_generation"]
        and _state["forced_graph_fallback"]
        and not _state["fallback_iteration_recorded"]
        and is_draft_extend
    )
    if is_forced_eagle_fallback:
        iteration = _state["fallback_iteration"]
        if iteration is None:
            raise RuntimeError("forced graph fallback has no assigned iteration")
        _delay_before_fallback_boundary()
        result = original(runner, *args, **kwargs)
        _record_marker("read", iteration)
        _append_unrelated_forward_tail(iteration)
        _state["pending_iterations"].append(iteration)
        _state["fallback_iteration_recorded"] = True
        _write_diagnostic("fallback_iteration_queued")
        return result
    if _mode == "plain_eager_fallback" and is_decode and _armed() and not _state["injected"]:
        _state["injected"] = True
        _delay_before_fallback_boundary()
        result = original(runner, *args, **kwargs)
        _record_fallback_read_boundary()
        _append_unrelated_forward_tail()
        return result
    return original(runner, *args, **kwargs)


def _around_ngram_forward(original, worker, *args, **kwargs):
    batch = args[0] if args else kwargs.get("batch")
    is_decode = bool(
        batch is not None
        and getattr(batch, "forward_mode", None) is not None
        and batch.forward_mode.is_decode()
    )
    if _mode == "ngram_fallback" and is_decode and _armed() and not _state["injected"]:
        _state["injected"] = True
        _delay_before_fallback_boundary()
        result = original(worker, *args, **kwargs)
        _record_fallback_read_boundary()
        _append_unrelated_forward_tail()
        return result
    return original(worker, *args, **kwargs)


def _dependency_payload(scheduler) -> dict | None:
    if len(_state["paired_iterations"]) < 2:
        return None
    trace = _state["trace"]
    pairs = []
    for iteration in _state["paired_iterations"][:2]:
        read = trace.marker("read", iteration)
        tail = trace.marker("tail", iteration)
        write = trace.marker("write", iteration)
        if read is None or tail is None or write is None:
            return None
        pairs.append(
            {
                "iteration": iteration,
                "kind": _state["iteration_kinds"].get(iteration),
                "read_happens_before_write": trace.has_path(read, write),
                "tail_happens_before_write": trace.has_path(tail, write),
            }
        )
    safety = all(pair["read_happens_before_write"] for pair in pairs)
    tail_dependency = all(pair["tail_happens_before_write"] for pair in pairs)
    return {
        "probe_mode": _mode,
        "dependency_pairs": pairs,
        "read_boundary_complete_at_schedule_marker": safety,
        "tail_was_complete_at_schedule_marker": tail_dependency,
        "shared_write_marker_queued": True,
        "shared_write_progress_before_metadata_read": 0 if safety else 1,
        "trace_node_count": trace.node_count,
        "trace_edge_count": trace.edge_count,
        "unresolved_wait_count": len(trace.unresolved_waits),
        "forced_graph_fallback": _state["forced_graph_fallback"],
        "fallback_iteration_recorded": _state["fallback_iteration_recorded"],
        "scheduler_type": "sglang.srt.managers.scheduler.Scheduler",
        "used_initialized_runner": bool(
            hasattr(scheduler.model_worker, "server_args") and hasattr(scheduler, "req_to_token_pool")
        ),
    }


def _fallback_payload(scheduler) -> dict | None:
    read_boundary_done = _state["read_boundary_done"]
    tail_done = _state["tail_done"]
    if read_boundary_done is None or tail_done is None:
        return None
    schedule_marker = torch.cuda.Event()
    schedule_marker.record(scheduler.schedule_stream)
    schedule_marker.synchronize()
    payload = {
        "probe_mode": _mode,
        "read_boundary_complete_at_schedule_marker": bool(read_boundary_done.query()),
        "tail_was_complete_at_schedule_marker": bool(tail_done.query()),
        "scheduler_type": "sglang.srt.managers.scheduler.Scheduler",
        "used_initialized_runner": bool(
            hasattr(scheduler.model_worker, "server_args") and hasattr(scheduler, "req_to_token_pool")
        ),
    }
    tail_done.synchronize()
    return payload


def _after_real_scheduling_pass(original, scheduler, *args, **kwargs):
    result = original(scheduler, *args, **kwargs)
    if _state["observed"]:
        return result
    if _is_eagle_mode():
        payload = _dependency_payload(scheduler)
    else:
        payload = _fallback_payload(scheduler)
    if payload is not None:
        _state["observed"] = True
        _write_result(payload)
    return result


def activate() -> None:
    if _is_eagle_mode():
        _patch_cuda_dependency_apis()
        HookRegistry.register(
            "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.forward_batch_generation",
            _around_eagle_generation,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.layers.attention.flashinfer_backend."
            "FlashInferAttnBackend.init_forward_metadata_out_graph",
            _around_eagle_metadata_boundary,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner."
            "EAGLEDraftExtendCudaGraphRunner.execute",
            _around_eagle_execute,
            HookType.AROUND,
        )
        if _mode == _EAGLE_GRAPH_TO_FALLBACK_MODE:
            HookRegistry.register(
                "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner."
                "EAGLEDraftExtendCudaGraphRunner.can_run_graph",
                _around_eagle_can_run_graph,
                HookType.AROUND,
            )
            HookRegistry.register(
                "sglang.srt.model_executor.model_runner.ModelRunner.forward",
                _around_eager_forward,
                HookType.AROUND,
            )
        HookRegistry.register(
            "sglang.srt.speculative.spec_utils.assign_req_to_token_pool_func",
            _around_eagle_shared_write,
            HookType.AROUND,
        )
    elif _mode == "plain_eager_fallback":
        HookRegistry.register(
            "sglang.srt.model_executor.model_runner.ModelRunner.forward",
            _around_eager_forward,
            HookType.AROUND,
        )
    elif _mode == "ngram_fallback":
        HookRegistry.register(
            "sglang.srt.speculative.ngram_worker.NGRAMWorker.forward_batch_generation",
            _around_ngram_forward,
            HookType.AROUND,
        )
    else:
        raise RuntimeError(f"unsupported event-order probe mode: {_mode!r}")

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
        _after_real_scheduling_pass,
        HookType.AROUND,
    )
    _write_diagnostic("plugin_activated")
