# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any


def _emit(event: str, **fields: Any) -> None:
    path = os.environ.get("SGLANG_PCG_TRACE")
    if not path:
        return
    payload = {
        "event": event,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        **fields,
    }
    trace_dir = Path(path)
    trace_dir.mkdir(parents=True, exist_ok=True)
    event_path = trace_dir / f"{payload['time_ns']}-{payload['pid']}-{event}.json"
    event_path.write_bytes(json.dumps(payload, sort_keys=True).encode())


def _install_trace() -> None:
    try:
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            PiecewiseCudaGraphRunner,
        )
    except Exception as error:
        _emit("instrumentation_import_error", error=repr(error))
        return

    original_init = PiecewiseCudaGraphRunner.__init__
    original_capture = PiecewiseCudaGraphRunner.capture
    original_replay = PiecewiseCudaGraphRunner.replay

    @functools.wraps(original_init)
    def traced_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _emit("runner_init_begin")
        original_init(self, *args, **kwargs)
        _emit(
            "runner_init_end",
            capture_num_tokens=list(getattr(self, "capture_num_tokens", ())),
            max_num_tokens=int(getattr(self, "max_num_tokens", -1)),
            is_multimodal=bool(getattr(self, "is_multimodal", False)),
        )

    @functools.wraps(original_capture)
    def traced_capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        _emit("capture_begin", capture_num_tokens=list(getattr(self, "capture_num_tokens", ())))
        result = original_capture(self, *args, **kwargs)
        _emit("capture_end")
        return result

    @functools.wraps(original_replay)
    def traced_replay(self: Any, forward_batch: Any, *args: Any, **kwargs: Any) -> Any:
        input_ids = getattr(forward_batch, "input_ids", ())
        _emit(
            "replay_begin",
            num_tokens=len(input_ids),
            has_input_embeds=getattr(forward_batch, "input_embeds", None) is not None,
            return_logprob=bool(getattr(forward_batch, "return_logprob", False)),
        )
        result = original_replay(self, forward_batch, *args, **kwargs)
        _emit("replay_end", num_tokens=len(input_ids))
        return result

    PiecewiseCudaGraphRunner.__init__ = traced_init
    PiecewiseCudaGraphRunner.capture = traced_capture
    PiecewiseCudaGraphRunner.replay = traced_replay
    _emit("instrumentation_installed")


if os.environ.get("SGLANG_PCG_TRACE"):
    _install_trace()
