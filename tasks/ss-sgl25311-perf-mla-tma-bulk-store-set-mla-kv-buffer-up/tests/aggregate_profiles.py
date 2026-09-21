#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed aggregation for independent sgl25311 profiler processes."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

MIN_RUNS = 5
MAX_RUNS = 20
EXPECTED_BATCH_SIZES = (768, 2048, 8192, 16384)
RESULT_RE = re.compile(
    r"^RESULT bs=(\d+) cand_ms=([\d.]+) ref_ms=([\d.]+) speedup=([\d.]+)$",
    re.MULTILINE,
)


def _load_run(path: Path) -> dict[int, dict[str, float]]:
    matches = list(RESULT_RE.finditer(path.read_text()))
    if len(matches) != len(EXPECTED_BATCH_SIZES):
        raise ValueError(
            f"{path} has {len(matches)} RESULT lines; expected {len(EXPECTED_BATCH_SIZES)}"
        )

    results: dict[int, dict[str, float]] = {}
    for match in matches:
        batch_size = int(match.group(1))
        candidate_ms, reference_ms, speedup = map(float, match.groups()[1:])
        if batch_size in results or batch_size not in EXPECTED_BATCH_SIZES:
            raise ValueError(f"{path} contains duplicate or unexpected batch size {batch_size}")
        if not all(
            math.isfinite(value) and value > 0
            for value in (candidate_ms, reference_ms, speedup)
        ):
            raise ValueError(f"{path} contains a non-positive or non-finite measurement")
        if not math.isclose(speedup, reference_ms / candidate_ms, rel_tol=0.02, abs_tol=0.01):
            raise ValueError(f"{path} contains an inconsistent speedup for batch size {batch_size}")
        results[batch_size] = {
            "candidate_ms": candidate_ms,
            "reference_ms": reference_ms,
            "speedup": speedup,
        }

    if set(results) != set(EXPECTED_BATCH_SIZES):
        raise ValueError(f"{path} does not contain the exact expected batch-size set")
    return results


def main() -> int:
    run_count = len(sys.argv) - 2
    if not MIN_RUNS <= run_count <= MAX_RUNS:
        print(
            f"usage: {sys.argv[0]} OUTPUT_JSON RUN_1 ... RUN_N "
            f"({MIN_RUNS} <= N <= {MAX_RUNS})",
            file=sys.stderr,
        )
        return 2

    output_path = Path(sys.argv[1])
    run_paths = [Path(argument) for argument in sys.argv[2:]]
    if len(set(run_paths)) != run_count:
        raise ValueError(f"expected {run_count} distinct profiler logs")

    runs = [_load_run(path) for path in run_paths]
    shapes: dict[str, dict[str, object]] = {}
    for batch_size in EXPECTED_BATCH_SIZES:
        speedups = [run[batch_size]["speedup"] for run in runs]
        median_speedup = statistics.median(speedups)
        minimum = min(speedups)
        maximum = max(speedups)
        relative_span = (maximum - minimum) / median_speedup
        shapes[str(batch_size)] = {
            "speedups": speedups,
            "median_speedup": median_speedup,
            "minimum_speedup": minimum,
            "maximum_speedup": maximum,
            "relative_span": relative_span,
        }
        # Keep a backwards-readable RESULT summary while making the JSON sidecar
        # authoritative. Normalizing candidate_ms to one makes the printed ratio
        # exact without pretending independently aggregated timings are paired.
        print(
            f"RESULT bs={batch_size} cand_ms=1.00000 "
            f"ref_ms={median_speedup:.5f} speedup={median_speedup:.5f}"
        )
        print(
            f"AGGREGATE bs={batch_size} runs={run_count} statistic=median "
            f"min={minimum:.5f} max={maximum:.5f} relative_span={relative_span:.6f}"
        )

    payload = {
        "schema_version": 1,
        "processes": run_count,
        "statistic": "median",
        "run_logs": [str(path) for path in run_paths],
        "shapes": shapes,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
