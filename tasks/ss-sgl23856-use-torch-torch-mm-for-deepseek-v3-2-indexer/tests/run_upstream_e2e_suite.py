#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run scored maintainer P2P groups from the immutable task-base source."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"empty upstream E2E manifest: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"duplicate upstream E2E entry: {path}")
    return entries


def _maintainer_nodes(p2p_path: Path) -> list[str]:
    nodes = [
        node
        for node in _read_manifest(p2p_path)
        if node.startswith("test/registered/") and node.count("::") == 2
    ]
    if not nodes:
        raise ValueError(f"no exact maintainer nodes in {p2p_path}")
    return nodes


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: run_upstream_e2e_suite.py P2P_MANIFEST RESULT_JSON GROUP_RUNNER"
        )

    p2p_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    group_runner = Path(sys.argv[3])
    expected = _maintainer_nodes(p2p_path)
    groups = list(dict.fromkeys("::".join(node.split("::")[:2]) for node in expected))
    if not group_runner.is_file():
        raise FileNotFoundError(group_runner)

    groups_result: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(groups):
        group_result_path = result_path.with_name(f"{result_path.stem}-group-{index}.json")
        completed = subprocess.run(
            [sys.executable, "-I", str(group_runner), group, str(group_result_path)],
            check=False,
        )
        if not group_result_path.is_file():
            groups_result.append(
                {"logical_group": group, "runner_exit_code": completed.returncode, "missing": True}
            )
            continue
        group_result = json.loads(group_result_path.read_text())
        group_result["runner_exit_code"] = completed.returncode
        groups_result.append(group_result)
        for nodeid, outcome in group_result.get("nodes", {}).items():
            if nodeid in nodes:
                raise ValueError(f"upstream E2E node collected by multiple groups: {nodeid}")
            nodes[nodeid] = outcome

    missing = sorted(set(expected) - set(nodes))
    extra = sorted(set(nodes) - set(expected))
    passed = sorted(nodeid for nodeid, outcome in nodes.items() if outcome.get("passed") is True)
    failed = sorted(nodeid for nodeid in expected if nodeid not in passed)
    result = {
        "schema_version": 1,
        "groups": groups_result,
        "expected": expected,
        "nodes": nodes,
        "missing": missing,
        "extra": extra,
        "passed": passed,
        "failed": failed,
        "collection_complete": not missing and not extra,
        "all_passed": not missing and not extra and not failed,
    }
    _write_result(result_path, result)

    return 0 if result["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
