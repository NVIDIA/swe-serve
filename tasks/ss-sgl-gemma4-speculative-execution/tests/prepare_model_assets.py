#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_MOUNTED_CACHE = Path("/hf-cache/hub")


def _load_manifest() -> dict[str, Any]:
    with Path("/tests/model_assets.json").open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError("model asset manifest must be a JSON object")
    return value


def _snapshot_path(cache_root: Path, asset: dict[str, Any]) -> Path:
    repository = str(asset["repository"]).replace("/", "--")
    return cache_root / f"models--{repository}" / "snapshots" / str(asset["revision"])


def _is_complete(path: Path, asset: dict[str, Any]) -> bool:
    if not (path / "config.json").is_file():
        return False
    return all(
        (path / shard["file"]).is_file() and (path / shard["file"]).stat().st_size == int(shard["size_bytes"])
        for shard in asset["weight_shards"]
    )


def _resolve(asset: dict[str, Any]) -> Path:
    snapshot = _snapshot_path(_MOUNTED_CACHE, asset)
    if not _is_complete(snapshot, asset):
        raise RuntimeError(
            f"required prepared model snapshot is missing or incomplete: {snapshot}; "
            "run scripts/preflight.py and mount the verified task view at /hf-cache"
        )
    return snapshot


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"target", "draft"}:
        raise SystemExit("usage: prepare_model_assets.py {target|draft}")
    asset = _load_manifest()["assets"][sys.argv[1]]
    print(_resolve(asset))


if __name__ == "__main__":
    main()
