#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_path(asset: dict[str, Any]) -> Path:
    repository = asset["repository"].replace("/", "--")
    return Path("/hf-cache/hub") / f"models--{repository}" / "snapshots" / asset["revision"]


def _verify_asset(name: str, asset: dict[str, Any]) -> None:
    model_path = _asset_path(asset)
    config = _load_json(model_path / "config.json")
    if config.get("model_type") != asset["config_model_type"]:
        raise RuntimeError(f"unexpected model_type in {model_path / 'config.json'}")
    if asset["config_architecture"] not in config.get("architectures", []):
        raise RuntimeError(f"missing checkpoint architecture in {model_path / 'config.json'}")

    # Hash every load-relevant file the runtime actually consumes, not the weights
    # alone: config.json (architecture/dtype/head), the tokenizer (tokenizer.json,
    # tokenizer_config.json, vocab.json, merges.txt) and generation_config.json all
    # change scoring behaviour, so a swap of any of them must fail closed.
    checked = list(asset["weight_shards"]) + list(asset.get("load_relevant_files", []))
    for entry in checked:
        file_path = model_path / entry["file"]
        if not file_path.is_file():
            raise RuntimeError(f"required checkpoint file missing: {file_path}")
        actual_size = file_path.stat().st_size
        if actual_size != entry["size_bytes"]:
            raise RuntimeError(
                f"size mismatch for {file_path}: {actual_size} != {entry['size_bytes']}"
            )
        actual_sha256 = _sha256(file_path)
        if actual_sha256 != entry["sha256"]:
            raise RuntimeError(
                f"hash mismatch for {file_path}: {actual_sha256} != {entry['sha256']}"
            )

    print(
        "validated authentic offline checkpoint "
        f"{name}={asset['repository']}@{asset['revision']} "
        f"({len(asset['weight_shards'])} shards + "
        f"{len(asset.get('load_relevant_files', []))} config/tokenizer files)"
    )


def main() -> None:
    manifest = _load_json(Path("/tests/model_assets.json"))
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required for reproducible scoring")
    for name in manifest["common_assets"]:
        _verify_asset(name, manifest["assets"][name])


if __name__ == "__main__":
    main()
