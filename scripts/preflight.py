#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight the pinned Hugging Face / data assets required by the release tasks."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "scripts/configs/hf_cache_requirements.json"
TASK_SET_PATH = REPO_ROOT / "task_list.csv"
NO_ASSETS = "no_external_assets"
READINESS_NAME = "readiness.json"
PROFILE_BY_GPUS = {"0": "cpu", "1": "h100_1"}
_READINESS_IDENTITY = ("schema_version", "manifest_sha256", "requirements_sha256")
_READINESS_REQUIREMENTS = ("schema_version", "task_manifest", "tasks", "views", "assets")
_REGISTRY_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class RequirementsError(ValueError):
    """The fixed requirements or task manifest is invalid."""


class PreflightError(RuntimeError):
    """Preparation or verification cannot safely complete."""

    def __init__(
        self,
        message: str,
        *,
        affected_task_ids: Sequence[str] = (),
        asset_errors: dict[str, PreflightError] | None = None,
        view_errors: dict[str, PreflightError] | None = None,
        task_errors: dict[str, PreflightError] | None = None,
        verified_task_ids: Sequence[str] = (),
    ) -> None:
        self.affected_task_ids = tuple(affected_task_ids)
        self.asset_errors = dict(asset_errors or {})
        self.view_errors = dict(view_errors or {})
        self.task_errors = dict(task_errors or {})
        self.verified_task_ids = tuple(verified_task_ids)
        super().__init__(message)


class InsufficientDiskError(PreflightError):
    """Selected assets do not fit on their target filesystem."""


class MissingAssetError(PreflightError):
    """Required local content is absent or partial."""


class CorruptAssetError(PreflightError):
    """Required local content does not match its declaration."""


@dataclass(frozen=True)
class ReleaseContract:
    """One validated HF/data requirements file plus its authoritative task CSV."""

    manifest: dict[str, Any]
    task_set_path: Path
    roster: tuple[str, ...]
    profiles: dict[str, str]


@dataclass(frozen=True)
class ReadyTask:
    """Verified HF/data inputs for one selected task."""

    task_id: str
    hf_view: Path | None


@dataclass(frozen=True)
class AssetReadinessDiagnostic:
    """One manifest-derived asset fact attached to a failed task selection."""

    repository: str
    revision: str
    gated: bool
    state: str


@dataclass(frozen=True)
class TaskReadinessFailure:
    """One selected task that cannot safely reach benchmark launch."""

    task_id: str
    reason: str
    assets: tuple[AssetReadinessDiagnostic, ...]


class SelectionReadinessError(PreflightError):
    """Selected release content is not ready for benchmark execution."""

    def __init__(
        self,
        *,
        requirements_path: Path,
        cache_root: Path,
        selected_task_ids: Sequence[str],
        failures: Sequence[TaskReadinessFailure],
    ) -> None:
        self.requirements_path = requirements_path.resolve()
        self.cache_root = cache_root.resolve()
        self.selected_task_ids = tuple(selected_task_ids)
        self.failures = tuple(failures)
        super().__init__(self._render())

    def _command(
        self,
        *,
        verify_only: bool,
        without_token: bool,
        dockerless: bool = False,
    ) -> str:
        repo_root = next(
            (
                parent
                for parent in self.requirements_path.parents
                if (parent / "scripts/preflight.py").is_file()
            ),
            REPO_ROOT,
        )
        python = repo_root / ".venv/bin/python3"
        command = []
        if without_token:
            command.extend(("/usr/bin/env", "-u", "HF_TOKEN"))
        command.extend(
            [
                str(python),
                str(repo_root / "scripts/preflight.py"),
                "--cache-root",
                str(self.cache_root),
            ]
        )
        task_ids = (
            self.selected_task_ids if verify_only else tuple(failure.task_id for failure in self.failures)
        )
        for task_id in task_ids:
            command.extend(("--task", task_id))
        if verify_only:
            command.append("--verify-only")
        rendered = shlex.join(command)
        if dockerless:
            rendered += ' --image-storage-root "$IMAGE_STORAGE_ROOT"'
        return rendered

    def _render(self) -> str:
        lines = [
            "selected HF/data readiness verification failed before Harbor or any workload "
            "container was launched:"
        ]
        diagnostic_assets = []
        for failure in self.failures:
            lines.append(f"  [affected-task] {failure.task_id}: {failure.reason}")
            if not failure.assets:
                lines.append(f"    [no-asset] {NO_ASSETS}")
                continue
            for asset in failure.assets:
                diagnostic_assets.append(asset)
                access = "gated" if asset.gated else "public"
                lines.append(f"    [{access}-asset {asset.state}] {asset.repository}@{asset.revision}")

        gated_states = {asset.state for asset in diagnostic_assets if asset.gated}
        if any(not asset.gated for asset in diagnostic_assets):
            lines.append(
                "Public selected assets do not require HF_TOKEN; valid local content is reused "
                "and missing content can be prepared anonymously on the host."
            )
        gated_states_allowing_token = gated_states - {"verified"}
        preparation_allows_token = bool(gated_states_allowing_token)
        if gated_states & {"missing", "corrupt"}:
            lines.append(
                "Set HF_TOKEN in the trusted host environment before the preparation command "
                "for the listed missing or corrupt gated assets."
            )
        elif gated_states_allowing_token:
            lines.append(
                "Make HF_TOKEN available only to trusted host-side preparation if the listed "
                "gated content is absent; already valid cached content is reused without it."
            )
        elif "verified" in gated_states:
            lines.append("The selected recovery does not require HF_TOKEN; verified gated assets are reused.")
        else:
            lines.append("The selected recovery does not require HF_TOKEN.")
        lines.extend(
            (
                "Oracle and NOP do not require LLM-provider credentials. Any execution mode "
                "requires its selected task assets to be present in the verified release cache. "
                "When missing selected assets are gated, HF_TOKEN is required only for host-side "
                "cache preparation and is never passed to benchmark execution. Verifiers receive "
                "the resulting verified read-only assets, never HF_TOKEN.",
                "Host-side preparation command:",
                f"  {self._command(verify_only=False, without_token=not preparation_allows_token)}",
                "Dockerless host-side preparation command (set IMAGE_STORAGE_ROOT first):",
                "  "
                + self._command(
                    verify_only=False,
                    without_token=not preparation_allows_token,
                    dockerless=True,
                ),
                "Offline --verify-only recovery command:",
                f"  {self._command(verify_only=True, without_token=True)}",
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class HfPlan:
    """Checked HF/data work ready to stage."""

    requirements: dict[str, Any]
    task_ids: tuple[str, ...]
    cache_root: Path
    token: str | None = field(repr=False, compare=False)
    asset_ids: tuple[str, ...]
    views: dict[str, list[str]]
    missing: tuple[str, ...]
    facts: dict[str, dict[str, Any]]
    required_bytes: int
    verify_only: bool
    reuse: bool

    def __post_init__(self) -> None:
        if self.missing and (self.reuse or self.verify_only):
            raise ValueError("missing assets cannot be reused or treated as verified")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(
    path: Path = REQUIREMENTS_PATH,
    *,
    expected_task_set: Path | None = None,
) -> ReleaseContract:
    """Load and validate the single HF/data requirements and roster contract."""
    try:
        manifest = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RequirementsError(f"HF/data requirements are unreadable: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RequirementsError(f"HF/data requirements must be a JSON object: {path}")
    if manifest.get("schema_version") != 1:
        raise RequirementsError(f"unsupported HF/data requirements schema: {path}")
    declaration = manifest.get("task_manifest")
    if not isinstance(declaration, dict):
        raise RequirementsError("HF/data requirements have no task_manifest object")
    relative = declaration.get("path")
    expected_sha256 = declaration.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise RequirementsError("HF/data task_manifest declaration is invalid")
    task_set = next(
        (parent / relative for parent in path.parents if (parent / relative).is_file()),
        None,
    )
    if task_set is None:
        raise RequirementsError(f"authoritative task CSV is missing: {relative}")
    if expected_task_set is not None and task_set.resolve() != expected_task_set.resolve():
        raise RequirementsError(f"HF/data requirements must use {expected_task_set}")
    try:
        task_set_raw = task_set.read_bytes()
        with task_set.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, strict=True))
        roster = tuple(row["task_id"] for row in rows)
        profiles = {row["task_id"]: PROFILE_BY_GPUS[row["gpus"]] for row in rows}
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError) as exc:
        raise RequirementsError(f"authoritative task CSV is unreadable: {task_set}: {exc}") from exc
    if hashlib.sha256(task_set_raw).hexdigest() != expected_sha256:
        raise RequirementsError("authoritative task CSV digest does not match the requirements")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not roster or len(profiles) != len(rows) or list(tasks) != list(roster):
        raise RequirementsError("HF/data tasks do not exactly match the authoritative CSV")
    views = manifest.get("views")
    if not isinstance(views, dict) or any(
        not isinstance(view, dict) or view.get("layout") != "hub_subdir" for view in views.values()
    ):
        raise RequirementsError("HF/data views must use the normalized Hugging Face Hub layout")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or any(
        not isinstance(asset, dict) or "view_path" in asset for asset in assets.values()
    ):
        raise RequirementsError("HF/data assets must not declare legacy view paths")
    return ReleaseContract(
        manifest=manifest,
        task_set_path=task_set.resolve(),
        roster=roster,
        profiles=profiles,
    )


def readiness_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Identify the authoritative CSV and HF/data requirements."""
    try:
        hf_requirements = {key: manifest[key] for key in _READINESS_REQUIREMENTS}
        requirements = json.dumps(hf_requirements, sort_keys=True, separators=(",", ":")).encode()
        manifest_sha256 = manifest["task_manifest"]["sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RequirementsError("HF/data requirements are incomplete") from exc
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "requirements_sha256": hashlib.sha256(requirements).hexdigest(),
    }


def build_readiness(
    identity: dict[str, Any],
    facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build readiness from verified HF/data facts."""
    return {**identity, "tasks": facts}


def read_readiness(path: Path) -> dict[str, Any]:
    try:
        readiness = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"readiness is missing or unreadable: {path}") from exc
    if not isinstance(readiness, dict) or not isinstance(readiness.get("tasks"), dict):
        raise PreflightError(f"readiness has an invalid structure: {path}")
    return readiness


def verify_readiness(path: Path, expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify selected task records and return them from cumulative readiness."""
    try:
        actual = read_readiness(path)
    except PreflightError as error:
        raise PreflightError(
            str(error),
            affected_task_ids=tuple(expected["tasks"]),
        ) from error
    if any(actual.get(key) != expected[key] for key in _READINESS_IDENTITY):
        raise PreflightError(
            "readiness is stale for the selected release",
            affected_task_ids=tuple(expected["tasks"]),
        )
    selected = {}
    stale = []
    for task_id, task in expected["tasks"].items():
        entry = actual["tasks"].get(task_id)
        if entry != task:
            stale.append(task_id)
            continue
        selected[task_id] = entry
    if stale:
        raise PreflightError(
            "readiness is stale or missing for task: " + ", ".join(stale),
            affected_task_ids=tuple(stale),
        )
    return selected


def merge_readiness(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        actual = read_readiness(path)
    except PreflightError:
        return expected
    if any(actual.get(key) != expected[key] for key in _READINESS_IDENTITY):
        return expected
    return {**expected, "tasks": {**actual["tasks"], **expected["tasks"]}}


def write_readiness(path: Path, value: dict[str, Any]) -> None:
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".readiness.", delete=False
        ) as output:
            temporary = output.name
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def invalidate_readiness(path: Path, identity: dict[str, Any], task_ids: Sequence[str]) -> None:
    """Remove failed selected tasks without shrinking unrelated valid readiness."""
    try:
        actual = read_readiness(path)
    except PreflightError:
        path.unlink(missing_ok=True)
        return
    if any(actual.get(key) != identity[key] for key in _READINESS_IDENTITY):
        path.unlink(missing_ok=True)
        return
    for task_id in task_ids:
        actual["tasks"].pop(task_id, None)
    if actual["tasks"]:
        write_readiness(path, actual)
    else:
        path.unlink(missing_ok=True)


def _safe_ready_view(cache_root: Path, relative: Any, task_id: str) -> Path | None:
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PreflightError(f"invalid ready_view in readiness for task: {task_id}")
    root = cache_root.resolve()
    view = (root / relative).resolve()
    try:
        view.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"ready_view escapes cache root for task: {task_id}") from exc
    if not view.is_dir():
        raise PreflightError(f"ready_view is missing for task {task_id}: {view}")
    return view


def select_tasks(
    roster: Sequence[str],
    profiles: dict[str, str],
    requested: Sequence[str],
    profile: str | None = None,
) -> list[str]:
    unknown = sorted(set(requested) - set(roster))
    if unknown:
        raise RequirementsError(
            "--task is not in the release roster (exports/task_list.csv): " + ", ".join(unknown)
        )
    selected = set(requested)
    return [
        task_id
        for task_id in roster
        if (not requested or task_id in selected) and (profile is None or profiles[task_id] == profile)
    ]


def _selection(requirements: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    declaration = requirements["tasks"][task_id]
    return None if declaration == NO_ASSETS else declaration


def _selected(
    requirements: dict[str, Any], task_ids: Sequence[str]
) -> tuple[list[str], dict[str, list[str]]]:
    asset_ids: list[str] = []
    views: dict[str, list[str]] = {}
    for task_id in task_ids:
        selection = _selection(requirements, task_id)
        if selection is None:
            continue
        view_assets = views.setdefault(selection["view"], [])
        for asset_id in selection["assets"]:
            if asset_id not in asset_ids:
                asset_ids.append(asset_id)
            if asset_id not in view_assets:
                view_assets.append(asset_id)
    return asset_ids, views


def asset_snapshot(cache_root: Path, asset: dict[str, Any]) -> Path:
    repository = asset["repository"].replace("/", "--")
    repo_type = asset["repo_type"]
    if repo_type not in {"model", "dataset"}:
        raise RequirementsError(f"unsupported Hugging Face repo_type: {repo_type!r}")
    kind = "models" if repo_type == "model" else "datasets"
    return cache_root / "hub" / f"{kind}--{repository}" / "snapshots" / asset["revision"]


def _validate_asset(cache_root: Path, asset_id: str, asset: dict[str, Any]) -> None:
    snapshot = asset_snapshot(cache_root, asset)
    if not snapshot.is_dir():
        raise MissingAssetError(f"missing pinned snapshot {asset['repository']}@{asset['revision']}")
    for relative, metadata in asset["required_files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise MissingAssetError(f"missing required file for {asset_id}: {relative}")
        if "size_bytes" in metadata and path.stat().st_size != metadata["size_bytes"]:
            raise CorruptAssetError(
                f"size mismatch for {asset_id}:{relative}: {path}; remove {snapshot} and rerun preparation"
            )
        if "sha256" in metadata and _sha256(path) != metadata["sha256"]:
            raise CorruptAssetError(
                f"SHA-256 mismatch for {asset_id}:{relative}: {path}; remove {snapshot} and rerun preparation"
            )
    for pattern, expected in asset.get("required_globs", {}).items():
        found = sum(path.is_file() for path in snapshot.glob(pattern))
        if found < expected:
            raise MissingAssetError(f"file-count mismatch for {asset_id}:{pattern}: {found} < {expected}")
        if found > expected:
            raise CorruptAssetError(
                f"file-count mismatch for {asset_id}:{pattern}: {found} > {expected}; "
                f"remove {snapshot} and rerun preparation"
            )
    if "snapshot_size_bytes" in asset:
        found = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
        if found != asset["snapshot_size_bytes"]:
            raise CorruptAssetError(
                f"snapshot-size mismatch for {asset_id}: {snapshot}; remove {snapshot} and rerun preparation"
            )


def _main_ref(cache_root: Path, asset: dict[str, Any]) -> Path:
    return asset_snapshot(cache_root, asset).parents[1] / "refs" / "main"


def _validate_main_ref(cache_root: Path, asset_id: str, asset: dict[str, Any]) -> None:
    ref = _main_ref(cache_root, asset)
    refs = ref.parent
    if refs.is_symlink() or (refs.exists() and not refs.is_dir()):
        raise CorruptAssetError(f"invalid refs directory for {asset_id}: {refs}")
    if ref.is_symlink() or (ref.exists() and not ref.is_file()):
        raise CorruptAssetError(f"invalid main ref for {asset_id}: {ref}")
    if not ref.is_file():
        raise MissingAssetError(f"missing main ref for {asset_id}: {ref}")
    try:
        revision = ref.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CorruptAssetError(f"unreadable main ref for {asset_id}: {ref}") from exc
    if revision != asset["revision"]:
        raise CorruptAssetError(f"stale or malformed main ref for {asset_id}: {ref}")


def _prepare_main_ref(cache_root: Path, asset_id: str, asset: dict[str, Any]) -> None:
    try:
        _validate_main_ref(cache_root, asset_id, asset)
        return
    except MissingAssetError:
        pass
    ref = _main_ref(cache_root, asset)
    temporary = ""
    try:
        ref.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=ref.parent, prefix=".main.", delete=False
        ) as output:
            temporary = output.name
            output.write(asset["revision"])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, ref)
        temporary = ""
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    _validate_main_ref(cache_root, asset_id, asset)


def _no_exist_paths(cache_root: Path, asset: dict[str, Any]) -> list[Path]:
    root = asset_snapshot(cache_root, asset).parents[1] / ".no_exist" / asset["revision"]
    return [root / relative for relative in asset.get("no_exist_files", [])]


def _validate_no_exist(cache_root: Path, asset_id: str, asset: dict[str, Any]) -> None:
    markers = _no_exist_paths(cache_root, asset)
    if not markers:
        return
    root = markers[0].parent
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise CorruptAssetError(f"invalid no-exist directory for {asset_id}: {root}")
    for marker in markers:
        if marker.is_symlink() or (marker.exists() and (not marker.is_file() or marker.stat().st_size)):
            raise CorruptAssetError(f"invalid no-exist marker for {asset_id}: {marker}")
        if not marker.is_file():
            raise MissingAssetError(f"missing no-exist marker for {asset_id}: {marker}")


def _prepare_no_exist(cache_root: Path, asset_id: str, asset: dict[str, Any]) -> None:
    try:
        _validate_no_exist(cache_root, asset_id, asset)
        return
    except MissingAssetError:
        pass
    for marker in _no_exist_paths(cache_root, asset):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
    _validate_no_exist(cache_root, asset_id, asset)


def _hub_repo_size(asset: dict[str, Any], token: str | None) -> int:
    from huggingface_hub import HfApi

    info = HfApi(token=token).repo_info(
        repo_id=asset["repository"],
        repo_type=asset["repo_type"],
        revision=asset["revision"],
        files_metadata=True,
        token=token,
    )
    if info.sha != asset["revision"]:
        raise PreflightError(f"Hub resolved {asset['repository']} to an unexpected revision")
    allow_patterns = asset.get("allow_patterns")
    if allow_patterns is None and "source_path" in asset:
        allow_patterns = [asset["source_path"]]
    sizes = []
    for sibling in info.siblings:
        filename = sibling.rfilename
        if allow_patterns and not any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns):
            continue
        if any(fnmatch.fnmatch(filename, pattern) for pattern in asset.get("ignore_patterns", [])):
            continue
        if not isinstance(sibling.size, int):
            raise PreflightError(f"Hub did not report a size for {asset['repository']}:{filename}")
        sizes.append(sibling.size)
    if not sizes:
        raise PreflightError(f"Hub reported no selected files for {asset['repository']}")
    return sum(sizes)


def _validate_hf_token(token: str) -> None:
    from huggingface_hub import HfApi

    HfApi().whoami(token=token)


def _validate_gated_access(asset: dict[str, Any], token: str) -> None:
    from huggingface_hub import auth_check

    auth_check(
        repo_id=asset["repository"],
        repo_type=asset["repo_type"],
        token=token,
    )


def _hub_download(asset: dict[str, Any], hub_cache: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    allow_patterns = asset.get("allow_patterns")
    if allow_patterns is None and "source_path" in asset:
        allow_patterns = [asset["source_path"]]
    snapshot_download(
        repo_id=asset["repository"],
        repo_type=asset["repo_type"],
        revision=asset["revision"],
        cache_dir=str(hub_cache),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=asset.get("ignore_patterns") or None,
    )


def _redact(exc: BaseException, token: str | None) -> str:
    message = str(exc)
    return message.replace(token, "<redacted>") if token else message


def _view_links(
    cache_root: Path, view_id: str, asset_ids: Sequence[str], requirements: dict[str, Any]
) -> list[tuple[Path, str, bool]]:
    layout = requirements["views"][view_id]["layout"]
    if layout != "hub_subdir":
        raise RequirementsError(f"view {view_id!r} must use the normalized Hugging Face Hub layout")
    links = []
    for asset_id in asset_ids:
        asset = requirements["assets"][asset_id]
        if "view_path" in asset:
            raise RequirementsError(f"asset {asset_id!r} declares a legacy view path")
        snapshot = asset_snapshot(cache_root, asset)
        repository_cache = snapshot.parents[1]
        blobs = repository_cache / "blobs"
        snapshots = repository_cache / "snapshots"
        refs = repository_cache / "refs"
        if not blobs.is_dir() or not snapshots.is_dir() or (refs.exists() and not refs.is_dir()):
            raise CorruptAssetError(
                f"prepared repository has an invalid Hub cache structure: {repository_cache}"
            )
        links.append((repository_cache, f"hub/{repository_cache.name}", True))
    return links


def _view_path(cache_root: Path, view_id: str) -> Path:
    if not view_id or view_id in {".", ".."} or Path(view_id).name != view_id:
        raise RequirementsError(f"view ID must be a path leaf: {view_id!r}")
    return cache_root / "views" / "h100_1" / view_id


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _view_link_is_internal(path: Path, repository: Path) -> bool:
    """Check the immediate link destination without following the blob-store link."""
    target = Path(os.path.abspath(path.parent.resolve() / os.readlink(path)))
    return _within(target, repository)


def _hardlink_view_path(source: Path, target: Path, preserve_symlinks: bool) -> None:
    """Materialize one self-contained view path without duplicating file bytes."""
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source.resolve(), target)
        return
    if not source.is_dir():
        raise MissingAssetError(f"view source is missing: {source}")
    safe_root = source if preserve_symlinks else source.parents[1]
    # huggingface_hub >= 1.3 links blobs/<sha> into a store shared across repositories under the
    # hub cache (hub/blobs/<xx>/<sha>). A symlink may therefore leave the repository cache but
    # must stay inside the hub cache. Materialize those blob entries, but preserve snapshot links
    # to repository-local blobs: task verifiers may identify weights by the blob's LFS hash name.
    hub_root = source.parent if preserve_symlinks else safe_root
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_symlink():
            link = Path(os.readlink(path))
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(hub_root.resolve())
            except (OSError, RuntimeError, ValueError) as exc:
                raise CorruptAssetError(f"view source symlink escapes its cache: {path}") from exc
            if link.is_absolute():
                raise CorruptAssetError(f"view source symlink is absolute: {path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if preserve_symlinks and _view_link_is_internal(path, safe_root):
                destination.symlink_to(link)
            elif resolved.is_file():
                os.link(resolved, destination)
            else:
                raise CorruptAssetError(f"view source symlink is not a file: {path}")
        elif path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path.resolve(), destination)


def _build_view(
    cache_root: Path, view_id: str, asset_ids: Sequence[str], requirements: dict[str, Any]
) -> None:
    for asset_id in asset_ids:
        _prepare_main_ref(cache_root, asset_id, requirements["assets"][asset_id])
        _prepare_no_exist(cache_root, asset_id, requirements["assets"][asset_id])
    view = _view_path(cache_root, view_id)
    if view.exists() or view.is_symlink():
        shutil.rmtree(view) if view.is_dir() and not view.is_symlink() else view.unlink()
    view.mkdir(parents=True)
    for source, relative, preserve_symlinks in _view_links(cache_root, view_id, asset_ids, requirements):
        _hardlink_view_path(source, view / relative, preserve_symlinks)


def _validate_view(
    cache_root: Path, view_id: str, asset_ids: Sequence[str], requirements: dict[str, Any]
) -> None:
    view = _view_path(cache_root, view_id)
    if not view.is_dir():
        raise MissingAssetError(f"missing ready view: views/h100_1/{view_id}")
    for asset_id in asset_ids:
        _validate_main_ref(cache_root, asset_id, requirements["assets"][asset_id])
        _validate_no_exist(cache_root, asset_id, requirements["assets"][asset_id])
    for source, relative, preserve_symlinks in _view_links(cache_root, view_id, asset_ids, requirements):
        target = view / relative
        if source.is_file():
            current = [(source, False)]
        else:
            current = [
                (path, preserve_symlinks and path.is_symlink())
                for path in source.rglob("*")
                if path.is_file() or path.is_symlink()
            ]
        for path, preserve_symlink in current:
            candidate = target if source.is_file() else target / path.relative_to(source)
            try:
                if preserve_symlink and _view_link_is_internal(path, source):
                    source_link = Path(os.readlink(path))
                    candidate_link = Path(os.readlink(candidate))
                    candidate_target = candidate.resolve(strict=True)
                    candidate_target.relative_to(target.resolve())
                    valid = (
                        not source_link.is_absolute()
                        and candidate_link == source_link
                        and os.path.samefile(path.resolve(strict=True), candidate_target)
                    )
                elif preserve_symlink:
                    # A link that left the repository cache (shared blob store) was materialized.
                    valid = candidate.is_file() and os.path.samefile(path.resolve(strict=True), candidate)
                else:
                    valid = (
                        not candidate.is_symlink()
                        and candidate.is_file()
                        and os.path.samefile(path.resolve(), candidate)
                    )
            except (OSError, RuntimeError, ValueError):
                valid = False
            if not valid:
                raise CorruptAssetError(f"stale content in view {view_id}: {relative}")


def task_facts(
    requirements: dict[str, Any],
    task_ids: Sequence[str],
    profiles: dict[str, str],
) -> dict[str, dict[str, Any]]:
    tasks = {}
    for task_id in task_ids:
        selection = _selection(requirements, task_id)
        if selection is None:
            tasks[task_id] = {"status": NO_ASSETS, "profile": profiles[task_id]}
        else:
            tasks[task_id] = {
                "status": "ready",
                "profile": profiles[task_id],
                "ready_view": f"views/h100_1/{selection['view']}",
                "assets": [
                    {
                        "repository": requirements["assets"][asset_id]["repository"],
                        "revision": requirements["assets"][asset_id]["revision"],
                    }
                    for asset_id in selection["assets"]
                ],
            }
    return tasks


def _verify_ready_tasks(
    release: ReleaseContract,
    cache_root: Path,
    task_ids: Sequence[str],
) -> dict[str, ReadyTask]:
    """Check selected readiness records and return their task views."""
    facts = task_facts(release.manifest, task_ids, release.profiles)
    entries = verify_readiness(
        cache_root / READINESS_NAME,
        build_readiness(readiness_identity(release.manifest), facts),
    )
    ready = {}
    view_errors = {}
    for task_id in task_ids:
        try:
            ready[task_id] = ReadyTask(
                task_id=task_id,
                hf_view=_safe_ready_view(cache_root, entries[task_id].get("ready_view"), task_id),
            )
        except PreflightError as error:
            view_errors[task_id] = error
    if view_errors:
        raise PreflightError(
            "selected prepared views failed readiness: "
            + "; ".join(f"{task_id}: {failure}" for task_id, failure in view_errors.items()),
            affected_task_ids=tuple(view_errors),
            task_errors=view_errors,
        )
    return ready


def verify_ready_task(
    requirements_path: Path,
    cache_root: Path,
    task_id: str,
) -> ReadyTask:
    return verify_ready_tasks(requirements_path, cache_root, [task_id])[task_id]


def _selection_readiness_failures(
    release: ReleaseContract,
    task_ids: Sequence[str],
    error: PreflightError,
    *,
    default_affected: Sequence[str] = (),
) -> list[TaskReadinessFailure]:
    explicitly_affected = set(error.affected_task_ids)
    if not explicitly_affected and not (error.asset_errors or error.view_errors or error.task_errors):
        explicitly_affected.update(default_affected)
    failures = []
    for task_id in task_ids:
        selection = _selection(release.manifest, task_id)
        selected_asset_ids = () if selection is None else tuple(selection["assets"])
        relevant_asset_ids = tuple(
            asset_id for asset_id in selected_asset_ids if asset_id in error.asset_errors
        )
        view_id = None if selection is None else selection["view"]
        assets_verified = task_id in error.verified_task_ids

        if relevant_asset_ids:
            reason = "; ".join(str(error.asset_errors[asset_id]) for asset_id in relevant_asset_ids)
            diagnostic_asset_ids = relevant_asset_ids
        elif view_id in error.view_errors:
            reason = str(error.view_errors[view_id])
            diagnostic_asset_ids = selected_asset_ids
            assets_verified = True
        elif task_id in error.task_errors:
            reason = str(error.task_errors[task_id])
            diagnostic_asset_ids = selected_asset_ids
        elif task_id in explicitly_affected:
            reason = str(error)
            diagnostic_asset_ids = selected_asset_ids
        else:
            continue

        diagnostics = []
        for asset_id in diagnostic_asset_ids:
            asset = release.manifest["assets"][asset_id]
            asset_error = error.asset_errors.get(asset_id)
            if isinstance(asset_error, CorruptAssetError):
                state = "corrupt"
            elif isinstance(asset_error, MissingAssetError):
                state = "missing"
            elif assets_verified:
                state = "verified"
            else:
                state = "unverified"
            diagnostics.append(
                AssetReadinessDiagnostic(
                    repository=asset["repository"],
                    revision=asset["revision"],
                    gated=bool(asset.get("gated")),
                    state=state,
                )
            )
        failures.append(
            TaskReadinessFailure(
                task_id=task_id,
                reason=reason,
                assets=tuple(diagnostics),
            )
        )
    return failures


def _selection_readiness_error(
    release: ReleaseContract,
    requirements_path: Path,
    cache_root: Path,
    task_ids: Sequence[str],
    error: PreflightError,
) -> SelectionReadinessError:
    failures = _selection_readiness_failures(
        release,
        task_ids,
        error,
        default_affected=task_ids,
    )
    return SelectionReadinessError(
        requirements_path=requirements_path,
        cache_root=cache_root,
        selected_task_ids=task_ids,
        failures=failures,
    )


def verify_ready_tasks(
    requirements_path: Path,
    cache_root: Path,
    task_ids: Sequence[str],
) -> dict[str, ReadyTask]:
    """Check exact selected readiness records before benchmark submission."""
    release = load_release(requirements_path)
    if not task_ids:
        raise RequirementsError("selected task IDs must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise RequirementsError("selected task IDs must be unique")
    selected = select_tasks(release.roster, release.profiles, task_ids)
    try:
        return _verify_ready_tasks(release, cache_root, selected)
    except PreflightError as error:
        raise _selection_readiness_error(
            release,
            requirements_path,
            cache_root,
            selected,
            error,
        ) from error


def _asset_inventory(
    requirements: dict[str, Any],
    task_ids: Sequence[str],
    cache_root: Path,
) -> list[dict[str, Any]]:
    asset_ids, _ = _selected(requirements, task_ids)
    assets = []
    for asset_id in asset_ids:
        asset = requirements["assets"][asset_id]
        estimated = asset.get("snapshot_size_bytes")
        if estimated is None:
            declared_sizes = [item.get("size_bytes") for item in asset["required_files"].values()]
            estimated = sum(size for size in declared_sizes if size is not None) or None
        assets.append(
            {
                "repository": asset["repository"],
                "revision": asset["revision"],
                "cache_snapshot": asset_snapshot(cache_root, asset).as_posix(),
                "requires_hf_token": asset.get("gated", False),
                "estimated_bytes": estimated,
            }
        )
    return assets


def dry_run_plan(
    requirements: dict[str, Any],
    task_ids: Sequence[str],
    profiles: dict[str, str],
    requested: Sequence[str],
    profile: str | None,
    cache_root: Path,
) -> dict[str, Any]:
    assets = _asset_inventory(requirements, task_ids, cache_root)
    external = [
        {
            "task_id": task_id,
            "profile": profiles[task_id],
            "ready_view": f"views/h100_1/{_selection(requirements, task_id)['view']}",
        }
        for task_id in task_ids
        if _selection(requirements, task_id) is not None
    ]
    hf_bytes = sum(asset["estimated_bytes"] or 0 for asset in assets)
    image_storage = image_disk_plan(requirements, task_ids)
    return {
        "manifest": requirements["task_manifest"],
        "profile_filter": profile,
        "task_filter": list(requested) or None,
        "selected_task_count": len(task_ids),
        "external_asset_tasks": external,
        "no_external_assets_count": len(task_ids) - len(external),
        "estimated_hf_data_bytes": hf_bytes,
        "estimated_compressed_image_bytes": image_storage["deduplicated_compressed_bytes"],
        "estimated_combined_bytes": hf_bytes + image_storage["deduplicated_compressed_bytes"],
        "assets_without_size_estimate": sum(asset["estimated_bytes"] is None for asset in assets),
        "unique_assets": assets,
        "container_image_storage": image_storage,
    }


def _image_references(requirements: dict[str, Any], task_ids: Sequence[str]) -> list[str]:
    try:
        return list(
            dict.fromkeys(
                reference for task_id in task_ids for reference in requirements["task_images"][task_id]
            )
        )
    except (KeyError, TypeError) as exc:
        raise RequirementsError("container image inventory is incomplete") from exc


def image_disk_plan(requirements: dict[str, Any], task_ids: Sequence[str]) -> dict[str, Any]:
    """Sum each selected Linux/amd64 compressed image layer digest once."""
    aliases = requirements.get("image_aliases", {})
    references = list(
        dict.fromkeys(
            aliases.get(reference, reference) for reference in _image_references(requirements, task_ids)
        )
    )
    images = requirements.get("container_images")
    if not isinstance(images, dict):
        raise RequirementsError("container image inventory is incomplete")

    unique_layers: dict[str, int] = {}
    naive_bytes = 0
    selected_images = []
    for reference in references:
        try:
            image = images[reference]
            manifest_digest = image["manifest_digest"]
            layers = image["layers"]
        except (KeyError, TypeError) as exc:
            raise RequirementsError(f"container image inventory is incomplete for: {reference}") from exc
        if image.get("platform") != "linux/amd64":
            raise RequirementsError(f"Linux/amd64 image inventory is missing for: {reference}")
        if not isinstance(manifest_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_digest):
            raise RequirementsError(f"container image manifest digest is invalid for: {reference}")
        if not isinstance(layers, list):
            raise RequirementsError(f"container image layer inventory is invalid for: {reference}")
        image_bytes = 0
        for layer in layers:
            digest = layer.get("digest") if isinstance(layer, dict) else None
            size = layer.get("size_bytes") if isinstance(layer, dict) else None
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                or not isinstance(size, int)
                or size < 0
            ):
                raise RequirementsError(f"container image layer is invalid for: {reference}")
            if digest in unique_layers and unique_layers[digest] != size:
                raise RequirementsError(f"container image layer size conflicts for: {digest}")
            unique_layers[digest] = size
            image_bytes += size
        naive_bytes += image_bytes
        selected_images.append(
            {
                "reference": reference,
                "manifest_digest": manifest_digest,
                "layer_count": len(layers),
                "compressed_bytes": image_bytes,
            }
        )

    deduplicated_bytes = sum(unique_layers.values())
    return {
        "platform": "linux/amd64",
        "image_count": len(references),
        "layer_descriptor_count": sum(image["layer_count"] for image in selected_images),
        "unique_layer_count": len(unique_layers),
        "naive_compressed_bytes": naive_bytes,
        "deduplicated_compressed_bytes": deduplicated_bytes,
        "shared_layer_savings_bytes": naive_bytes - deduplicated_bytes,
        "images": selected_images,
    }


def _registry_coordinates(reference: str) -> tuple[str, str, str]:
    name, separator, digest = reference.partition("@")
    if separator:
        manifest_ref = digest
    last_slash = name.rfind("/")
    last_colon = name.rfind(":")
    if last_colon > last_slash:
        if not separator:
            manifest_ref = name[last_colon + 1 :]
        name = name[:last_colon]
    elif not separator:
        manifest_ref = "latest"

    components = name.split("/")
    first = components[0]
    if len(components) > 1 and ("." in first or ":" in first or first == "localhost"):
        registry = first
        repository = "/".join(components[1:])
    else:
        registry = "registry-1.docker.io"
        repository = name if "/" in name else f"library/{name}"
    if registry in {"docker.io", "index.docker.io"}:
        registry = "registry-1.docker.io"
    if not repository or not manifest_ref:
        raise RequirementsError(f"container image reference is invalid: {reference}")
    return registry, repository, manifest_ref


def _registry_bearer_token(challenge: str) -> str:
    if not challenge.lower().startswith("bearer "):
        raise PreflightError("container registry did not offer bearer authentication")
    parameters = dict(re.findall(r'([A-Za-z_]+)="([^"]*)"', challenge))
    realm = parameters.pop("realm", None)
    if not realm:
        raise PreflightError("container registry bearer challenge has no realm")
    url = realm
    if parameters:
        url += "?" + urllib.parse.urlencode(parameters)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"container registry token request failed: {exc}") from exc
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise PreflightError("container registry token response has no token")
    return token


def _registry_manifest(
    registry: str,
    repository: str,
    manifest_ref: str,
    tokens: dict[tuple[str, str], str],
) -> tuple[dict[str, Any], str]:
    quoted_repository = urllib.parse.quote(repository, safe="/")
    quoted_reference = urllib.parse.quote(manifest_ref, safe=":")
    url = f"https://{registry}/v2/{quoted_repository}/manifests/{quoted_reference}"
    key = (registry, repository)

    for attempt in range(2):
        headers = {
            "Accept": _REGISTRY_MANIFEST_ACCEPT,
            "User-Agent": "swe-serve-taskgen-preflight",
        }
        if token := tokens.get(key):
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                reported_digest = response.headers.get("Docker-Content-Digest")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and attempt == 0:
                challenge = exc.headers.get("WWW-Authenticate", "")
                tokens[key] = _registry_bearer_token(challenge)
                continue
            raise PreflightError(
                f"container registry access failed for {repository}@{manifest_ref}: HTTP {exc.code}"
            ) from exc
        except OSError as exc:
            raise PreflightError(
                f"container registry access failed for {repository}@{manifest_ref}: {exc}"
            ) from exc
    else:  # pragma: no cover - the loop either returns or raises
        raise PreflightError(f"container registry access failed for {repository}@{manifest_ref}")

    actual_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if reported_digest and reported_digest != actual_digest:
        raise PreflightError(
            f"container registry returned inconsistent digest for {repository}@{manifest_ref}"
        )
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(
            f"container registry returned an invalid manifest for {repository}@{manifest_ref}"
        ) from exc
    if not isinstance(manifest, dict):
        raise PreflightError(
            f"container registry returned an invalid manifest for {repository}@{manifest_ref}"
        )
    return manifest, actual_digest


def _inspect_registry_image(reference: str, expected_digest: str) -> tuple[str, list[dict[str, Any]]]:
    registry, repository, manifest_ref = _registry_coordinates(reference)
    tokens: dict[tuple[str, str], str] = {}
    manifest, actual_digest = _registry_manifest(registry, repository, manifest_ref, tokens)

    if isinstance(manifest.get("manifests"), list):
        candidates = [
            descriptor
            for descriptor in manifest["manifests"]
            if descriptor.get("platform", {}).get("os") == "linux"
            and descriptor.get("platform", {}).get("architecture") == "amd64"
            and descriptor.get("platform", {}).get("variant") in (None, "")
        ]
        if len(candidates) != 1:
            raise PreflightError(
                f"container registry has {len(candidates)} Linux/amd64 manifests for {reference}"
            )
        if candidates[0].get("digest") != expected_digest:
            raise PreflightError(
                f"container image digest changed for {reference}: "
                f"{candidates[0].get('digest')} != {expected_digest}"
            )
        manifest, actual_digest = _registry_manifest(
            registry,
            repository,
            expected_digest,
            tokens,
        )

    if actual_digest != expected_digest:
        raise PreflightError(
            f"container image digest changed for {reference}: {actual_digest} != {expected_digest}"
        )
    config_digest = manifest.get("config", {}).get("digest")
    layers = manifest.get("layers")
    if (
        not isinstance(config_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest)
        or not isinstance(layers, list)
    ):
        raise PreflightError(f"container registry manifest is incomplete for {reference}")
    normalized_layers = []
    for layer in layers:
        digest = layer.get("digest")
        size = layer.get("size")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise PreflightError(f"container registry layer is invalid for {reference}")
        normalized_layers.append({"digest": digest, "size_bytes": size})
    return config_digest, normalized_layers


def _local_docker_storage_root() -> Path | None:
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except OSError:
        return None
    root = result.stdout.strip()
    return Path(root) if result.returncode == 0 and root else None


def _local_image_present(reference: str, config_digest: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == config_digest


def image_precheck(
    requirements: dict[str, Any],
    task_ids: Sequence[str],
    *,
    image_storage_root: Path | None = None,
) -> dict[str, Any]:
    """Verify selected registry manifests and report deduplicated missing layer bytes."""
    storage = image_disk_plan(requirements, task_ids)
    references = _image_references(requirements, task_ids)
    aliases = requirements.get("image_aliases", {})
    docker_storage_root = None if image_storage_root is not None else _local_docker_storage_root()
    docker_available = docker_storage_root is not None
    storage_root = image_storage_root or docker_storage_root
    all_layers: dict[str, int] = {}
    present_layers: set[str] = set()
    present_images = 0

    for reference in references:
        recorded = requirements["container_images"][aliases.get(reference, reference)]
        config_digest, registry_layers = _inspect_registry_image(
            reference,
            recorded["manifest_digest"],
        )
        if registry_layers != recorded["layers"]:
            raise PreflightError(f"container image layers changed for {reference}")
        for layer in registry_layers:
            all_layers[layer["digest"]] = layer["size_bytes"]
        if docker_available and _local_image_present(reference, config_digest):
            present_images += 1
            present_layers.update(layer["digest"] for layer in registry_layers)
        print(f"[image-access] {reference}@{recorded['manifest_digest']} (linux/amd64)")

    missing_layers = set(all_layers) - present_layers
    missing_bytes = sum(all_layers[digest] for digest in missing_layers)
    if docker_available:
        local = f"{present_images}/{len(references)} exact image input(s) present"
    elif image_storage_root is not None:
        local = "operator image store selected; all selected layers treated as missing"
    else:
        local = "local Docker store unavailable; all selected layers treated as missing"
    print(
        f"[image-storage] {missing_bytes} compressed bytes across "
        f"{len(missing_layers)} unique missing layer(s); {local}"
    )
    return {
        **storage,
        "registry_verified_image_count": len(references),
        "local_docker_store_checked": docker_available,
        "image_storage_root": str(storage_root) if storage_root is not None else None,
        "local_exact_image_count": present_images,
        "missing_unique_layer_count": len(missing_layers),
        "deduplicated_missing_compressed_bytes": missing_bytes,
    }


def precheck(
    requirements: dict[str, Any],
    task_ids: Sequence[str],
    profiles: dict[str, str],
    cache_root: Path,
    *,
    verify_only: bool,
    token: str | None,
) -> HfPlan:
    """Validate local state and remote access without transferring HF/data."""
    asset_ids, views = _selected(requirements, task_ids)
    facts = task_facts(requirements, task_ids, profiles)

    missing: list[str] = []
    asset_errors: dict[str, PreflightError] = {}
    for asset_id in asset_ids:
        try:
            _validate_asset(cache_root, asset_id, requirements["assets"][asset_id])
        except MissingAssetError as error:
            missing.append(asset_id)
            asset_errors[asset_id] = error
        except CorruptAssetError as error:
            if not verify_only:
                raise
            asset_errors[asset_id] = error
    if verify_only:
        if asset_errors:
            details = "; ".join(f"{asset_id}: {error}" for asset_id, error in asset_errors.items())
            if any(isinstance(error, CorruptAssetError) for error in asset_errors.values()):
                raise CorruptAssetError(
                    f"selected assets failed verification: {details}",
                    asset_errors=asset_errors,
                )
            raise MissingAssetError(
                f"missing or partial assets: {', '.join(missing)}",
                asset_errors=asset_errors,
            )
        view_errors: dict[str, PreflightError] = {}
        for view_id, view_assets in views.items():
            try:
                _validate_view(cache_root, view_id, view_assets, requirements)
            except PreflightError as error:
                view_errors[view_id] = error
        if view_errors:
            details = "; ".join(f"{view_id}: {error}" for view_id, error in view_errors.items())
            if any(isinstance(error, CorruptAssetError) for error in view_errors.values()):
                raise CorruptAssetError(
                    f"selected prepared views failed verification: {details}",
                    view_errors=view_errors,
                )
            raise MissingAssetError(
                f"selected prepared views are missing: {details}",
                view_errors=view_errors,
            )
        return HfPlan(
            requirements=requirements,
            task_ids=tuple(task_ids),
            cache_root=cache_root,
            token=token,
            asset_ids=tuple(asset_ids),
            views=views,
            missing=(),
            facts=facts,
            required_bytes=0,
            verify_only=True,
            reuse=True,
        )

    reuse = not missing
    if reuse:
        for asset_id in asset_ids:
            try:
                _validate_main_ref(cache_root, asset_id, requirements["assets"][asset_id])
                _validate_no_exist(cache_root, asset_id, requirements["assets"][asset_id])
            except MissingAssetError as exc:
                print(f"[rebuild] {exc}", file=sys.stderr)
                reuse = False
                break
    if reuse:
        try:
            for view_id, view_assets in views.items():
                _validate_view(cache_root, view_id, view_assets, requirements)
        except PreflightError as exc:
            print(f"[rebuild] {exc}", file=sys.stderr)
            reuse = False

    gated_assets: dict[tuple[str, str], list[str]] = {}
    for asset_id in missing:
        asset = requirements["assets"][asset_id]
        if asset.get("gated"):
            key = (asset["repo_type"], asset["repository"])
            gated_assets.setdefault(key, []).append(asset_id)
    if gated_assets:
        gated_repositories = [repository for _repo_type, repository in gated_assets]
        if not token:
            raise PreflightError(
                "HF_TOKEN is required to stage gated repositories: " + ", ".join(gated_repositories)
            )
        try:
            _validate_hf_token(token)
        except Exception as exc:
            raise PreflightError(
                "HF_TOKEN validation failed before staging gated repositories: " + _redact(exc, token)
            ) from exc
        for asset_ids_for_repository in gated_assets.values():
            asset = requirements["assets"][asset_ids_for_repository[0]]
            affected_tasks = [
                task_id
                for task_id in task_ids
                if (selection := _selection(requirements, task_id)) is not None
                and any(asset_id in selection["assets"] for asset_id in asset_ids_for_repository)
            ]
            try:
                _validate_gated_access(asset, token)
            except Exception as exc:
                task_label = "task" if len(affected_tasks) == 1 else "tasks"
                detail = PreflightError(
                    f"gated repository authorization failed for {asset['repository']} "
                    f"(affected selected {task_label}: {', '.join(affected_tasks)}). "
                    "Confirm that the token has read access to this repository; for a fine-grained "
                    "token, grant access to this repository or to public gated repositories. "
                    "Accept the repository terms or wait for publisher approval, then rerun "
                    f"preparation: {_redact(exc, token)}"
                )
                raise PreflightError(
                    str(detail),
                    affected_task_ids=affected_tasks,
                    asset_errors={asset_id: detail for asset_id in asset_ids_for_repository},
                ) from exc

    sizes = {}
    for asset_id in missing:
        asset = requirements["assets"][asset_id]
        try:
            sizes[asset_id] = _hub_repo_size(asset, token)
        except Exception as exc:
            raise PreflightError(
                f"access check failed for {asset['repository']}@{asset['revision']}: {_redact(exc, token)}"
            ) from exc
        print(f"[access] {asset['repository']}@{asset['revision']}")

    return HfPlan(
        requirements=requirements,
        task_ids=tuple(task_ids),
        cache_root=cache_root,
        token=token,
        asset_ids=tuple(asset_ids),
        views=views,
        missing=tuple(missing),
        facts=facts,
        required_bytes=sum(sizes.values()),
        verify_only=False,
        reuse=reuse,
    )


def stage(plan: HfPlan) -> dict[str, dict[str, Any]]:
    """Apply one checked HF/data plan."""
    if plan.verify_only:
        return plan.facts
    if plan.reuse:
        print(f"[reuse] {len(plan.asset_ids)} pinned asset(s); local HF/data is current")
        return plan.facts

    hub_cache = plan.cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    for asset_id in plan.missing:
        asset = plan.requirements["assets"][asset_id]
        print(f"[download] {asset['repository']}@{asset['revision']}")
        try:
            _hub_download(asset, hub_cache, plan.token)
        except Exception as exc:
            raise PreflightError(
                f"download failed for {asset['repository']}@{asset['revision']}: {_redact(exc, plan.token)}"
            ) from exc
        _validate_asset(plan.cache_root, asset_id, asset)
    for view_id, view_assets in plan.views.items():
        _build_view(plan.cache_root, view_id, view_assets, plan.requirements)
        _validate_view(plan.cache_root, view_id, view_assets, plan.requirements)
    print(f"[ready] {len(plan.task_ids)} task(s), {len(plan.asset_ids)} unique pinned asset(s)")
    return plan.facts


def _disk_anchor(path: Path) -> Path:
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return anchor


def check_disk(path: Path, required_bytes: int, *, label: str = "HF/data assets") -> None:
    """Reject the plan before starting a large transfer."""
    available = shutil.disk_usage(_disk_anchor(path)).free
    if available < required_bytes:
        raise InsufficientDiskError(
            f"insufficient disk for {label}: need {required_bytes} bytes, have {available}"
        )


def _same_filesystem(first: Path, second: Path) -> bool:
    return _disk_anchor(first).stat().st_dev == _disk_anchor(second).stat().st_dev


def check_release_disk(cache_root: Path, hf_bytes: int, image_plan: dict[str, Any]) -> None:
    """Gate the HF and container-image targets before either large transfer starts."""
    image_bytes = image_plan["deduplicated_missing_compressed_bytes"]
    if not image_bytes:
        check_disk(cache_root, hf_bytes)
        return
    declared_root = image_plan["image_storage_root"]
    if not declared_root:
        raise PreflightError(
            "cannot check container-image disk: local Docker store is unavailable; "
            "pass --image-storage-root for the launcher image cache"
        )
    image_root = Path(declared_root)
    if _same_filesystem(cache_root, image_root):
        check_disk(
            cache_root,
            hf_bytes + image_bytes,
            label="HF/data and container-image assets",
        )
    else:
        check_disk(cache_root, hf_bytes)
        check_disk(image_root, image_bytes, label="container-image assets")


def prepare_or_verify(
    manifest: dict[str, Any],
    task_ids: Sequence[str],
    profiles: dict[str, str],
    cache_root: Path,
    *,
    verify_only: bool,
    token: str | None,
    image_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Precheck, stage, and record one selected HF/data preparation."""
    identity = readiness_identity(manifest)
    readiness_path = cache_root / READINESS_NAME
    try:
        plan = precheck(
            manifest,
            task_ids,
            profiles,
            cache_root,
            verify_only=verify_only,
            token=token,
        )
        try:
            if image_plan is None:
                check_disk(cache_root, plan.required_bytes)
            else:
                check_release_disk(cache_root, plan.required_bytes, image_plan)
        except InsufficientDiskError as error:
            if len(task_ids) == 1:
                scope = f"task {task_ids[0]} requires more disk"
            else:
                scope = f"aggregate selected cohort of {len(task_ids)} tasks is too large"
            raise PreflightError(f"{scope}: {error}") from error
        expected = build_readiness(identity, stage(plan))
        if verify_only:
            try:
                verify_readiness(readiness_path, expected)
            except PreflightError as error:
                affected = error.affected_task_ids or tuple(task_ids)
                raise PreflightError(
                    str(error),
                    affected_task_ids=affected,
                    verified_task_ids=affected,
                ) from error
        else:
            write_readiness(readiness_path, merge_readiness(readiness_path, expected))
    except PreflightError:
        if not verify_only:
            invalidate_readiness(readiness_path, identity, task_ids)
        raise
    return expected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, help="shared prepared HF/data cache root")
    parser.add_argument(
        "--image-storage-root",
        help="launcher image-cache filesystem when local Docker storage is not the target",
    )
    parser.add_argument("--task", action="append", default=[], help="selected task filter; repeatable")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_BY_GPUS.values()),
        help="CSV-derived release profile filter",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan without calls or writes")
    mode.add_argument("--verify-only", action="store_true", help="verify without network calls or writes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = None if args.verify_only else os.environ.get("HF_TOKEN") or None
    try:
        release = load_release(expected_task_set=TASK_SET_PATH)
        selected = select_tasks(release.roster, release.profiles, args.task, args.profile)
        if not selected:
            raise RequirementsError("task and profile filters select no release tasks")
        cache_root = Path(args.cache_root).expanduser().resolve()
        image_storage_root = (
            Path(args.image_storage_root).expanduser().resolve() if args.image_storage_root else None
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection": dry_run_plan(
                            release.manifest,
                            selected,
                            release.profiles,
                            args.task,
                            args.profile,
                            cache_root,
                        ),
                    },
                    indent=2,
                )
            )
        else:
            image_plan = None
            if not args.verify_only:
                image_plan = image_precheck(
                    release.manifest,
                    selected,
                    image_storage_root=image_storage_root,
                )
            try:
                prepare_or_verify(
                    release.manifest,
                    selected,
                    release.profiles,
                    cache_root,
                    verify_only=args.verify_only,
                    token=token,
                    image_plan=image_plan,
                )
            except PreflightError as error:
                if args.verify_only:
                    raise _selection_readiness_error(
                        release,
                        REQUIREMENTS_PATH,
                        cache_root,
                        selected,
                        error,
                    ) from error
                raise
            if args.verify_only:
                for asset in _asset_inventory(release.manifest, selected, cache_root):
                    print(f"[verified] {asset['repository']}@{asset['revision']}")
                    print(f"  path: {asset['cache_snapshot']}")
                    access = (
                        "gated; HF_TOKEN required for initial preparation"
                        if asset["requires_hf_token"]
                        else "public; no HF_TOKEN required"
                    )
                    print(f"  access: {access}")
                    if asset["estimated_bytes"] is not None:
                        print(f"  estimated_bytes: {asset['estimated_bytes']}")
                print(f"verified {len(selected)} task(s) for the release")
        return 0
    except (RequirementsError, PreflightError, OSError) as exc:
        print(f"error: {_redact(exc, token)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
