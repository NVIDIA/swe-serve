# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-capable Harbor Docker environment for SWE-Serve GPU tasks.

Stock Harbor's Docker environment does not allocate GPUs. This drop-in
subclass enables them via public extension surface only: it reports GPU
capability and appends the standard compose device-reservation overlay
(configs/gpu_compose.yaml, relative to this file). It also accepts two optional
environment kwargs:

- ``shared_mounts_json``: a JSON list of Docker bind-mount objects, used to
  attach the read-only model cache and related task data to the containers.
- ``base_snapshot_host``: an existing absolute host directory. Right after the
  agent container starts, the pristine ``/code`` checkout is copied there, so
  speed-scored verifiers can compare the agent's work against an immutable
  ``/base`` snapshot (mounted via ``shared_mounts_json``).

Most users never pass these by hand — ``run_task.py`` composes them per task.
Direct usage:

    harbor run -p tasks/<task-id> --agent oracle \
        --environment-import-path scripts.gpu_env:GpuDockerEnvironment
"""

import json
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment

GPU_COMPOSE = Path(__file__).resolve().parent / "configs" / "gpu_compose.yaml"


def _parse_shared_mounts(raw):
    if raw is None:
        return []
    mounts = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(mounts, list) or not all(isinstance(m, dict) for m in mounts):
        raise ValueError("shared_mounts_json must be a JSON list of mount objects")
    for mount in mounts:
        if not isinstance(mount.get("source"), str) or not isinstance(mount.get("target"), str):
            raise ValueError("every shared mount needs string source and target fields")
    return mounts


class GpuDockerEnvironment(DockerEnvironment):
    """DockerEnvironment with GPU allocation, shared mounts, and /base capture."""

    def __init__(self, *args, shared_mounts_json=None, base_snapshot_host=None, **kwargs):
        environment_dir = Path(kwargs.get("environment_dir", args[0] if args else "")).resolve()

        extra = list(kwargs.pop("extra_docker_compose", None) or [])
        if str(GPU_COMPOSE) not in [str(path) for path in extra]:
            extra.append(GPU_COMPOSE)
        kwargs["extra_docker_compose"] = extra

        mounts = list(kwargs.pop("mounts", None) or [])
        for mount in _parse_shared_mounts(shared_mounts_json):
            if mount not in mounts:
                mounts.append(mount)
        if mounts:
            kwargs["mounts"] = mounts

        super().__init__(*args, **kwargs)

        self._base_snapshot_host = Path(base_snapshot_host) if base_snapshot_host else None
        if self._base_snapshot_host is not None and not (
            self._base_snapshot_host.is_absolute() and self._base_snapshot_host.is_dir()
        ):
            raise ValueError("base_snapshot_host must be an existing absolute directory")
        self._capture_base_snapshot = environment_dir.name == "environment"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"gpus": True})

    async def start(self, force_build: bool):
        """Start the container, then preserve its pristine checkout at /base."""
        await super().start(force_build)
        if self._base_snapshot_host is not None and self._capture_base_snapshot:
            result = await self._run_docker_compose_command(
                ["cp", "main:/code/.", str(self._base_snapshot_host)],
                check=False,
            )
            if result.return_code != 0:
                detail = (result.stderr or result.stdout or "no output").strip()[-800:]
                raise RuntimeError(f"failed to snapshot /code to /base: {detail}")
