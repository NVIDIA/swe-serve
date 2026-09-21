# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unique SDAR day-zero configuration and compatibility contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _server_args(architecture: str):
    return SimpleNamespace(
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=None,
        dllm_block_size=None,
        max_running_requests=3,
        model_path=f"synthetic-{architecture}",
        revision=None,
    )


def test_sdar_architecture_defaults(monkeypatch) -> None:
    import sglang.srt.dllm.config as config_module
    from sglang.srt.models.registry import ModelRegistry

    for architecture in ("SDARForCausalLM", "SDARMoeForCausalLM"):
        model_class, resolved_architecture = ModelRegistry.resolve_model_cls([architecture])
        assert resolved_architecture == architecture
        assert issubclass(model_class, torch.nn.Module)
        monkeypatch.setattr(
            config_module.ModelConfig,
            "from_server_args",
            lambda *args, architecture=architecture, **kwargs: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            ),
        )
        config = config_module.DllmConfig.from_server_args(_server_args(architecture))
        assert config.block_size == 4
        assert config.mask_id == 151669
        assert config.max_running_requests == 3


def test_existing_llada_defaults_are_preserved(monkeypatch) -> None:
    import sglang.srt.dllm.config as config_module

    monkeypatch.setattr(
        config_module.ModelConfig,
        "from_server_args",
        lambda *args, **kwargs: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"])
        ),
    )
    config = config_module.DllmConfig.from_server_args(_server_args("LLaDA2MoeModelLM"))
    assert config.block_size == 32
    assert config.mask_id == 156895


def test_unknown_diffusion_architecture_is_rejected(monkeypatch) -> None:
    import sglang.srt.dllm.config as config_module

    monkeypatch.setattr(
        config_module.ModelConfig,
        "from_server_args",
        lambda *args, **kwargs: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["AutoregressiveOnlyModel"])
        ),
    )
    with pytest.raises(RuntimeError, match="Unknown diffusion LLM"):
        config_module.DllmConfig.from_server_args(_server_args("AutoregressiveOnlyModel"))
