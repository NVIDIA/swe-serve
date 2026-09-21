# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def test_dense_qwen3_5_checkpoint_architecture_is_discoverable() -> None:
    from sglang.srt.models.registry import ModelRegistry

    architecture = "Qwen3_5ForConditionalGeneration"
    model_cls, resolved_architecture = ModelRegistry.resolve_model_cls(architecture)

    assert resolved_architecture == architecture
    assert model_cls.__name__ == architecture


def test_qwen3_next_causal_model_remains_discoverable() -> None:
    from sglang.srt.models.registry import ModelRegistry

    architecture = "Qwen3NextForCausalLM"
    model_cls, resolved_architecture = ModelRegistry.resolve_model_cls(architecture)

    assert resolved_architecture == architecture
    assert model_cls.__name__ == architecture
