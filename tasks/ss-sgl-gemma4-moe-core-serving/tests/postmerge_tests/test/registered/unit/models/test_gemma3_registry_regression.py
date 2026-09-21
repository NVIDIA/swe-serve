# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def test_gemma3_conditional_generation_remains_discoverable() -> None:
    from sglang.srt.models.registry import ModelRegistry

    architecture = "Gemma3ForConditionalGeneration"
    model_cls, resolved_architecture = ModelRegistry.resolve_model_cls(architecture)

    assert resolved_architecture == architecture
    assert model_cls.__name__ == architecture
