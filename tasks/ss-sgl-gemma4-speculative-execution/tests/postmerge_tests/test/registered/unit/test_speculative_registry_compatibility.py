# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_existing_dflash_algorithm_registry_contract() -> None:
    algorithm = SpeculativeAlgorithm.from_string("dflash")
    assert algorithm is SpeculativeAlgorithm.DFLASH
    assert algorithm.is_dflash()
    assert algorithm.is_speculative()
    assert algorithm.supports_target_verify_for_draft()
