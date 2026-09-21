# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch


def test_existing_cuda_profile_remains_available() -> None:
    expected_visible = int(os.environ["TRANSFORMERS5_MOE_EXPECTED_VISIBLE_GPUS"])
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == expected_visible
    assert int(os.environ["TRANSFORMERS5_MOE_TP_SIZE"]) in {1, 4}
    probe = torch.arange(16, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        probe.square(), torch.tensor([float(index * index) for index in range(16)], device="cuda")
    )
