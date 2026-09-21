# Copyright 2023-2024 SGLang Team
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin production-wrapper controls beside the canonical kernel matrix."""

import torch
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

DEVICE = "cuda"
CACHE_SIZE = 4096


def _reference_scatter(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    n_loc = loc.shape[0]
    kv_view = kv_buffer.view(kv_buffer.shape[0], -1)
    kv_view[loc.long(), :nope_dim] = cache_k_nope.reshape(n_loc, -1)
    kv_view[loc.long(), nope_dim:] = cache_k_rope.reshape(n_loc, -1)


def _assert_wrapper_matches_reference(batch_size, nope_dim, rope_dim, dtype):
    cache_k_nope = torch.randn(
        (batch_size, 1, nope_dim), dtype=dtype, device=DEVICE
    )
    cache_k_rope = torch.randn(
        (batch_size, 1, rope_dim), dtype=dtype, device=DEVICE
    )
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim), dtype=dtype, device=DEVICE
    )
    kv_reference = kv_buffer.clone()
    loc = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _reference_scatter(kv_reference, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_reference)


def test_wrapper_supported_layout_across_tma_threshold():
    for batch_size in (767, 768, 1024):
        _assert_wrapper_matches_reference(batch_size, 512, 64, torch.bfloat16)


def test_wrapper_tma_incompatible_layout_falls_back():
    _assert_wrapper_matches_reference(768, 13, 8, torch.bfloat16)
