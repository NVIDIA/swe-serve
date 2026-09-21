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
# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/tests/lora/test_moe_lora_align_sum.py, will optimize in future refactor
import random
import sys

import pytest
import torch

# ---------------------------------------------------------
# IMPORT PREBUILT KERNEL
# ---------------------------------------------------------
def moe_lora_align_block_size(*args, **kwargs):
    """Defer the feature import so the no-op side still collects every case."""
    from sglang.jit_kernel.moe_lora_align import (
        moe_lora_align_block_size as source_op,
    )

    return source_op(*args, **kwargs)


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def expand_token_lora_ids(seg_indptr, req_to_lora):
    """Adapt the maintainer request fixture to the public per-token API."""
    segment_lengths = (seg_indptr[1:] - seg_indptr[:-1]).to(torch.int64)
    return torch.repeat_interleave(req_to_lora, segment_lengths)


def sample_data(num_experts, max_loras, num_tokens, topk_num):
    # 1. Generate TopK IDs (Flattened tokens)
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]

    # 2. Generate Random Requests (Segments)
    # We split num_tokens into random chunks to simulate a batch of requests
    remaining_tokens = num_tokens
    seg_lens = []
    while remaining_tokens > 0:
        # Random length between 1 and remaining
        length = random.randint(1, min(32, remaining_tokens))
        if remaining_tokens - length < 0:
            length = remaining_tokens
        seg_lens.append(length)
        remaining_tokens -= length

    # Ensure we cover the full range exactly (cleanup last segment)
    if sum(seg_lens) < num_tokens:
        seg_lens.append(num_tokens - sum(seg_lens))

    # 3. Build seg_indptr [0, len1, len1+len2, ...]
    seg_indptr = torch.cumsum(
        torch.tensor([0] + seg_lens, dtype=torch.int32), dim=0
    ).to(dtype=torch.int32)

    # 4. Assign a LoRA ID to each Request
    num_reqs = len(seg_lens)
    req_to_lora = torch.randint(0, max_loras, (num_reqs,), dtype=torch.int32)

    return (topk_ids.to("cuda"), seg_indptr.to("cuda"), req_to_lora.to("cuda"))


@pytest.mark.parametrize("num_tokens", [100, 200, 1024, 4096])
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [2, 32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(
    num_tokens, topk_num, num_experts, max_loras, block_size
):
    # sample data
    random.seed(1)
    torch.manual_seed(1)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping moe_lora_align_block_size test.")
    # UPDATED: Get the new 3-step mapping tensors
    topk_ids, seg_indptr, req_to_lora = sample_data(
        num_experts, max_loras, num_tokens, topk_num
    )

    # compute paddings
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # The upstream fixture models request ownership compactly. Expand it to
    # the token-level ownership tensor required by the benchmark's public API.
    token_lora_ids = expand_token_lora_ids(seg_indptr, req_to_lora)
    assert token_lora_ids.shape == (num_tokens,)

    sorted_token_ids, expert_ids, num_tokens_post_pad = moe_lora_align_block_size(
        topk_ids,
        token_lora_ids,
        block_size,
        num_experts,
        max_loras,
    )
    assert sorted_token_ids.shape == (max_loras, max_num_tokens_padded)
    assert expert_ids.shape == (max_loras, max_num_m_blocks)
    assert num_tokens_post_pad.shape == (max_loras,)

    # verify values
    sorted_token_ids = sorted_token_ids.view(max_loras, -1, block_size)

    for lora_idx in range(max_loras):
        # Count how many tokens actually belong to this LoRA
        expected_count = (token_lora_ids == lora_idx).sum().item()

        # Verify the kernel processed a reasonable number of tokens (sanity check)
        # Note: num_tokens_post_pad includes padding, so it might be larger than expected_count
        assert num_tokens_post_pad[lora_idx].item() >= expected_count * topk_num

        for token_idx in range(sorted_token_ids.size(1)):
            block = sorted_token_ids[lora_idx][token_idx]
            # Valid indices are those less than total numel
            indices = block[block != topk_ids.numel()]

            if indices.numel() > 0:
                # 1. Verify routing: Does the token actually route to this expert?
                expert_id = expert_ids[lora_idx][token_idx]
                assert torch.all(topk_ids.view(-1)[indices] == expert_id)

                # 2. Verify ownership: Did the kernel grab the correct tokens for this LoRA?
                # The indices in 'sorted_token_ids' point to the flattened [token, topk] array.
                # We divide by topk_num to get the original token index.
                original_token_indices = indices // topk_num

                # Check that all tokens in this block truly belong to 'lora_idx'
                actual_owners = token_lora_ids[original_token_indices]
                assert torch.all(
                    actual_owners == lora_idx
                ), f"Kernel put tokens from LoRA {actual_owners} into block for LoRA {lora_idx}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
