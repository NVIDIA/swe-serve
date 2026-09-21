# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter, defaultdict

import torch


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise AssertionError("K1 requires a CUDA GPU")
    return torch.device("cuda")


def _round_up(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def _eager_delta(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_ids: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_ranks: torch.Tensor,
    num_experts: int,
    apply_routed_weight: bool,
) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    num_loras = lora_a.shape[0]
    output = torch.zeros(
        (num_tokens, top_k, lora_b.shape[2]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for token in range(num_tokens):
        lora_id = int(token_lora_ids[token])
        if not 0 <= lora_id < num_loras:
            continue
        rank = int(lora_ranks[lora_id])
        if rank == 0:
            continue
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            if not 0 <= expert < num_experts:
                continue
            a_expert = 0 if lora_a.shape[1] == 1 else expert
            b_expert = 0 if lora_b.shape[1] == 1 else expert
            a = lora_a[lora_id, a_expert, :rank].float()
            b = lora_b[lora_id, b_expert, :, :rank].float()
            delta = hidden_states[token].float() @ a.T @ b.T
            if apply_routed_weight:
                delta = delta * topk_weights[token, route].float()
            output[token, route] = delta.to(output.dtype)
    return output


def _call_fused(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_ids: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_ranks: torch.Tensor,
    num_experts: int,
    apply_routed_weight: bool = True,
) -> torch.Tensor:
    from sglang.srt.lora.triton_ops import fused_moe_lora

    return fused_moe_lora(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_ids=token_lora_ids,
        lora_a=lora_a,
        lora_b=lora_b,
        lora_ranks=lora_ranks,
        num_experts=num_experts,
        apply_routed_weight=apply_routed_weight,
    )


def test_alignment_conserves_every_valid_route_with_large_expert_space() -> None:
    from sglang.jit_kernel.moe_lora_align import moe_lora_align_block_size

    device = _cuda()
    topk_ids = torch.tensor(
        [
            [0, 1499, -1],
            [7, 7, 1500],
            [1499, 2, 3],
            [4, 5, 6],
            [1, -1, 2],
            [1498, 0, 8],
            [0, 1, 2],
        ],
        dtype=torch.int32,
        device=device,
    )
    token_lora_ids = torch.tensor([0, 1, 1, -1, 0, 2, 3], dtype=torch.int32, device=device)
    block_size = 4
    num_experts = 1500
    num_loras = 3
    sentinel = topk_ids.numel()

    sorted_routes, expert_ids, post_pad = moe_lora_align_block_size(
        topk_ids,
        token_lora_ids,
        block_size,
        num_experts,
        num_loras,
    )

    capacity = _round_up(sentinel + num_experts * (block_size - 1), block_size)
    assert sorted_routes.shape == (num_loras, capacity)
    assert expert_ids.shape == (num_loras, capacity // block_size)
    assert post_pad.shape == (num_loras,)
    for tensor in (sorted_routes, expert_ids, post_pad):
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.int32

    expected: dict[int, list[int]] = defaultdict(list)
    expected_counts: dict[tuple[int, int], int] = defaultdict(int)
    flat = topk_ids.reshape(-1)
    for route_id in range(sentinel):
        token = route_id // topk_ids.shape[1]
        lora_id = int(token_lora_ids[token])
        expert = int(flat[route_id])
        if 0 <= lora_id < num_loras and 0 <= expert < num_experts:
            expected[lora_id].append(route_id)
            expected_counts[(lora_id, expert)] += 1

    for lora_id in range(num_loras):
        used = int(post_pad[lora_id])
        expected_used = sum(
            _round_up(count, block_size) for (owner, _), count in expected_counts.items() if owner == lora_id
        )
        assert used == expected_used
        assert used % block_size == 0

        row = sorted_routes[lora_id]
        prefix = row[:used].cpu().tolist()
        real_routes = [route for route in prefix if route != sentinel]
        assert Counter(real_routes) == Counter(expected[lora_id])
        assert len(real_routes) == len(set(real_routes))
        assert torch.all(row[used:] == sentinel)

        used_blocks = used // block_size
        for block_index in range(used_blocks):
            expert = int(expert_ids[lora_id, block_index])
            assert 0 <= expert < num_experts
            block = row[block_index * block_size : (block_index + 1) * block_size]
            emitted = block[block != sentinel].to(torch.int64)
            assert emitted.numel() > 0
            assert torch.all(flat[emitted] == expert)
            assert torch.all(token_lora_ids[emitted // topk_ids.shape[1]] == lora_id)
        assert torch.all(expert_ids[lora_id, used_blocks:] == -1)


def test_fused_delta_matches_routed_bfloat16_eager() -> None:
    device = _cuda()
    torch.manual_seed(17)
    tokens, top_k, experts, loras, hidden, rank, output = 7, 3, 6, 3, 16, 8, 12
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) / 4
    topk_ids = torch.tensor(
        [[0, 2, 5], [1, 4, 3], [5, 0, 2], [3, 2, 1], [4, 5, 0], [2, 1, 3], [0, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    token_lora_ids = torch.tensor([0, 1, 2, 0, 2, 1, 0], dtype=torch.int32, device=device)
    lora_a = torch.randn(loras, experts, rank, hidden, device=device, dtype=torch.bfloat16) / 5
    lora_b = torch.randn(loras, experts, output, rank, device=device, dtype=torch.bfloat16) / 5
    ranks = torch.full((loras,), rank, dtype=torch.int32, device=device)

    actual = _call_fused(x, topk_ids, topk_weights, token_lora_ids, lora_a, lora_b, ranks, experts)
    expected = _eager_delta(x, topk_ids, topk_weights, token_lora_ids, lora_a, lora_b, ranks, experts, True)
    assert actual.shape == expected.shape
    assert actual.device.type == "cuda" and actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_mixed_ranks_ignore_nonzero_tails_and_inactive_routes() -> None:
    device = _cuda()
    torch.manual_seed(29)
    tokens, top_k, experts, loras, hidden, max_rank, output = 8, 2, 5, 3, 12, 7, 9
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16) / 3
    topk_ids = torch.tensor(
        [[0, 1], [2, -1], [4, 5], [3, 0], [1, 4], [2, 3], [-1, 0], [4, 2]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(tokens, top_k, device=device)
    token_lora_ids = torch.tensor([0, 1, 1, 2, -1, 0, 3, 1], dtype=torch.int32, device=device)
    ranks = torch.tensor([2, 5, 0], dtype=torch.int32, device=device)
    lora_a = torch.randn(loras, experts, max_rank, hidden, device=device, dtype=torch.bfloat16) / 5
    lora_b = torch.randn(loras, experts, output, max_rank, device=device, dtype=torch.bfloat16) / 5
    for lora_id, active_rank in enumerate(ranks.cpu().tolist()):
        if active_rank < max_rank:
            lora_a[lora_id, :, active_rank:] += 4
            lora_b[lora_id, :, :, active_rank:] -= 3

    actual = _call_fused(x, topk_ids, topk_weights, token_lora_ids, lora_a, lora_b, ranks, experts)
    expected = _eager_delta(x, topk_ids, topk_weights, token_lora_ids, lora_a, lora_b, ranks, experts, True)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[3]) == 0
    assert torch.count_nonzero(actual[4]) == 0
    assert torch.count_nonzero(actual[6]) == 0


def test_shared_and_per_expert_axes_are_independent() -> None:
    device = _cuda()
    torch.manual_seed(41)
    tokens, top_k, experts, loras, hidden, max_rank, output = 6, 2, 4, 2, 10, 6, 8
    x = torch.randn(tokens, hidden, device=device, dtype=torch.float16) / 4
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [3, 0], [1, 2], [0, 3], [2, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(tokens, top_k, device=device)
    token_lora_ids = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int32, device=device)
    ranks = torch.tensor([3, 6], dtype=torch.int32, device=device)

    cases = [
        (
            torch.randn(loras, 1, max_rank, hidden, device=device, dtype=torch.float16) / 6,
            torch.randn(loras, experts, output, max_rank, device=device, dtype=torch.float16) / 6,
        ),
        (
            torch.randn(loras, experts, max_rank, hidden, device=device, dtype=torch.float16) / 6,
            torch.randn(loras, 1, output, max_rank, device=device, dtype=torch.float16) / 6,
        ),
        (
            torch.randn(loras, 1, max_rank, hidden, device=device, dtype=torch.float16) / 6,
            torch.randn(loras, 1, output, max_rank, device=device, dtype=torch.float16) / 6,
        ),
    ]
    for lora_a, lora_b in cases:
        actual = _call_fused(
            x,
            topk_ids,
            topk_weights,
            token_lora_ids,
            lora_a,
            lora_b,
            ranks,
            experts,
            apply_routed_weight=False,
        )
        expected = _eager_delta(
            x,
            topk_ids,
            topk_weights,
            token_lora_ids,
            lora_a,
            lora_b,
            ranks,
            experts,
            False,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
