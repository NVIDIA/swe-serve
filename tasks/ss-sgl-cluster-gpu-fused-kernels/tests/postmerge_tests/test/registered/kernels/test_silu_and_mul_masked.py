# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the masked SiLU-and-multiply MoE activation kernel.

Compares the Triton ``silu_and_mul_masked_fwd`` op against an eager torch
reference that mirrors the kernel's arithmetic: SiLU(gate) is computed in fp32
then cast back to bf16 before multiplying by the (bf16) up half, and only the
``masked_m`` valid rows per expert are written.

Run with::

    pytest test/registered/kernels/test_silu_and_mul_masked.py -v

Requires a CUDA-capable GPU; the kernel is CUDA-only Triton and skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="silu_and_mul_masked_fwd is a CUDA-only Triton kernel",
)


def _reference(input: torch.Tensor, masked_m: torch.Tensor, size_n: int) -> torch.Tensor:
    """Eager equivalent honoring masked_m.

    The kernel computes ``silu`` in fp32, casts the activated gate back to the
    input dtype (bf16), then multiplies by the (bf16) up half. We replicate that
    cast order so the comparison is apples-to-apples. Rows >= masked_m[e] are
    padding and left as zeros (the kernel never writes them).
    """
    expert_num = input.shape[0]
    out = torch.zeros(
        (expert_num, input.shape[1], size_n),
        dtype=input.dtype,
        device=input.device,
    )
    for e in range(expert_num):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        gate = input[e, :m, :size_n]
        up = input[e, :m, size_n:]
        gate_f32 = gate.to(torch.float32)
        act = gate_f32 / (1.0 + torch.exp(-gate_f32))
        act = act.to(input.dtype)
        out[e, :m, :] = up * act
    return out


@pytest.mark.parametrize("expert_num,token_num,size_n", [(1, 64, 256), (4, 128, 512), (8, 96, 128)])
def test_masked_parity(expert_num, token_num, size_n):
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_fwd

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(
        expert_num, token_num, 2 * size_n, dtype=dtype, device=device
    ).contiguous()
    masked_m = torch.randint(
        0, token_num + 1, (expert_num,), dtype=torch.int32, device=device
    )
    output = torch.empty(
        expert_num, token_num, size_n, dtype=dtype, device=device
    ).contiguous()

    silu_and_mul_masked_fwd(input, output, masked_m)

    ref = _reference(input, masked_m, size_n)

    # Compare only the valid (unmasked) rows per expert; padding rows are
    # uninitialized in the kernel output by design.
    for e in range(expert_num):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        torch.testing.assert_close(
            output[e, :m, :], ref[e, :m, :], atol=1e-2, rtol=1e-2
        )


def test_full_mask():
    """All tokens valid for every expert (masked_m == token_num)."""
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_fwd

    expert_num, token_num, size_n = 2, 32, 128
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(
        expert_num, token_num, 2 * size_n, dtype=dtype, device=device
    ).contiguous()
    masked_m = torch.full((expert_num,), token_num, dtype=torch.int32, device=device)
    output = torch.empty(
        expert_num, token_num, size_n, dtype=dtype, device=device
    ).contiguous()

    silu_and_mul_masked_fwd(input, output, masked_m)
    ref = _reference(input, masked_m, size_n)
    torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
