# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused Gemma4 dual-RMSNorm + residual + scalar kernel.

Compares the Triton ``gemma_dual_rmsnorm_residual_scalar`` op against an eager
torch reference that mirrors the kernel's arithmetic. The kernel loads every
input as float32, performs all three RMSNorms, the sum, the residual add and the
scalar multiply in float32, and casts back to the input dtype (bf16) exactly
once at the store. The reference replicates that single-cast-at-the-end order so
the comparison is apples-to-apples.

Run with::

    pytest test/registered/kernels/test_gemma_dual_rmsnorm_residual.py -v

Requires a CUDA-capable GPU; the kernel is CUDA-only Triton and skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gemma_dual_rmsnorm_residual_scalar is a CUDA-only Triton kernel",
)


def _rmsnorm_f32(x_f32: torch.Tensor, w_f32: torch.Tensor, eps: float) -> torch.Tensor:
    """Row-wise RMSNorm in float32: x * rsqrt(mean(x^2) + eps) * w.

    Mirrors the kernel: ``var = sum(x*x) / N`` (population mean over the last
    dim), then ``x * rsqrt(var + eps) * w``. All math stays in float32.
    """
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps) * w_f32


def _reference(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float,
    eps2: float,
    eps3: float,
) -> torch.Tensor:
    """Eager equivalent: (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + r) * scalar.

    Everything runs in float32 and is cast back to ``x1.dtype`` once at the end,
    matching the kernel's load-as-f32 / store-as-input-dtype contract.
    """
    out_dtype = x1.dtype
    x1f = x1.to(torch.float32)
    x2f = x2.to(torch.float32)
    w1f = weight1.to(torch.float32)
    w2f = weight2.to(torch.float32)
    w3f = weight3.to(torch.float32)
    rf = residual.to(torch.float32)
    sf = scalar.to(torch.float32)

    norm1 = _rmsnorm_f32(x1f, w1f, eps1)
    norm2 = _rmsnorm_f32(x2f, w2f, eps2)
    combined = norm1 + norm2
    norm3 = _rmsnorm_f32(combined, w3f, eps3)
    out = (norm3 + rf) * sf
    return out.to(out_dtype)


def _make_inputs(M, N, device, dtype, seed):
    torch.manual_seed(seed)
    x1 = torch.randn(M, N, dtype=dtype, device=device).contiguous()
    x2 = torch.randn(M, N, dtype=dtype, device=device).contiguous()
    weight1 = torch.randn(N, dtype=dtype, device=device)
    weight2 = torch.randn(N, dtype=dtype, device=device)
    weight3 = torch.randn(N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device).contiguous()
    scalar = torch.tensor([1.5], dtype=dtype, device=device)
    return x1, weight1, x2, weight2, weight3, residual, scalar


@pytest.mark.parametrize("rows,hidden", [(16, 256), (64, 512), (128, 1024)])
def test_dual_rmsnorm_parity(rows, hidden):
    from sglang.srt.layers.gemma4_fused_ops import gemma_dual_rmsnorm_residual_scalar

    device = "cuda"
    dtype = torch.bfloat16
    eps1 = eps2 = eps3 = 1e-6

    x1, weight1, x2, weight2, weight3, residual, scalar = _make_inputs(
        rows, hidden, device, dtype, seed=0
    )

    out = gemma_dual_rmsnorm_residual_scalar(
        x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
    )

    ref = _reference(
        x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
    )

    assert out.shape == x1.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_dual_rmsnorm_default_eps():
    """Exercise the default eps path (op defaults eps1=eps2=eps3=1e-6)."""
    from sglang.srt.layers.gemma4_fused_ops import gemma_dual_rmsnorm_residual_scalar

    device = "cuda"
    dtype = torch.bfloat16
    rows, hidden = 32, 384

    x1, weight1, x2, weight2, weight3, residual, scalar = _make_inputs(
        rows, hidden, device, dtype, seed=1
    )

    out = gemma_dual_rmsnorm_residual_scalar(
        x1, weight1, x2, weight2, weight3, residual, scalar
    )
    ref = _reference(
        x1, weight1, x2, weight2, weight3, residual, scalar, 1e-6, 1e-6, 1e-6
    )
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
