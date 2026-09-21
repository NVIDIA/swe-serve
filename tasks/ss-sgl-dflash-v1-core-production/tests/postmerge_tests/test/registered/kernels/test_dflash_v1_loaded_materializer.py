# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loaded-checkpoint numerical coverage for DFLASH fused KV materialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


DRAFT_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
DRAFT_REVISION = "d3af30def9601abdd10810aba220d692f0e803f0"
DRAFT_SNAPSHOT = Path(
    "/hf-cache/hub/models--z-lab--LLaMA3.1-8B-Instruct-DFlash-UltraChat/"
    f"snapshots/{DRAFT_REVISION}"
)


def _init_single_gpu_parallel_state() -> None:
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.utils.network import get_open_port

    torch.cuda.set_device(0)
    set_global_server_args_for_scheduler(
        ServerArgs(model_path=str(DRAFT_SNAPSHOT), device="cuda")
    )
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def _close_parallel_state() -> None:
    from sglang.srt.distributed import destroy_model_parallel

    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _load_draft_model():
    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
    from sglang.srt.models.dflash import DFlashDraftModel
    from transformers import AutoConfig

    snapshot = DRAFT_SNAPSHOT
    assert (snapshot / "config.json").is_file(), snapshot
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = DFlashDraftModel(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    weight_files = sorted(str(path) for path in snapshot.glob("*.safetensors"))
    assert weight_files, snapshot
    model.load_weights(safetensors_weights_iterator(weight_files))
    return model


def _independent_reference(layer, hidden: torch.Tensor, positions: torch.Tensor):
    attention = layer.self_attn
    kv_size = int(attention.kv_size)
    weight = attention.qkv_proj.weight[
        int(attention.q_size) : int(attention.q_size) + 2 * kv_size
    ]
    projected = F.linear(hidden, weight)
    key, value = projected.split([kv_size, kv_size], dim=-1)

    num_heads = int(attention.num_kv_heads)
    head_dim = int(attention.head_dim)
    key = key.reshape(hidden.shape[0], num_heads, head_dim)
    value = value.reshape_as(key)

    key_float = key.float()
    eps = float(attention.k_norm.variance_epsilon)
    inv_rms = torch.rsqrt(key_float.square().mean(dim=-1, keepdim=True) + eps)
    norm_weight = attention.k_norm.weight.float().reshape(1, 1, head_dim)
    normalized = key_float * inv_rms * norm_weight

    rotary = attention.rotary_emb
    ensure_length = getattr(rotary, "_ensure_cos_sin_cache_length", None)
    if callable(ensure_length):
        ensure_length(int(positions.max().item()))
    rotary_dim = int(getattr(rotary, "rotary_dim", head_dim))
    half = rotary_dim // 2
    cos_sin = rotary.cos_sin_cache.index_select(0, positions)
    cos = cos_sin[:, :half].float().unsqueeze(1)
    sin = cos_sin[:, half:rotary_dim].float().unsqueeze(1)
    first = normalized[..., :half]
    second = normalized[..., half:rotary_dim]
    rotated = torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            normalized[..., rotary_dim:],
        ),
        dim=-1,
    ).to(dtype=hidden.dtype)
    return rotated, value


def test_loaded_checkpoint_layers_match_fused_materialization() -> None:
    assert torch.cuda.is_available(), "CUDA is required for fused DFLASH materialization"

    from sglang.srt.speculative.triton_ops import FusedKVMaterializeHelper

    _init_single_gpu_parallel_state()
    model = None
    try:
        model = _load_draft_model()
        layers = list(model.layers)
        assert layers and all(layer.self_attn.qkv_proj.weight.is_cuda for layer in layers)
        torch.manual_seed(20260709)
        hidden = torch.randn(
            5,
            int(model.config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        positions = torch.tensor([0, 11, 3, 7, 1], device="cuda", dtype=torch.int64)
        expected = [
            _independent_reference(layer, hidden, positions) for layer in layers
        ]

        first_attention = layers[0].self_attn
        helper = FusedKVMaterializeHelper(
            layers=layers,
            rotary_emb=first_attention.rotary_emb,
            num_kv_heads=int(first_attention.num_kv_heads),
            head_dim=int(first_attention.head_dim),
            device=torch.device("cuda"),
        )
        writes = []
        helper.materialize(
            hidden,
            positions,
            lambda layer_id, key, value: writes.append((layer_id, key, value)),
        )

        assert [layer_id for layer_id, _, _ in writes] == list(range(len(layers)))
        for layer_id, actual_key, actual_value in writes:
            expected_key, expected_value = expected[layer_id]
            assert actual_key.shape == expected_key.shape
            assert actual_value.shape == expected_value.shape
            torch.testing.assert_close(actual_key, expected_key, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(
                actual_value, expected_value, rtol=2e-2, atol=2e-2
            )

        empty_writes = []
        helper.materialize(
            hidden[:0],
            positions[:0],
            lambda *values: empty_writes.append(values),
        )
        assert empty_writes == []

        with pytest.raises(ValueError, match="num_kv_heads mismatch"):
            FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=first_attention.rotary_emb,
                num_kv_heads=int(first_attention.num_kv_heads) + 1,
                head_dim=int(first_attention.head_dim),
                device=torch.device("cuda"),
            )
    finally:
        del model
        _close_parallel_state()
        torch.cuda.empty_cache()
