# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HiSparse decode lifecycle requires CUDA and pinned host memory",
)

_PAGE_SIZE = 64
_DEVICE_BUFFER_SIZE = 64
_POOL_SIZE = 512
_LAYERS = 2
_KV_DIM = 16


@dataclass
class _Runtime:
    req_pool: object
    device_pool: object
    allocator: object
    coordinator: object
    req: object
    logical_locs: torch.Tensor
    initial_allocator_available: int
    initial_host_available: int


def _world_group():
    import torch.distributed as dist

    if not dist.is_initialized():
        rendezvous = os.path.join(
            tempfile.gettempdir(),
            f"hisparse-lifecycle-{os.getpid()}-{uuid.uuid4().hex}",
        )
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
    return dist.group.WORLD


def _token_pattern(token: int, layer: int) -> float:
    return float(layer * 1000 + token + 1)


def _make_runtime(seq_len: int) -> _Runtime:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.hisparse_memory_pool import (
        HiSparseNSATokenToKVPool,
        HiSparseTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.sampling.sampling_params import SamplingParams

    torch.cuda.set_device(0)
    req_pool = ReqToTokenPool(
        size=4,
        max_context_len=320,
        device="cuda",
        enable_memory_saver=False,
    )
    device_pool = HiSparseNSATokenToKVPool(
        size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        kv_lora_rank=8,
        dtype=torch.bfloat16,
        qk_rope_head_dim=8,
        layer_num=_LAYERS,
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=_KV_DIM,
        host_to_device_ratio=2,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        kvcache=device_pool,
        need_sort=False,
        host_to_device_ratio=2,
    )
    initial_allocator_available = allocator.available_size()
    coordinator = HiSparseCoordinator(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        top_k=3,
        device_buffer_size=_DEVICE_BUFFER_SIZE,
        device="cuda",
        tp_group=_world_group(),
        host_to_device_ratio=2,
    )
    initial_host_available = coordinator.mem_pool_host.available_size()

    req = Req(
        rid=f"lifecycle-{seq_len}",
        origin_input_text="",
        origin_input_ids=list(range(seq_len)),
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    assert req_pool.alloc([req]) == [req.req_pool_idx]
    req.fill_ids = list(range(seq_len))
    req.kv_allocated_len = seq_len
    req.kv_committed_len = seq_len

    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
    prefix_lens = prefix_lens_cpu.to("cuda")
    seq_lens = seq_lens_cpu.to("cuda")
    last_loc = torch.tensor([-1], dtype=torch.int64, device="cuda")
    logical_locs = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        seq_len,
    )
    assert logical_locs is not None and logical_locs.numel() == seq_len
    req_pool.write((req.req_pool_idx, slice(0, seq_len)), logical_locs)

    sparse_locs = allocator.full_to_hisparse_device_index_mapping[logical_locs].long()
    assert torch.all(sparse_locs > 0)
    for layer, buffer in enumerate(device_pool.kv_buffer):
        values = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.float32)
        values = values.add(layer * 1000).to(buffer.dtype).view(seq_len, 1, 1)
        buffer.index_copy_(0, sparse_locs, values.expand(-1, 1, _KV_DIM))

    return _Runtime(
        req_pool=req_pool,
        device_pool=device_pool,
        allocator=allocator,
        coordinator=coordinator,
        req=req,
        logical_locs=logical_locs,
        initial_allocator_available=initial_allocator_available,
        initial_host_available=initial_host_available,
    )


def _stage_and_collect(runtime: _Runtime) -> None:
    runtime.coordinator.admit_request_into_staging(runtime.req)
    runtime.coordinator.write_staging_stream.synchronize()
    ready = runtime.coordinator.collect_ready_reqs()
    assert ready == [runtime.req]
    assert runtime.req.hisparse_staging is False


def _install_later_request(runtime: _Runtime, seq_len: int) -> None:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid=f"lifecycle-reuse-{seq_len}",
        origin_input_text="",
        origin_input_ids=list(range(seq_len)),
        sampling_params=SamplingParams(max_new_tokens=4),
    )
    assert runtime.req_pool.alloc([req]) == [req.req_pool_idx]
    req.fill_ids = list(range(seq_len))
    req.kv_allocated_len = seq_len
    req.kv_committed_len = seq_len

    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
    logical_locs = runtime.allocator.alloc_extend(
        prefix_lens_cpu.to("cuda"),
        prefix_lens_cpu,
        seq_lens_cpu.to("cuda"),
        seq_lens_cpu,
        torch.tensor([-1], dtype=torch.int64, device="cuda"),
        seq_len,
    )
    assert logical_locs is not None and logical_locs.numel() == seq_len
    runtime.req_pool.write((req.req_pool_idx, slice(0, seq_len)), logical_locs)

    sparse_locs = runtime.allocator.full_to_hisparse_device_index_mapping[
        logical_locs
    ].long()
    assert torch.all(sparse_locs > 0)
    for layer, buffer in enumerate(runtime.device_pool.kv_buffer):
        values = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.float32)
        values = values.add(layer * 1000).to(buffer.dtype).view(seq_len, 1, 1)
        buffer.index_copy_(0, sparse_locs, values.expand(-1, 1, _KV_DIM))

    runtime.req = req
    runtime.logical_locs = logical_locs


def _assert_host_tokens(runtime: _Runtime, positions: list[int]) -> None:
    req_idx = runtime.req.req_pool_idx
    host_locs = runtime.coordinator.req_to_host_pool[
        req_idx, torch.tensor(positions, device="cuda")
    ].long()
    assert torch.all(host_locs >= 0)
    for layer, host_buffer in enumerate(runtime.coordinator.mem_pool_host.kv_buffer):
        actual = host_buffer[host_locs.cpu()].float()
        expected = torch.tensor(
            [_token_pattern(position, layer) for position in positions],
            dtype=host_buffer.dtype,
        ).float().view(-1, 1, 1)
        assert torch.equal(actual, expected.expand(-1, 1, _KV_DIM))


def _cleanup(runtime: _Runtime, all_logical_locs: torch.Tensor | None = None) -> None:
    logical_locs = runtime.logical_locs if all_logical_locs is None else all_logical_locs
    runtime.coordinator.request_finished(runtime.req)
    runtime.coordinator.request_finished(runtime.req)
    runtime.allocator.free(logical_locs)
    runtime.req_pool.free(runtime.req)
    torch.cuda.synchronize()
    assert runtime.allocator.available_size() == runtime.initial_allocator_available
    assert (
        runtime.coordinator.mem_pool_host.available_size()
        == runtime.initial_host_available
    )


def test_decode_backup_grows_by_one_page_before_reusing_reserved_slot() -> None:
    runtime = _make_runtime(seq_len=128)
    _stage_and_collect(runtime)
    req_idx = runtime.req.req_pool_idx
    req_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
    host_available_after_prefill = runtime.coordinator.mem_pool_host.available_size()

    first_decode = runtime.allocator.alloc_decode(
        torch.tensor([129], dtype=torch.int64, device="cuda"),
        torch.tensor([129], dtype=torch.int64),
        runtime.logical_locs[-1:],
    )
    assert first_decode is not None
    runtime.req_pool.write((req_idx, 128), first_decode[0])
    runtime.coordinator.map_last_loc_to_buffer(
        torch.tensor([129], dtype=torch.int64, device="cuda"),
        first_decode,
        req_indices,
        torch.tensor([129], dtype=torch.int64),
    )

    reserved = runtime.allocator.full_to_hisparse_device_index_mapping[
        first_decode
    ].long()
    assert torch.all(reserved > 0)
    for layer, buffer in enumerate(runtime.device_pool.kv_buffer):
        buffer.index_fill_(0, reserved, _token_pattern(128, layer))

    second_decode = runtime.allocator.alloc_decode(
        torch.tensor([130], dtype=torch.int64, device="cuda"),
        torch.tensor([130], dtype=torch.int64),
        first_decode,
    )
    assert second_decode is not None
    runtime.req_pool.write((req_idx, 129), second_decode[0])
    runtime.coordinator.map_last_loc_to_buffer(
        torch.tensor([130], dtype=torch.int64, device="cuda"),
        second_decode,
        req_indices,
        torch.tensor([130], dtype=torch.int64),
    )
    torch.cuda.synchronize()

    second_reserved = runtime.allocator.full_to_hisparse_device_index_mapping[
        second_decode
    ].long()
    assert torch.equal(second_reserved, reserved)
    for layer, buffer in enumerate(runtime.device_pool.kv_buffer):
        buffer.index_fill_(0, second_reserved, _token_pattern(129, layer))
    torch.cuda.synchronize()

    assert (
        runtime.coordinator.mem_pool_host.available_size()
        == host_available_after_prefill - _PAGE_SIZE
    )
    _assert_host_tokens(runtime, [128])

    runtime.req.kv_allocated_len = 130
    all_logical_locs = torch.cat((runtime.logical_locs, first_decode, second_decode))
    _cleanup(runtime, all_logical_locs)


def test_inflight_retraction_frees_pages_once_and_reuses_the_same_pools() -> None:
    runtime = _make_runtime(seq_len=70)
    runtime.coordinator.admit_request_into_staging(runtime.req)
    runtime.coordinator.write_staging_stream.synchronize()
    assert (
        runtime.coordinator.mem_pool_host.available_size()
        == runtime.initial_host_available - 128
    )

    runtime.coordinator.retract_req(runtime.req)
    runtime.allocator.free(runtime.logical_locs)
    runtime.req_pool.free(runtime.req)
    torch.cuda.synchronize()
    assert runtime.allocator.available_size() == runtime.initial_allocator_available
    assert (
        runtime.coordinator.mem_pool_host.available_size()
        == runtime.initial_host_available
    )

    _install_later_request(runtime, seq_len=70)
    _stage_and_collect(runtime)
    assert (
        runtime.coordinator.mem_pool_host.available_size()
        == runtime.initial_host_available - 128
    )
    _assert_host_tokens(runtime, [0, 63, 64, 69])
    _cleanup(runtime)


@pytest.mark.parametrize("seq_len", [32, 128])
def test_existing_public_staging_restore_and_cleanup(seq_len: int) -> None:
    runtime = _make_runtime(seq_len=seq_len)
    _stage_and_collect(runtime)
    _assert_host_tokens(runtime, [0, seq_len - 1])

    if seq_len > _DEVICE_BUFFER_SIZE:
        selected = torch.tensor([[7, 79, 111]], dtype=torch.int32, device="cuda")
    else:
        selected = torch.tensor([[1, 7, 19]], dtype=torch.int32, device="cuda")
    runtime.coordinator.num_real_reqs.fill_(1)
    device_locs = runtime.coordinator.swap_in_selected_pages(
        torch.tensor([runtime.req.req_pool_idx], dtype=torch.int64, device="cuda"),
        torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        selected,
        layer_id=0,
    )
    torch.cuda.synchronize()

    assert torch.all(device_locs > 0)
    restored = runtime.device_pool.kv_buffer[0][device_locs[0].long()].float()
    expected = selected[0].float().add(1).view(-1, 1, 1)
    assert torch.equal(restored, expected.expand(-1, 1, _KV_DIM))

    _cleanup(runtime)
