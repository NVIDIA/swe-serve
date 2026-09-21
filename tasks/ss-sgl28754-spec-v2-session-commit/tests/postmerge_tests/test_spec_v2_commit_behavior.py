# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral probes for speculative KV commitment and session transfer."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.srt.speculative import eagle_info_v2, spec_utils
from sglang.srt.speculative.eagle_info_v2 import EagleDraftInputV2Mixin


class _Grammar:
    def __init__(self, terminate_after: int):
        self.accepted = []
        self.terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self.terminate_after


class _Algorithm:
    def __init__(self, dflash: bool):
        self.dflash = dflash

    def is_dflash(self) -> bool:
        return self.dflash


class _SpecReq:
    def __init__(
        self,
        *,
        committed: int = 7,
        retracted: bool = False,
        finished: bool = False,
        grammar=None,
    ):
        self.kv_committed_len = committed
        self.is_retracted = retracted
        self._finished = finished
        self.grammar = grammar
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.histogram = []

    def finished(self) -> bool:
        return self._finished

    def update_spec_correct_drafts_histogram(self, value: int):
        self.histogram.append(value)


def _processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _settle(req: _SpecReq, *, dflash: bool, tokens=(101, 102, 103)):
    result = SimpleNamespace(
        next_token_ids=torch.tensor([*tokens, 0], dtype=torch.long),
        accept_lens=torch.tensor([len(tokens)], dtype=torch.long),
        speculative_num_draft_tokens=len(tokens) + 1,
        num_correct_drafts=None,
        num_correct_drafts_per_req_cpu=None,
    )
    batch = SimpleNamespace(reqs=[req], spec_algorithm=_Algorithm(dflash))
    output = _processor()._resolve_spec_v2_tokens(result, batch)
    return output, req


def test_eagle_and_dflash_settle_the_same_retained_run():
    eagle_output, eagle = _settle(_SpecReq(), dflash=False)
    dflash_output, dflash = _settle(_SpecReq(), dflash=True)
    assert eagle_output == dflash_output == [[101, 102, 103]]
    assert eagle.kv_committed_len == dflash.kv_committed_len == 10


def test_dflash_commits_the_full_retained_run():
    output, req = _settle(_SpecReq(committed=11), dflash=True)
    assert output == [[101, 102, 103]]
    assert req.kv_committed_len == 14


def test_finished_eagle_result_does_not_rewind_committed_boundary():
    output, req = _settle(_SpecReq(committed=19, finished=True), dflash=False)
    assert output == [[101, 102, 103]]
    assert req.kv_committed_len == 19


def test_finished_dflash_result_does_not_move_committed_boundary():
    _, req = _settle(_SpecReq(committed=19, finished=True), dflash=True)
    assert req.kv_committed_len == 19


def test_retracted_result_does_not_move_committed_boundary():
    _, req = _settle(_SpecReq(committed=0, retracted=True), dflash=False)
    assert req.kv_committed_len == 0


def test_grammar_commits_only_the_terminating_prefix():
    grammar = _Grammar(terminate_after=2)
    output, req = _settle(_SpecReq(committed=5, grammar=grammar), dflash=False)
    assert output == [[101, 102]]
    assert req.kv_committed_len == 7


def test_grammar_commits_all_nonterminating_tokens():
    grammar = _Grammar(terminate_after=99)
    output, req = _settle(_SpecReq(committed=5, grammar=grammar), dflash=False)
    assert output == [[101, 102, 103]]
    assert req.kv_committed_len == 8


def test_grammar_termination_preserves_only_the_emitted_prefix():
    grammar = _Grammar(terminate_after=2)
    output, _ = _settle(_SpecReq(grammar=grammar), dflash=False)
    assert output == [[101, 102]]
    assert grammar.accepted == [101, 102]


def test_nonterminating_grammar_preserves_all_accepted_output():
    grammar = _Grammar(terminate_after=99)
    output, _ = _settle(_SpecReq(grammar=grammar), dflash=False)
    assert output == [[101, 102, 103]]
    assert grammar.accepted == [101, 102, 103]


def test_eagle_prepare_does_not_advance_committed_boundary(monkeypatch):
    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("draft preparation must not globally synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden_sync)
    monkeypatch.setattr(eagle_info_v2, "get_alloc_reserve_per_decode", lambda: 4)
    monkeypatch.setattr(
        eagle_info_v2,
        "alloc_token_slots",
        lambda _tree_cache, count: torch.arange(count, dtype=torch.int32),
    )
    monkeypatch.setattr(
        spec_utils, "assign_req_to_token_pool_func", lambda *args, **kwargs: None
    )

    req = SimpleNamespace(kv_allocated_len=10, kv_committed_len=7, decode_batch_idx=0)
    batch = SimpleNamespace(
        maybe_evict_swa=lambda: None,
        batch_size=lambda: 1,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        reqs=[req],
        device="cpu",
        tree_cache=object(),
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((1, 32), dtype=torch.int32)
        ),
    )

    EagleDraftInputV2Mixin().prepare_for_decode(batch)

    assert req.kv_committed_len == 7
    assert req.kv_allocated_len >= 10


def test_commit_paths_do_not_require_global_or_device_synchronization(monkeypatch):
    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("commit bookkeeping must not globally synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden_sync)

    output, req = _settle(_SpecReq(committed=7), dflash=False)
    assert output == [[101, 102, 103]]
    assert req.kv_committed_len == 10


class _Allocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


class _InnerCache:
    def __init__(self, *, page_size: int = 1):
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(256, dtype=torch.int32).reshape(2, 128),
            free_slots=[],
        )
        self.token_to_kv_pool_allocator = _Allocator()
        self.page_size = page_size


class _SessionReq:
    def __init__(
        self,
        *,
        committed: int,
        allocated: int,
        origin_len: int,
        finished_len: int,
    ):
        self.session = SimpleNamespace(
            session_id="session-a",
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
        )
        self.req_pool_idx = 0
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(finished_len))
        self.finished_len = finished_len
        self.finished_reason = FINISH_LENGTH(length=finished_len)
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None

    def finished(self) -> bool:
        return self.finished_reason is not None

    def pop_committed_kv_cache(self):
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


def test_finished_session_inherits_authoritative_completed_boundary():
    tree_cache = StreamingSession(_InnerCache())
    req = _SessionReq(committed=37, allocated=40, origin_len=26, finished_len=12)

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 38
    assert slot.kv_allocated_len == 38


def test_finished_session_never_inherits_past_physical_allocation():
    tree_cache = StreamingSession(_InnerCache())
    req = _SessionReq(committed=39, allocated=40, origin_len=30, finished_len=15)

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 40
    assert slot.kv_allocated_len == 40


def test_page_aligned_tail_cleanup_preserves_the_partial_page():
    inner = _InnerCache(page_size=4)
    tree_cache = StreamingSession(inner)
    slot = SessionSlot(req_pool_idx=0, kv_committed_len=7, kv_allocated_len=12)
    req = SimpleNamespace(
        kv_committed_len=7,
        kv_allocated_len=12,
        swa_evicted_seqlen=0,
    )

    tree_cache._free_tail(slot, req, prefix_len=5)

    assert len(inner.token_to_kv_pool_allocator.freed) == 1
    assert inner.token_to_kv_pool_allocator.freed[0].tolist() == [8, 9, 10, 11]
    assert slot.kv_committed_len == slot.kv_allocated_len == 5
    assert req.kv_committed_len == req.kv_allocated_len == 5


def test_overshoot_trim_caps_all_visible_request_state():
    inner = _InnerCache(page_size=1)
    tree_cache = StreamingSession(inner)
    req = _SessionReq(committed=40, allocated=44, origin_len=26, finished_len=12)
    req.output_ids = list(range(14))
    req.swa_evicted_seqlen = 42

    tree_cache._trim_overshoot(req, finished_len=12)

    assert req.kv_committed_len == 38
    assert req.kv_allocated_len == 38
    assert req.swa_evicted_seqlen == 38
    assert len(req.output_ids) == 12
    assert inner.token_to_kv_pool_allocator.freed[0].tolist() == list(range(38, 44))
