# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from bcg_engine_fixtures import (
    assert_snapshot_detached,
    calibrate_cuda_graph_observer,
    contract_value,
    cuda_graph_launch_count,
    diagnostic_snapshot,
    eager_expected,
    initialize_runner,
    make_batch,
    make_runner,
    observe_factory_operations,
    public_contracts,
    require_cuda,
)


def assert_capacity_present(capacity):
    assert capacity is not None
    if isinstance(capacity, (int, float)):
        assert capacity > 0
    elif hasattr(capacity, "__len__"):
        assert len(capacity) > 0


def test_segmented_replay_updates_and_capture_state():
    require_cuda()
    calibrate_cuda_graph_observer()
    _factory, diagnostics = public_contracts()
    runner = make_runner(capture_sizes=[4, 8])
    runner.attn_backend.diagnostic_subject = runner
    initialize_runner(runner)

    snapshot = diagnostic_snapshot(diagnostics, runner)
    assert contract_value(snapshot, "backend") == "breakable"
    assert contract_value(snapshot, "graph_count") >= 2
    assert contract_value(snapshot, "segment_count") >= 2
    assert contract_value(snapshot, "eager_break_count") >= 1
    assert True in runner.attn_backend.capture_states
    assert True in runner.attn_backend.diagnostic_capture_states
    previous_replay_count = contract_value(snapshot, "replay_count")
    dense_calls = runner.model.model.forward_calls

    cases = (
        ([1, 3, 5], [1, 2]),
        ([9, 2, 8, 4], [4]),
        ([4, 7, 6, 5, 2], [1, 1, 3]),
    )
    live_batch_sizes = []
    for values, seq_lens in cases:
        before = len(runner.attn_backend.capture_states)
        diagnostic_before = len(runner.attn_backend.diagnostic_capture_states)
        batch_before = len(runner.attn_backend.batch_sizes)
        result = []
        launches = cuda_graph_launch_count(
            lambda: result.append(runner.forward_extend(make_batch(runner, values, seq_lens=seq_lens)))
        )
        output, used_graph = result[0]
        assert used_graph is True
        assert launches >= 2
        assert runner.model.model.forward_calls == dense_calls
        assert runner.attn_backend.capture_states[before:] == [True]
        assert runner.attn_backend.diagnostic_capture_states[diagnostic_before:] == [True]
        observed_batch_sizes = runner.attn_backend.batch_sizes[batch_before:]
        assert observed_batch_sizes == [len(seq_lens)]
        live_batch_sizes.extend(observed_batch_sizes)
        torch.testing.assert_close(output.next_token_logits, eager_expected(values))
        current_replay_count = contract_value(
            diagnostic_snapshot(diagnostics, runner), "replay_count"
        )
        assert current_replay_count > previous_replay_count
        previous_replay_count = current_replay_count

    assert live_batch_sizes == [2, 1, 3]
    assert (
        contract_value(diagnostic_snapshot(diagnostics, runner), "capture_active")
        is False
    )


def test_factory_construction_and_typed_shape_identity(monkeypatch):
    require_cuda()
    factory, diagnostics = public_contracts()
    observed = observe_factory_operations(monkeypatch)
    runner = initialize_runner(make_runner(capture_sizes=[4, 8]))
    assert len(observed["create"]) >= 1
    observed_sizes = {
        args[0] if args else kwargs.get("size")
        for args, kwargs in observed["shape_identity"]
        if isinstance(args[0] if args else kwargs.get("size"), int)
    }
    assert {4, 8} <= observed_sizes

    small = factory.shape_identity(4)
    assert small != factory.shape_identity(8)
    assert small != factory.shape_identity(4, stream_idx=1)
    assert small != factory.shape_identity(4, variant_label="alternate")
    snapshot = diagnostic_snapshot(diagnostics, runner)
    assert_snapshot_detached(diagnostics, runner, snapshot)


def test_shared_largest_view_and_untouched_tail():
    require_cuda()
    _factory, diagnostics = public_contracts()
    runner = initialize_runner(make_runner(capture_sizes=[4, 8]))

    large_values = [8, 7, 6, 5, 4, 3, 2, 1]
    large, used_graph = runner.forward_extend(make_batch(runner, large_values))
    assert used_graph is True
    torch.cuda.synchronize()
    large_view = large.next_token_logits
    storage_identity = large_view.untyped_storage().data_ptr()
    tail = large_view[4:].clone()
    materialized = diagnostic_snapshot(diagnostics, runner)
    allocation_identity = contract_value(materialized, "allocation_identity")
    assert allocation_identity is not None
    assert_capacity_present(contract_value(materialized, "allocation_capacity"))

    for values in ([1, 2, 3, 4], [11, 7, 5, 3], [2, 9, 1, 8]):
        small, used_graph = runner.forward_extend(make_batch(runner, values))
        torch.cuda.synchronize()
        assert used_graph is True
        assert small.next_token_logits.untyped_storage().data_ptr() == storage_identity
        assert small.next_token_logits.shape == (4, 4)
        torch.testing.assert_close(small.next_token_logits, eager_expected(values))
        torch.testing.assert_close(large_view[4:], tail)

    snapshot = diagnostic_snapshot(diagnostics, runner)
    assert contract_value(snapshot, "allocation_identity") == allocation_identity
    assert_capacity_present(contract_value(snapshot, "allocation_capacity"))
