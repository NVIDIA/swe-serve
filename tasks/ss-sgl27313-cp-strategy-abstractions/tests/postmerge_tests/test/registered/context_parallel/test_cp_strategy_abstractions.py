# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import sglang.srt.layers.cp as cp
import sglang.srt.layers.cp.utils as cp_utils
import sglang.srt.server_args as server_args_module
from sglang.srt.environ import EnvBool, envs
from sglang.srt.layers.cp import (
    BaseContextParallelMetadata,
    ContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
)
from sglang.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def _reset_cp_singleton():
    init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))
    yield
    init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))


def test_enums_environment_backend_and_exports(monkeypatch) -> None:
    assert isinstance(envs.SGLANG_ENABLE_CP_V2, EnvBool)
    monkeypatch.delenv("SGLANG_ENABLE_CP_V2", raising=False)
    assert envs.SGLANG_ENABLE_CP_V2.get() is False
    monkeypatch.setenv("SGLANG_ENABLE_CP_V2", "1")
    assert envs.SGLANG_ENABLE_CP_V2.get() is True

    assert ContextParallelStrategyKind.ZIGZAG.value == 1
    assert ContextParallelStrategyKind.INTERLEAVE.value == 2
    assert ContextParallelStrategyKind.NONE.cli_value == "none"
    for invalid_strategy in ("striped", "none", "NONE", "ZIGZAG", "INTERLEAVE"):
        with pytest.raises(ValueError):
            ContextParallelStrategyKind.from_string(invalid_strategy)

    assert CPAttentionBackendKind.FLASH_ATTENTION.value == 0
    assert CPAttentionBackendKind.from_string("fa3") == CPAttentionBackendKind.FLASH_ATTENTION
    assert CPAttentionBackendKind.from_string("flashinfer") == CPAttentionBackendKind.FLASH_ATTENTION
    for invalid_backend in ("triton", "none", "FA3", "FLASHINFER"):
        with pytest.raises(ValueError):
            CPAttentionBackendKind.from_string(invalid_backend)

    public_exports = {
        "BaseContextParallelMetadata": BaseContextParallelMetadata,
        "CPAttentionBackendKind": CPAttentionBackendKind,
        "ContextParallelMetadata": ContextParallelMetadata,
        "ContextParallelStrategy": ContextParallelStrategy,
        "ContextParallelStrategyKind": ContextParallelStrategyKind,
        "InterleaveCPStrategy": InterleaveCPStrategy,
        "InterleaveContextParallelMetadata": InterleaveContextParallelMetadata,
        "ZigzagCPStrategy": ZigzagCPStrategy,
        "ZigzagContextParallelMetadata": ZigzagContextParallelMetadata,
        "get_cp_strategy": get_cp_strategy,
        "get_cp_strategy_kind": get_cp_strategy_kind,
        "init_cp_strategy": init_cp_strategy,
        "is_cp_enabled": is_cp_enabled,
        "is_interleave": is_interleave,
        "is_zigzag": is_zigzag,
    }
    for name, expected in public_exports.items():
        assert getattr(cp, name) is expected

    helper_names = {
        "get_cp_strategy",
        "get_cp_strategy_kind",
        "init_cp_strategy",
        "is_cp_enabled",
        "is_interleave",
        "is_zigzag",
    }
    for name, expected in public_exports.items():
        if name in helper_names:
            assert not hasattr(cp_utils, name)
        else:
            assert getattr(cp_utils, name) is expected


def test_strategy_singleton_lifecycle_residual_identity_and_env_timing(monkeypatch) -> None:
    for disabled_size in (1, 0, -3):
        assert (
            init_cp_strategy(
                SimpleNamespace(
                    enable_prefill_cp=True,
                    cp_strategy="zigzag",
                    attn_cp_size=disabled_size,
                )
            )
            is None
        )
        assert get_cp_strategy() is None
        assert get_cp_strategy_kind() == ContextParallelStrategyKind.NONE
        assert not is_cp_enabled()

    strategy = None
    for enabled_value in ("0", "1"):
        assert init_cp_strategy(SimpleNamespace(enable_prefill_cp=False)) is None
        assert get_cp_strategy() is None
        monkeypatch.setenv("SGLANG_ENABLE_CP_V2", enabled_value)
        assert (
            init_cp_strategy(
                SimpleNamespace(
                    enable_prefill_cp=True,
                    cp_strategy="zigzag",
                    attn_cp_size=4,
                )
            )
            is None
        )
        initialized = get_cp_strategy()
        assert isinstance(initialized, ZigzagCPStrategy)
        assert initialized.cp_size == 4
        assert get_cp_strategy() is initialized
        strategy = initialized

    assert (
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="interleave",
                attn_cp_size=2,
            )
        )
        is None
    )
    replacement = get_cp_strategy()
    assert isinstance(replacement, InterleaveCPStrategy)
    assert replacement.cp_size == 2
    assert replacement is not strategy
    assert get_cp_strategy() is replacement
    assert not is_zigzag()

    assert init_cp_strategy(SimpleNamespace(enable_prefill_cp=False)) is None
    assert get_cp_strategy() is None

    with pytest.raises(ValueError):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="striped",
                attn_cp_size=2,
            )
        )


def test_lazy_worker_recovery_and_unconfigured_fallback(monkeypatch) -> None:
    worker_args = SimpleNamespace(
        enable_prefill_cp=True,
        cp_strategy="interleave",
        attn_cp_size=4,
    )
    server_args_module.set_global_server_args_for_scheduler(worker_args)
    recovered = get_cp_strategy()
    assert isinstance(recovered, InterleaveCPStrategy)
    assert get_cp_strategy() is recovered
    assert is_interleave()

    init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))
    server_args_module.set_global_server_args_for_scheduler(SimpleNamespace(enable_prefill_cp=False))
    assert get_cp_strategy() is None

    init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def _unconfigured_global_args():
        raise ValueError("global server args are not configured")

    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        _unconfigured_global_args,
    )
    assert get_cp_strategy() is None
    assert get_cp_strategy_kind() == ContextParallelStrategyKind.NONE


def test_server_context_handler_zigzag_and_disabled_states() -> None:
    cases = (
        (True, "zigzag", 2, ZigzagCPStrategy),
        (False, "zigzag", 2, None),
        (True, "zigzag", 1, None),
    )
    for enabled, strategy_name, cp_size, expected_type in cases:
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))
        server_args = object.__new__(ServerArgs)
        values = {
            "enable_prefill_context_parallel": False,
            "enable_dsa_prefill_context_parallel": False,
            "enable_prefill_cp": enabled,
            "cp_strategy": strategy_name,
            "attn_cp_size": cp_size,
            "tp_size": 2,
            "dp_size": 1,
            "moe_dp_size": 1,
            "ep_size": 1,
            "pp_size": 1,
            "enable_aiter_allreduce_fusion": False,
        }
        for name, value in values.items():
            setattr(server_args, name, value)

        server_args._handle_context_parallelism()

        selected = get_cp_strategy()
        if expected_type is None:
            assert selected is None
        else:
            assert isinstance(selected, expected_type)
            assert selected.cp_size == cp_size
            assert get_cp_strategy() is selected


class _ForwardMode:
    def __init__(self, is_cp_extend: bool):
        self.is_cp_extend = is_cp_extend

    def is_context_parallel_extend(self) -> bool:
        return self.is_cp_extend


def test_strategy_applicability_boundaries_and_forward_mode() -> None:
    assert inspect.isabstract(ContextParallelStrategy)
    assert issubclass(ZigzagCPStrategy, ContextParallelStrategy)
    assert issubclass(InterleaveCPStrategy, ContextParallelStrategy)
    assert ZigzagCPStrategy.name == "zigzag"
    assert ZigzagCPStrategy.kind == ContextParallelStrategyKind.ZIGZAG
    assert InterleaveCPStrategy.name == "interleave"
    assert InterleaveCPStrategy.kind == ContextParallelStrategyKind.INTERLEAVE

    zigzag = ZigzagCPStrategy(cp_size=4)
    interleave = InterleaveCPStrategy(cp_size=4)
    implicit_extend = SimpleNamespace()
    explicit_implicit_extend = SimpleNamespace(forward_mode=None)
    cp_extend = SimpleNamespace(forward_mode=_ForwardMode(True))
    decode = SimpleNamespace(forward_mode=_ForwardMode(False))

    assert not zigzag.can_apply(7, implicit_extend)
    assert zigzag.can_apply(8, implicit_extend)
    assert zigzag.can_apply(8, explicit_implicit_extend)
    assert zigzag.can_apply(8, cp_extend)
    assert not zigzag.can_apply(8, decode)

    assert not interleave.can_apply(3, implicit_extend)
    assert interleave.can_apply(4, implicit_extend)
    assert interleave.can_apply(4, explicit_implicit_extend)
    assert interleave.can_apply(4, cp_extend)
    assert not interleave.can_apply(4, decode)

    for disabled_size in (1, 0, -1):
        assert not ZigzagCPStrategy(cp_size=disabled_size).can_apply(100, implicit_extend)
        assert not InterleaveCPStrategy(cp_size=disabled_size).can_apply(100, implicit_extend)


def test_metadata_summaries_and_zigzag_compatibility_alias() -> None:
    base_metadata = BaseContextParallelMetadata()
    assert base_metadata.total_seq_lens == 0
    assert base_metadata.bs == 1
    assert ContextParallelMetadata is ZigzagContextParallelMetadata
    assert issubclass(ZigzagContextParallelMetadata, BaseContextParallelMetadata)
    assert issubclass(InterleaveContextParallelMetadata, BaseContextParallelMetadata)
    assert [field.name for field in dataclasses.fields(InterleaveContextParallelMetadata)] == [
        field.name for field in dataclasses.fields(BaseContextParallelMetadata)
    ]

    cases = (
        (ZigzagCPStrategy(cp_size=4), ZigzagContextParallelMetadata),
        (InterleaveCPStrategy(cp_size=2), InterleaveContextParallelMetadata),
    )
    for strategy, expected_type in cases:
        from_extend = strategy.build_metadata(
            num_tokens=100,
            seqs_len=[50, 50],
            extend_seqs_len=[7, 11, 13],
        )
        assert isinstance(from_extend, expected_type)
        assert from_extend.total_seq_lens == 31
        assert from_extend.bs == 3

        from_sequences = strategy.build_metadata(
            num_tokens=100,
            seqs_len=[8, 9],
            extend_seqs_len=None,
        )
        assert isinstance(from_sequences, expected_type)
        assert from_sequences.total_seq_lens == 17
        assert from_sequences.bs == 2

        from_empty_extend = strategy.build_metadata(
            num_tokens=100,
            seqs_len=[8, 9],
            extend_seqs_len=[],
        )
        assert isinstance(from_empty_extend, expected_type)
        assert from_empty_extend.total_seq_lens == 17
        assert from_empty_extend.bs == 2

        from_tokens = strategy.build_metadata(
            num_tokens=23,
            seqs_len=None,
            extend_seqs_len=None,
        )
        assert isinstance(from_tokens, expected_type)
        assert from_tokens.total_seq_lens == 23
        assert from_tokens.bs == 1

        from_empty_lists = strategy.build_metadata(
            num_tokens=23,
            seqs_len=[],
            extend_seqs_len=[],
        )
        assert isinstance(from_empty_lists, expected_type)
        assert from_empty_lists.total_seq_lens == 23
        assert from_empty_lists.bs == 1

    zigzag_defaults = ZigzagContextParallelMetadata()
    assert zigzag_defaults.total_seq_lens == 0
    assert zigzag_defaults.bs == 1
    for field in (
        "split_list",
        "zigzag_index",
        "cp_reverse_index",
        "reverse_split_len",
        "per_rank_actual_token",
        "max_rank_len",
        "kv_len_prev_tensor",
        "kv_len_next_tensor",
        "actual_seq_q_prev_tensor",
        "actual_seq_q_next_tensor",
        "cu_seqlens_q_prev_tensor",
        "cu_seqlens_q_next_tensor",
        "kv_len_prev_list",
        "kv_len_next_list",
        "actual_seq_q_prev_list",
        "actual_seq_q_next_list",
    ):
        assert getattr(zigzag_defaults, field) is None
    for field in (
        "total_q_prev_tokens",
        "total_q_next_tokens",
        "max_seqlen_q_prev",
        "max_seqlen_q_next",
    ):
        assert getattr(zigzag_defaults, field) == 0


def test_strategy_topology_properties_delegate_to_runtime_state(monkeypatch) -> None:
    strategy = ZigzagCPStrategy(cp_size=4)
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.get_attn_context_model_parallel_rank",
        lambda: 3,
    )
    assert strategy.cp_rank == 3

    for cp_enabled, dsa_arch, expected in (
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ):
        monkeypatch.setattr(
            "sglang.srt.server_args.get_global_server_args",
            lambda cp_enabled=cp_enabled, dsa_arch=dsa_arch: SimpleNamespace(
                enable_prefill_cp=cp_enabled,
                _is_dsa_model_arch=dsa_arch,
            ),
        )
        assert strategy.per_layer_attn_cp_comm is expected


def test_runtime_followup_operations_fail_closed() -> None:
    required_signatures = {
        "can_apply": ("self", "num_tokens", "forward_batch"),
        "build_metadata": ("self", "num_tokens", "seqs_len", "extend_seqs_len"),
        "shard_hidden_states": ("self", "x", "forward_batch"),
        "shard_position_ids": ("self", "positions", "forward_batch"),
        "gather_hidden_states": ("self", "x", "forward_batch", "stream"),
        "gather_kv_cache": ("self", "x", "forward_batch", "stream"),
        "shard_per_request": ("self", "extend_seqs_cpu", "extend_seqs"),
        "split_before_forward": (
            "self",
            "forward_batch",
            "input_ids",
            "positions",
            "input_embeds",
        ),
        "run_attention": (
            "self",
            "q",
            "forward_batch",
            "device",
            "attn_fn",
            "attention_backend",
        ),
        "materialize_full_kv": ("self", "forward_batch", "layer", "k", "v"),
        "reindex_attn_metadata": ("self", "core_attn_metadata"),
    }
    for method_name, parameter_names in required_signatures.items():
        assert (
            tuple(inspect.signature(getattr(ContextParallelStrategy, method_name)).parameters)
            == parameter_names
        )

    base_parameters = inspect.signature(ContextParallelStrategy.build_metadata).parameters
    assert base_parameters["extend_seqs_len"].default is None
    assert (
        inspect.signature(ContextParallelStrategy.gather_hidden_states).parameters["stream"].default is None
    )
    assert inspect.signature(ContextParallelStrategy.gather_kv_cache).parameters["stream"].default is None
    assert (
        inspect.signature(ContextParallelStrategy.split_before_forward).parameters["input_embeds"].default
        is None
    )
    assert (
        inspect.signature(ContextParallelStrategy.run_attention).parameters["attention_backend"].default
        == CPAttentionBackendKind.FLASH_ATTENTION
    )

    batch = SimpleNamespace()
    for strategy in (ZigzagCPStrategy(cp_size=4), InterleaveCPStrategy(cp_size=4)):
        with pytest.raises(NotImplementedError):
            strategy.shard_hidden_states(object(), batch)
        with pytest.raises(NotImplementedError):
            strategy.shard_position_ids(object(), batch)
        with pytest.raises(NotImplementedError):
            strategy.gather_hidden_states(object(), batch)
        with pytest.raises(NotImplementedError):
            strategy.gather_hidden_states(object(), batch, stream=object())
        with pytest.raises(NotImplementedError):
            strategy.gather_kv_cache(object(), batch)
        with pytest.raises(NotImplementedError):
            strategy.gather_kv_cache(object(), batch, stream=object())
        with pytest.raises(NotImplementedError):
            strategy.run_attention(object(), batch, object(), lambda *args: None)
        with pytest.raises(NotImplementedError):
            strategy.run_attention(
                object(),
                batch,
                object(),
                lambda *args: None,
                attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
            )
        with pytest.raises(NotImplementedError):
            strategy.materialize_full_kv(batch, object(), object(), object())
        with pytest.raises(NotImplementedError):
            strategy.shard_per_request([], object())
        with pytest.raises(NotImplementedError):
            strategy.split_before_forward(batch, object(), object())
        with pytest.raises(NotImplementedError):
            strategy.split_before_forward(
                batch,
                object(),
                object(),
                input_embeds=object(),
            )
        assert strategy.reindex_attn_metadata(object()) is None
