# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_CAPACITY_CASES = (
    ((1, 3, 7), 8),
    ((2, 5), 6),
)

# These are exact-base lifecycle helpers unrelated to adaptive capacity.
# Stubbing this fixed list keeps the real non-dummy __post_init__ path while
# avoiding hardware, model-download, and backend initialization.
_BASE_LIFECYCLE_STUBS = (
    "_maybe_download_model_for_runai",
    "_handle_load_balance_method",
    "_handle_multimodal",
    "_handle_ssl_validation",
    "_handle_asr_validation",
    "_validate_prefill_only_disable_kv_cache_args",
    "_handle_deprecated_args",
    "_handle_prefill_delayer_env_compat",
    "_handle_missing_default_values",
    "_handle_hpu_backends",
    "_handle_cpu_backends",
    "_handle_npu_backends",
    "_handle_mps_backends",
    "_handle_xpu_backends",
    "_handle_piecewise_cuda_graph",
    "_handle_gpu_memory_settings",
    "_handle_model_specific_adjustments",
    "_handle_sampling_backend",
    "_handle_deterministic_inference",
    "_handle_attention_backend_compatibility",
    "_handle_mamba_backend",
    "_handle_linear_attn_backend",
    "_handle_kv4_compatibility",
    "_handle_page_size",
    "_handle_amd_specifics",
    "_handle_nccl_pre_warm",
    "_handle_grammar_backend",
    "_handle_multi_item_scoring",
    "_handle_prefill_only_disable_kv_cache",
    "_handle_hicache",
    "_handle_data_parallelism",
    "_handle_context_parallelism",
    "_handle_moe_kernel_config",
    "_handle_a2a_moe",
    "_handle_eplb_and_dispatch",
    "_handle_expert_distribution_metrics",
    "_handle_elastic_ep",
    "_handle_pipeline_parallelism",
    "_handle_encoder_disaggregation",
    "_handle_tokenizer_batching",
    "_handle_environment_variables",
    "_handle_cache_compatibility",
    "_handle_dllm_inference",
    "_handle_debug_utils",
    "_handle_other_validations",
)


def _noop(*_args, **_kwargs):
    return None


def _server_args(tmp_path, *, candidate_steps=(1, 7), **overrides):
    import sglang.srt.arg_groups.pd_disaggregation_hook as pd_hook
    import sglang.srt.arg_groups.speculative_hook as speculative_hook
    import sglang.srt.server_args as server_args_module
    import sglang.srt.utils.hf_transformers_utils as hf_utils

    ServerArgs = server_args_module.ServerArgs

    step_slug = "-".join(str(step) for step in candidate_steps)
    config = tmp_path / f"adaptive-{step_slug}.json"
    config.write_text(json.dumps({"candidate_steps": list(candidate_steps)}))
    values = {
        "speculative_algorithm": "EAGLE",
        "speculative_draft_model_path": "draft",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
        "speculative_adaptive": True,
        "speculative_adaptive_config": str(config),
        "page_size": 1,
    }
    values.update(overrides)

    trace = []
    hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    model_config = SimpleNamespace(hf_config=hf_config)
    original_hook = speculative_hook.handle_speculative_decoding

    def traced_hook(args):
        trace.append("hook-enter")
        original_hook(args)
        trace.append("hook-exit")

    def post_hook_checkpoint(self):
        trace.append("post-hook")

    with ExitStack() as stack:
        for name in _BASE_LIFECYCLE_STUBS:
            stack.enter_context(patch.object(ServerArgs, name, _noop))
        stack.enter_context(patch.object(ServerArgs, "_handle_load_format", post_hook_checkpoint))
        stack.enter_context(patch.object(ServerArgs, "get_model_config", return_value=model_config))
        stack.enter_context(
            patch.object(
                server_args_module,
                "get_device_memory_capacity",
                return_value=80 * 1024 * 1024 * 1024,
            )
        )
        stack.enter_context(
            patch.object(
                server_args_module,
                "current_platform",
                SimpleNamespace(apply_server_args_defaults=lambda _args: None),
                create=True,
            )
        )
        stack.enter_context(patch.object(pd_hook, "handle_pd_disaggregation", _noop))
        stack.enter_context(patch.object(hf_utils, "get_config", return_value=hf_config))
        stack.enter_context(
            patch.object(
                speculative_hook,
                "handle_speculative_decoding",
                traced_hook,
            )
        )
        # This must not be "dummy": dummy returns before the production hook.
        args = ServerArgs(model_path="verifier-model", **values)
    assert trace == ["hook-enter", "hook-exit", "post-hook"]
    return args, config


class _ModelConfigFactory:
    @classmethod
    def from_server_args(cls, _server_args):
        return SimpleNamespace(
            is_generation=True,
            context_len=100,
            image_token_id=None,
            is_multimodal=False,
            hf_config=SimpleNamespace(),
        )


def _tokenizer_reserved_tokens(args):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = args
    manager.model_config_class = _ModelConfigFactory
    manager.init_model_config()
    return manager.num_reserved_tokens


class _StopAfterRequestPool(RuntimeError):
    pass


def _model_runner_max_context(args, monkeypatch):
    import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_mixin

    captured = {}

    def fake_request_pool(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(kv_mixin, "ReqToTokenPool", fake_request_pool)
    monkeypatch.setattr(kv_mixin, "is_deepseek_dsa", lambda _config: False)
    monkeypatch.setattr(kv_mixin, "is_deepseek_v4", lambda _config: False)

    def stop_after_request_pool(*_args):
        raise _StopAfterRequestPool

    runner = SimpleNamespace(
        max_running_requests=4,
        req_to_token_pool=None,
        server_args=args,
        mambaish_config=None,
        model_config=SimpleNamespace(context_len=100, hf_config=SimpleNamespace()),
        device="cpu",
        is_draft_worker=False,
        _validate_prefill_only_disable_kv_cache_pool_family=stop_after_request_pool,
    )
    with pytest.raises(_StopAfterRequestPool):
        kv_mixin.ModelRunnerKVCacheMixin._init_pools(runner)
    return captured["max_context_len"]


def test_adaptive_tokenizer_reserves_largest_reachable_draft(tmp_path):
    for candidate_steps, expected in _CAPACITY_CASES:
        args, _ = _server_args(tmp_path, candidate_steps=candidate_steps)
        assert _tokenizer_reserved_tokens(args) == expected


def test_adaptive_decode_allocation_uses_largest_reachable_draft(tmp_path):
    from sglang.srt.managers.utils import get_alloc_len_per_decode

    for candidate_steps, expected in _CAPACITY_CASES:
        args, _ = _server_args(tmp_path, candidate_steps=candidate_steps)
        assert get_alloc_len_per_decode(args) == expected


def test_adaptive_capacity_is_stable_for_one_launch(tmp_path, monkeypatch):
    from sglang.srt.managers.utils import get_alloc_len_per_decode

    for candidate_steps, expected in _CAPACITY_CASES:
        args, config = _server_args(tmp_path, candidate_steps=candidate_steps)
        first = (
            _tokenizer_reserved_tokens(args),
            get_alloc_len_per_decode(args),
            _model_runner_max_context(args, monkeypatch),
        )
        config.write_text(json.dumps({"candidate_steps": [1, 2]}))
        second = (
            _tokenizer_reserved_tokens(args),
            get_alloc_len_per_decode(args),
            _model_runner_max_context(args, monkeypatch),
        )
        expected_tuple = (expected, expected, 104 + expected)
        assert first == second == expected_tuple


def test_static_speculative_decode_allocation_is_unchanged(tmp_path):
    from sglang.srt.managers.utils import get_alloc_len_per_decode

    args, _ = _server_args(
        tmp_path,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_adaptive=False,
    )
    assert get_alloc_len_per_decode(args) == 4


def test_non_speculative_decode_allocation_is_one(tmp_path):
    from sglang.srt.managers.utils import get_alloc_len_per_decode

    args, _ = _server_args(tmp_path, speculative_algorithm=None)
    assert get_alloc_len_per_decode(args) == 1


def test_static_tokenizer_reservation_is_unchanged(tmp_path):
    args, _ = _server_args(
        tmp_path,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_adaptive=False,
    )
    assert _tokenizer_reserved_tokens(args) == 4


def test_model_runner_context_headroom_keeps_adaptive_maximum(tmp_path, monkeypatch):
    for candidate_steps, expected in _CAPACITY_CASES:
        args, _ = _server_args(tmp_path, candidate_steps=candidate_steps)
        assert _model_runner_max_context(args, monkeypatch) == 104 + expected


def test_adaptive_downshift_never_requests_negative_slots(monkeypatch):
    import sglang.srt.speculative.eagle_info_v2 as eagle_v2
    import sglang.srt.speculative.spec_utils as spec_utils

    allocated = []
    req = SimpleNamespace(
        output_ids=[1],
        origin_input_ids=[1],
        kv_committed_len=10,
        kv_allocated_len=30,
        decode_batch_idx=0,
    )
    batch = SimpleNamespace(
        maybe_evict_swa=lambda: None,
        batch_size=lambda: 1,
        sampling_info=SimpleNamespace(penalizer_orchestrator=SimpleNamespace(is_required=False)),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        reqs=[req],
        device="cpu",
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((1, 64), dtype=torch.int64)),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
    )
    monkeypatch.setattr(
        eagle_v2,
        "get_alloc_len_per_decode",
        lambda *_args, **_kwargs: 2,
    )

    def alloc_slots(_cache, count):
        allocated.append(count)
        return torch.arange(count, dtype=torch.int64)

    monkeypatch.setattr(eagle_v2, "alloc_token_slots", alloc_slots)
    monkeypatch.setattr(
        spec_utils,
        "assign_req_to_token_pool_func",
        lambda *_args, **_kwargs: None,
    )
    eagle_v2.EagleDraftInputV2Mixin.prepare_for_decode(SimpleNamespace(), batch)
    assert all(count >= 0 for count in allocated)
    assert req.kv_allocated_len == 30
