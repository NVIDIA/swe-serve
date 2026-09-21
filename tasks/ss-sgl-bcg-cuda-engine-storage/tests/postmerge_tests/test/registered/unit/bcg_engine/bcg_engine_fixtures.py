# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch
from torch import nn


def public_contracts():
    try:
        module = importlib.import_module("sglang.srt.model_executor.prefill_cuda_graph")
    except ImportError as exc:
        pytest.fail(f"prefill CUDA-graph public contracts are unavailable: {exc}")
    factory_type = module.PrefillCudaGraphFactory
    try:
        factory = factory_type()
    except TypeError:
        factory = factory_type
    return factory, module.get_prefill_cuda_graph_diagnostics


def contract_value(record, name):
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def diagnostic_snapshot(diagnostics, subject):
    try:
        return diagnostics(subject)
    except TypeError as subject_error:
        try:
            return diagnostics()
        except TypeError:
            raise subject_error


def assert_snapshot_detached(diagnostics, subject, snapshot):
    backend = contract_value(snapshot, "backend")
    if isinstance(snapshot, Mapping):
        with pytest.raises(TypeError):
            snapshot["backend"] = "mutated"
    else:
        with pytest.raises((AttributeError, TypeError)):
            setattr(snapshot, "backend", "mutated")
    assert contract_value(diagnostic_snapshot(diagnostics, subject), "backend") == backend


def observe_factory_operations(monkeypatch):
    module = importlib.import_module("sglang.srt.model_executor.prefill_cuda_graph")
    factory_type = module.PrefillCudaGraphFactory
    observed = {"create": [], "shape_identity": []}

    def install(name):
        descriptor = inspect.getattr_static(factory_type, name)
        if isinstance(descriptor, classmethod):
            original = descriptor.__func__

            def delegated(cls, *args, **kwargs):
                observed[name].append((args, kwargs))
                return original(cls, *args, **kwargs)

            replacement = classmethod(delegated)
        elif isinstance(descriptor, staticmethod):
            original = descriptor.__func__

            def delegated(*args, **kwargs):
                observed[name].append((args, kwargs))
                return original(*args, **kwargs)

            replacement = staticmethod(delegated)
        else:
            original = descriptor

            def delegated(instance, *args, **kwargs):
                observed[name].append((args, kwargs))
                return original(instance, *args, **kwargs)

            replacement = delegated
        monkeypatch.setattr(factory_type, name, replacement)

    install("create")
    install("shape_identity")
    return observed


def cuda_graph_launch_count(call):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return sum(
        event.count for event in prof.key_averages() if "graphlaunch" in event.key.replace("_", "").lower()
    )


def calibrate_cuda_graph_observer():
    source = torch.zeros(1, device="cuda")
    target = torch.zeros_like(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        target.copy_(source + 1)
    assert cuda_graph_launch_count(graph.replay) >= 1
    assert cuda_graph_launch_count(lambda: target.copy_(source + 2)) == 0


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("real CUDA capture is required")


def ensure_production_runtime():
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.utils.network import get_free_port

    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_free_port()}",
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(backend="nccl")


@dataclass
class TinyHFConfig:
    architectures: List[str] = field(default_factory=lambda: ["TinyForCausalLM"])


@dataclass
class TinyModelConfig:
    num_hidden_layers: int = 1
    hidden_size: int = 4
    vocab_size: int = 4
    dtype: object = torch.float32
    is_encoder_decoder: bool = False
    is_multimodal: bool = False
    hf_config: TinyHFConfig = field(default_factory=TinyHFConfig)


class TinyReqToTokenPool:
    size = 16


class TinyTokenToKVPool:
    pass


class TinyLoraManager:
    def prepare_lora_batch(self, _forward_batch):
        return None


class TinyGroup:
    ca_comm = None

    def barrier(self):
        return None


class TinyAttentionBackend:
    def __init__(self):
        self.capture_states = []
        self.diagnostic_capture_states = []
        self.batch_sizes = []
        self.diagnostic_subject = None

    def init_forward_metadata(self, _forward_batch):
        return None

    def init_cuda_graph_state(self, _max_bs, _max_num_token):
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_capture_cuda_graph(self, *_args, **_kwargs):
        return None

    def init_forward_metadata_replay_cuda_graph(self, *_args, **_kwargs):
        return None

    def forward(
        self,
        query,
        _key,
        _value,
        _attention_layer,
        forward_batch,
        _save_kv_cache=True,
        **_kwargs,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        self.capture_states.append(get_is_capture_mode())
        self.batch_sizes.append(forward_batch.batch_size)
        if self.diagnostic_subject is not None:
            _factory, diagnostics = public_contracts()
            self.diagnostic_capture_states.append(
                bool(
                    contract_value(
                        diagnostic_snapshot(diagnostics, self.diagnostic_subject),
                        "capture_active",
                    )
                )
            )
        return query + 3


class TinyAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        from sglang.srt.layers.radix_attention import RadixAttention

        self.attn = RadixAttention(
            num_heads=1,
            head_dim=4,
            scaling=1.0,
            num_kv_heads=1,
            layer_id=0,
        )

    def forward(self, hidden_states, forward_batch):
        return self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            forward_batch,
        ).reshape(-1, 4)


class TinyLayerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.layers = nn.ModuleList([TinyAttentionBlock()])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        del input_embeds
        self.forward_calls += 1
        hidden = input_ids.float().unsqueeze(1).expand(-1, 4)
        hidden = hidden + positions.float().unsqueeze(1)
        return self.layers[0](hidden, forward_batch).clone()


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyLayerStack()
        self.quant_config = None

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        hidden = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=kwargs.get("input_embeds"),
        )
        return LogitsProcessorOutput(next_token_logits=hidden, hidden_states=hidden)


def engine_server_args(*, capture_sizes=None):
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    args = ServerArgs(model_path="dummy")
    args.disable_piecewise_cuda_graph = False
    args.piecewise_cuda_graph_tokens = list(capture_sizes or [4, 8])
    args.piecewise_cuda_graph_compiler = "eager"
    args.cuda_graph_bs = [1, 4]
    set_global_server_args_for_scheduler(args)
    return args


def make_runner(*, capture_sizes=None, server_args=None):
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    ensure_production_runtime()
    runner = object.__new__(ModelRunner)
    runner.device = "cuda"
    runner.gpu_id = 0
    runner.dtype = torch.float32
    runner.model_config = TinyModelConfig()
    runner.server_args = server_args or engine_server_args(capture_sizes=capture_sizes)
    if capture_sizes is not None:
        runner.server_args.piecewise_cuda_graph_tokens = list(capture_sizes)
    runner.model = TinyCausalLM()
    runner.req_to_token_pool = TinyReqToTokenPool()
    runner.token_to_kv_pool = TinyTokenToKVPool()
    runner.token_to_kv_pool_allocator = runner.token_to_kv_pool
    runner.lora_manager = TinyLoraManager()
    runner.attn_backend = TinyAttentionBackend()
    runner.decode_attn_backend = runner.attn_backend
    runner.decode_attn_backend_group = [runner.attn_backend]
    runner.tp_group = TinyGroup()
    runner.is_multimodal = False
    runner.is_generation = True
    runner.is_draft_worker = False
    runner.is_hybrid_swa = False
    runner.support_pp = False
    runner.spec_algorithm = SpeculativeAlgorithm.NONE
    runner.piecewise_cuda_graph_runner = None
    runner.graph_runner = None
    runner.memory_pool_config = None
    runner.use_ngram_embedding = False
    runner.eagle_use_aux_hidden_state = False
    runner.dflash_use_aux_hidden_state = False
    runner.hisparse_coordinator = None
    runner.eplb_manager = None
    runner.forward_pass_id = 0
    runner.token_table = None
    return runner


def initialize_runner(runner):
    runner.init_piecewise_cuda_graphs()
    return runner


def initialize_device_graph_runner(runner):
    runner.server_args.disable_piecewise_cuda_graph = True
    runner.server_args.cuda_graph_bs = [1, 4]
    runner.init_device_graphs()
    return runner


def make_batch(runner, values, *, seq_lens=None):
    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    values = torch.as_tensor(values, device="cuda", dtype=torch.int64)
    num_tokens = values.numel()
    seq_lens = list(seq_lens or [num_tokens])
    batch_size = len(seq_lens)
    starts = []
    offset = 0
    for length in seq_lens:
        starts.append(offset)
        offset += length
    assert offset == num_tokens
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=values,
        req_pool_indices=torch.arange(batch_size, device="cuda", dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int64),
        out_cache_loc=torch.arange(num_tokens, device="cuda", dtype=torch.int64),
        seq_lens_sum=num_tokens,
        orig_seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        positions=torch.arange(num_tokens, device="cuda", dtype=torch.int64),
        extend_num_tokens=num_tokens,
        extend_seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int64),
        extend_prefix_lens=torch.zeros(batch_size, device="cuda", dtype=torch.int64),
        extend_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int64),
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=seq_lens,
        extend_logprob_start_lens_cpu=seq_lens,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool=runner.token_to_kv_pool,
        attn_backend=runner.attn_backend,
        dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        global_forward_mode=ForwardMode.EXTEND,
        spec_algorithm=runner.spec_algorithm,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        num_token_non_padded_cpu=num_tokens,
    )


def make_decode_batch(runner, values):
    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    values = torch.as_tensor(values, device="cuda", dtype=torch.int64)
    batch_size = values.numel()
    seq_lens = torch.ones(batch_size, device="cuda", dtype=torch.int32)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=values,
        req_pool_indices=torch.arange(batch_size, device="cuda", dtype=torch.int64),
        seq_lens=seq_lens,
        out_cache_loc=torch.arange(batch_size, device="cuda", dtype=torch.int64),
        seq_lens_sum=batch_size,
        orig_seq_lens=seq_lens,
        seq_lens_cpu=torch.ones(batch_size, dtype=torch.int32),
        positions=torch.arange(batch_size, device="cuda", dtype=torch.int64),
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool=runner.token_to_kv_pool,
        attn_backend=runner.attn_backend,
        dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        global_forward_mode=ForwardMode.DECODE,
        spec_algorithm=runner.spec_algorithm,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        num_token_non_padded=torch.tensor(batch_size, device="cuda"),
        num_token_non_padded_cpu=batch_size,
    )


def eager_expected(values):
    values = torch.as_tensor(values, device="cuda", dtype=torch.float32)
    positions = torch.arange(values.numel(), device="cuda", dtype=torch.float32)
    return (values + positions).unsqueeze(1).expand(-1, 4) + 3
