# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import requests
import torch

MASK_ID = 156895
BLOCK_SIZE = 4
VOCAB_SIZE = 32


def _real_config(monkeypatch, tmp_path, algorithm="LowConfidence", **settings):
    import sglang.srt.dllm.config as config_module

    monkeypatch.setattr(
        config_module.ModelConfig,
        "from_server_args",
        lambda *args, **kwargs: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"])
        ),
    )
    config_path = None
    if settings:
        config_path = tmp_path / f"{algorithm}.yaml"
        config_path.write_text("".join(f"{key}: {value}\n" for key, value in settings.items()))
    args = SimpleNamespace(
        dllm_algorithm=algorithm,
        dllm_algorithm_config=str(config_path) if config_path else None,
        dllm_block_size=BLOCK_SIZE,
        max_running_requests=None,
        model_path="immutable-public-llada",
        revision=None,
    )
    return config_module.DllmConfig.from_server_args(args)


def _forward_batch(input_ids: torch.Tensor, batch_size: int):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    block_size = input_ids.numel() // batch_size
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=input_ids,
        req_pool_indices=torch.arange(batch_size, dtype=torch.int32),
        seq_lens=torch.full((batch_size,), block_size, dtype=torch.int32),
        out_cache_loc=torch.arange(input_ids.numel(), dtype=torch.int64),
        seq_lens_sum=input_ids.numel(),
        positions=torch.arange(input_ids.numel(), dtype=torch.int64),
        extend_num_tokens=input_ids.numel(),
        extend_seq_lens=torch.full((batch_size,), block_size, dtype=torch.int32),
        extend_prefix_lens=torch.zeros(batch_size, dtype=torch.int32),
        extend_start_loc=torch.arange(batch_size, dtype=torch.int32) * block_size,
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[block_size] * batch_size,
    )


def _logits_output(logits: torch.Tensor):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    return LogitsProcessorOutput(next_token_logits=None, full_logits=logits)


def _position_logits(num_tokens: int) -> torch.Tensor:
    logits = torch.full((num_tokens, VOCAB_SIZE), -12.0)
    targets = torch.arange(num_tokens).remainder(7).add(1)
    logits.scatter_(1, targets[:, None], 12.0)
    return logits


class _StaticRunner:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.calls = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.calls += 1
        return _logits_output(self.logits.clone()), False


class _TwoStageRunner:
    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.calls = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.calls += 1
        target = 1 if self.calls == 1 else 2
        logits = torch.full((self.num_tokens, VOCAB_SIZE), -12.0)
        logits[:, target] = 12.0
        return _logits_output(logits), False


def _generated_count(generated) -> int:
    if generated is None:
        return 0
    if isinstance(generated, torch.Tensor):
        return generated.numel()
    return sum(torch.as_tensor(row).numel() for row in generated)


def test_low_confidence_multiblock_independent_threshold(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    local_mask = 7
    ids = torch.tensor([9, local_mask, local_mask, local_mask, 8, local_mask, local_mask, local_mask])
    config = _real_config(monkeypatch, tmp_path, threshold=0.9, block_size=BLOCK_SIZE)
    config.mask_id = local_mask
    runner = _StaticRunner(_position_logits(ids.numel()))
    _, generated, _ = get_algorithm(config).run(runner, _forward_batch(ids, 2))

    assert ids.tolist() == [9, 2, 3, 4, 8, 6, 7, 1]
    assert _generated_count(generated) == 6


def test_low_confidence_fallback_progresses_every_active_block(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    local_mask = 17
    ids = torch.tensor([9, local_mask, local_mask, local_mask, 8, local_mask, local_mask, local_mask])
    config = _real_config(monkeypatch, tmp_path, threshold=1.1, block_size=BLOCK_SIZE)
    config.mask_id = local_mask
    runner = _StaticRunner(_position_logits(ids.numel()))
    get_algorithm(config).run(runner, _forward_batch(ids, 2))

    assert local_mask not in ids.tolist()
    assert ids[0].item() == 9 and ids[4].item() == 8
    assert runner.calls <= BLOCK_SIZE + 1


def test_joint_threshold_m2t_t2t_preserves_prompt(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    local_mask = 7
    ids = torch.tensor([9, local_mask, local_mask, local_mask])
    config = _real_config(
        monkeypatch,
        tmp_path,
        algorithm="JointThreshold",
        threshold=0.5,
        edit_threshold=0.5,
        max_post_edit_steps=2,
        block_size=BLOCK_SIZE,
    )
    config.mask_id = local_mask
    _, generated, _ = get_algorithm(config).run(_TwoStageRunner(ids.numel()), _forward_batch(ids, 1))

    assert ids.tolist() == [9, 2, 2, 2]
    assert _generated_count(generated) == 3


def test_joint_threshold_mask_free_avoids_iterative_refinement(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    ids = torch.tensor([9, 1, 2, 3])
    config = _real_config(monkeypatch, tmp_path, algorithm="JointThreshold")
    config.mask_id = 7
    runner = _StaticRunner(_position_logits(ids.numel()))
    _, generated, _ = get_algorithm(config).run(runner, _forward_batch(ids, 1))

    assert ids.tolist() == [9, 1, 2, 3]
    assert _generated_count(generated) == 0
    assert runner.calls <= 1


@contextmanager
def _server(
    *,
    model: str,
    dllm: bool,
    max_running: int | None,
    graph_bs: tuple[int, ...] | None,
    mem_fraction: float,
    kv_cache_dtype: str | None = None,
):
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import find_available_port, popen_launch_server

    port = find_available_port(30000)
    base_url = f"http://127.0.0.1:{port}"
    args = [
        "--trust-remote-code",
        "--tp",
        "1",
        "--mem-fraction-static",
        str(mem_fraction),
    ]
    if max_running is not None:
        args.extend(["--max-running-requests", str(max_running)])
    if dllm:
        args.extend(
            [
                "--attention-backend",
                "flashinfer",
                "--dllm-algorithm",
                "LowConfidence",
                "--grammar-backend",
                "none",
            ]
        )
    if graph_bs is None:
        args.append("--disable-cuda-graph")
    else:
        args.append("--cuda-graph-bs")
        args.extend(str(size) for size in graph_bs)
    if kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", kv_cache_dtype])

    process = popen_launch_server(model, base_url, timeout=600, other_args=args)
    try:
        yield base_url
    finally:
        kill_process_tree(process.pid)


def _generate(
    base_url: str,
    *,
    rid,
    input_ids=None,
    text=None,
    max_new_tokens=32,
    timeout=600,
):
    payload = {
        "rid": rid,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if input_ids is not None:
        payload["input_ids"] = input_ids
    else:
        payload["text"] = text
    response = requests.post(base_url + "/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _flush(base_url: str):
    response = requests.post(base_url + "/flush_cache", timeout=60)
    response.raise_for_status()


def test_public_llada_default_and_over_capacity_mixed_waves_recovery_quality():
    assert torch.cuda.is_available()
    model = os.environ["SGLANG_TEST_LLADA_MODEL"]
    base_prompt = [100 + i for i in range(32)]

    with _server(
        model=model,
        dllm=True,
        max_running=None,
        graph_bs=None,
        mem_fraction=0.82,
    ) as base_url:
        result = _generate(
            base_url,
            rid="default-capacity",
            input_ids=base_prompt,
            max_new_tokens=32,
        )
        assert result["meta_info"]["completion_tokens"] > 0

    with _server(
        model=model,
        dllm=True,
        max_running=4,
        graph_bs=None,
        mem_fraction=0.82,
    ) as base_url:
        lengths = [32, 96, 64, 32, 96, 64, 32, 96]

        def one(index):
            return _generate(
                base_url,
                rid=f"wave-a-{index}",
                input_ids=base_prompt[:-1] + [500 + index],
                max_new_tokens=lengths[index],
            )

        with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
            first_wave = [
                future.result() for future in as_completed([pool.submit(one, i) for i in range(len(lengths))])
            ]
        assert len(first_wave) == len(lengths)
        assert all(item["meta_info"]["completion_tokens"] > 0 for item in first_wave)

        with ThreadPoolExecutor(max_workers=4) as pool:
            followup = list(
                pool.map(
                    lambda i: _generate(
                        base_url,
                        rid=f"wave-b-{i}",
                        input_ids=base_prompt[:-1] + [700 + i],
                        max_new_tokens=32,
                    ),
                    range(4),
                )
            )
        assert len(followup) == 4
        assert all(item["meta_info"]["completion_tokens"] > 0 for item in followup)

        from sglang.test.few_shot_gsm8k import run_eval

        quality = run_eval(
            SimpleNamespace(
                host="http://127.0.0.1",
                port=int(base_url.rsplit(":", 1)[1]),
                data_path=os.environ["SGLANG_TEST_GSM8K"],
                num_questions=200,
                num_shots=5,
                max_new_tokens=512,
                parallel=128,
                temperature=0.0,
            )
        )
        assert quality["accuracy"] > 0.88


def test_public_llada_complete_block_radix_reuse():
    assert torch.cuda.is_available()
    model = os.environ["SGLANG_TEST_LLADA_MODEL"]
    common = [1000 + i for i in range(64)]

    with _server(
        model=model,
        dllm=True,
        max_running=1,
        graph_bs=None,
        mem_fraction=0.82,
        kv_cache_dtype="fp8_e4m3",
    ) as base_url:
        cached = {}
        cold = {}
        uncached_work = {}
        for boundary in (31, 32, 33, 64):
            probe = common[:boundary] + [4000 + boundary] + [5000 + i for i in range(64 - boundary)]

            _flush(base_url)
            cold_result = _generate(
                base_url,
                rid=f"cold-{boundary}",
                input_ids=probe,
                max_new_tokens=32,
            )
            cold[boundary] = int(cold_result["meta_info"].get("cached_tokens", 0))

            _flush(base_url)
            _generate(
                base_url,
                rid=f"warm-{boundary}",
                input_ids=common + [3000],
                max_new_tokens=32,
            )
            warm_result = _generate(
                base_url,
                rid=f"probe-{boundary}",
                input_ids=probe,
                max_new_tokens=32,
            )
            meta = warm_result["meta_info"]
            cached[boundary] = int(meta.get("cached_tokens", 0))
            uncached_work[boundary] = int(meta["prompt_tokens"]) - cached[boundary]
            assert meta["completion_tokens"] > 0

    assert cached[31] == cold[31]
    assert cached[32] > cold[32]
    assert cached[33] == cached[32]
    assert cached[64] > cached[33]
    assert uncached_work[64] < uncached_work[32]


def _batch_generate(base_url: str, size: int, *, prefix: str):
    prompts = [[100 + i for i in range(32)] + [900 + row] for row in range(size)]
    response = _generate(
        base_url,
        rid=[f"{prefix}-{row}" for row in range(size)],
        input_ids=prompts,
        max_new_tokens=32,
    )
    assert isinstance(response, list) and len(response) == size
    return [item["output_ids"] for item in response]


def _profile_event_names(profile_dir: Path):
    names = []
    files = list(profile_dir.glob("*.trace.json.gz"))
    assert files, f"no profiler trace under {profile_dir}"
    for path in files:
        with gzip.GzipFile(filename=path, mode="rb") as compressed:
            with io.TextIOWrapper(compressed) as text:
                payload = json.load(text)
        events = payload.get("traceEvents", payload if isinstance(payload, list) else [])
        names.extend(str(event.get("name", "")) for event in events)
    return names


def _cuda_launch_api_count(names):
    return sum(
        name.casefold().replace(" ", "").startswith("cudalaunchkernel")
        for name in names
    )


def test_public_llada_eager_graph_parity_bs_1_2_4():
    assert torch.cuda.is_available()
    model = os.environ["SGLANG_TEST_LLADA_MODEL"]
    eager = {}
    eager_profile_dir = (
        Path(os.environ["SGLANG_TEST_OUTPUT"]) / f"eager-profile-{time.time_ns()}"
    )
    with _server(
        model=model,
        dllm=True,
        max_running=4,
        graph_bs=None,
        mem_fraction=0.82,
    ) as base_url:
        for size in (1, 2, 4):
            _batch_generate(base_url, size, prefix=f"eager-warm-{size}")
        start = requests.post(
            base_url + "/start_profile",
            json={
                "output_dir": str(eager_profile_dir),
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
            },
            timeout=60,
        )
        start.raise_for_status()
        for size in (1, 2, 4):
            eager[size] = _batch_generate(base_url, size, prefix=f"eager-{size}")
        stop = requests.post(base_url + "/stop_profile", timeout=300)
        stop.raise_for_status()

    profile_dir = Path(os.environ["SGLANG_TEST_OUTPUT"]) / f"graph-profile-{time.time_ns()}"
    with _server(
        model=model,
        dllm=True,
        max_running=4,
        graph_bs=(1, 2, 4),
        mem_fraction=0.82,
    ) as base_url:
        for size in (1, 2, 4):
            _batch_generate(base_url, size, prefix=f"graph-warm-{size}")
        start = requests.post(
            base_url + "/start_profile",
            json={
                "output_dir": str(profile_dir),
                "activities": ["CPU", "GPU"],
                "with_stack": False,
                "record_shapes": False,
            },
            timeout=60,
        )
        start.raise_for_status()
        for size in (1, 2, 4):
            assert _batch_generate(base_url, size, prefix=f"graph-{size}") == eager[size]
        stop = requests.post(base_url + "/stop_profile", timeout=300)
        stop.raise_for_status()

    eager_names = _profile_event_names(eager_profile_dir)
    graph_names = _profile_event_names(profile_dir)
    eager_lowered = [name.casefold().replace(" ", "") for name in eager_names]
    graph_lowered = [name.casefold().replace(" ", "") for name in graph_names]

    assert not any(
        "cudagraphlaunch" in name or "cudagraph::replay" in name
        for name in eager_lowered
    )
    assert any(
        "cudagraphlaunch" in name or "cudagraph::replay" in name
        for name in graph_lowered
    )

    eager_launches = _cuda_launch_api_count(eager_names)
    graph_launches = _cuda_launch_api_count(graph_names)
    assert eager_launches > 0
    assert graph_launches * 2 < eager_launches, (
        "graph-configured diffusion inference did not replace eager model kernel launches: "
        f"eager={eager_launches}, graph={graph_launches}"
    )


def test_legacy_low_confidence_single_block(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    local_mask = 7
    ids = torch.tensor([9, local_mask, local_mask, local_mask])
    config = _real_config(monkeypatch, tmp_path, block_size=BLOCK_SIZE)
    config.mask_id = local_mask
    runner = _StaticRunner(_position_logits(ids.numel()))
    _, generated, _ = get_algorithm(config).run(runner, _forward_batch(ids, 1))
    assert local_mask not in ids.tolist()
    assert _generated_count(generated) == 3


def test_low_confidence_mask_free_fast_path(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    ids = torch.tensor([9, 1, 2, 3])
    config = _real_config(monkeypatch, tmp_path)
    config.mask_id = 7
    runner = _StaticRunner(_position_logits(ids.numel()))
    _, generated, _ = get_algorithm(config).run(runner, _forward_batch(ids, 1))
    assert ids.tolist() == [9, 1, 2, 3]
    assert _generated_count(generated) == 0
    assert runner.calls == 1


def test_disabled_diffusion_configuration_remains_none():
    from sglang.srt.dllm.config import DllmConfig

    assert DllmConfig.from_server_args(SimpleNamespace(dllm_algorithm=None)) is None


def test_llada_defaults_and_low_confidence_identifier(monkeypatch, tmp_path):
    from sglang.srt.dllm.algorithm import get_algorithm

    config = _real_config(monkeypatch, tmp_path)
    assert config.algorithm == "LowConfidence"
    assert config.block_size in {BLOCK_SIZE, 32}
    assert config.mask_id == MASK_ID
    algorithm = get_algorithm(config)
    assert algorithm.__class__.__name__ == "LowConfidence"


def test_pinned_llada_public_single_generation():
    assert torch.cuda.is_available()
    with _server(
        model=os.environ["SGLANG_TEST_LLADA_MODEL"],
        dllm=True,
        max_running=1,
        graph_bs=None,
        mem_fraction=0.82,
    ) as base_url:
        result = _generate(
            base_url,
            rid="llada-p2p",
            input_ids=[100 + i for i in range(32)],
            max_new_tokens=32,
        )
        assert result["output_ids"]
        assert result["meta_info"]["completion_tokens"] > 0


def test_ordinary_ar_public_batch_generation():
    assert torch.cuda.is_available()
    with _server(
        model=os.environ["SGLANG_TEST_AR_MODEL"],
        dllm=False,
        max_running=4,
        graph_bs=None,
        mem_fraction=0.55,
    ) as base_url:
        result = _generate(
            base_url,
            rid=["ar-a", "ar-b"],
            text=["The capital of France is", "Two plus two equals"],
            max_new_tokens=8,
        )
        assert isinstance(result, list) and len(result) == 2
        assert all(item["meta_info"]["completion_tokens"] == 8 for item in result)


def test_ordinary_ar_public_prefix_cache_followup():
    assert torch.cuda.is_available()
    prefix = [100 + i for i in range(64)]
    with _server(
        model=os.environ["SGLANG_TEST_AR_MODEL"],
        dllm=False,
        max_running=2,
        graph_bs=None,
        mem_fraction=0.55,
    ) as base_url:
        _flush(base_url)
        _generate(
            base_url,
            rid="ar-warm",
            input_ids=prefix + [1000],
            max_new_tokens=8,
        )
        result = _generate(
            base_url,
            rid="ar-followup",
            input_ids=prefix + [1001],
            max_new_tokens=8,
        )
        assert result["meta_info"].get("cached_tokens", 0) > 0
        assert result["meta_info"]["completion_tokens"] == 8
