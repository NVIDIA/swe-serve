# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public worker-factory contracts for unified EAGLE V2 dispatch."""

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _worker(algorithm: SpeculativeAlgorithm, *, disable_overlap: bool):
    args = ServerArgs(
        model_path="dummy",
        speculative_algorithm=algorithm.name,
        disable_overlap_schedule=disable_overlap,
    )
    return algorithm.create_worker(args)


def test_eagle_family_uses_one_worker_across_overlap_modes() -> None:
    for algorithm in (
        SpeculativeAlgorithm.EAGLE,
        SpeculativeAlgorithm.EAGLE3,
        SpeculativeAlgorithm.STANDALONE,
    ):
        overlap_worker = _worker(algorithm, disable_overlap=False)
        synchronous_worker = _worker(algorithm, disable_overlap=True)
        assert overlap_worker is synchronous_worker


def test_ngram_worker_remains_distinct() -> None:
    ngram_worker = _worker(SpeculativeAlgorithm.NGRAM, disable_overlap=True)
    eagle_worker = _worker(SpeculativeAlgorithm.EAGLE3, disable_overlap=True)
    assert ngram_worker is not eagle_worker
    assert SpeculativeAlgorithm.from_string("ngram") is SpeculativeAlgorithm.NGRAM
    assert SpeculativeAlgorithm.from_string("eagle3") is SpeculativeAlgorithm.EAGLE3

