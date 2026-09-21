# Copyright 2023-2024 SGLang Team
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""topk > 1 tree drafting (EAGLE3 topk16 + EAGLE/Llama-2 topk8).

topk > 1 routes to spec v1, except page_size==1 which can also stay on spec v2
(overlap). flashinfer is pinned because this runs on the cheap (5090) runner,
where fa3 (Hopper-only) isn't available -- functional sanity only, no perf/stress.
(topk > 1 on fa3 is covered on the Hopper runner in test_spec_eagle_fa3.py.)
"""

import os
import unittest

from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base


class TestEagle3Topk16(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit, SpecLogprobKit):
    """EAGLE3 topk=16 tree (spec v1): correctness + gsm8k + logprob losslessness."""

    spec_topk = 16
    spec_tokens = 64
    disable_overlap = True  # topk>1 -> spec v1
    cuda_graph_max_bs = 5
    acc_length_thres = 3.1
    batch_accept_len_thres = 1.75
    gsm8k_accept_len_thres = 2.4  # EAGLE3 topk16 gsm8k accept ~2.48


class TestEagle3Topk16SpecV2(TestEagle3Topk16, SpecFeatureKit):
    """EAGLE3 topk=16 tree on spec v2 (overlap, page1): guards the v2 tree path's
    accepted-path compaction, validated by logprob_spec_v2_match."""

    model = os.environ["EAGLE_V2_TARGET_MODEL"]
    draft_model = os.environ["EAGLE_V2_DRAFT_MODEL"]
    disable_overlap = False


class TestEagleLlama2Suite(
    EagleLlama2Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=8 full coverage (kits listed in bases)."""


class TestEagleLlama2Chunked4(EagleLlama2Base, SpecCorrectnessKit):
    """Correctness under tiny chunked prefill."""

    chunked_prefill_size = 4


class TestEagleLlama3TokenMap(EagleLlama2Base, SpecAccuracyKit):
    """EAGLE on Llama-3-8B with a FR-Spec token map (topk=4)."""

    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model = "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B"
    spec_topk = 4
    spec_tokens = 8
    cuda_graph_max_bs = 5
    gsm8k_accept_len_thres = 2.5  # FR-Spec token map lowers accept (~2.57)
    extra_args = (
        "--speculative-token-map",
        "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
    )


if __name__ == "__main__":
    unittest.main()
