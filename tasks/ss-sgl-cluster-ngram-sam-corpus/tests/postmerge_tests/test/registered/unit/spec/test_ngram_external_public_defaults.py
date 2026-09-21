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
"""Supplemental public-default and backward-compatibility coverage for external SAM."""

from unittest.mock import patch

from sglang.srt.server_args import prepare_server_args
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.test.test_utils import CustomTestCase


class TestNgramExternalPublicDefaults(CustomTestCase):
    def test_public_defaults_and_optional_budget_are_backward_compatible(self):
        with patch("sglang.srt.server_args.get_device", return_value="cuda"):
            server_args = prepare_server_args(
                [
                    "--model-path",
                    "dummy",
                    "--speculative-algorithm",
                    "NGRAM",
                ]
            )
        self.assertIsNone(server_args.speculative_ngram_external_corpus_path)
        self.assertEqual(server_args.speculative_ngram_external_sam_budget, 0)
        self.assertEqual(
            server_args.speculative_ngram_external_corpus_max_tokens, 10_000_000
        )

        # Preserve the pre-feature calling pattern: external_sam_budget is optional.
        corpus = NgramCorpus(
            max_trie_depth=12,
            min_bfs_breadth=1,
            max_bfs_breadth=8,
            draft_token_num=4,
            match_type="BFS",
            capacity=100_000,
            external_corpus_max_tokens=32,
        )
        self.assertEqual(corpus.list_external_corpora(), {})
        self.assertEqual(corpus.remaining_token_budget, 32)


if __name__ == "__main__":
    import unittest

    unittest.main()
