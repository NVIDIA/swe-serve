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
"""Engine API tests for the /v1/score scoring pipeline.

Two model types, two scoring modes:

  TestCausalLMScoring        — CausalLM, single-item and batched multi-item
  TestSeqClsScoring          — SequenceClassification, single-item mode
  TestSeqClsMISScoring       — SequenceClassification, MIS delimiter mode

The Engine (Python API) is the right layer for correctness testing: it
exercises tokenization, forward pass, pooling, and score extraction without
the HTTP serialization overhead.  HTTP-layer tests live in test_score_api.py.
"""

import json
import os
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine

# Env overrides allow running against a locally cached checkpoint (offline CI).
_CAUSAL_LM_MODEL = os.environ.get("TEST_MODEL_NAME", "Qwen/Qwen3-0.6B")
_SEQCLS_MODEL = os.environ.get(
    "TEST_CLASSIFICATION_BASE_MODEL", "Qwen/Qwen3-0.6B"
)  # backbone; arch overridden to SeqCls below
# <|endoftext|> for Qwen3 tokenizer — used as MIS delimiter
_QWEN3_EOT_TOKEN_ID = 151643


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class TestCausalLMScoring(unittest.TestCase):
    """CausalLM scoring via Engine — correctness, batching, and edge cases.

    A single Engine instance is shared across all test methods (class-level
    setup) so model loading happens once, not once per test.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(model_path=_CAUSAL_LM_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hf_scores(self, query, items, label_token_ids, item_first=False):
        """Reference scores computed directly with HuggingFace (CPU inference)."""
        tokenizer = AutoTokenizer.from_pretrained(
            _CAUSAL_LM_MODEL, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            _CAUSAL_LM_MODEL, trust_remote_code=True
        )
        try:
            scores = []
            for item in items:
                text = f"{item}{query}" if item_first else f"{query}{item}"
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    last_logits = model(**inputs).logits[0, -1]
                target_probs = torch.softmax(last_logits[label_token_ids], dim=-1)
                scores.append([p.item() for p in target_probs])
            return scores
        finally:
            model.cpu()
            del model, tokenizer
            torch.cuda.empty_cache()

    def _assert_scores_close(self, hf, sgl, tol=0.01):
        self.assertEqual(len(hf), len(sgl))
        for hf_row, sgl_row in zip(hf, sgl):
            self.assertEqual(len(hf_row), len(sgl_row))
            for h, s in zip(hf_row, sgl_row):
                self.assertLessEqual(abs(h - s), tol, f"HF={h:.6f} SGLang={s:.6f}")
            self.assertAlmostEqual(sum(sgl_row), 1.0, places=6)

    # ------------------------------------------------------------------
    # Multi-item / batching
    # ------------------------------------------------------------------

    def test_score_batch_sizes(self):
        """Correct output count and shape for batch sizes 1, 2, 4, 8."""
        label_token_ids = [1, 2, 3]
        for n in [1, 2, 4, 8]:
            with self.subTest(n=n):
                scores = self.engine.score(
                    query="The test was",
                    items=[f"test {i}" for i in range(n)],
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), n)
                for row in scores:
                    self.assertEqual(len(row), len(label_token_ids))
                    self.assertTrue(all(isinstance(v, float) for v in row))
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_empty_items(self):
        """Empty items list → empty scores and zero prompt_tokens."""
        result = self.engine.score(
            query="Test query", items=[], label_token_ids=[1, 2], apply_softmax=True
        )
        self.assertEqual(len(result.scores), 0)
        self.assertEqual(result.prompt_tokens, 0)

    def test_score_without_softmax(self):
        """apply_softmax=False returns raw logits (not probability-constrained)."""
        scores = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad", "Neutral"],
            label_token_ids=[1, 2, 3],
            apply_softmax=False,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertEqual(len(row), 3)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_varying_label_token_sets(self):
        """Different label_token_ids lengths all produce correct-shaped output."""
        for n_labels in [1, 2, 4, 8]:
            with self.subTest(n_labels=n_labels):
                scores = self.engine.score(
                    query="Choose:",
                    items=["Option A", "Option B"],
                    label_token_ids=list(range(1, n_labels + 1)),
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), 2)
                for row in scores:
                    self.assertEqual(len(row), n_labels)
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_unicode(self):
        """Unicode query and items do not crash and produce valid scores."""
        scores = self.engine.score(
            query="选择最佳选项：",
            items=["选项A", "选项B", "选项C"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_deterministic(self):
        """Identical calls return numerically equivalent scores (within GPU float tolerance)."""
        kwargs = dict(query="Choose:", items=["A", "B", "C"], label_token_ids=[1, 2, 3])
        scores_a = self.engine.score(**kwargs).scores
        scores_b = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores_a), len(scores_b))
        for row_a, row_b in zip(scores_a, scores_b):
            self.assertEqual(len(row_a), len(row_b))
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=5)

    def test_score_error_handling(self):
        """Invalid argument types raise ValueError or TypeError."""
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query="Q", items=["X"], label_token_ids="bad", apply_softmax=True
            )
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query="Q", items=None, label_token_ids=[1, 2], apply_softmax=True
            )


# ---------------------------------------------------------------------------
# SequenceClassification — single-item mode
# ---------------------------------------------------------------------------


class TestSeqClsScoring(unittest.TestCase):
    """SequenceClassification scoring via Engine — no MIS delimiter.

    Uses json_model_override_args to load Qwen3-0.6B backbone weights into
    Qwen3ForSequenceClassification.  The classification head is randomly
    initialised; shape/pipeline correctness is what matters here.
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_score_shape(self):
        """Each item gets a score vector of length num_labels."""
        scores = self.engine.score(
            query="Rate each option:",
            items=["Option A", "Option B"],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 2)
        for i, row in enumerate(scores):
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_score_single_item_edge_case(self):
        """Single item in the list."""
        scores = self.engine.score(
            query="Evaluate:", items=["Only item"], apply_softmax=True
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_score_without_softmax(self):
        """Without softmax, returns raw logits (no probability constraints)."""
        scores = self.engine.score(
            query="Evaluate:", items=["Alpha", "Beta"], apply_softmax=False
        ).scores
        self.assertEqual(len(scores), 2)
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_deterministic(self):
        """Identical inputs yield near-identical scores (fp16 tolerance)."""
        kwargs = dict(query="Evaluate:", items=["alpha", "beta", "gamma"])
        scores1 = self.engine.score(**kwargs).scores
        scores2 = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores1), len(scores2))
        for s1, s2 in zip(scores1, scores2):
            for v1, v2 in zip(s1, s2):
                self.assertAlmostEqual(v1, v2, places=1)

    def test_score_tokenized_inputs(self):
        """Pre-tokenized query/items match text input scores."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        query, items = "Rate this:", ["Good", "Bad"]

        text_scores = self.engine.score(
            query=query, items=items, apply_softmax=True
        ).scores
        token_scores = self.engine.score(
            query=tok.encode(query),
            items=[tok.encode(i) for i in items],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(text_scores), len(token_scores))
        for ts, ks in zip(text_scores, token_scores):
            for t, k in zip(ts, ks):
                # Fresh H100 verifier processes can differ by a few 1e-4 for
                # this randomly initialized low-precision classification head.
                self.assertAlmostEqual(t, k, delta=5e-4)


# ---------------------------------------------------------------------------
# SequenceClassification — MIS (delimiter) mode
# ---------------------------------------------------------------------------


class TestSeqClsMISScoring(unittest.TestCase):
    """SeqCls MIS: all items packed into one sequence separated by delimiter token.

    score_and_pool() extracts per-item scores at delimiter positions.
    Two sub-cases are tested:
      - NUM_LABELS=2  — standard binary classification head
      - NUM_LABELS=12 — stress-tests 2-D tensor indexing in score_and_pool()
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=_QWEN3_EOT_TOKEN_ID,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_mis_one_vector_per_item(self):
        """MIS produces exactly one score vector per item."""
        items = ["Option A", "Option B", "Option C"]
        scores = self.engine.score(
            query="Rate each option:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for i, row in enumerate(scores):
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_mis_single_item_edge_case(self):
        """Single item through MIS path."""
        scores = self.engine.score(
            query="Evaluate:", items=["Single item"], apply_softmax=True
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_mis_many_items(self):
        """10 items all return valid probability vectors."""
        items = [f"Item {i}" for i in range(10)]
        scores = self.engine.score(
            query="Classify each:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_mis_tokenized_inputs(self):
        """Pre-tokenized MIS inputs agree with equivalent text inputs."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        query, items = "Rate each:", ["Good", "Bad", "Neutral"]

        text_scores = self.engine.score(
            query=query, items=items, apply_softmax=True
        ).scores
        token_scores = self.engine.score(
            query=tok.encode(query),
            items=[tok.encode(item) for item in items],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(text_scores), len(token_scores))
        for text_row, token_row in zip(text_scores, token_scores):
            self.assertEqual(len(text_row), len(token_row))
            for text_score, token_score in zip(text_row, token_row):
                # Fresh H100 verifier processes can differ by a few 1e-4 for
                # this randomly initialized low-precision classification head.
                self.assertAlmostEqual(text_score, token_score, delta=5e-4)

    def test_mis_items_produce_distinct_scores(self):
        """Different items must yield different score vectors.

        Catches bugs where all delimiter positions share the same pooled
        hidden state (e.g. off-by-one in score_and_pool indexing).
        """
        items = [
            "Option A is about cats",
            "Option B is about dogs",
            "Option C is about fish",
        ]
        scores = self.engine.score(query="Rate each option:", items=items).scores
        self.assertEqual(len(scores), len(items))
        self.assertFalse(
            all(scores[0] == s for s in scores[1:]),
            f"All items returned identical scores — delimiter indexing is likely broken. "
            f"Scores: {scores[0]}",
        )

    def test_mis_deterministic(self):
        """Identical MIS requests return identical scores."""
        kwargs = dict(query="Evaluate:", items=["alpha", "beta", "gamma"])
        self.assertEqual(
            self.engine.score(**kwargs).scores, self.engine.score(**kwargs).scores
        )


# ---------------------------------------------------------------------------
# SequenceClassification — MIS with many labels (tensor shape stress test)
# ---------------------------------------------------------------------------


class TestSeqClsMISAdvancedScoring(unittest.TestCase):
    """SeqCls MIS with 12 labels — stresses the 2-D tensor path in score_and_pool.

    Kept in a separate class (own Engine instance) so it doesn't fight the
    2-label class-level engine for GPU memory.
    """

    NUM_LABELS = 12

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=_QWEN3_EOT_TOKEN_ID,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_many_labels_correct_shape(self):
        """5 items × 12 labels — each score vector has the right length."""
        items = [f"Item {i}" for i in range(5)]
        scores = self.engine.score(
            query="Classify:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_many_items_produce_distinct_scores(self):
        """15 items should not all return identical score vectors."""
        items = [f"City {i}" for i in range(15)]
        scores = self.engine.score(query="Classify each city:", items=items).scores
        self.assertEqual(len(scores), len(items))
        self.assertGreater(
            len({tuple(s) for s in scores}),
            1,
            "All 15 items returned identical scores",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
