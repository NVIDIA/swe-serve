# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GenerationBatchResult.copy_to_cpu hidden-states D2H guard.

copy_to_cpu must be able to skip the device->host copy of hidden_states when the
caller does not request them, so that overlap scheduling does not pay an unnecessary
GPU->CPU transfer for unused hidden states.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_result():
    """Build a GenerationBatchResult that exercises only the hidden_states path.

    GenerationBatchResult is a plain dataclass with all-default fields, so it is
    constructible standalone (no model / no GPU). copy_to_cpu unconditionally
    touches next_token_ids and copy_done, so both are mocked; logits_output is a
    mock whose hidden_states is a tracked tensor-like object.
    """
    logits_output = MagicMock()
    # Skip the return_logprob branch entirely: these are not read when
    # return_logprob=False, so leaving them as auto-mocks is harmless.
    hidden_states = MagicMock(name="hidden_states")
    logits_output.hidden_states = hidden_states

    result = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=MagicMock(name="next_token_ids"),
        copy_done=MagicMock(name="copy_done"),
    )
    return result, hidden_states


class TestCopyToCpuHiddenStates(CustomTestCase):
    def test_skips_hidden_states_d2h_when_not_requested(self):
        """return_hidden_states=False must NOT copy hidden_states to CPU."""
        result, hidden_states = _make_result()
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
        hidden_states.to.assert_not_called()

    def test_copies_hidden_states_d2h_when_requested(self):
        """return_hidden_states=True must copy hidden_states to CPU."""
        result, hidden_states = _make_result()
        result.copy_to_cpu(return_logprob=False, return_hidden_states=True)
        hidden_states.to.assert_called_once_with("cpu", non_blocking=True)


if __name__ == "__main__":
    unittest.main(verbosity=3)
