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
"""Non-overlapping task guards beside the scored SGLang maintainer suites."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser

try:
    from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
except ImportError as production_import_error:
    _MISSING_DETECTOR_ERROR = production_import_error

    class DeepSeekV4Detector:  # type: ignore[no-redef]
        def __getattr__(self, name):
            raise ImportError(
                "sglang.srt.function_call.deepseekv4_detector is absent"
            ) from _MISSING_DETECTOR_ERROR


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_favorite_tourist_spot",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )
    ]


def test_deepseek_v4_plain_text_has_no_calls():
    result = DeepSeekV4Detector().detect_and_parse(
        "Here is a plain answer with no tool calls whatsoever.", _tools()
    )
    assert result.calls == []
    assert "plain answer" in result.normal_text


def test_deepseek_v32_rejects_v4_wrapper():
    text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_favorite_tourist_spot">
<｜DSML｜parameter name="city" string="true">London</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
    result = DeepSeekV32Detector().detect_and_parse(text, _tools())
    assert result.calls == []
    assert "tool_calls" in result.normal_text


def test_deepseek_v32_registry_dispatch():
    parser = FunctionCallParser(_tools(), "deepseekv32")
    text = """Sure - let me look that up.
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_favorite_tourist_spot">
<｜DSML｜parameter name="city" string="true">London</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""
    normal_text, calls = parser.parse_non_stream(text)
    assert len(calls) == 1
    assert calls[0].name == "get_favorite_tourist_spot"
    assert json.loads(calls[0].parameters)["city"] == "London"
    assert "let me look that up" in normal_text
