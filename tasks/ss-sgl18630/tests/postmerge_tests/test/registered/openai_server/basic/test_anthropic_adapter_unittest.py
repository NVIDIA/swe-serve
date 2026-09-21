# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Anthropic <-> OpenAI request/response conversion layer.

These tests exercise the pure conversion methods on ``AnthropicServing``
(``_convert_to_chat_completion_request`` and ``_convert_response``). Both methods
operate purely on the Anthropic/OpenAI Pydantic protocol models and do not touch
the underlying ``OpenAIServingChat`` instance, so the handler is constructed with
a ``None`` delegate. No model weights, GPU, or running server are required.
"""

import pytest

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
    AnthropicToolChoice,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FunctionResponse,
    ToolCall,
    ToolChoice,
    UsageInfo,
)


@pytest.fixture
def serving():
    # The conversion methods never dereference the delegate, so None is safe.
    return AnthropicServing(openai_serving_chat=None)


# --------------------------------------------------------------------------- #
# Request conversion: Anthropic Messages request -> OpenAI ChatCompletion      #
# --------------------------------------------------------------------------- #


def test_convert_request_simple_text(serving):
    req = AnthropicMessagesRequest(
        model="test-model",
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="Hello there")],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert chat.model == "test-model"
    assert chat.max_tokens == 64
    assert chat.stream is False
    assert len(chat.messages) == 1
    assert chat.messages[0].role == "user"
    assert chat.messages[0].content == "Hello there"


def test_convert_request_string_system_message(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        system="You are helpful.",
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert chat.messages[0].role == "system"
    assert chat.messages[0].content == "You are helpful."
    assert chat.messages[1].role == "user"
    assert chat.messages[1].content == "hi"


def test_convert_request_block_system_message(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        system=[
            AnthropicContentBlock(type="text", text="line1"),
            AnthropicContentBlock(type="text", text="line2"),
        ],
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert chat.messages[0].role == "system"
    assert chat.messages[0].content == "line1\nline2"


def test_convert_request_image_block_to_image_url(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicContentBlock(type="text", text="describe"),
                    AnthropicContentBlock(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "QUJD",
                        },
                    ),
                ],
            )
        ],
    )
    chat = serving._convert_to_chat_completion_request(req)

    content = chat.messages[0].content
    parts = [
        c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else c
        for c in content
    ]
    text_parts = [c for c in parts if c["type"] == "text"]
    assert text_parts and text_parts[0]["text"] == "describe"
    image_parts = [c for c in parts if c["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_convert_request_tool_use_block_to_tool_calls(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(
                        type="tool_use",
                        id="toolu_1",
                        name="get_weather",
                        input={"city": "Paris"},
                    )
                ],
            )
        ],
    )
    chat = serving._convert_to_chat_completion_request(req)

    msg = chat.messages[0]
    assert msg.role == "assistant"
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "toolu_1"
    assert tc.type == "function"
    assert tc.function.name == "get_weather"
    assert tc.function.arguments == '{"city": "Paris"}'


def test_convert_request_tool_result_becomes_tool_message(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicContentBlock(
                        type="tool_result",
                        tool_use_id="toolu_1",
                        content="sunny",
                    )
                ],
            )
        ],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert len(chat.messages) == 1
    msg = chat.messages[0]
    assert msg.role == "tool"
    assert msg.tool_call_id == "toolu_1"
    assert msg.content == "sunny"


def test_convert_request_sampling_params(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        stop_sequences=["STOP", "END"],
        messages=[AnthropicMessage(role="user", content="hi")],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert chat.temperature == 0.5
    assert chat.top_p == 0.9
    assert chat.top_k == 40
    assert chat.stop == ["STOP", "END"]


def test_convert_request_tools_and_default_tool_choice(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[AnthropicMessage(role="user", content="hi")],
        tools=[
            AnthropicTool(
                name="get_weather",
                description="Get the weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ],
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert chat.tools is not None
    assert len(chat.tools) == 1
    tool = chat.tools[0]
    assert tool.function.name == "get_weather"
    assert tool.function.description == "Get the weather"
    assert tool.function.parameters["properties"]["city"]["type"] == "string"
    # Tools present with no explicit choice -> defaults to "auto".
    assert chat.tool_choice == "auto"


def test_convert_request_tool_choice_any_maps_to_required(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[AnthropicMessage(role="user", content="hi")],
        tools=[
            AnthropicTool(name="t", input_schema={"type": "object"}),
        ],
        tool_choice=AnthropicToolChoice(type="any"),
    )
    chat = serving._convert_to_chat_completion_request(req)
    assert chat.tool_choice == "required"


def test_convert_request_tool_choice_specific_tool(serving):
    req = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[AnthropicMessage(role="user", content="hi")],
        tools=[
            AnthropicTool(name="get_weather", input_schema={"type": "object"}),
        ],
        tool_choice=AnthropicToolChoice(type="tool", name="get_weather"),
    )
    chat = serving._convert_to_chat_completion_request(req)

    assert isinstance(chat.tool_choice, ToolChoice)
    assert chat.tool_choice.type == "function"
    assert chat.tool_choice.function.name == "get_weather"


# --------------------------------------------------------------------------- #
# Response conversion: OpenAI ChatCompletion response -> Anthropic response     #
# --------------------------------------------------------------------------- #


def test_convert_response_text(serving):
    resp = ChatCompletionResponse(
        id="cmpl-1",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="The capital is Paris."),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=11, completion_tokens=5, total_tokens=16),
    )
    out = serving._convert_response(resp)

    assert out.model == "test-model"
    assert out.role == "assistant"
    assert out.type == "message"
    assert out.stop_reason == "end_turn"
    assert len(out.content) == 1
    assert out.content[0].type == "text"
    assert out.content[0].text == "The capital is Paris."
    assert out.usage.input_tokens == 11
    assert out.usage.output_tokens == 5


def test_convert_response_length_maps_to_max_tokens(serving):
    resp = ChatCompletionResponse(
        id="cmpl-2",
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="truncated"),
                finish_reason="length",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    out = serving._convert_response(resp)
    assert out.stop_reason == "max_tokens"


def test_convert_response_tool_calls_to_tool_use(serving):
    resp = ChatCompletionResponse(
        id="cmpl-3",
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionResponse(
                                name="get_weather",
                                arguments='{"city": "Paris"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=UsageInfo(prompt_tokens=7, completion_tokens=4, total_tokens=11),
    )
    out = serving._convert_response(resp)

    assert out.stop_reason == "tool_use"
    tool_blocks = [b for b in out.content if b.type == "tool_use"]
    assert len(tool_blocks) == 1
    block = tool_blocks[0]
    assert block.id == "call_1"
    assert block.name == "get_weather"
    assert block.input == {"city": "Paris"}
