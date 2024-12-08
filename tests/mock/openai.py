# test_chat_completion_mock.py
import pytest
from unittest.mock import Mock, patch
from easyai4all.client import Client
from easyai4all.chat_completion import ChatCompletionResponse


@pytest.fixture
def weather_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.fixture
def mock_successful_response():
    return {
        "id": "mock-123",
        "object": "chat.completion",
        "created": 1699000000,
        "model": "openai/gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris", "unit": "c"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


class TestChatCompletionMock:
    def test_successful_api_call(self, weather_tools, mock_successful_response):
        """Test a successful API call with mocked response"""
        with patch("easyai4all.client.Client.create") as mock_create:
            # Setup mock
            mock_create.return_value = ChatCompletionResponse(
                **mock_successful_response
            )

            # Make the call
            client = Client()
            completion = client.create(
                model="openai/gpt-4",
                messages=[
                    {"role": "user", "content": "What's the weather like in Paris?"}
                ],
                tools=weather_tools,
            )

            # Verify the response
            assert isinstance(completion, ChatCompletionResponse)
            assert completion.id == "mock-123"
            assert len(completion.choices) == 1
            assert (
                completion.choices[0].message.tool_calls[0].function.name
                == "get_weather"
            )

    def test_api_error(self, weather_tools):
        """Test API error handling"""
        with patch("easyai4all.client.Client.create") as mock_create:
            # Setup mock to raise an exception
            mock_create.side_effect = Exception("API Error")

            # Verify error handling
            with pytest.raises(Exception) as exc_info:
                client = Client()
                client.create(
                    model="openai/gpt-4",
                    messages=[{"role": "user", "content": "What's the weather?"}],
                    tools=weather_tools,
                )
            assert str(exc_info.value) == "API Error"

    def test_invalid_model(self, weather_tools):
        """Test handling of invalid model specification"""
        with patch("easyai4all.client.Client.create") as mock_create:
            mock_create.side_effect = ValueError("Invalid model")

            with pytest.raises(ValueError):
                client = Client()
                client.create(
                    model="invalid-model",
                    messages=[{"role": "user", "content": "What's the weather?"}],
                    tools=weather_tools,
                )

    def test_empty_response(self, weather_tools):
        """Test handling of empty response"""
        empty_response = {
            "id": "mock-124",
            "object": "chat.completion",
            "created": 1699000000,
            "model": "openai/gpt-4",
            "choices": [],
        }

        with patch("easyai4all.client.Client.create") as mock_create:
            mock_create.return_value = ChatCompletionResponse(**empty_response)

            client = Client()
            completion = client.create(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=weather_tools,
            )

            assert len(completion.choices) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
