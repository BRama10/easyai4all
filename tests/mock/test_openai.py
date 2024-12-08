# mock/openai.py
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
        "created": 1733646711,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "tool_calls": [
                        {
                            "id": "call_xqS71iZdmLoJ6MQ6EFmMEY36",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris","unit":"c"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
                "audio": None,
            }
        ],
        "usage": {
            "prompt_tokens": 55,
            "completion_tokens": 18,
            "total_tokens": 73,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
                "audio_tokens": 0,
            },
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        },
        "system_fingerprint": "fp_9d50cd990b",
        "service_tier": None,
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
                model="openai/gpt-4o",
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
                    model="openai/gpt-4o",
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
            "id": "chatcmpl-Ac6zfPoYlboym43uN0RPZSDlS1gR2",
            "object": "chat.completion",
            "created": 1733646711,
            "model": "gpt-4o-2024-08-06",
            "choices": [],
            "usage": {
                "prompt_tokens": 55,
                "completion_tokens": 18,
                "total_tokens": 73,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                    "audio_tokens": 0,
                },
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            },
            "system_fingerprint": "fp_9d50cd990b",
            "service_tier": None,
        }

        with patch("easyai4all.client.Client.create") as mock_create:
            mock_create.return_value = ChatCompletionResponse(**empty_response)

            client = Client()
            completion = client.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=weather_tools,
            )

            assert len(completion.choices) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
