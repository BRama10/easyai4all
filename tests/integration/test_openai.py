# test_chat_completion_integration.py
import pytest
import os
import time
import json
from easyai4all.client import Client
from easyai4all.chat_completion import ChatCompletionResponse

# Skip all tests in this file if no API key is present
pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="Integration tests require OPENAI_API_KEY environment variable",
)


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
def client():
    return Client()


class TestChatCompletionIntegration:
    def test_real_api_call(self, client: Client, weather_tools):
        """Test actual API call and response"""
        completion = client.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
            tools=weather_tools,
        )

        # Verify real response
        assert isinstance(completion, ChatCompletionResponse)
        assert completion.id is not None  # Real API should provide an ID
        assert len(completion.choices) > 0

        # Verify tool call structure
        tool_calls = completion.choices[0].message.tool_calls
        assert len(tool_calls) > 0

        # Verify function call
        function_call = tool_calls[0].function
        assert function_call.name == "get_weather"

        # Verify arguments
        args = json.loads(function_call.arguments)
        assert "location" in args
        assert "unit" in args
        assert args["location"].lower() == "paris"
        assert args["unit"] in ["c", "f"]

    def test_api_performance(self, client, weather_tools):
        """Test API response time"""
        start_time = time.time()

        client.create(
            model="openai/gpt-4",
            messages=[
                {"role": "user", "content": "What's the weather like in London?"}
            ],
            tools=weather_tools,
        )

        end_time = time.time()
        response_time = end_time - start_time

        # API should respond within reasonable time
        assert (
            response_time < 10
        ), f"API took too long to respond: {response_time} seconds"

    @pytest.mark.parametrize(
        "location", ["New York", "Tokyo", "Sydney", "London", "Paris"]
    )
    def test_multiple_locations(self, client: Client, weather_tools, location):
        """Test API with different locations"""
        completion = client.create(
            model="openai/gpt-4",
            messages=[
                {"role": "user", "content": f"What's the weather like in {location}?"}
            ],
            tools=weather_tools,
        )

        tool_calls = completion.choices[0].message.tool_calls
        args = json.loads(tool_calls[0].function.arguments)

        assert location.lower() in args["location"].lower()

    def test_invalid_model_error(self, client, weather_tools):
        """Test error handling with invalid model"""
        with pytest.raises(Exception):
            client.create(
                model="nonexistent-model",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=weather_tools,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
