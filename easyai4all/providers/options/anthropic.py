from typing import Any, Dict, List, Optional
import time

from easyai4all.providers.base_provider import Provider
from easyai4all.chat_completion import ChatCompletionResponse


class Anthropic(Provider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(
            api_key=api_key,
            api_base="https://api.anthropic.com/v1/messages",
            env_api_key="ANTHROPIC_API_KEY",
        )

    @property
    def headers(self) -> Dict[str, str]:

        base_headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }

        return base_headers

    def _prepare_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

    def _process_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        """Process Anthropic API response into standardized format.

        Args:
            response: Raw Anthropic API response

        Returns:
            Standardized chat completion response
        """
        # Extract the text content from the response
        content = ""
        tool_calls = []

        for item in response["content"]:
            if item["type"] == "text":
                content += item["text"]
            elif item["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": item["id"],
                        "type": "function",  # Standard type for tool calls
                        "function": {"name": item["name"], "arguments": item["input"]},
                    }
                )

        # Build the standardized message
        message = {
            "role": response["role"],
            "content": content,
        }

        # Add tool calls if present
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Create completion tokens details
        completion_tokens_details = {
            "accepted_prediction_tokens": 0,
            "reasoning_tokens": response["usage"]["output_tokens"],
            "rejected_prediction_tokens": 0,
        }

        # Create prompt tokens details
        prompt_tokens_details = {"cached_tokens": 0}

        return ChatCompletionResponse.from_dict(
            {
                "id": response["id"],
                "object": "chat.completion",
                "created": int(time.time()),  # Current Unix timestamp
                "model": response["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": self._map_stop_reason(response["stop_reason"]),
                    }
                ],
                "usage": {
                    "prompt_tokens": response["usage"]["input_tokens"],
                    "completion_tokens": response["usage"]["output_tokens"],
                    "total_tokens": response["usage"]["input_tokens"]
                    + response["usage"]["output_tokens"],
                    "completion_tokens_details": completion_tokens_details,
                    "prompt_tokens_details": prompt_tokens_details,
                },
                # Generated fingerprint
                "system_fingerprint": "anthropic-" + response["id"],
            }
        )

    def _map_stop_reason(self, stop_reason: str) -> str:
        """Map Anthropic stop reasons to standard finish reasons."""
        reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        return reason_map.get(stop_reason, "stop")
