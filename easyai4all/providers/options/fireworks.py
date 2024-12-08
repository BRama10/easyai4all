from typing import Any, Dict, List, Optional
import os
import time
from easyai4all.providers.base_provider import Provider
from easyai4all.chat_completion import ChatCompletionResponse


class FireworksAI(Provider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Fireworks AI provider with API key and base URL."""
        super().__init__(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base="https://api.fireworks.ai/inference/v1",
            env_api_key="FIREWORKS_API_KEY",
        )
        if not self.api_key:
            raise ValueError(
                "API key must be provided either directly or through the FIREWORKS_API_KEY environment variable."
            )

    @property
    def headers(self) -> Dict[str, str]:
        """Construct and return the request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare the API request payload."""
        return {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "max_tokens": max_tokens,
            **kwargs,
        }

    def _process_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        """Process Fireworks AI response into standardized format."""
        # Extract choices
        choices = []
        for choice in response["choices"]:
            # Process tool calls if available
            tool_calls = []
            if "tool_calls" in choice["message"]:
                tool_calls = [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        },
                    }
                    for tool_call in choice["message"]["tool_calls"]
                ]

            # Prepare the message
            message = {
                "role": choice["message"]["role"],
                "content": choice["message"]["content"],
                "tool_calls": tool_calls if tool_calls else None,
            }

            # Append the choice
            choices.append(
                {
                    "index": choice["index"],
                    "message": message,
                    "finish_reason": choice["finish_reason"],
                }
            )

        # Extract usage details
        usage = response.get("usage", {})
        completion_tokens_details = {
            "accepted_prediction_tokens": 0,
            "reasoning_tokens": usage.get("completion_tokens", 0),
            "rejected_prediction_tokens": 0,
        }
        prompt_tokens_details = {"cached_tokens": 0}

        # Return standardized response
        return ChatCompletionResponse.from_dict(
            {
                "id": response["id"],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response["model"],
                "choices": choices,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "completion_tokens_details": completion_tokens_details,
                    "prompt_tokens_details": prompt_tokens_details,
                },
                "system_fingerprint": f"fireworks-{response['id']}",
            }
        )

    def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> ChatCompletionResponse:
        """Execute a chat completion request."""
        endpoint = f"{self.api_base}/chat/completions"
        payload = self._prepare_request(model, messages, **kwargs)
        response = self.post(endpoint, json=payload, headers=self.headers)
        return self._process_response(response)
