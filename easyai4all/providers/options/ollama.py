import time
from typing import Any, Dict, List
from easyai4all.providers.base_provider import Provider
from easyai4all.chat_completion import ChatCompletionResponse

class Ollama(Provider):
    def __init__(self, api_base: str = "http://localhost:11434/api/chat") -> None:
        super().__init__(api_key=None, api_base=api_base)

    @property
    def headers(self) -> Dict[str, str]:
        """Headers for Ollama requests (no authorization needed)."""
        return {"Content-Type": "application/json"}

    def _prepare_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare the request payload for Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)
        return payload

    def _process_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        """Process Ollama API response into standardized format."""
        # Extract response content and tool calls
        message_content = response["message"]["content"]
        tool_calls = response["message"].get("tool_calls", [])

        # Process tool calls into standardized format
        standardized_tool_calls = []
        for call in tool_calls:
            tool_call = {
                "id": call.get("id", str(time.time())),  # Generate a unique ID if not provided
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                },
            }
            standardized_tool_calls.append(tool_call)

        # Build the standardized response
        fingerprint = f"ollama-{response['model']}-{response.get('id', 'unknown')}"
        return ChatCompletionResponse.from_dict(
            {
                "id": response.get("id", str(time.time())),
                "object": "chat.completion",
                "created": int(
                    time.mktime(time.strptime(response["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"))
                ),
                "model": response["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response["message"]["role"],
                            "content": message_content,
                            "tool_calls": standardized_tool_calls if tool_calls else None,
                        },
                        "finish_reason": "stop",  # Fixed as "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0)
                    + response.get("eval_count", 0),
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "reasoning_tokens": response.get("eval_count", 0),
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                    },
                },
                "system_fingerprint": fingerprint,
            }
        )
