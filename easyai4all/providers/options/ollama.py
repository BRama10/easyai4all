import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from easyai4all.providers.base_provider import Provider
from easyai4all.chat_completion import ChatCompletionResponse


class Ollama(Provider):
    def __init__(self, api_base: Optional[str] = None) -> None:
        """
        Initialize the Ollama provider.

        Args:
            api_base: The base URL for the Ollama API.
        """
        # Set the system-wide environment variable for the API key to None
        os.environ["OLLAMA_API_KEY"] = "None"
        super().__init__(
            api_key=os.getenv("OLLAMA_API_KEY"),
            api_base=api_base or "http://localhost:11434/api/chat",
            env_api_key="OLLAMA_API_KEY",
        )

    @property
    def headers(self) -> Dict[str, str]:
        """Headers for Ollama API."""
        return {"Content-Type": "application/json"}

    def _prepare_request(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare the request payload for the Ollama API.

        Args:
            model: The model name to use.
            messages: List of chat messages.
            **kwargs: Additional parameters for Ollama.

        Returns:
            A dictionary representing the request payload.
        """
        return {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
            **kwargs,
        }

    def _process_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        """
        Process the response from the Ollama API into a standardized format.

        Args:
            response: The raw JSON response from Ollama.

        Returns:
            A standardized `ChatCompletionResponse` object.
        """

        def parse_timestamp(timestamp: str) -> int:
            """Convert timestamp to a Unix time integer."""
            if "." in timestamp:
                base, fraction = timestamp.split(".")
                fraction = fraction[:6]
                timestamp = f"{base}.{fraction}Z"
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            return int(dt.timestamp())

        # Extract tool calls if present
        tool_calls = response["message"].get("tool_calls", [])
        processed_tool_calls = [
            {
                "id": tool_call["function"]["name"],  # Use function name as id
                "type": "function",
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                },
            }
            for tool_call in tool_calls
        ]

        # Build the message
        message = {
            "role": response["message"]["role"],
            "content": response["message"].get("content", ""),
        }
        if processed_tool_calls:
            message["tool_calls"] = processed_tool_calls

        # Map the response into a standardized format
        return ChatCompletionResponse.from_dict(
            {
                "id": response.get("id", ""),
                "object": "chat.completion",
                "created": parse_timestamp(response["created_at"]),
                "model": response["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": response.get("done_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0)
                    + response.get("eval_count", 0),
                },
                # Generate a fingerprint for debugging and traceability
                "system_fingerprint": f"ollama-{response['model']}-{response.get('id', '')}",
            }
        )
