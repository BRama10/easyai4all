from typing import Optional, List, Dict, Any

from easyai4all.providers.base_provider import Provider
from easyai4all.chat_completion import ChatCompletionResponse


class OpenAI(Provider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(
            api_key=api_key,
            api_base="https://api.openai.com/v1/chat/completions",
            env_api_key="OPENAI_API_KEY",
        )

    def _prepare_request(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        return {"model": model, "messages": messages, **kwargs}

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return ChatCompletionResponse.from_dict(response)
