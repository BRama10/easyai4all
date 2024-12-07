from providers.base_provider import Provider
from providers.options.anthropic import Anthropic
from providers.options.gemini import Gemini
from providers.options.openai import OpenAI


from typing import Any, Dict, List, Optional


class Client:
    PROVIDER_MAPPING: dict[str, Provider] = {
        "openai": OpenAI,
        "gemini": Gemini,
        "anthropic": Anthropic,
    }

    def __init__(self, provider_configs: Optional[Dict[str, Dict]] = None):
        self.provider_configs = provider_configs or {}

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        try:
            provider, model_name = model.split("/", 1)
            provider_class = self.PROVIDER_MAPPING.get(provider)

            if not provider_class:
                raise ValueError(f"Unsupported provider: {provider}")

            config = self.provider_configs.get(provider, {})
            provider_instance = provider_class(**config)

            return provider_instance.create(model_name, messages, **kwargs)

        except Exception as e:
            raise RuntimeError(f"Error creating completion: {str(e)}")
