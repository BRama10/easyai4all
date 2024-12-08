from abc import ABC, abstractmethod
import os
from typing import List, Dict, Any, Optional
import httpx

from easyai4all.chat_completion.response import ChatCompletionResponse


class Provider(ABC):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = None,
        env_api_key: str = None,
        timeout: int = 60,
    ) -> None:
        """Initialize provider with API configuration.

        Args:
            api_key: API key for authentication. If not provided, checks environment
            api_base: Base URL for API endpoint
            env_api_key: Name of environment variable containing API key
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.api_base = api_base

        self.api_key = api_key or (os.getenv(env_api_key) if env_api_key else None)
        if not self.api_key:
            raise ValueError(
                f"API key must be provided either directly or through {env_api_key} environment variable"
            )

    @property
    def headers(self) -> Dict[str, str]:
        """Default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @abstractmethod
    def _prepare_request(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Prepare the request payload.

        Args:
            model: Model identifier
            messages: List of chat messages
            **kwargs: Additional model parameters

        Returns:
            Request payload dictionary
        """
        pass

    @abstractmethod
    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the API response.

        Args:
            response: Raw API response

        Returns:
            Processed response in standardized format
        """
        pass

    def create(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> ChatCompletionResponse:
        """Make a synchronous chat completion request.

        Args:
            model: Model identifier
            messages: List of chat messages
            **kwargs: Additional model parameters

        Returns:
            Processed chat completion response

        Raises:
            httpx.HTTPError: If the request fails
        """
        payload = self._prepare_request(model, messages, **kwargs)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.api_base, headers=self.headers, json=payload)
            response.raise_for_status()
            return self._process_response(response.json())

    async def acreate(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Make an asynchronous chat completion request.

        Args:
            model: Model identifier
            messages: List of chat messages
            **kwargs: Additional model parameters

        Returns:
            Processed chat completion response

        Raises:
            httpx.HTTPError: If the request fails
        """
        payload = self._prepare_request(model, messages, **kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_base, headers=self.headers, json=payload
            )
            response.raise_for_status()
            return self._process_response(response.json())
