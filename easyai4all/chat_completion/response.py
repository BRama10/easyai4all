# chat_completion/response.py
from dataclasses import dataclass
from typing import List, Optional, Literal
from datetime import datetime
from .models.base import BaseModel
from .models.message import Choice
from .models.usage import Usage
from .exceptions import InvalidDataError


@dataclass
class ChatCompletionResponse(BaseModel):
    """Represents a chat completion response returned by the model."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: str
    service_tier: Optional[str] = None

    @property
    def created_datetime(self) -> datetime:
        """Convert the Unix timestamp to a datetime object."""
        return datetime.fromtimestamp(self.created)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletionResponse":
        """Create a ChatCompletionResponse instance from a dictionary."""
        try:
            # Create Usage instance
            usage = Usage.from_dict(data["usage"])

            # Create Choice instances
            choices = [Choice.from_dict(choice) for choice in data["choices"]]

            return cls(
                id=data["id"],
                object=data["object"],
                created=data["created"],
                model=data["model"],
                choices=choices,
                usage=usage,
                system_fingerprint=data["system_fingerprint"],
                service_tier=data.get("service_tier"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise InvalidDataError(f"Invalid data format: {str(e)}")
