# chat_completion/models/usage.py
from dataclasses import dataclass
from typing import Optional
from .base import BaseModel


@dataclass
class CompletionTokensDetails(BaseModel):
    """Detailed breakdown of tokens used in the completion."""

    accepted_prediction_tokens: int = 0
    reasoning_tokens: int = 0
    rejected_prediction_tokens: int = 0
    audio_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionTokensDetails":
        return cls(
            accepted_prediction_tokens=data.get("accepted_prediction_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            rejected_prediction_tokens=data.get("rejected_prediction_tokens", 0),
            audio_tokens=data.get("audio_tokens"),
        )


@dataclass
class PromptTokensDetails(BaseModel):
    """Detailed breakdown of tokens used in the prompt."""

    cached_tokens: int = 0
    audio_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PromptTokensDetails":
        return cls(
            cached_tokens=data.get("cached_tokens", 0),
            audio_tokens=data.get("audio_tokens"),
        )


@dataclass
class Usage(BaseModel):
    """Usage statistics for the completion request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: Optional[PromptTokensDetails] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        """Create a Usage instance from a dictionary."""
        completion_tokens_details = CompletionTokensDetails.from_dict(
            data.get("completion_tokens_details", {})
        )

        prompt_tokens_details = None
        if "prompt_tokens_details" in data:
            prompt_tokens_details = PromptTokensDetails.from_dict(
                data["prompt_tokens_details"]
            )

        return cls(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            completion_tokens_details=completion_tokens_details,
            prompt_tokens_details=prompt_tokens_details,
        )
