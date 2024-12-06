# chat_completion/models/audio.py
from dataclasses import dataclass
from .base import BaseModel


@dataclass
class AudioResponse(BaseModel):
    """Represents an audio response from the model."""

    id: str
    expires_at: int
    data: str
    transcript: str
