# chat_completion/models/tokens.py
from dataclasses import dataclass
from typing import List, Optional
from .base import BaseModel


@dataclass
class TokenInfo(BaseModel):
    """Represents detailed information about a token."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None


@dataclass
class LogProbs(BaseModel):
    """Log probability information for the choice."""

    content: Optional[List[TokenInfo]] = None
    refusal: Optional[List[TokenInfo]] = None
    top_logprobs: Optional[List[TokenInfo]] = None
