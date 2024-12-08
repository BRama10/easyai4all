# chat_completion/models/tools.py
from dataclasses import dataclass
from typing import Literal, Dict, Any
from .base import BaseModel
import json


@dataclass
class Function(BaseModel):
    """Represents a function that the model called."""

    name: str
    arguments: Dict[str, Any]  # Changed from str to Dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> "Function":
        # Parse JSON string arguments into a dictionary if it's a string
        arguments = data["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return cls(name=data["name"], arguments=arguments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            "name": self.name,
            "arguments": self.arguments,  # Will be a raw dict, not JSON string
        }


@dataclass
class ToolCall(BaseModel):
    """Represents a tool call generated by the model."""

    id: str
    type: Literal["function"]
    function: Function

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(
            id=data["id"],
            type=data["type"],
            function=Function.from_dict(data["function"]),
        )
