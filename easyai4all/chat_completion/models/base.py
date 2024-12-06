# chat_completion/models/base.py
from dataclasses import dataclass, fields, asdict
from typing import TypeVar, Dict, Any, Type

T = TypeVar("T")


@dataclass
class BaseModel:
    """Base class for all models with common functionality."""

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Convert dictionary to model instance."""
        field_types = {f.name: f.type for f in fields(cls)}
        processed_data = {}
        for key, value in data.items():
            if key in field_types:
                processed_data[key] = value
        return cls(**processed_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return asdict(self)
