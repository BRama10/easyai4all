# tools.py
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Literal


@dataclass
class FunctionParameters:
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionParameters":
        return cls(
            type=data.get("type", "object"),
            properties=data["properties"],
            required=data.get("required", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Function:
    name: str
    description: Optional[str] = None
    parameters: Optional[FunctionParameters] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Function":
        parameters = data.get("parameters")
        return cls(
            name=data["name"],
            description=data.get("description"),
            parameters=FunctionParameters.from_dict(parameters) if parameters else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = self.parameters.to_dict()
        return result


@dataclass
class Tool:
    type: Literal["function"] = "function"
    function: Function

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        return cls(type=data["type"], function=Function.from_dict(data["function"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "function": self.function.to_dict()}


@dataclass
class ToolChoice:
    type: Literal["function"] = "function"
    function: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolChoice":
        return cls(type=data["type"], function=data["function"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
