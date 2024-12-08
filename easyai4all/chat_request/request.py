# chat_request.py
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Literal
from .messages import Message
from .tools import Tool, ToolChoice


@dataclass
class ChatRequest:
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoice]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatRequest":
        # Process messages
        messages = [Message.from_dict(msg) for msg in data["messages"]]

        # Process tools if present
        tools = None
        if data.get("tools"):
            tools = [Tool.from_dict(tool) for tool in data["tools"]]

        # Process tool_choice if present
        tool_choice = data.get("tool_choice")
        if isinstance(tool_choice, dict):
            tool_choice = ToolChoice.from_dict(tool_choice)

        return cls(
            model=data["model"],
            messages=messages,
            max_tokens=data.get("max_tokens"),
            tools=tools,
            tool_choice=tool_choice,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            n=data.get("n"),
            stream=data.get("stream"),
            stop=data.get("stop"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            logit_bias=data.get("logit_bias"),
            user=data.get("user"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
        }

        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.tools:
            result["tools"] = [tool.to_dict() for tool in self.tools]
        if self.tool_choice:
            if isinstance(self.tool_choice, ToolChoice):
                result["tool_choice"] = self.tool_choice.to_dict()
            else:
                result["tool_choice"] = self.tool_choice
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.n is not None:
            result["n"] = self.n
        if self.stream is not None:
            result["stream"] = self.stream
        if self.stop is not None:
            result["stop"] = self.stop
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            result["logit_bias"] = self.logit_bias
        if self.user is not None:
            result["user"] = self.user

        return result
