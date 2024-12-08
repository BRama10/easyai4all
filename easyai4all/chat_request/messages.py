# messages.py
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any, Literal, Type
from .content import TextContent, ImageUrlContent


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageUrlContent]]]
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        content = data["content"]
        if isinstance(content, list):
            processed_content = []
            for item in content:
                if item["type"] == "text":
                    processed_content.append(TextContent.from_dict(item))
                elif item["type"] == "image_url":
                    processed_content.append(ImageUrlContent.from_dict(item))
            content = processed_content

        return cls(role=data["role"], content=content, name=data.get("name"))

    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role}

        if isinstance(self.content, list):
            result["content"] = [item.to_dict() for item in self.content]
        else:
            result["content"] = self.content

        if self.name:
            result["name"] = self.name

        return result
