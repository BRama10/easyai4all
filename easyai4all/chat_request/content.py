# content.py
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any


@dataclass
class ImageUrl:
    url: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageUrl":
        return cls(url=data["url"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextContent":
        return cls(type=data["type"], text=data["text"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageUrlContent:
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageUrlContent":
        return cls(type=data["type"], image_url=ImageUrl.from_dict(data["image_url"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "image_url": self.image_url.to_dict()}
