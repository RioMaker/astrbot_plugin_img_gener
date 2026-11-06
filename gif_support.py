"""GIF request defaults and bounded decoding for static consumers."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .config import get_config
from .errors import ConfigurationError, ImageAPIError


@dataclass(frozen=True, slots=True)
class GifOptions:
    model: str = "gpt-image-2.6"
    size: str = "512x512"
    quality: str = "low"
    frame_count: int = 16
    frame_duration_ms: int = 40

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("请配置 GIF 生图模型。")
        if self.size not in {"512x512", "640x640", "768x768", "1024x1024"}:
            raise ConfigurationError("GIF 尺寸请选择 512、640、768 或 1024 的方图。")
        if self.quality not in {"auto", "low", "medium", "high"}:
            raise ConfigurationError("GIF 质量只支持 auto、low、medium 或 high。")
        if type(self.frame_count) is not int or not 2 <= self.frame_count <= 16:
            raise ConfigurationError("GIF 帧数必须是 2～16 的整数。")
        if (
            type(self.frame_duration_ms) is not int
            or not 20 <= self.frame_duration_ms <= 2000
            or self.frame_duration_ms % 10
        ):
            raise ConfigurationError("GIF 每帧时长必须为 20～2000 毫秒，且为 10 的倍数。")

    @classmethod
    def from_config(cls, config):
        defaults = cls()
        return cls(
            **{
                key: get_config(config, "animation", key, default=getattr(defaults, key))
                for key in cls.__dataclass_fields__
            }
        )

    def request_fields(self) -> dict:
        return {
            "model": self.model.strip(),
            "size": self.size,
            "quality": self.quality,
            "output_format": "gif",
            "frame_count": self.frame_count,
            "frame_duration_ms": self.frame_duration_ms,
        }


def first_frame_png(content: bytes) -> bytes:
    """Decode only the first frame for references and vision evaluation."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("image exceeds decode limit")
            image.seek(0)
            output = io.BytesIO()
            image.convert("RGBA").save(output, "PNG")
            return output.getvalue()
    except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
        raise ImageAPIError("无法读取 GIF 首帧，图片可能损坏或像素过大。") from exc


def validate_gif(content: bytes) -> None:
    """Reject malformed or static results before counting a GIF job as successful."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "GIF" or not 2 <= image.n_frames <= 256:
                raise ValueError("not an animated GIF")
            if image.width * image.height * image.n_frames > 40_000_000:
                raise ValueError("animation exceeds decode limit")
            for index in range(image.n_frames):
                image.seek(index)
                image.load()
    except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
        raise ImageAPIError("生图服务未返回有效的 GIF 动图，请检查模型和上游 GIF 支持。") from exc
