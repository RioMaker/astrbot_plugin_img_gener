"""GIF request defaults and bounded decoding for static consumers."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .config import as_int, get_config
from .errors import ConfigurationError, ImageAPIError

MAX_GIF_OUTPUT_PIXELS = 40_000_000
MAX_GIF_SOURCE_FRAME_PIXELS = 4096 * 4096
MAX_GIF_SOURCE_TOTAL_PIXELS = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class GifOptions:
    model: str = "gpt-image-2.6"
    size: str = "512x512"
    quality: str = "low"
    frame_count: int = 16
    frame_duration_ms: int = 40
    timeout_seconds: int = 300

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
        if type(self.timeout_seconds) is not int or not 10 <= self.timeout_seconds <= 1_000_000:
            raise ConfigurationError("GIF 超时必须为 0（沿用生图超时）或 10～1000000 秒的整数。")

    @classmethod
    def from_config(cls, config):
        defaults = cls()
        values = {
            key: get_config(config, "animation", key, default=getattr(defaults, key))
            for key in cls.__dataclass_fields__
        }
        timeout = get_config(config, "animation", "timeout_seconds", default=0)
        if type(timeout) is int and timeout == 0:
            timeout = as_int(
                get_config(config, "api", "timeout_seconds", default=300), 300, minimum=10
            )
        values["timeout_seconds"] = timeout
        return cls(**values)

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


def validate_gif(
    content: bytes, *, options: GifOptions | None = None, operation: str = "生动图"
) -> None:
    """Validate returned GIFs and retain non-sensitive metadata in failure diagnostics."""
    _process_gif(content, options=options, operation=operation)


def prepare_gif(content: bytes, *, options: GifOptions, operation: str) -> bytes:
    """Downsize oversized upstream animations before applying the output decode budget."""
    target = tuple(int(part) for part in options.size.split("x"))
    return _process_gif(content, options=options, operation=operation, resize_to=target)


def _process_gif(
    content: bytes,
    *,
    options: GifOptions | None,
    operation: str,
    resize_to: tuple[int, int] | None = None,
) -> bytes:
    metadata = {
        "operation": operation,
        "bytes": len(content),
        "format": "unknown",
        "size": "unknown",
        "frames": "unknown",
    }
    if options is not None:
        metadata.update(
            model=options.model,
            requested_size=options.size,
            requested_frames=options.frame_count,
        )

    def failure(reason: str, message: str, cause: Exception | None = None) -> ImageAPIError:
        details = {"reason": reason, **metadata}
        if cause is not None:
            details["error"] = type(cause).__name__
            details["decode_error"] = " ".join(str(cause).split())[:200]
        return ImageAPIError(
            message, detail="gif_validation " + " ".join(f"{k}={v}" for k, v in details.items())
        )

    try:
        with Image.open(io.BytesIO(content)) as image:
            metadata.update(format=image.format or "unknown", size=f"{image.width}x{image.height}")
            if image.format != "GIF":
                raise failure(
                    "unexpected_format",
                    f"已请求 GIF 动图，但上游返回了 {image.format or '其他'} 格式图片"
                    f"（{image.width}×{image.height}），本次未作为动图归档。",
                )
            frames = image.n_frames
            metadata["frames"] = frames
            if frames < 2:
                raise failure(
                    "single_frame", "上游返回的 GIF 只有 1 帧，是静态图，本次未作为动图归档。"
                )
            if frames > 256:
                raise failure(
                    "frame_limit", f"上游返回 GIF 共 {frames} 帧，超过本地校验上限 256 帧。"
                )
            original_size = image.size
            resize = resize_to is not None and (
                image.width > resize_to[0] or image.height > resize_to[1]
            )
            pixels = image.width * image.height
            if resize and (
                pixels > MAX_GIF_SOURCE_FRAME_PIXELS
                or pixels * frames > MAX_GIF_SOURCE_TOTAL_PIXELS
            ):
                raise failure(
                    "source_pixel_limit",
                    f"上游返回 GIF 为 {image.width}×{image.height}、{frames} 帧，"
                    "超过逐帧缩放的安全解码上限，未读取完整帧数据。",
                )
            output_pixels = resize_to[0] * resize_to[1] if resize else pixels
            if output_pixels * frames > MAX_GIF_OUTPUT_PIXELS:
                raise failure(
                    "pixel_limit",
                    f"上游返回 GIF 为 {image.width}×{image.height}、{frames} 帧，"
                    "超过本地总像素上限 4000 万；请降低尺寸或帧数。",
                )
            resized_frames = []
            durations = []
            loop = image.info.get("loop")
            for index in range(frames):
                metadata["checking_frame"] = index + 1
                image.seek(index)
                if image.size != original_size:
                    raise failure("canvas_change", "上游 GIF 的画布尺寸在帧间变化，无法安全处理。")
                image.load()
                if resize:
                    # Keep only small composited frames; never retain all source-resolution frames.
                    frame = image.convert("RGBA")
                    frame.thumbnail(resize_to, Image.Resampling.LANCZOS)
                    if frame.getextrema()[3] == (255, 255):
                        # RGB adaptive quantization retains subtle opaque-frame color changes.
                        frame = frame.convert("RGB")
                    resized_frames.append(frame)
                    durations.append(image.info.get("duration", options.frame_duration_ms))
            if resize:
                output = io.BytesIO()
                save_options = {"loop": loop} if loop is not None else {}
                resized_frames[0].save(
                    output,
                    "GIF",
                    save_all=True,
                    append_images=resized_frames[1:],
                    duration=durations,
                    disposal=2,
                    optimize=False,
                    **save_options,
                )
                prepared = output.getvalue()
                validate_gif(prepared, options=options, operation=operation)
                return prepared
            return content
    except Image.DecompressionBombError as exc:
        raise failure("decode_pixel_limit", "返回图片的像素过大，已停止 GIF 解码。", exc) from exc
    except (OSError, ValueError, EOFError) as exc:
        raise failure(
            "decode_error", "返回的 GIF 文件损坏、内容不完整或无法解码，本次未作为动图归档。", exc
        ) from exc
