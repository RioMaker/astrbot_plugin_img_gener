from __future__ import annotations

from dataclasses import dataclass

from .errors import ImageGeneratorError

MAX_GALLERY_IMAGES = 20
MAX_SQLITE_INTEGER = (1 << 63) - 1
MAX_GALLERY_ARGUMENT_LENGTH = 512
GALLERY_USAGE = (
    "用法：/画廊（本人最近 20 张）\n"
    "/画廊 10（本人最近 10 张，数量 1～20）\n"
    "/画廊 -5（从本人最近第 5 张起取 20 张）\n"
    "/画廊 [123 151 488]（按图库编号顺序选图，最多 20 张）"
)


@dataclass(frozen=True, slots=True)
class GalleryRequest:
    start_rank: int = 1
    count: int = MAX_GALLERY_IMAGES
    image_ids: tuple[int, ...] = ()


def _invalid(message: str) -> ImageGeneratorError:
    return ImageGeneratorError(f"{message}\n{GALLERY_USAGE}")


def _positive_integer(value: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise _invalid("图片数量、图库编号和起始位置必须是正整数。")
    if len(value) > 19:
        raise _invalid("图库编号或起始位置超出支持范围。")
    number = int(value)
    if number == 0:
        raise _invalid("图片数量、图库编号和起始位置必须大于 0。")
    if number > MAX_SQLITE_INTEGER:
        raise _invalid("图库编号或起始位置超出支持范围。")
    return number


def parse_gallery_request(arguments: str) -> GalleryRequest:
    """Parse a recent count, starting rank, or bracketed archive-ID queue."""

    if len(arguments) > MAX_GALLERY_ARGUMENT_LENGTH:
        raise _invalid("画廊参数过长，请缩短输入。")
    tail = arguments.strip()
    if "[" in tail or "]" in tail:
        if (
            not tail.startswith("[")
            or not tail.endswith("]")
            or "[" in tail[1:-1]
            or "]" in tail[1:-1]
        ):
            raise _invalid("指定图库编号时，请使用完整的一对方括号，且不要混入其他参数。")
        parts = tail[1:-1].split()
        if not parts:
            raise _invalid("方括号内至少填写一个图库编号。")
        if len(parts) > MAX_GALLERY_IMAGES:
            raise _invalid(f"一次画廊最多选择 {MAX_GALLERY_IMAGES} 张图片。")
        image_ids = tuple(_positive_integer(part) for part in parts)
        return GalleryRequest(count=len(image_ids), image_ids=image_ids)
    parts = tail.split()
    if not parts:
        return GalleryRequest()
    if any(part.startswith("-") for part in parts):
        if len(parts) != 1:
            raise _invalid("最近范围 -N 必须单独使用，不能与图库编号混用。")
        return GalleryRequest(start_rank=_positive_integer(parts[0][1:]))
    if len(parts) != 1:
        raise _invalid("指定多个图库编号时，请放入方括号，例如 /画廊 [123 151]。")
    count = _positive_integer(parts[0])
    if count > MAX_GALLERY_IMAGES:
        raise _invalid(f"最近图片数量必须在 1～{MAX_GALLERY_IMAGES} 之间。")
    return GalleryRequest(count=count)
