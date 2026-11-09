from __future__ import annotations

import io
import math
import re
from pathlib import Path

from .contact_sheet import ContactSheet
from .errors import ImageGeneratorError
from .gallery_selection import MAX_GALLERY_ARGUMENT_LENGTH, MAX_GALLERY_IMAGES, MAX_SQLITE_INTEGER

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:  # pragma: no cover - reported when the command is used.
    Image = ImageDraw = ImageOps = None


MAX_PUZZLE_IMAGES = MAX_GALLERY_IMAGES
MAX_PUZZLE_EDGE = 1024
MAX_SOURCE_PIXELS = 40_000_000
PUZZLE_USAGE = (
    f"用法：/随机拼图 [2～{MAX_PUZZLE_IMAGES}]（省略数量时为 2 张）\n"
    "2 张左右各取半幅，4 张各取一个象限；其他偶数按宫格，奇数按扇形拼接。"
)
SELECTED_PUZZLE_USAGE = (
    f"用法：/拼图 <图库编号1> <图库编号2> ...（2～{MAX_PUZZLE_IMAGES} 项，空格分隔）\n"
    "例如：/拼图 233 123；按输入顺序裁切，偶数宫格、奇数扇形。"
)


def parse_puzzle_count(arguments: str) -> int:
    tail = arguments.strip()
    if not tail:
        return 2
    if len(tail) > 2 or re.fullmatch(r"[0-9]{1,2}", tail) is None:
        raise ImageGeneratorError(PUZZLE_USAGE)
    count = int(tail)
    if not 2 <= count <= MAX_PUZZLE_IMAGES:
        raise ImageGeneratorError(PUZZLE_USAGE)
    return count


def parse_puzzle_ids(arguments: str) -> tuple[int, ...]:
    """Parse explicit archive IDs, retaining the caller's order and repeated IDs."""
    if len(arguments) > MAX_GALLERY_ARGUMENT_LENGTH:
        raise ImageGeneratorError(f"拼图参数过长，请缩短输入。\n{SELECTED_PUZZLE_USAGE}")
    parts = arguments.split()
    if not 2 <= len(parts) <= MAX_PUZZLE_IMAGES:
        raise ImageGeneratorError(SELECTED_PUZZLE_USAGE)
    if any(re.fullmatch(r"[0-9]{1,19}", part) is None for part in parts):
        raise ImageGeneratorError(f"图库编号必须是正整数。\n{SELECTED_PUZZLE_USAGE}")
    image_ids = tuple(map(int, parts))
    if any(not 1 <= image_id <= MAX_SQLITE_INTEGER for image_id in image_ids):
        raise ImageGeneratorError(f"图库编号超出有效范围。\n{SELECTED_PUZZLE_USAGE}")
    return image_ids


def _load_frame(path: Path):
    """Read one oriented frame, limiting decode size and retained working memory."""
    try:
        with Image.open(path) as source:
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError("image exceeds puzzle decode limit")
            source.seek(0)
            prepared = ImageOps.exif_transpose(source)
            prepared.thumbnail((MAX_PUZZLE_EDGE * 2, MAX_PUZZLE_EDGE), Image.Resampling.LANCZOS)
            rgba = prepared.convert("RGBA")
            frame = Image.new("RGB", rgba.size, "white")
            frame.paste(rgba, (0, 0), rgba.getchannel("A"))
            return frame
    except (OSError, ValueError, OverflowError, Image.DecompressionBombError) as exc:
        raise ImageGeneratorError(
            "图库图片缺失、损坏或尺寸过大，请更换图片或联系管理员检查图库。",
            detail=f"random_puzzle path={path} error={type(exc).__name__}: {exc}",
        ) from exc


def _join_halves(left, right):
    # Match height without stretching either image; retain exactly its corresponding half.
    height = min(left.height, right.height, MAX_PUZZLE_EDGE)
    width_at_height = (left.width / left.height + right.width / right.height) * height / 2
    if width_at_height > MAX_PUZZLE_EDGE * 2:
        height = max(1, int(height * MAX_PUZZLE_EDGE * 2 / width_at_height))
    halves = []
    for index, frame in enumerate((left, right)):
        width = max(2, round(frame.width * height / frame.height))
        fitted = frame.resize((width, height), Image.Resampling.LANCZOS)
        middle = width // 2
        box = (0, 0, middle, height) if index == 0 else (middle, 0, width, height)
        halves.append(fitted.crop(box))
    canvas = Image.new("RGB", (sum(half.width for half in halves), height), "white")
    canvas.paste(halves[0], (0, 0))
    canvas.paste(halves[1], (halves[0].width, 0))
    return canvas


def _grid_shape(count: int) -> tuple[int, int]:
    rows = math.isqrt(count)
    while count % rows:
        rows -= 1
    return count // rows, rows


def build_random_puzzle(image_paths: tuple[Path, ...]) -> ContactSheet:
    """Join pieces in the supplied image order, retaining repeats for explicit selections."""
    if Image is None:
        raise ImageGeneratorError("拼图需要 Pillow 依赖，请确认插件依赖已完整安装。")
    count = len(image_paths)
    if not 2 <= count <= MAX_PUZZLE_IMAGES:
        raise ImageGeneratorError(PUZZLE_USAGE)
    first = _load_frame(image_paths[0])
    if count == 2:
        canvas = _join_halves(first, _load_frame(image_paths[1]))
    else:
        # Use the first image's aspect ratio as the common full frame.
        first.thumbnail((MAX_PUZZLE_EDGE, MAX_PUZZLE_EDGE), Image.Resampling.LANCZOS)
        width, height = first.size
        if min(width, height) < count:
            raise ImageGeneratorError("用于拼图的图片尺寸过小或过于狭长，请更换图片后重试。")
        canvas = Image.new("RGB", first.size, "white")
        columns, rows = _grid_shape(count) if count % 2 == 0 else (0, 0)
        radius = math.ceil(math.hypot(width, height))
        circle = (
            width / 2 - radius,
            height / 2 - radius,
            width / 2 + radius,
            height / 2 + radius,
        )
        for index, path in enumerate(image_paths):
            frame = first if index == 0 else _load_frame(path)
            fitted = ImageOps.fit(frame, first.size, method=Image.Resampling.LANCZOS)
            if count % 2 == 0:
                row, column = divmod(index, columns)
                box = (
                    column * width // columns,
                    row * height // rows,
                    (column + 1) * width // columns,
                    (row + 1) * height // rows,
                )
                canvas.paste(fitted.crop(box), box[:2])
            else:
                # An enclosing circle extends wedges to every corner of the rectangular frame.
                mask = Image.new("L", first.size, 0)
                ImageDraw.Draw(mask).pieslice(
                    circle,
                    start=-90 + index * 360 / count,
                    end=-90 + (index + 1) * 360 / count,
                    fill=255,
                )
                canvas.paste(fitted, (0, 0), mask)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return ContactSheet(content=output.getvalue())
