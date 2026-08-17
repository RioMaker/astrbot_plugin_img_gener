from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .errors import ImageGeneratorError

try:  # Pillow is optional until the sheet mode is actually used.
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


MAX_SHEET_CELLS = 12
SHEET_CELL_SIZE = 1024
SHEET_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class ContactSheet:
    content: bytes
    media_type: str = "image/png"


def build_contact_sheet(image_paths: tuple[Path, ...]) -> ContactSheet:
    """Arrange reference images into one PNG grid so a single image field carries
    every character, which sidesteps gateways that drop repeated image fields.
    """

    if Image is None:
        raise ImageGeneratorError(
            "拼合参考图模式需要 Pillow 依赖；请确认插件依赖已完整安装。"
        )
    if not image_paths:
        raise ImageGeneratorError("没有可拼合的参考图。")
    paths = list(image_paths)[:MAX_SHEET_CELLS]
    columns = max(1, min(SHEET_COLUMNS, len(paths)))
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * SHEET_CELL_SIZE, rows * SHEET_CELL_SIZE), "white"
    )
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                prepared = ImageOps.exif_transpose(image).convert("RGB")
        except OSError as exc:
            raise ImageGeneratorError(f"参考图无法读取：{path.name}") from exc
        prepared.thumbnail((SHEET_CELL_SIZE, SHEET_CELL_SIZE), Image.LANCZOS)
        cell_x = (index % columns) * SHEET_CELL_SIZE
        cell_y = (index // columns) * SHEET_CELL_SIZE
        offset_x = cell_x + (SHEET_CELL_SIZE - prepared.width) // 2
        offset_y = cell_y + (SHEET_CELL_SIZE - prepared.height) // 2
        canvas.paste(prepared, (offset_x, offset_y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return ContactSheet(content=buffer.getvalue())
