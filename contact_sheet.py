from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from .errors import ImageGeneratorError

try:  # Pillow is optional until the sheet mode is actually used.
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageOps = None


MAX_SHEET_CELLS = 12
SHEET_CELL_SIZE = 1024
SHEET_COLUMNS = 2
SHEET_LABEL_HEIGHT = 72
MAX_GALLERY_GRID_CELLS = 9
GALLERY_GRID_CELL_SIZE = 768


@dataclass(frozen=True, slots=True)
class ContactSheet:
    content: bytes
    media_type: str = "image/png"


def build_gallery_grid(image_paths: tuple[Path, ...]) -> ContactSheet:
    """Build an unlabeled adaptive 1-9 image grid for random gallery results."""

    if Image is None or ImageOps is None:
        raise ImageGeneratorError(
            "随机图库拼图需要 Pillow 依赖；请确认插件依赖已完整安装。"
        )
    selected = image_paths[:MAX_GALLERY_GRID_CELLS]
    if not selected:
        raise ImageGeneratorError("图库中没有可用于拼图的图片。")
    count = len(selected)
    columns = count if count <= 3 else (2 if count == 4 else 3)
    rows = (count + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * GALLERY_GRID_CELL_SIZE, rows * GALLERY_GRID_CELL_SIZE),
        (24, 24, 24),
    )
    for index, path in enumerate(selected):
        try:
            with Image.open(path) as source:
                fitted = ImageOps.fit(
                    source.convert("RGB"),
                    (GALLERY_GRID_CELL_SIZE, GALLERY_GRID_CELL_SIZE),
                    method=Image.Resampling.LANCZOS,
                )
        except (OSError, ValueError) as exc:
            raise ImageGeneratorError(
                "图库中有图片无法读取，请联系管理员检查长期存储。",
                detail=f"gallery_grid path={path} error={type(exc).__name__}: {exc}",
            ) from exc
        x = (index % columns) * GALLERY_GRID_CELL_SIZE
        y = (index // columns) * GALLERY_GRID_CELL_SIZE
        canvas.paste(fitted, (x, y))
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return ContactSheet(content=output.getvalue())


def _sheet_label_font():
    candidates = (
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "DejaVuSans.ttf",
    )
    for candidate in candidates:
        if "/" in candidate and not Path(candidate).is_file():
            continue
        try:
            return ImageFont.truetype(candidate, 32)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet_from_sources(
    sources: list[tuple[str, bytes, str]],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> ContactSheet:
    """Arrange upload sources into a grid with optional non-overlapping labels."""

    if Image is None or ImageDraw is None or ImageFont is None:
        raise ImageGeneratorError(
            "拼合参考图模式需要 Pillow 依赖；请确认插件依赖已完整安装。"
        )
    if not sources:
        raise ImageGeneratorError("没有可拼合的参考图。")
    selected = sources[:MAX_SHEET_CELLS]
    columns = max(1, min(SHEET_COLUMNS, len(selected)))
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * SHEET_CELL_SIZE, rows * SHEET_CELL_SIZE), "white"
    )
    has_labels = labels is not None
    content_height = (
        SHEET_CELL_SIZE - SHEET_LABEL_HEIGHT if has_labels else SHEET_CELL_SIZE
    )
    draw = ImageDraw.Draw(canvas) if has_labels else None
    font = _sheet_label_font() if has_labels else None
    for index, (name, content, _media_type) in enumerate(selected):
        try:
            with Image.open(io.BytesIO(content)) as image:
                prepared = ImageOps.exif_transpose(image).convert("RGB")
        except OSError as exc:
            raise ImageGeneratorError(f"参考图无法读取：{name}") from exc
        prepared.thumbnail((SHEET_CELL_SIZE, content_height), Image.LANCZOS)
        cell_x = (index % columns) * SHEET_CELL_SIZE
        cell_y = (index // columns) * SHEET_CELL_SIZE
        offset_x = cell_x + (SHEET_CELL_SIZE - prepared.width) // 2
        offset_y = cell_y + (content_height - prepared.height) // 2
        canvas.paste(prepared, (offset_x, offset_y))
        if draw is not None:
            label = (
                str(labels[index]).strip()
                if labels is not None and index < len(labels)
                else f"REF-{index + 1:02d}"
            )
            label_top = cell_y + content_height
            draw.rectangle(
                (
                    cell_x,
                    label_top,
                    cell_x + SHEET_CELL_SIZE - 1,
                    cell_y + SHEET_CELL_SIZE - 1,
                ),
                fill=(24, 24, 24),
            )
            try:
                draw.text(
                    (cell_x + 24, label_top + 15),
                    label,
                    fill="white",
                    font=font,
                )
            except UnicodeEncodeError:
                draw.text(
                    (cell_x + 24, label_top + 15),
                    label.split("|", 1)[0].strip(),
                    fill="white",
                    font=ImageFont.load_default(),
                )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return ContactSheet(content=buffer.getvalue())


def build_contact_sheet(image_paths: tuple[Path, ...]) -> ContactSheet:
    """Arrange reference images into one PNG grid so a single image field carries
    every character, which sidesteps gateways that drop repeated image fields.
    """

    if not image_paths:
        raise ImageGeneratorError("没有可拼合的参考图。")
    sources: list[tuple[str, bytes, str]] = []
    for path in image_paths:
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ImageGeneratorError(f"参考图无法读取：{path.name}") from exc
        sources.append((path.name, content, "application/octet-stream"))
    return build_contact_sheet_from_sources(sources)
