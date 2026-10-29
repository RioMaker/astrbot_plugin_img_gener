from __future__ import annotations

import io
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from .contact_sheet import ContactSheet
from .errors import ImageGeneratorError
from .gallery import GalleryImage

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:  # pragma: no cover - reported when the command is used.
    Image = ImageDraw = ImageFont = ImageOps = None


MAX_PERSONAL_GALLERY_IMAGES = 20
_FONT_PATH = Path(__file__).parent / "assets" / "fonts" / "NotoSansSC-Regular.otf"
_BEIJING = timezone(timedelta(hours=8))
_BACKGROUND = (250, 248, 242)
_WELL = (241, 239, 233)
_INK = (37, 40, 41)
_MUTED = (106, 112, 120)
_TEAL = (41, 126, 137)


def _columns(count: int) -> int:
    if count <= 4:
        return 1 if count == 1 else 2
    return 3 if count <= 9 else 4


@lru_cache(maxsize=24)
def _font(size: int):
    try:
        return ImageFont.truetype(str(_FONT_PATH), size)
    except OSError as exc:
        raise ImageGeneratorError(
            "画廊字体资源缺失或损坏，请重新安装完整插件。",
            detail=f"gallery font unavailable: {_FONT_PATH}",
        ) from exc


def _single_line(value: str) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())


def _elide(text: str, font, width: int, *, suffix: str = "") -> str:
    if font.getlength(text + suffix) <= width:
        return text + suffix
    while text and font.getlength(text + "…" + suffix) > width:
        text = text[:-1]
    return text + "…" + suffix


def _wrap(text: str, font, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for char in text:
        if line and font.getlength(line + char) > width:
            lines.append(line.rstrip())
            line = char.lstrip()
        else:
            line += char
    if line:
        lines.append(line)
    return lines


def _image_well(path: Path, size: int):
    """Contain the full oriented original, compositing alpha onto an ivory well."""
    well = Image.new("RGB", (size, size), _WELL)
    try:
        with Image.open(path) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("image exceeds gallery decode limit")
            prepared = ImageOps.exif_transpose(source)
            prepared.thumbnail((size, size), Image.Resampling.LANCZOS)
            rgba = prepared.convert("RGBA")
            offset = ((size - rgba.width) // 2, (size - rgba.height) // 2)
            well.paste(rgba, offset, rgba.getchannel("A"))
        return well, True
    except (OSError, ValueError, OverflowError, Image.DecompressionBombError):
        return well, False


def _date(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp, tz=_BEIJING).strftime("%m/%d")
    except (OSError, OverflowError, ValueError):
        return "--/--"


def build_personal_gallery(
    records: tuple[GalleryImage, ...],
    *,
    user_name: str,
    subtitle: str,
    command: str,
    next_command: str | None = None,
    show_latest: bool = False,
) -> ContactSheet:
    """Render an ordered 1–20 image gallery without cropping or making API calls."""
    if Image is None:
        raise ImageGeneratorError("画廊需要 Pillow 依赖，请确认插件依赖已完整安装。")
    count = len(records)
    if not 1 <= count <= MAX_PERSONAL_GALLERY_IMAGES:
        raise ImageGeneratorError("画廊每次需要 1～20 张图片。")

    columns = _columns(count)
    rows = math.ceil(count / columns)
    width = {1: 1000, 2: 1440, 3: 1800, 4: 2000}[columns]
    margin, gap, header = 44, 24, 190
    card_width = (width - margin * 2 - gap * (columns - 1)) // columns
    caption_height, radius = 66, 12
    card_height = card_width + caption_height
    content_bottom = header + rows * card_height + (rows - 1) * gap
    title_font = _font(64 if columns >= 2 else 54)
    subtitle_font, label_font, date_font = _font(30), _font(34), _font(27)
    footer_font, small_font = _font(27), _font(23)
    footer_lines = _wrap(f"本次指令：{_single_line(command)}", footer_font, width - margin * 2)
    footer_lines.append("查看原图：/生图历史 编号")
    if next_command:
        footer_lines.extend(
            _wrap(f"下一组：{_single_line(next_command)}", footer_font, width - margin * 2)
        )
    footer_height = 66 + len(footer_lines) * 40
    canvas = Image.new("RGB", (width, content_bottom + footer_height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    title = _elide(
        _single_line(user_name)[:80] or "用户",
        title_font,
        width - margin * 2 - 4,
        suffix="の画廊",
    )
    draw.text((margin, 28), title, font=title_font, fill=_INK, stroke_width=1, anchor="lt")
    draw.text(
        (margin, 112),
        _elide(_single_line(subtitle), subtitle_font, width - margin * 2),
        font=subtitle_font,
        fill=_MUTED,
        anchor="lt",
    )
    draw.line((margin, 164, width - margin, 164), fill=(182, 185, 184), width=2)

    unreadable = 0
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        row_count = min(columns, count - row * columns)
        row_width = row_count * card_width + (row_count - 1) * gap
        x = (width - row_width) // 2 + column * (card_width + gap)
        y = header + row * (card_height + gap)
        draw.rounded_rectangle(
            (x - 2, y + 3, x + card_width + 2, y + card_height + 5),
            radius=radius + 2,
            fill=(233, 230, 224),
        )
        draw.rounded_rectangle(
            (x, y, x + card_width - 1, y + card_height - 1),
            radius=radius,
            fill="white",
        )
        well, readable = _image_well(record.image_path, card_width)
        # Only the upper corners of the image well are rounded.
        mask = Image.new("L", (card_width, card_width), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle((0, 0, card_width - 1, card_width - 1), radius, fill=255)
        mask_draw.rectangle((0, radius, card_width - 1, card_width - 1), fill=255)
        canvas.paste(well, (x, y), mask)
        if not readable:
            unreadable += 1
            text = "图片无法读取"
            text_x = x + (card_width - label_font.getlength(text)) / 2
            draw.text(
                (text_x, y + card_width / 2 - 20),
                text,
                font=label_font,
                fill=_MUTED,
                anchor="lt",
            )
        elif show_latest and index == 0:
            draw.rounded_rectangle((x + 14, y + 14, x + 108, y + 56), 16, fill=_TEAL)
            draw.text((x + 34, y + 21), "最新", font=small_font, fill="white", anchor="lt")
        date = _date(record.created_at)
        date_width = date_font.getlength(date)
        label = f"#{record.image_id}"
        id_font = label_font
        id_size = 34
        while id_font.getlength(label) > card_width - date_width - 56 and id_size > 12:
            id_size -= 1
            id_font = _font(id_size)
        draw.text(
            (x + 19, y + card_width + 14),
            label,
            font=id_font,
            fill=_INK,
            stroke_width=0,
            anchor="lt",
        )
        draw.text(
            (x + card_width - 19 - date_width, y + card_width + 20),
            date,
            font=date_font,
            fill=_MUTED,
            anchor="lt",
        )

    divider_y = content_bottom + 26
    draw.line((margin, divider_y, width - margin, divider_y), fill=(182, 185, 184), width=2)
    for index, line in enumerate(footer_lines):
        draw.text(
            (margin, divider_y + 19 + index * 40),
            line,
            font=footer_font,
            fill=_MUTED,
            anchor="lt",
        )
    if unreadable:
        # The card placeholders retain each selected record's position and ID.
        note = f"{unreadable} 张图片无法读取"
        note_width = small_font.getlength(note)
        if footer_font.getlength(footer_lines[-1]) + note_width + 40 < width - margin * 2:
            draw.text(
                (width - margin - note_width, divider_y + 19 + (len(footer_lines) - 1) * 40),
                note,
                font=small_font,
                fill=_MUTED,
                anchor="lt",
            )

    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return ContactSheet(content=output.getvalue())
