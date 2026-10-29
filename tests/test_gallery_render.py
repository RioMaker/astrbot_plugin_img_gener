from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from astrbot_plugin_img_gener import gallery_render
from astrbot_plugin_img_gener.errors import ImageGeneratorError
from astrbot_plugin_img_gener.gallery import GalleryImage

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _record(path, *, image_id=1, created_at=0):
    return GalleryImage(
        image_id=image_id,
        job_id=None,
        requester_qq="123",
        requester_name="用户",
        group_id=None,
        prompt="画一张",
        size="1024x1024",
        quality="medium",
        media_type="image/png",
        image_path=path,
        created_at=created_at,
        duration_seconds=1,
    )


def _render(records, **kwargs):
    arguments = {
        "user_name": "小明",
        "subtitle": f"指定图片 · 共 {len(records)} 张 · 按输入顺序排列",
        "command": "/画廊 [250 252 266 600]",
    }
    arguments.update(kwargs)
    result = gallery_render.build_personal_gallery(tuple(records), **arguments)
    assert result.media_type == "image/png"
    return Image.open(io.BytesIO(result.content))


@pytest.mark.parametrize(
    ("count", "columns", "rows"), [(1, 1, 1), (4, 2, 2), (9, 3, 3), (20, 4, 5)]
)
def test_gallery_adapts_and_keeps_requested_order(tmp_path, count, columns, rows):
    records = []
    colors = []
    for index in range(count):
        color = (30 + index * 10, 70, 170)
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (80, 80), color).save(path)
        colors.append(color)
        records.append(_record(path, image_id=count - index))

    result = _render(records)
    assert result.width <= 2000
    # Scan image centers, not labels: every input must appear once in row-major order.
    found = []
    for y in range(result.height):
        row_colors = []
        for column in range(columns):
            pixel = result.getpixel((int((column + 0.5) * result.width / columns), y))
            if pixel in colors and (not row_colors or row_colors[-1] != pixel):
                row_colors.append(pixel)
        if len(row_colors) == columns and (not found or row_colors != found[-1]):
            found.append(row_colors)
    assert len(found) == rows
    assert [color for row in found for color in row] == colors


@pytest.mark.parametrize("size", [(400, 100), (100, 400)])
def test_image_well_preserves_all_edges_of_landscape_and_portrait(tmp_path, size):
    source = Image.new("RGB", size, "red")
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, size[0] - 1, 15), fill="blue")
    draw.rectangle((0, size[1] - 16, size[0] - 1, size[1] - 1), fill="green")
    draw.rectangle((0, 16, 15, size[1] - 17), fill="yellow")
    path = tmp_path / "wide_or_tall.png"
    source.save(path)
    well, readable = gallery_render._image_well(path, 400)
    assert readable
    colors = well.getcolors(400 * 400)
    assert colors is not None
    found = {color for _, color in colors}
    assert {(255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 255, 0)} <= found
    assert gallery_render._WELL in found


def test_image_well_composites_alpha_and_applies_exif(tmp_path):
    transparent = tmp_path / "alpha.png"
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(transparent)
    well, readable = gallery_render._image_well(transparent, 100)
    assert readable
    assert well.getpixel((50, 50)) == gallery_render._WELL

    rotated = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (100, 40), (220, 30, 20)).save(rotated, exif=exif)
    well, readable = gallery_render._image_well(rotated, 100)
    assert readable
    assert well.getpixel((0, 50)) == gallery_render._WELL
    assert well.getpixel((50, 0)) != gallery_render._WELL


def test_missing_and_corrupt_images_keep_cards_and_ids(tmp_path, monkeypatch):
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"not an image")
    records = [_record(tmp_path / "missing.png", image_id=250), _record(corrupt, image_id=600)]
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def track_text(self, xy, text, *args, **kwargs):
        texts.append(text)
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track_text)
    result = _render(records)
    assert result.width == 1440
    assert texts.count("图片无法读取") == 2
    assert "#250" in texts and "#600" in texts
    assert "2 张图片无法读取" in texts


def test_dates_are_beijing_and_long_text_stays_inside_canvas(tmp_path, monkeypatch):
    path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10), "red").save(path)
    timestamp = datetime(2026, 10, 29, 20, tzinfo=timezone.utc).timestamp()
    record = _record(path, image_id=9223372036854775807, created_at=timestamp)
    texts = []
    original_text = ImageDraw.ImageDraw.text

    def track_text(self, xy, text, *args, **kwargs):
        texts.append((xy, text, kwargs["font"].getlength(text)))
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track_text)
    command = "/画廊 [" + " ".join(["9223372036854775807"] * 20) + "]"
    result = _render(
        [replace(record, image_id=record.image_id - index) for index in range(20)],
        user_name="很长的用户昵称" * 40,
        command=command,
    )
    assert any(text.endswith("の画廊") for _, text, _ in texts)
    assert sum(text == "10/30" for _, text, _ in texts) == 20
    assert any(text == "#9223372036854775807" for _, text, _ in texts)
    assert all(xy[0] >= 0 and xy[0] + length <= result.width for xy, _, length in texts)
    footer_start = next(index for index, (_, text, _) in enumerate(texts) if "本次指令" in text)
    footer_end = next(index for index, (_, text, _) in enumerate(texts) if "查看原图" in text)
    footer = "".join(text for _, text, _ in texts[footer_start:footer_end])
    assert footer.replace("本次指令：", "").replace(" ", "") == command.replace(" ", "")


@pytest.mark.parametrize("count", [0, 21])
def test_gallery_rejects_out_of_range_count(tmp_path, count):
    records = [_record(tmp_path / "missing.png")] * count
    with pytest.raises(ImageGeneratorError, match="1～20"):
        _render(records)


def test_bundled_font_renders_chinese_and_kana_without_system_fonts():
    font = gallery_render._font(30)
    assert str(font.path).endswith("NotoSansSC-Regular.otf")
    missing = bytes(font.getmask("\U0010ffff"))
    assert bytes(font.getmask("画")) != missing
    assert bytes(font.getmask("の")) != missing
