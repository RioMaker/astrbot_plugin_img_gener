from __future__ import annotations

import io

import pytest
from astrbot_plugin_img_gener.contact_sheet import build_contact_sheet
from astrbot_plugin_img_gener.errors import ImageGeneratorError

Image = pytest.importorskip("PIL.Image")

def _fake_image(path, color=(200, 30, 30), size=(800, 600)) -> None:
    Image.new("RGB", size, color).save(path)


def test_build_contact_sheet_arranges_cells_in_grid(tmp_path) -> None:
    first = tmp_path / "coco.png"
    second = tmp_path / "mushroom.png"
    _fake_image(first, color=(200, 30, 30))
    _fake_image(second, color=(30, 30, 200))
    sheet = build_contact_sheet((first, second))
    assert sheet.media_type == "image/png"
    image = Image.open(io.BytesIO(sheet.content))
    assert image.format == "PNG"
    assert image.size == (2048, 1024)


def test_build_contact_sheet_rejects_empty_input(tmp_path) -> None:
    with pytest.raises(ImageGeneratorError):
        build_contact_sheet(())


def test_build_contact_sheet_rejects_unreadable_file(tmp_path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ImageGeneratorError):
        build_contact_sheet((broken,))
