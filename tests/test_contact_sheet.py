from __future__ import annotations

import io

import pytest
from astrbot_plugin_img_gener.contact_sheet import (
    build_contact_sheet,
    build_contact_sheet_from_sources,
    build_gallery_grid,
)
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


def test_build_contact_sheet_accepts_in_memory_sources(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _fake_image(first, color=(200, 30, 30))
    _fake_image(second, color=(30, 30, 200))

    sheet = build_contact_sheet_from_sources(
        [
            ("first.png", first.read_bytes(), "image/png"),
            ("second.png", second.read_bytes(), "image/png"),
        ],
        labels=["REF-01 | 可可子", "REF-02 | 菌菌"],
    )

    image = Image.open(io.BytesIO(sheet.content))
    assert image.size == (2048, 1024)
    assert image.getpixel((100, 1000)) == (24, 24, 24)
    assert image.getpixel((1124, 1000)) == (24, 24, 24)


def test_build_contact_sheet_rejects_empty_input(tmp_path) -> None:
    with pytest.raises(ImageGeneratorError):
        build_contact_sheet(())


def test_build_contact_sheet_rejects_unreadable_file(tmp_path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ImageGeneratorError):
        build_contact_sheet((broken,))
@pytest.mark.parametrize(
    ("count", "expected_size"),
    [
        (3, (2304, 768)),
        (4, (1536, 1536)),
        (6, (2304, 1536)),
        (9, (2304, 2304)),
    ],
)
def test_gallery_grid_adapts_to_requested_count(
    tmp_path, count, expected_size
) -> None:
    paths = []
    for index in range(count):
        path = tmp_path / f"gallery_{index}.png"
        _fake_image(path, color=(index * 20, 30, 200), size=(300, 500))
        paths.append(path)

    grid = build_gallery_grid(tuple(paths))

    image = Image.open(io.BytesIO(grid.content))
    assert image.size == expected_size
