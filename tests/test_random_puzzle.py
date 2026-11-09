from __future__ import annotations

import io
import math

import pytest
from astrbot_plugin_img_gener import random_puzzle
from astrbot_plugin_img_gener.errors import ImageGeneratorError

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _save(tmp_path, name, color, size=(301, 241)):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return path


def _render(paths):
    result = random_puzzle.build_random_puzzle(tuple(paths))
    assert result.media_type == "image/png"
    with Image.open(io.BytesIO(result.content)) as source:
        source.load()
        return source.copy()


@pytest.mark.parametrize("argument,count", [("", 2), ("  ", 2), (" 4 ", 4), ("3", 3), ("20", 20)])
def test_count_defaults_and_bounds(argument, count):
    assert random_puzzle.parse_puzzle_count(argument) == count


@pytest.mark.parametrize(
    "argument", ["0", "1", "21", "-2", "+4", "2.0", "2 4", "三", "²", "9" * 5000]
)
def test_invalid_counts_are_rejected(argument):
    with pytest.raises(ImageGeneratorError, match="/随机拼图"):
        random_puzzle.parse_puzzle_count(argument)


@pytest.mark.parametrize(
    "argument,expected",
    [
        ("233 123", (233, 123)),
        (" 233\t123\n456 ", (233, 123, 456)),
        ("233 123 233", (233, 123, 233)),
        ("001 002", (1, 2)),
        ("1 9223372036854775807", (1, 9223372036854775807)),
        (" ".join(map(str, range(1, 21))), tuple(range(1, 21))),
    ],
)
def test_explicit_ids_preserve_order_and_repeats(argument, expected):
    assert random_puzzle.parse_puzzle_ids(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        "",
        " ",
        "233",
        "0 1",
        "-1 2",
        "+1 2",
        "1.5 2",
        "1,2",
        "[1 2]",
        "one two",
        "１ ２",
        "1 9223372036854775808",
        "1 " + "9" * 5000,
        " ".join(["1"] * 21),
    ],
)
def test_invalid_explicit_ids_return_the_selected_command_usage(argument):
    with pytest.raises(ImageGeneratorError) as error:
        random_puzzle.parse_puzzle_ids(argument)
    assert "/拼图" in error.value.public_message
    assert "/随机拼图" not in error.value.public_message


def test_two_images_keep_exact_halves_at_matching_height(tmp_path):
    left = Image.new("RGB", (240, 120), "red")
    ImageDraw.Draw(left).rectangle((120, 0, 239, 119), fill="blue")
    right = Image.new("RGB", (80, 240), "yellow")
    ImageDraw.Draw(right).rectangle((40, 0, 79, 239), fill="green")
    paths = [tmp_path / "left.png", tmp_path / "right.png"]
    left.save(paths[0])
    right.save(paths[1])
    originals = [path.read_bytes() for path in paths]

    result = _render(paths)

    assert result.size == (140, 120)
    assert result.getpixel((10, 60)) == (255, 0, 0)
    assert result.getpixel((119, 60)) == (255, 0, 0)
    assert result.getpixel((135, 60)) == (0, 128, 0)
    assert [path.read_bytes() for path in paths] == originals


def test_four_images_keep_only_their_corresponding_quadrant(tmp_path):
    colors = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (200, 200, 10)]
    paths = []
    for index, color in enumerate(colors):
        source = Image.new("RGB", (200, 160), "magenta")
        x, y = index % 2 * 100, index // 2 * 80
        ImageDraw.Draw(source).rectangle((x, y, x + 99, y + 79), fill=color)
        path = tmp_path / f"{index}.png"
        source.save(path)
        paths.append(path)

    result = _render(paths)

    assert result.size == (200, 160)
    assert set(result.getdata()) == set(colors)
    for index, color in enumerate(colors):
        assert result.getpixel((index % 2 * 100 + 50, index // 2 * 80 + 40)) == color


@pytest.mark.parametrize("count", range(4, 21, 2))
def test_even_grids_use_every_image_and_cover_every_pixel(tmp_path, count):
    colors = [(10 + index * 10, 40, 160) for index in range(count)]
    result = _render([_save(tmp_path, f"{index}.png", c) for index, c in enumerate(colors)])

    assert result.size == (301, 241)
    assert set(result.getdata()) == set(colors)
    # Independent expected layouts, including non-square counts and indivisible dimensions.
    columns, rows = {
        4: (2, 2),
        6: (3, 2),
        8: (4, 2),
        10: (5, 2),
        12: (4, 3),
        14: (7, 2),
        16: (4, 4),
        18: (6, 3),
        20: (5, 4),
    }[count]
    for index, color in enumerate(colors):
        x = int((index % columns + 0.5) * result.width / columns)
        y = int((index // columns + 0.5) * result.height / rows)
        assert result.getpixel((x, y)) == color


@pytest.mark.parametrize("count", range(3, 20, 2))
def test_odd_wedges_start_at_top_and_fill_the_rectangle(tmp_path, count):
    colors = [(10 + index * 10, 70, 150) for index in range(count)]
    result = _render([_save(tmp_path, f"{index}.png", c) for index, c in enumerate(colors)])

    assert set(result.getdata()) == set(colors)  # No white seams, gaps, or unused corners.
    for index, color in enumerate(colors):
        angle = math.radians(-90 + (index + 0.5) * 360 / count)
        point = (
            round(result.width / 2 + 85 * math.cos(angle)),
            round(result.height / 2 + 85 * math.sin(angle)),
        )
        assert result.getpixel(point) == color


def test_mixed_aspect_ratios_are_center_cropped_for_quadrants(tmp_path):
    first = _save(tmp_path, "first.png", "blue", (200, 200))
    wide = Image.new("RGB", (400, 200), "magenta")
    ImageDraw.Draw(wide).rectangle((100, 0, 299, 199), fill="green")
    path = tmp_path / "wide.png"
    wide.save(path)

    result = _render([first, path, first, path])

    assert set(result.getdata()) == {(0, 0, 255), (0, 128, 0)}


def test_gif_uses_only_first_frame_and_preserves_original(tmp_path):
    path = tmp_path / "animation.gif"
    Image.new("RGB", (100, 100), "red").save(
        path,
        save_all=True,
        append_images=[Image.new("RGB", (100, 100), "blue")],
        duration=80,
        loop=0,
    )
    original = path.read_bytes()
    result = _render([path, _save(tmp_path, "right.png", "green", (100, 100))])

    assert result.getpixel((20, 50)) == (255, 0, 0)
    assert result.getpixel((80, 50)) == (0, 128, 0)
    assert path.read_bytes() == original
    with Image.open(path) as image:
        assert image.n_frames == 2


def test_alpha_and_exif_orientation(tmp_path):
    alpha = tmp_path / "alpha.png"
    Image.new("RGBA", (80, 120), (255, 0, 0, 0)).save(alpha)
    rotated = tmp_path / "rotated.jpg"
    source = Image.new("RGB", (120, 80), "blue")
    exif = source.getexif()
    exif[274] = 6
    source.save(rotated, exif=exif)

    result = _render([alpha, rotated])

    assert result.size == (80, 120)
    assert result.getpixel((20, 60)) == (255, 255, 255)
    assert result.getpixel((60, 60))[2] > 250


@pytest.mark.parametrize("kind", ["missing", "corrupt", "oversized"])
def test_unreadable_sources_fail_without_partial_output_or_exposing_paths(
    tmp_path, monkeypatch, kind
):
    valid = _save(tmp_path, "valid.png", "green")
    bad = tmp_path / "private-path.png"
    if kind == "corrupt":
        bad.write_bytes(b"not an image")
    elif kind == "oversized":
        _save(tmp_path, bad.name, "red", (400, 400))
        monkeypatch.setattr(random_puzzle, "MAX_SOURCE_PIXELS", 80_000)

    with pytest.raises(ImageGeneratorError) as error:
        _render([valid, bad])

    assert "图库图片" in error.value.public_message
    assert "private-path" not in error.value.public_message


@pytest.mark.parametrize("count", [0, 1, 21])
def test_renderer_rejects_invalid_count_before_opening_files(tmp_path, count):
    with pytest.raises(ImageGeneratorError, match="/随机拼图"):
        _render([tmp_path / "missing.png"] * count)


def test_output_dimensions_are_bounded_for_large_sources(tmp_path):
    path = _save(tmp_path, "large.png", "teal", (3000, 1800))
    result = _render([path, path])
    assert result.height <= 1024 and result.width <= 2048
    result = _render([path] * 4)
    assert max(result.size) <= 1024
