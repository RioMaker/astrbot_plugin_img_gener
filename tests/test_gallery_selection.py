from __future__ import annotations

import pytest
from astrbot_plugin_img_gener.errors import ImageGeneratorError
from astrbot_plugin_img_gener.gallery_selection import (
    GALLERY_USAGE,
    MAX_SQLITE_INTEGER,
    GalleryRequest,
    parse_gallery_request,
)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("", GalleryRequest()),
        ("  \t\n", GalleryRequest()),
        ("1", GalleryRequest(count=1)),
        ("10", GalleryRequest(count=10)),
        (" 20 ", GalleryRequest(count=20)),
        ("-1", GalleryRequest(start_rank=1)),
        ("-5", GalleryRequest(start_rank=5)),
        ("-21", GalleryRequest(start_rank=21)),
        (f"-{MAX_SQLITE_INTEGER}", GalleryRequest(start_rank=MAX_SQLITE_INTEGER)),
        ("[10]", GalleryRequest(count=1, image_ids=(10,))),
        (
            " [123 151 488 622] ",
            GalleryRequest(count=4, image_ids=(123, 151, 488, 622)),
        ),
        ("[3\n1\t3 2]", GalleryRequest(count=4, image_ids=(3, 1, 3, 2))),
        (
            f"[{MAX_SQLITE_INTEGER}]",
            GalleryRequest(count=1, image_ids=(MAX_SQLITE_INTEGER,)),
        ),
    ],
)
def test_gallery_request_distinguishes_count_rank_and_ordered_ids(arguments, expected) -> None:
    assert parse_gallery_request(arguments) == expected


def test_gallery_request_keeps_twenty_repeated_slots() -> None:
    request = parse_gallery_request("[" + " ".join(["123"] * 20) + "]")
    assert request.count == 20
    assert request.image_ids == (123,) * 20


@pytest.mark.parametrize(
    "arguments",
    [
        "0",
        "21",
        "-0",
        "-",
        "--5",
        "+5",
        "1.5",
        "最新",
        "１０",
        "10 20",
        "-5 10",
        "10 -5",
        "[]",
        "[ ]",
        "[1 2",
        "1 2]",
        "[[1]]",
        "[1] [2]",
        "[1] 2",
        "2 [1]",
        "-5 [1]",
        "[1,2]",
        "[1，2]",
        "[1 -2]",
        "[0]",
        "[1 text]",
        "9" * 20,
        "-" + "9" * 20,
        str(MAX_SQLITE_INTEGER + 1),
        f"-{MAX_SQLITE_INTEGER + 1}",
        f"[{MAX_SQLITE_INTEGER + 1}]",
        "[" + " ".join(["1"] * 21) + "]",
        " " * 513,
        "[" + "1" * 5000 + "]",
    ],
)
def test_invalid_gallery_request_has_actionable_usage(arguments) -> None:
    with pytest.raises(ImageGeneratorError) as error:
        parse_gallery_request(arguments)
    assert GALLERY_USAGE in error.value.public_message


def test_bare_id_queue_explains_required_brackets() -> None:
    with pytest.raises(ImageGeneratorError) as error:
        parse_gallery_request("123 151 488")
    assert "方括号" in error.value.public_message
