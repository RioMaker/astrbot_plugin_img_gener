from __future__ import annotations

from astrbot_plugin_img_gener.gallery import GalleryStore


def _add(store: GalleryStore, *, requester: str, now: float) -> int:
    record = store.add_image(
        b"image-bytes",
        extension=".png",
        media_type="image/png",
        requester_qq=requester,
        requester_name=f"用户{requester}",
        group_id="100",
        prompt="雨夜车站的中文提示词",
        size="816x816",
        quality="medium",
        duration_seconds=12.5,
        job_id=7,
        now=now,
    )
    return record.image_id


def test_gallery_uses_permanent_auto_increment_ids_and_metadata(tmp_path) -> None:
    root = tmp_path / "gallery"
    store = GalleryStore(root)

    first = _add(store, requester="123", now=100)
    second = _add(store, requester="456", now=200)

    reopened = GalleryStore(root)
    record = reopened.get(second)
    assert (first, second) == (1, 2)
    assert reopened.count() == 2
    assert record is not None
    assert record.requester_qq == "456"
    assert record.requester_name == "用户456"
    assert record.prompt == "雨夜车站的中文提示词"
    assert record.duration_seconds == 12.5
    assert record.image_path.name == "2.png"
    assert record.image_path.read_bytes() == b"image-bytes"


def test_gallery_random_is_bounded_to_nine(tmp_path) -> None:
    store = GalleryStore(tmp_path / "gallery")
    for index in range(12):
        _add(store, requester=str(index), now=100 + index)

    records = store.random(99)

    assert len(records) == 9
    assert len({record.image_id for record in records}) == 9
