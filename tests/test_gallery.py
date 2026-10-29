from __future__ import annotations

import pytest
from astrbot_plugin_img_gener.gallery import GalleryStore


def _add(store: GalleryStore, *, requester: str, now: float, group_id: str | None = "100") -> int:
    record = store.add_image(
        b"image-bytes",
        extension=".png",
        media_type="image/png",
        requester_qq=requester,
        requester_name=f"用户{requester}",
        group_id=group_id,
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


def test_recent_gallery_filters_requester_across_groups_and_pages_after_reopen(tmp_path) -> None:
    root = tmp_path / "gallery"
    store = GalleryStore(root)
    requester_ids = []
    for index in range(25):
        requester_ids.append(
            _add(
                store,
                requester="123",
                now=100 - index,
                group_id=("100", "200", None)[index % 3],
            )
        )
        _add(store, requester="456", now=100 + index)

    reopened = GalleryStore(root)
    total, latest = reopened.recent_for_requester("123")
    assert total == 25
    assert [record.image_id for record in latest] == requester_ids[::-1][:20]
    assert {record.group_id for record in latest} == {"100", "200", None}
    assert {record.requester_qq for record in latest} == {"123"}

    total, from_fifth = reopened.recent_for_requester("123", offset=4)
    assert total == 25
    assert [record.image_id for record in from_fifth] == requester_ids[::-1][4:24]

    total, final_page = reopened.recent_for_requester("123", offset=20)
    assert total == 25
    assert [record.image_id for record in final_page] == requester_ids[::-1][20:]
    assert reopened.recent_for_requester("123", offset=25) == (25, ())
    assert reopened.recent_for_requester("unknown") == (0, ())


def test_recent_gallery_respects_requested_count_and_caps_large_limits(tmp_path) -> None:
    store = GalleryStore(tmp_path / "gallery")
    ids = [_add(store, requester="123", now=index) for index in range(21)]

    total, recent_ten = store.recent_for_requester("123", limit=10)
    assert total == 21
    assert [record.image_id for record in recent_ten] == ids[::-1][:10]
    assert len(store.recent_for_requester("123", limit=100)[1]) == 20


def test_recent_gallery_count_and_page_share_one_snapshot(tmp_path, monkeypatch) -> None:
    root = tmp_path / "gallery"
    store = GalleryStore(root)
    first = _add(store, requester="123", now=1)
    second = _add(store, requester="123", now=2)
    writer = GalleryStore(root)
    original_connect = store._connect
    inserted = False

    def insert_between_count_and_page(statement: str) -> None:
        nonlocal inserted
        if "ORDER BY image_id DESC LIMIT" in statement and not inserted:
            inserted = True
            _add(writer, requester="123", now=3)

    def reader_connection():
        connection = original_connect()
        connection.set_trace_callback(insert_between_count_and_page)
        return connection

    monkeypatch.setattr(store, "_connect", reader_connection)
    total, page = store.recent_for_requester("123")

    assert inserted
    assert total == 2
    assert [record.image_id for record in page] == [second, first]
    assert writer.count() == 3


def test_get_many_uses_global_ids_in_input_order_with_duplicates_and_missing_ids(tmp_path) -> None:
    root = tmp_path / "gallery"
    store = GalleryStore(root)
    first = _add(store, requester="123", now=1)
    second = _add(store, requester="456", now=2, group_id="200")
    third = _add(store, requester="789", now=3, group_id=None)

    records = GalleryStore(root).get_many((third, 999, first, third, second))

    assert [record.image_id for record in records] == [third, first, third, second]
    assert [record.requester_qq for record in records] == ["789", "123", "789", "456"]
    assert store.get_many(()) == ()
    assert store.get_many((998, 999)) == ()
    assert [record.image_id for record in store.get_many((first,) * 20)] == [first] * 20


def test_gallery_queries_reject_unsafe_bounds(tmp_path) -> None:
    store = GalleryStore(tmp_path / "gallery")
    with pytest.raises(ValueError):
        store.get_many((1,) * 21)
    for invalid in (0, -1, 1 << 63):
        with pytest.raises(ValueError):
            store.get_many((invalid,))
    for invalid_offset in (-1, 1 << 63):
        with pytest.raises(ValueError):
            store.recent_for_requester("123", offset=invalid_offset)
