from __future__ import annotations

import asyncio
import importlib
import io
import sys
from pathlib import Path

import pytest
from test_main_background import _install_astrbot_stubs


class GalleryEvent:
    def __init__(self, sender_id="10001", name="画廊用户"):
        self.sender_id = sender_id
        self.name = name
        self.stopped = False

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return self.name

    def stop_event(self):
        self.stopped = True

    def plain_result(self, message):
        return message

    def chain_result(self, parts):
        return parts


@pytest.fixture
def gallery_plugin(tmp_path):
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.gallery_store = module.GalleryStore(tmp_path / "gallery")
    plugin.output_store = module.OutputStore(tmp_path / "outputs")
    plugin._gallery_render_lock = asyncio.Lock()
    return module, plugin


def add_image(plugin, *, requester="10001", group_id="200"):
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGB", (60, 100), "teal").save(buffer, "PNG")
    return plugin.gallery_store.add_image(
        buffer.getvalue(),
        extension=".png",
        media_type="image/png",
        requester_qq=requester,
        requester_name="归档时昵称",
        group_id=group_id,
        prompt="示例图片",
        size="60x100",
        quality="medium",
        duration_seconds=1,
    )


@pytest.mark.parametrize("query,count,start", [("", 20, 1), ("10", 10, 1), ("-5", 20, 5)])
def test_recent_gallery_uses_caller_and_current_name(
    gallery_plugin,
    monkeypatch,
    query,
    count,
    start,
):
    module, plugin = gallery_plugin
    own = []
    for index in range(26):
        own.append(add_image(plugin, group_id=str(index % 2)).image_id)
        add_image(plugin, requester="10002")
    rendered = []

    def render(records, **kwargs):
        rendered.append(([record.image_id for record in records], kwargs))
        return module.build_gallery_grid((records[0].image_path,))

    monkeypatch.setattr(module, "build_personal_gallery", render)
    event = GalleryEvent()

    async def scenario():
        return [result async for result in plugin.gallery_command(event, query)]

    results = asyncio.run(scenario())
    assert event.stopped and len(results) == 1
    assert Path(results[0][0].path).is_file()
    selected, kwargs = rendered[0]
    assert selected == list(reversed(own))[start - 1 : start - 1 + count]
    assert kwargs["user_name"] == "画廊用户"
    assert kwargs["subtitle"] == f"最近第 {start}—{start + count - 1} 张 · 共 26 张"
    assert kwargs["next_command"] == f"/画廊 -{start + count}"
    assert kwargs["show_latest"] is (start == 1)
    assert plugin.gallery_store.count() == 52  # The collage is temporary, never re-archived.


def test_selected_gallery_preserves_global_ids_order_and_repeats(gallery_plugin, monkeypatch):
    module, plugin = gallery_plugin
    first = add_image(plugin)
    second = add_image(plugin, requester="20002")
    seen = []

    def render(records, **kwargs):
        seen.append(([record.image_id for record in records], kwargs))
        return module.build_gallery_grid((records[0].image_path,))

    monkeypatch.setattr(module, "build_personal_gallery", render)

    async def scenario():
        return [
            result
            async for result in plugin.gallery_command(
                GalleryEvent(), f"[{second.image_id} {first.image_id} {second.image_id}]"
            )
        ]

    results = asyncio.run(scenario())
    assert len(results) == 1 and Path(results[0][0].path).is_file()
    ids, kwargs = seen[0]
    assert ids == [second.image_id, first.image_id, second.image_id]
    assert kwargs["command"] == "/画廊 [2 1 2]"
    assert kwargs["next_command"] is None and not kwargs["show_latest"]


def test_missing_ids_fail_as_a_batch_before_rendering(gallery_plugin, monkeypatch):
    module, plugin = gallery_plugin
    add_image(plugin)

    def unexpected_render(*args, **kwargs):
        pytest.fail("Missing selections must not render a partial gallery")

    monkeypatch.setattr(module, "build_personal_gallery", unexpected_render)

    async def scenario():
        return [result async for result in plugin.gallery_command(GalleryEvent(), "[1 88 99]")]

    results = asyncio.run(scenario())
    assert len(results) == 1 and "88、99" in results[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


@pytest.mark.parametrize("query", ["0", "21", "1 2", "-1 3", "[]", "[1", "[0]"])
def test_invalid_request_returns_usage_without_reading_store(gallery_plugin, query):
    _module, plugin = gallery_plugin
    plugin.gallery_store = None

    async def scenario():
        return [result async for result in plugin.gallery_command(GalleryEvent(), query)]

    result = asyncio.run(scenario())
    assert len(result) == 1 and "/画廊" in result[0]
    assert "尚未初始化" not in result[0]


def test_empty_and_out_of_range_gallery_return_clear_messages(gallery_plugin):
    _module, plugin = gallery_plugin

    async def scenario():
        empty = [result async for result in plugin.gallery_command(GalleryEvent())]
        add_image(plugin)
        outside = [result async for result in plugin.gallery_command(GalleryEvent(), "-2")]
        return empty, outside

    empty, outside = asyncio.run(scenario())
    assert "还没有成功归档" in empty[0]
    assert "1 张" in outside[0] and "超出范围" in outside[0]


def test_actual_gallery_render_handles_short_page_and_missing_file(gallery_plugin):
    _module, plugin = gallery_plugin
    for _ in range(4):
        record = add_image(plugin)
    record.image_path.unlink()

    async def scenario():
        return [result async for result in plugin.gallery_command(GalleryEvent(), "10")]

    results = asyncio.run(scenario())
    image = pytest.importorskip("PIL.Image")
    with image.open(results[0][0].path) as sheet:
        assert sheet.width > 500 and sheet.height > 500
        sheet.verify()
    assert plugin.gallery_store.count() == 4


def test_renderer_error_is_reported_without_raw_exception(gallery_plugin, monkeypatch):
    module, plugin = gallery_plugin
    add_image(plugin)

    def fail_render(*args, **kwargs):
        raise RuntimeError("private filesystem details")

    monkeypatch.setattr(module, "build_personal_gallery", fail_render)

    async def scenario():
        return [result async for result in plugin.gallery_command(GalleryEvent())]

    result = asyncio.run(scenario())
    assert "画廊拼图保存失败" in result[0]
    assert "private" not in result[0]
