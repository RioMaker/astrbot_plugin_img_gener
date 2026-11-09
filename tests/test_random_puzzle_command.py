from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from test_gallery_command import GalleryEvent, add_image
from test_gallery_command import gallery_plugin as gallery_plugin


@pytest.mark.parametrize("query,count", [("", 2), ("3", 3), ("4", 4), ("20", 20)])
def test_random_puzzle_reads_global_archive_and_only_saves_temporary_result(
    gallery_plugin, query, count
):
    _module, plugin = gallery_plugin
    for index in range(count):
        add_image(plugin, requester=str(10000 + index), group_id=str(index % 2))
    event = GalleryEvent()

    async def scenario():
        return [result async for result in plugin.random_puzzle_command(event, query)]

    results = asyncio.run(scenario())

    assert event.stopped and len(results) == 1
    image_path = Path(results[0][0].path)
    assert image_path.parent == plugin.output_store.output_dir
    image = pytest.importorskip("PIL.Image")
    with image.open(image_path) as sheet:
        assert sheet.format == "PNG"
        sheet.verify()
    assert plugin.gallery_store.count() == count
    # The minimal fixture has no API client, quota manager, or profile store.


@pytest.mark.parametrize("query", ["0", "1", "21", "2 4", "hello", "9" * 5000])
def test_invalid_arguments_stop_event_before_store_access(gallery_plugin, query):
    _module, plugin = gallery_plugin
    plugin.gallery_store = None
    event = GalleryEvent()

    async def scenario():
        return [result async for result in plugin.random_puzzle_command(event, query)]

    result = asyncio.run(scenario())
    assert event.stopped and "/随机拼图" in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


@pytest.mark.parametrize("available,requested", [(0, 2), (1, 2), (3, 4)])
def test_insufficient_archive_never_repeats_images_or_changes_layout(
    gallery_plugin, available, requested
):
    _module, plugin = gallery_plugin
    for _ in range(available):
        add_image(plugin)

    async def scenario():
        return [
            result async for result in plugin.random_puzzle_command(GalleryEvent(), str(requested))
        ]

    result = asyncio.run(scenario())
    assert f"只有 {available} 张" in result[0] and f"{requested} 张不同图片" in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


def test_missing_source_returns_error_and_leaves_no_partial_output(gallery_plugin):
    _module, plugin = gallery_plugin
    add_image(plugin)
    add_image(plugin).image_path.unlink()

    async def scenario():
        return [result async for result in plugin.random_puzzle_command(GalleryEvent(), "")]

    result = asyncio.run(scenario())
    assert "图库图片缺失" in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


@pytest.mark.parametrize("stage", ["query", "render", "save"])
def test_failures_return_safe_messages(gallery_plugin, monkeypatch, stage):
    module, plugin = gallery_plugin
    add_image(plugin)
    add_image(plugin)

    def fail(*args, **kwargs):
        raise RuntimeError("private filesystem details")

    if stage == "query":
        monkeypatch.setattr(plugin.gallery_store, "sample_random", fail)
    elif stage == "render":
        monkeypatch.setattr(module, "build_random_puzzle", fail)
    else:
        monkeypatch.setattr(plugin.output_store, "save_bytes", fail)

    async def scenario():
        return [result async for result in plugin.random_puzzle_command(GalleryEvent(), "")]

    result = asyncio.run(scenario())
    assert "随机拼图失败" in result[0] and "private" not in result[0]


@pytest.mark.parametrize("mixed_commands", [False, True])
def test_concurrent_renders_are_serialized_off_the_event_loop(
    gallery_plugin, monkeypatch, mixed_commands
):
    module, plugin = gallery_plugin
    add_image(plugin)
    add_image(plugin)
    render = module.build_random_puzzle
    active = 0
    thread_ids = []

    def tracked_render(paths):
        nonlocal active
        assert active == 0
        active += 1
        thread_ids.append(threading.get_ident())
        try:
            time.sleep(0.02)
            return render(paths)
        finally:
            active -= 1

    monkeypatch.setattr(module, "build_random_puzzle", tracked_render)

    async def invoke(selected=False):
        if selected:
            return [result async for result in plugin.puzzle_command(GalleryEvent(), "1 2")]
        return [result async for result in plugin.random_puzzle_command(GalleryEvent(), "")]

    async def scenario():
        return await asyncio.gather(invoke(), invoke(mixed_commands), invoke(mixed_commands))

    results = asyncio.run(scenario())
    assert len({result[0][0].path for result in results}) == 3
    assert len(thread_ids) == 3 and threading.get_ident() not in thread_ids
