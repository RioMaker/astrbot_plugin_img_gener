from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_gallery_command import GalleryEvent, add_image
from test_gallery_command import gallery_plugin as gallery_plugin


@pytest.mark.parametrize("ids", [(2, 1), (3, 1, 2), (2, 1, 3, 2), (3, 1) * 10])
def test_selected_puzzle_preserves_global_id_order_and_uses_the_shared_renderer(
    gallery_plugin, monkeypatch, ids
):
    module, plugin = gallery_plugin
    image = pytest.importorskip("PIL.Image")
    colors = [(200, 10, 10), (10, 200, 10), (10, 10, 200)]
    for index, color in enumerate(colors):
        record = add_image(plugin, requester=str(10000 + index), group_id=str(index % 2))
        image.new("RGB", (120, 120), color).save(record.image_path)
    rendered = []
    original = module.build_random_puzzle

    def render(paths):
        rendered.append(tuple(int(path.stem) for path in paths))
        return original(paths)

    monkeypatch.setattr(module, "build_random_puzzle", render)
    event = GalleryEvent()

    async def scenario():
        return [result async for result in plugin.puzzle_command(event, " ".join(map(str, ids)))]

    results = asyncio.run(scenario())

    assert event.stopped and len(results) == 1 and rendered == [ids]
    path = Path(results[0][0].path)
    assert path.parent == plugin.output_store.output_dir
    with image.open(path) as output:
        assert output.format == "PNG" and output.size == (120, 120)
        if len(ids) == 2:
            assert output.getpixel((30, 60)) == colors[1]
            assert output.getpixel((90, 60)) == colors[0]
        elif len(ids) == 3:
            assert output.getpixel((90, 30)) == colors[2]
            assert output.getpixel((60, 100)) == colors[0]
            assert output.getpixel((30, 30)) == colors[1]
        elif len(ids) == 4:
            for index, image_id in enumerate(ids):
                assert (
                    output.getpixel((30 + index % 2 * 60, 30 + index // 2 * 60))
                    == colors[image_id - 1]
                )
        output.load()
    # No generation API client or quota manager exists in this fixture.
    assert plugin.gallery_store.count() == 3


@pytest.mark.parametrize("query", ["", "233", "0 2", "1,2", "1 9223372036854775808"])
def test_invalid_selection_is_rejected_before_reading_the_gallery(gallery_plugin, query):
    _module, plugin = gallery_plugin
    plugin.gallery_store = None
    event = GalleryEvent()

    async def scenario():
        return [result async for result in plugin.puzzle_command(event, query)]

    result = asyncio.run(scenario())
    assert event.stopped and "/拼图" in result[0]
    assert "尚未初始化" not in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


def test_missing_and_deleted_ids_are_listed_once_without_partial_render(
    gallery_plugin, monkeypatch
):
    module, plugin = gallery_plugin
    add_image(plugin)
    deleted = add_image(plugin)
    with plugin.gallery_store._connect() as connection:
        connection.execute(
            "UPDATE gallery_images SET deleted_at = 1 WHERE image_id = ?", (deleted.image_id,)
        )

    def unexpected_render(*args, **kwargs):
        pytest.fail("A missing selection must not produce a partial puzzle")

    monkeypatch.setattr(module, "build_random_puzzle", unexpected_render)

    async def scenario():
        return [result async for result in plugin.puzzle_command(GalleryEvent(), "1 88 2 99 88")]

    result = asyncio.run(scenario())
    assert "不存在或已删除" in result[0] and "88、2、99" in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


def test_missing_source_reports_failure_without_sending_partial_puzzle(gallery_plugin):
    _module, plugin = gallery_plugin
    add_image(plugin)
    add_image(plugin).image_path.unlink()

    async def scenario():
        return [result async for result in plugin.puzzle_command(GalleryEvent(), "1 2")]

    result = asyncio.run(scenario())
    assert "图库图片缺失" in result[0]
    assert list(plugin.output_store.output_dir.iterdir()) == []


@pytest.mark.parametrize("stage", ["query", "render", "save"])
def test_selected_puzzle_failures_do_not_expose_internal_details(
    gallery_plugin, monkeypatch, stage
):
    module, plugin = gallery_plugin
    add_image(plugin)
    add_image(plugin)

    def fail(*args, **kwargs):
        raise RuntimeError("private filesystem details")

    if stage == "query":
        monkeypatch.setattr(plugin.gallery_store, "get_many", fail)
    elif stage == "render":
        monkeypatch.setattr(module, "build_random_puzzle", fail)
    else:
        monkeypatch.setattr(plugin.output_store, "save_bytes", fail)

    async def scenario():
        return [result async for result in plugin.puzzle_command(GalleryEvent(), "1 2")]

    result = asyncio.run(scenario())
    assert "拼图失败" in result[0] and "private" not in result[0]
