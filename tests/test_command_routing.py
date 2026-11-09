from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from command_routing_support import install_command_parser
from test_gallery_command import GalleryEvent, add_image
from test_main_background import _install_astrbot_stubs


class CommandEvent(GalleryEvent):
    def __init__(self, text, *, awake=True):
        super().__init__()
        # AstrBot's waking stage removes the wake prefix before CommandFilter runs.
        self.text = text.removeprefix("/")
        self.is_at_or_wake_command = awake
        self.extras = {}

    def get_message_str(self):
        return self.text

    def set_extra(self, name, value):
        self.extras[name] = value


@pytest.fixture
def routed_plugin(monkeypatch):
    _install_astrbot_stubs()
    parser = install_command_parser(monkeypatch)
    registered = {}

    def command(name, *, alias=None, **kwargs):
        def register(handler):
            registered[name] = parser.CommandFilter(
                name, alias=alias, handler_md=SimpleNamespace(handler=handler)
            )
            return handler

        return register

    monkeypatch.setattr(sys.modules["astrbot.api.event"].filter, "command", command)
    monkeypatch.delitem(sys.modules, "astrbot_plugin_img_gener.main", raising=False)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    return module, plugin, registered


def parsed_event(registered, name, message):
    event = CommandEvent(message)
    assert registered[name].filter(event, {})
    return event


def dispatch(plugin, command_filter, event):
    async def collect():
        return [
            reply
            async for reply in command_filter.handler_md.handler(
                plugin, event, **event.extras["parsed_params"]
            )
        ]

    return asyncio.run(collect())


@pytest.mark.parametrize(
    "name,message,argument,expected",
    [
        ("拼图", "/拼图 233 123", "query", "233 123"),
        ("拼图", "/拼图 233\t123\n456 789", "query", "233 123 456 789"),
        (
            "拼图",
            "/拼图 " + " ".join(map(str, range(1, 21))),
            "query",
            " ".join(map(str, range(1, 21))),
        ),
        ("随机拼图", "/随机拼图 4 5", "query", "4 5"),
        ("画廊", "/画廊 [233 123 456]", "query", "[233 123 456]"),
        ("生图", "/生图 一只 猫 坐在 窗边", "prompt", "一只 猫 坐在 窗边"),
        ("生图", "/draw 一只 猫", "prompt", "一只 猫"),
        ("生图", "/绘图 一只 猫", "prompt", "一只 猫"),
        ("生动图", "/生动图 猫 眨眼 无缝循环", "prompt", "猫 眨眼 无缝循环"),
        ("生图历史", "/生图历史 随机 4", "query", "随机 4"),
        ("生图参考图", "/生图参考图 设置 12345", "query", "设置 12345"),
    ],
)
def test_full_tail_survives_framework_command_parsing(
    routed_plugin, name, message, argument, expected
):
    _module, _plugin, registered = routed_plugin
    event = parsed_event(registered, name, message)
    assert event.extras["parsed_params"] == {argument: expected}


@pytest.mark.parametrize(
    "name", ["拼图", "随机拼图", "画廊", "生图", "生动图", "生图历史", "生图参考图"]
)
def test_bare_commands_receive_empty_tail_without_framework_missing_argument_error(
    routed_plugin, name
):
    _module, _plugin, registered = routed_plugin
    event = parsed_event(registered, name, "/" + name)
    assert list(event.extras["parsed_params"].values()) == [""]


@pytest.mark.parametrize(
    "message,awake", [("/拼图 1 2", False), ("/拼图表 1 2", True), ("/随机拼图 2", True)]
)
def test_puzzle_route_keeps_wake_and_command_boundaries(routed_plugin, message, awake):
    _module, _plugin, registered = routed_plugin
    event = CommandEvent(message, awake=awake)
    assert not registered["拼图"].filter(event, {})
    assert event.extras == {} and not event.stopped


@pytest.mark.parametrize("query", ["2 1", "3 1 2", "2 1 3 2"])
def test_puzzle_dispatch_from_framework_params_reaches_image_output(routed_plugin, tmp_path, query):
    module, plugin, registered = routed_plugin
    plugin.gallery_store = module.GalleryStore(tmp_path / "gallery")
    plugin.output_store = module.OutputStore(tmp_path / "outputs")
    plugin._gallery_render_lock = asyncio.Lock()
    for _ in range(3):
        add_image(plugin)
    event = parsed_event(registered, "拼图", "/拼图 " + query)

    replies = dispatch(plugin, registered["拼图"], event)

    assert event.stopped and len(replies) == 1
    assert isinstance(replies[0], list), replies
    assert Path(replies[0][0].path).is_file()
    assert plugin.gallery_store.count() == 3


@pytest.mark.parametrize("name", ["生图", "生动图"])
def test_generation_dispatch_preserves_full_prompt(routed_plugin, name):
    _module, plugin, registered = routed_plugin
    captured = []

    async def draw(event, prompt, *, animated=False):
        captured.append((prompt, animated))
        yield "accepted"

    plugin._draw_command = draw
    event = parsed_event(registered, name, "/" + name + " 一只 猫 坐在 窗边")
    assert dispatch(plugin, registered[name], event) == ["accepted"]
    assert captured == [("一只 猫 坐在 窗边", name == "生动图")]


def test_extra_random_count_argument_is_rejected_after_real_parsing(routed_plugin):
    _module, plugin, registered = routed_plugin
    event = parsed_event(registered, "随机拼图", "/随机拼图 4 5")
    replies = dispatch(plugin, registered["随机拼图"], event)
    assert "/随机拼图" in replies[0] and "尚未初始化" not in replies[0]
