from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest


def _install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    star_module = types.ModuleType("astrbot.api.star")
    components = types.ModuleType("astrbot.api.message_components")
    command_module = types.ModuleType("astrbot.core.star.filter.command")
    path_module = types.ModuleType("astrbot.core.utils.astrbot_path")

    class _Decorator:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            return lambda function: function

    class _Filter:
        llm_tool = _Decorator()
        command = _Decorator()

    class _Star:
        def __init__(self, context=None, config=None):
            self.context = context
            self.config = config

    class _Logger:
        def __getattr__(self, name):
            del name
            return lambda *args, **kwargs: None

    api.AstrBotConfig = dict
    api.logger = _Logger()
    event_module.AstrMessageEvent = object
    event_module.filter = _Filter()
    star_module.Context = object
    star_module.Star = _Star
    command_module.GreedyStr = str
    path_module.get_astrbot_plugin_data_path = lambda: "."

    modules = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_module,
        "astrbot.api.star": star_module,
        "astrbot.api.message_components": components,
        "astrbot.core.star.filter.command": command_module,
        "astrbot.core.utils.astrbot_path": path_module,
    }
    sys.modules.update(modules)


def test_llm_tool_returns_before_background_generation_finishes() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._background_tasks = set()
        plugin._is_allowed = lambda event: True
        release = asyncio.Event()

        async def fake_generate(event, prompt, **kwargs):
            del event, prompt, kwargs
            await release.wait()
            return "后台完成"

        plugin._generate_image = fake_generate

        class Event:
            def __init__(self):
                self.sent = []

            def plain_result(self, message):
                return message

            async def send(self, message):
                self.sent.append(message)

        event = Event()
        result = await plugin.generate_image(event, "test prompt")
        assert "任务已受理" in result
        await asyncio.sleep(0)
        assert len(plugin._background_tasks) == 1
        task = next(iter(plugin._background_tasks))
        assert not task.done()

        release.set()
        await task
        assert event.sent == ["后台完成"]

    asyncio.run(scenario())


def test_draw_command_yields_progress_before_starting_generation() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        calls = []

        async def fake_generate(event, prompt, **kwargs):
            del event, kwargs
            calls.append(prompt)
            return "生成完成"

        plugin._generate_image = fake_generate

        class Event:
            def __init__(self):
                self.stopped = False

            def stop_event(self):
                self.stopped = True

            def plain_result(self, message):
                return message

        event = Event()
        results = plugin.draw_command(event, "test prompt")
        progress = await anext(results)

        assert event.stopped
        assert "正在检查额度并进行安全审核" in progress
        assert calls == []

        completed = await anext(results)
        assert completed == "生成完成"
        assert calls == ["test prompt"]
        with pytest.raises(StopAsyncIteration):
            await anext(results)

    asyncio.run(scenario())


def test_review_client_uses_separate_endpoint_and_key() -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {
        "api": {"api_key": "image-key"},
        "safety": {
            "review_base_url": "https://uuapi.shop/v1",
            "review_api_key": "review-key",
            "review_model": "review-model",
        },
    }

    client = plugin._review_client()

    assert client.base_url == "https://uuapi.shop/v1"
    assert client.api_key == "review-key"
    assert client.api_key != plugin.config["api"]["api_key"]


def test_remote_review_requires_a_configured_model() -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {"safety": {"review_model": ""}}

    with pytest.raises(module.ConfigurationError):
        asyncio.run(plugin._remote_review(None, "safe prompt"))
