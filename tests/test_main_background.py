from __future__ import annotations

import asyncio
import importlib
import sys
import types


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
