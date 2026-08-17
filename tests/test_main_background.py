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
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.config = {"generation": {"start_message": "自定义受理话术"}}
        plugin._is_allowed = lambda event: True
        plugin.moderator = module.PromptModerator(enabled=False)
        release = asyncio.Event()

        async def fake_generate(event, prompt, **kwargs):
            del event, prompt, kwargs
            await release.wait()
            return "后台完成"

        plugin._generate_image = fake_generate

        class Event:
            def __init__(self):
                self.sent = []

            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "1"

            def get_group_id(self):
                return "100"

            def plain_result(self, message):
                return message

            async def send(self, message):
                self.sent.append(message)

        event = Event()
        result = await plugin.generate_image(event, "test prompt")
        assert result.startswith("自定义受理话术\n本次尺寸：816x816")
        assert "当前队列：全局 1 个" in result
        assert "预计生成约 3 分钟" in result
        assert "完成后会单独发送图片和简短评价" in result
        await asyncio.sleep(0)
        assert len(plugin._background_tasks) == 1
        task = next(iter(plugin._background_tasks))
        assert not task.done()
        assert len(plugin._active_jobs) == 1

        release.set()
        await task
        assert event.sent == ["后台完成"]
        assert plugin._active_jobs == {}

    asyncio.run(scenario())


def test_draw_command_yields_progress_and_continues_in_background() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._background_tasks = set()
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin._is_allowed = lambda event: True
        plugin.moderator = module.PromptModerator(enabled=False)
        calls = []
        options = []
        release = asyncio.Event()
        plugin.config = {
            "generation": {
                "start_message": "自定义开始反馈",
                "default_size": "640x1024",
            }
        }

        async def fake_generate(event, prompt, **kwargs):
            del event
            await release.wait()
            calls.append(prompt)
            options.append(kwargs)
            return "生成完成"

        plugin._generate_image = fake_generate

        class Event:
            def __init__(self):
                self.stopped = False
                self.sent = []

            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "2"

            def get_group_id(self):
                return "100"

            def stop_event(self):
                self.stopped = True

            def plain_result(self, message):
                return message

            async def send(self, message):
                self.sent.append(message)

        event = Event()
        results = plugin.draw_command(event, "竖图 一只猫")
        progress = await anext(results)

        assert event.stopped
        assert progress.startswith("自定义开始反馈\n本次尺寸：640x1024")
        assert "当前队列" in progress
        assert "预计生成约 3 分钟" in progress
        assert calls == []
        assert len(plugin._active_jobs) == 1
        with pytest.raises(StopAsyncIteration):
            await anext(results)

        task = next(iter(plugin._background_tasks))
        assert not task.done()
        release.set()
        await task
        assert calls == ["竖图 一只猫"]
        assert options[0]["size"] == "640x1024"
        assert event.sent == ["生成完成"]
        assert plugin._active_jobs == {}

    asyncio.run(scenario())


def test_draw_command_rejects_blocked_term_before_feedback_or_queue() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._background_tasks = set()
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin._is_allowed = lambda event: True
        plugin.config = {"generation": {"default_size": "816x816"}}
        plugin.moderator = module.PromptModerator(
            enabled=True,
            blocked_terms=["禁用画风"],
        )
        canceled = []

        class Limiter:
            async def acquire(self, user_id, group_id, *, bypass_limits=False):
                assert user_id == "qq:3"
                assert group_id == "qq:100"
                assert bypass_limits is False
                return types.SimpleNamespace(
                    allowed=True,
                    lease_id="local-review-lease",
                )

            async def cancel(self, lease_id):
                canceled.append(lease_id)

        plugin.rate_limiter = Limiter()

        class Event:
            def __init__(self):
                self.stopped = False

            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "3"

            def get_group_id(self):
                return "100"

            def is_admin(self):
                return False

            def stop_event(self):
                self.stopped = True

            def plain_result(self, message):
                return message

        event = Event()
        results = plugin.draw_command(event, "请使用禁用画风画一只猫")
        response = await anext(results)

        assert event.stopped
        assert "CUSTOM_BLOCKED_TERM" in response
        assert "管理员设置的禁用规则" in response
        assert "正在" not in response
        assert "本次尺寸" not in response
        assert plugin._active_jobs == {}
        assert plugin._background_tasks == set()
        assert canceled == ["local-review-lease"]
        with pytest.raises(StopAsyncIteration):
            await anext(results)

        llm_response = await plugin.generate_image(
            event,
            "也请使用禁用画风画一只狗",
        )
        assert "CUSTOM_BLOCKED_TERM" in llm_response
        assert "正在" not in llm_response
        assert "本次尺寸" not in llm_response
        assert plugin._active_jobs == {}
        assert canceled == ["local-review-lease", "local-review-lease"]

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


def test_duration_format_is_human_readable() -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")

    assert module.GPTImageGeneratorPlugin._format_duration(12.34) == "12.3 秒"
    assert module.GPTImageGeneratorPlugin._format_duration(72.34) == (
        "1 分 12.3 秒"
    )


def test_status_command_lists_active_jobs_with_elapsed_time() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.config = {
            "api": {
                "base_url": "https://uuapi.cc/v1",
                "api_key": "image-key",
                "model": "gpt-image-2",
            },
            "safety": {
                "review_api_key": "review-key",
                "review_model": "review-model",
            },
        }
        plugin.moderator = types.SimpleNamespace(enabled=True, fail_closed=True)
        plugin.references = types.SimpleNamespace(characters=[])

        limits = types.SimpleNamespace(
            user_max_attempts=3,
            user_attempt_window_seconds=600,
            user_max_generations=5,
            user_generation_window_seconds=3600,
            group_max_generations=20,
            group_generation_window_seconds=3600,
            global_max_concurrent=2,
            group_max_concurrent=1,
        )

        class Limiter:
            def __init__(self):
                self.limits = limits

            async def status(self, user_id, group_id):
                del user_id, group_id
                return {
                    "user_attempts": 1,
                    "user_generations": 1,
                    "group_generations": 2,
                    "global_inflight": 1,
                }

        plugin.rate_limiter = Limiter()

        class Event:
            def stop_event(self):
                pass

            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "1"

            def get_group_id(self):
                return "100"

            def plain_result(self, message):
                return message

        event = Event()
        job_id, _ = plugin._register_job(event, source="LLM", size="1024x1536")
        plugin._active_jobs[job_id].started_at -= 65
        plugin._set_job_stage(job_id, "生成图片")

        results = plugin.status_command(event)
        message = await anext(results)

        assert "【队列】" in message
        assert "全局：1 个" in message
        assert "【任务】" in message
        assert "1024x1536" in message
        assert "生成图片" in message
        assert "已用 1 分" in message
        assert "【配额】" in message
        assert "服务配置" not in message
        assert "密钥" not in message
        assert "审核模型" not in message
        assert "生图接口" not in message

    asyncio.run(scenario())


def test_custom_legacy_allowlist_does_not_reject_configured_default() -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {
        "generation": {
            "default_size": "816x816",
            "allowed_sizes": [
                "auto",
                "1024x1024",
                "1024x1536",
                "1536x1024",
                "800x600",
                "640x480",
            ],
        }
    }

    assert plugin._normalize_size(None) == "816x816"
    assert plugin._normalize_size("816x816") == "816x816"

    with pytest.raises(
        module.ConfigurationError,
        match="不支持该图片尺寸",
    ):
        plugin._normalize_size("1024x640")
