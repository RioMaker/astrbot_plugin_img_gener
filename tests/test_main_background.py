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

    class _Image:
        def __init__(self, path: str):
            self.path = path

        @classmethod
        def fromFileSystem(cls, path: str):
            return cls(path)

        async def convert_to_file_path(self):
            return self.path

    class _At:
        def __init__(self, qq: str):
            self.qq = str(qq)

    class _Plain:
        def __init__(self, text: str):
            self.text = text

    class _Reply:
        def __init__(self, chain=None):
            self.chain = list(chain or [])

    class _Logger:
        def __getattr__(self, name):
            del name
            return lambda *args, **kwargs: None

    components.Image = _Image
    components.Reply = _Reply
    components.At = _At
    components.Plain = _Plain
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
        result = await plugin.generate_image(event, "雨后的城市车站，电影感光影")
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


def test_llm_tool_uses_ai_generated_chinese_prompt_and_keeps_reference_text() -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._background_tasks = set()
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.config = {}
        plugin._is_allowed = lambda event: True
        plugin.moderator = module.PromptModerator(enabled=False)
        plugin.rate_limiter = types.SimpleNamespace(
            limits=types.SimpleNamespace(global_max_concurrent=2),
            status=lambda user_id, group_id: _status(user_id, group_id),
        )
        received = {}
        release = asyncio.Event()

        async def _status(user_id, group_id):
            del user_id, group_id
            return {"global_inflight": 1}

        async def fake_generate(event, prompt, **kwargs):
            del event
            received["prompt"] = prompt
            received.update(kwargs)
            await release.wait()
            return None

        plugin._generate_image = fake_generate

        class Event:
            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "1"

            def get_group_id(self):
                return None

            def get_message_str(self):
                return "请帮我画一张可可子在雨后的车站撑透明伞，竖图"

        event = Event()
        generated_prompt = (
            "雨后的火车站站台，可可子手持透明雨伞，地面倒映暖色灯光，"
            "细腻日系插画，竖向构图"
        )
        result = await plugin.generate_image(
            event,
            generated_prompt,
        )
        await asyncio.sleep(0)

        assert received["prompt"] == generated_prompt
        assert received["reference_prompt"] == (
            "请帮我画一张可可子在雨后的车站撑透明伞，竖图\n"
            + generated_prompt
        )
        assert "请帮我画一张" not in received["prompt"]

        release.set()
        await next(iter(plugin._background_tasks))

    asyncio.run(scenario())


def test_llm_tool_rejects_non_chinese_generated_prompt(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._background_tasks = set()
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.config = {}
        plugin.history_store = module.GenerationHistoryStore(tmp_path / "history.sqlite3")
        plugin._is_allowed = lambda event: True

        class Event:
            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "1"

            def get_group_id(self):
                return None

            def get_message_str(self):
                return "请帮我画一张雨后的车站"

        result = await plugin.generate_image(
            Event(),
            "A rainy train station with cinematic lighting",
        )

        assert "中文提示词" in result
        assert plugin._background_tasks == set()
        assert plugin._active_jobs == {}
        record = plugin.history_store.recent_for_user("qq:1")[0]
        assert record.status == "rejected"
        assert record.stage == "中文提示词校验失败"
        assert "中文提示词" in record.result_message

    asyncio.run(scenario())


def test_history_create_failure_cannot_overwrite_an_old_record(
    tmp_path, monkeypatch
) -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin._active_jobs = {}
    plugin._next_job_id = 0
    plugin.history_store = module.GenerationHistoryStore(tmp_path / "history.sqlite3")

    old_job_id = plugin.history_store.create_job(
        user_id="qq:old",
        group_id=None,
        source="LLM",
        prompt="旧任务",
        size="816x816",
        quality="auto",
    )
    plugin.history_store.finish_job(
        old_job_id,
        status="succeeded",
        stage="已完成",
        result_message="旧任务已完成。",
    )

    def fail_to_create(**kwargs):
        del kwargs
        raise OSError("temporary sqlite failure")

    monkeypatch.setattr(plugin.history_store, "create_job", fail_to_create)

    class Event:
        def get_platform_name(self):
            return "qq"

        def get_sender_id(self):
            return "new"

        def get_group_id(self):
            return None

    fallback_job_id, _ = plugin._register_job(
        Event(), source="LLM", size="816x816", prompt="新任务"
    )
    assert fallback_job_id == old_job_id
    assert not plugin._active_jobs[fallback_job_id].history_persisted

    plugin._finalize_job(
        fallback_job_id,
        status="failed",
        stage="生成图片失败",
        result_message="新任务失败。",
    )

    old_record = plugin.history_store.recent_for_user("qq:old")[0]
    assert old_record.status == "succeeded"
    assert old_record.stage == "已完成"
    assert old_record.result_message == "旧任务已完成。"


def test_quoted_reply_images_are_loaded_as_reference_sources(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.config = {
            "references": {
                "enable_quoted_images": True,
                "max_image_size_mb": 1,
            }
        }
        image_path = tmp_path / "quoted.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nquoted-image")
        quoted_image = module.Comp.Image(str(image_path))

        class Event:
            def get_messages(self):
                return [
                    module.Comp.Reply(chain=[quoted_image]),
                    module.Comp.Image(str(image_path)),
                ]

        components = plugin._quoted_image_components(Event())
        sources = await plugin._load_quoted_image_sources(components)

        assert components == (quoted_image,)
        assert sources == [
            ("quoted_1.png", image_path.read_bytes(), "image/png")
        ]
        augmented = plugin._augment_quoted_reference_prompt(
            "雨夜街道",
            start_index=2,
            count=1,
            use_sheet=True,
        )
        assert "REF-02（拼合图第 2 格）" in augmented
        assert "用户引用的历史聊天图片" in augmented

    asyncio.run(scenario())


def test_portrait_reference_command_and_tool_enforce_owner_permissions(
    tmp_path,
) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.config = {"references": {"max_image_size_mb": 1}}
        plugin.user_profiles = module.UserProfileStore(tmp_path / "profiles.sqlite3")

        first_path = tmp_path / "first.png"
        first_path.write_bytes(b"\x89PNG\r\n\x1a\nfirst")
        second_path = tmp_path / "second.jpg"
        second_path.write_bytes(b"\xff\xd8\xffsecond")

        class Event:
            def __init__(
                self,
                *,
                sender_id="10001",
                sender_name="小明",
                admin=False,
                messages=None,
            ):
                self.sender_id = sender_id
                self.sender_name = sender_name
                self.admin = admin
                self.messages = list(messages or [])
                self.sent = []
                self.stopped = False

            def get_sender_id(self):
                return self.sender_id

            def get_sender_name(self):
                return self.sender_name

            def get_group_id(self):
                return "200"

            def get_messages(self):
                return self.messages

            def is_admin(self):
                return self.admin

            def stop_event(self):
                self.stopped = True

            def plain_result(self, message):
                return message

            def chain_result(self, parts):
                return parts

            async def send(self, message):
                self.sent.append(message)

        owner = Event(messages=[module.Comp.Image(str(first_path))])
        set_results = plugin.portrait_reference_command(owner, "设置")
        set_chain = await anext(set_results)
        assert owner.stopped
        assert "已替换 QQ 10001" in next(
            part.text for part in set_chain if hasattr(part, "text")
        )
        profile = plugin.user_profiles.get("10001")
        assert profile is not None
        assert len(profile.image_paths) == 1
        assert profile.image_paths[0].read_bytes().endswith(b"first")

        owner.messages = [
            module.Comp.Reply(chain=[module.Comp.Image(str(second_path))])
        ]
        replace_results = plugin.portrait_reference_command(owner, "替换")
        replace_chain = await anext(replace_results)
        assert "已替换 QQ 10001" in next(
            part.text for part in replace_chain if hasattr(part, "text")
        )
        profile = plugin.user_profiles.get("10001")
        assert profile is not None
        assert len(profile.image_paths) == 1
        assert profile.image_paths[0].suffix == ".jpg"
        assert profile.image_paths[0].read_bytes().endswith(b"second")

        view_results = plugin.portrait_reference_command(owner, "")
        view_chain = await anext(view_results)
        assert any(isinstance(part, module.Comp.Image) for part in view_chain)

        tool_result = await plugin.get_user_portrait_reference(owner, "self")
        assert tool_result == "已向用户发送档案中当前唯一的本体参考图。"
        assert len(owner.sent) == 1
        assert any(isinstance(part, module.Comp.Image) for part in owner.sent[0])

        tool = module.GPTImageGeneratorPlugin.get_user_portrait_reference
        tool_annotations = tool.__annotations__
        tool_doc = tool.__doc__ or ""
        assert tool_annotations["target_user_qq"] == "str"
        assert "target_user_qq(string)" in tool_doc

        denied = await plugin.get_user_portrait_reference(owner, "10002")
        assert "只有 AstrBot 管理员" in denied

        plugin.user_profiles.ensure_profile("10002", display_name="小红")
        admin = Event(sender_id="99999", sender_name="管理员", admin=True)
        list_results = plugin.portrait_reference_command(admin, "列表")
        listing = await anext(list_results)
        assert "（10001）（小明）— 已设置" in listing
        assert "（10002）（小红）— 未设置" in listing

        admin_view_results = plugin.portrait_reference_command(admin, "10001")
        admin_view_chain = await anext(admin_view_results)
        assert any(isinstance(part, module.Comp.Image) for part in admin_view_chain)

    asyncio.run(scenario())


def test_generation_uploads_quoted_reply_image(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.config = {
            "references": {
                "enable_quoted_images": True,
                "max_quoted_images": 2,
                "max_image_size_mb": 1,
                "reference_mode": "multi",
            },
            "generation": {"enable_evaluation": False},
        }
        plugin._active_jobs = {}
        plugin.references = module.CharacterReferenceManager(
            [],
            data_dir=tmp_path,
            plugin_dir=tmp_path,
            max_total_images=4,
        )
        plugin.moderator = module.PromptModerator(enabled=False)
        plugin._is_allowed = lambda event: True

        class RateLimiter:
            def __init__(self):
                self.completed = []
                self.canceled = []

            async def acquire(self, user_id, group_id, *, bypass_limits=False):
                del user_id, group_id, bypass_limits
                return types.SimpleNamespace(allowed=True, lease_id="lease-1")

            async def complete(self, lease_id):
                self.completed.append(lease_id)

            async def cancel(self, lease_id):
                self.canceled.append(lease_id)

        plugin.rate_limiter = RateLimiter()

        captured = {}

        class ImageClient:
            async def edit(self, prompt, sources, *, size, quality):
                captured["prompt"] = prompt
                captured["sources"] = sources
                captured["size"] = size
                captured["quality"] = quality
                return types.SimpleNamespace(
                    content=b"\x89PNG\r\n\x1a\ngenerated",
                    media_type="image/png",
                    revised_prompt="",
                    extension=".png",
                )

            async def generate(self, prompt, *, size, quality):
                raise AssertionError("quoted image must use /images/edits")

        plugin._client = lambda: ImageClient()
        plugin.output_store = types.SimpleNamespace(
            save=lambda image: tmp_path / "generated.png"
        )
        plugin._completion_chain = lambda event, caption, path: (caption, path)

        async def no_evaluation(prompt, content, media_type):
            del prompt, content, media_type
            return None

        plugin._evaluate_image = no_evaluation

        quoted_path = tmp_path / "quoted.png"
        quoted_path.write_bytes(b"\x89PNG\r\n\x1a\nquoted")
        quoted_image = module.Comp.Image(str(quoted_path))

        class Event:
            def __init__(self):
                self.sent = []

            def get_messages(self):
                return [module.Comp.Reply(chain=[quoted_image])]

            def get_platform_name(self):
                return "qq"

            def get_sender_id(self):
                return "1"

            def get_group_id(self):
                return None

            def is_admin(self):
                return False

            async def send(self, message):
                self.sent.append(message)

        event = Event()
        result = await plugin._generate_image(event, "参考引用图片绘制雨夜街道")

        assert result is None
        assert captured["sources"][0][0] == "reference_01_user_quote_01.png"
        assert captured["sources"][0][1] == quoted_path.read_bytes()
        assert "用户引用图片映射" in captured["prompt"]
        assert "REF-01（文件 reference_01_user_quote_01.png）" in captured["prompt"]
        assert plugin.rate_limiter.completed == ["lease-1"]
        assert plugin.rate_limiter.canceled == []
        assert len(event.sent) == 1

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


def test_status_command_and_agent_tool_list_authoritative_state(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.history_store = module.GenerationHistoryStore(tmp_path / "history.sqlite3")
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
                    "user_attempts_remaining": 2,
                    "user_attempt_reset_after": 0,
                    "user_cooldown_remaining": 30,
                    "user_generations": 1,
                    "user_pending": 0,
                    "user_generations_remaining": 4,
                    "user_generation_reset_after": 0,
                    "group_generations": 2,
                    "group_pending": 0,
                    "group_generations_remaining": 18,
                    "group_generation_reset_after": 0,
                    "group_cooldown_remaining": 10,
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

        assert "【结论】" in message
        assert "你的任务确实在队列或处理中：1 个" in message
        assert "【队列与进度】" in message
        assert "全局 1 个" in message
        assert "1024x1536" in message
        assert "生成图片" in message
        assert "已用 1 分" in message
        assert "【余量与冷却】" in message
        assert "我的调用余量：2 次" in message
        assert "我的调用冷却：还需 30.0 秒" in message
        assert "我的生图余量：4 次" in message
        assert "本群生图余量：18 次" in message
        assert "【我的最近记录】" in message
        assert "服务配置" not in message
        assert "密钥" not in message
        assert "审核模型" not in message
        assert "生图接口" not in message

        tool = module.GPTImageGeneratorPlugin.get_image_generation_status
        tool_annotations = tool.__annotations__
        tool_doc = tool.__doc__ or ""
        assert tool_annotations["history_limit"] == "str"
        assert "history_limit(string)" in tool_doc
        assert "history_limit(integer)" not in tool_doc

        tool_message = await plugin.get_image_generation_status(
            event, include_history=True, history_limit="5"
        )
        assert "你的任务确实在队列或处理中：1 个" in tool_message
        assert "生成图片" in tool_message

    asyncio.run(scenario())


def test_agent_status_reports_no_active_task_and_recent_502_failure(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin._active_jobs = {}
        plugin._next_job_id = 0
        plugin.config = {}
        plugin.history_store = module.GenerationHistoryStore(tmp_path / "history.sqlite3")
        job_id = plugin.history_store.create_job(
            user_id="qq:1",
            group_id="qq:100",
            source="LLM",
            prompt="雨夜车站",
            size="816x816",
            quality="medium",
            now=100,
        )
        plugin.history_store.finish_job(
            job_id,
            status="failed",
            stage="生成图片失败",
            result_message="生图服务上游暂时异常（HTTP 502），请稍后重试。",
            now=120,
        )
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
                    "user_attempts_remaining": 2,
                    "user_cooldown_remaining": 0,
                    "user_generations": 0,
                    "user_pending": 0,
                    "user_generations_remaining": 5,
                    "group_generations": 0,
                    "group_pending": 0,
                    "group_generations_remaining": 20,
                    "group_cooldown_remaining": 0,
                    "global_inflight": 0,
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

            def is_admin(self):
                return False

            def plain_result(self, message):
                return message

        event = Event()
        report = await plugin.get_image_generation_status(
            event, include_history=True, history_limit="5"
        )

        assert "当前没有你的任务在队列或处理中" in report
        assert "最近任务 #0001 为“失败”" in report
        assert "HTTP 502" in report
        assert "我的生图余量：5 次" in report
        assert "本群生图余量：20 次" in report


    asyncio.run(scenario())


def test_first_qq_portrait_without_reference_uses_description_and_is_archived(
    tmp_path,
) -> None:
    async def scenario() -> None:
        image_module = pytest.importorskip("PIL.Image")
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.config = {
            "generation": {"enable_evaluation": False},
            "references": {"strict_references": True},
        }
        plugin._active_jobs = {}
        plugin.references = module.CharacterReferenceManager(
            [], data_dir=tmp_path, plugin_dir=tmp_path
        )
        plugin.user_profiles = module.UserProfileStore(tmp_path / "profiles.sqlite3")
        plugin.user_profiles.ensure_profile("10001", display_name="首次画像用户")
        plugin.gallery_store = module.GalleryStore(tmp_path / "gallery")
        plugin.output_store = module.OutputStore(tmp_path / "outputs")
        plugin.moderator = module.PromptModerator(enabled=False)
        plugin._is_allowed = lambda event: True

        class Limiter:
            def __init__(self):
                self.completed = []

            async def acquire(self, user_id, group_id, *, bypass_limits=False):
                del user_id, group_id, bypass_limits
                return types.SimpleNamespace(allowed=True, lease_id="portrait-lease")

            async def complete(self, lease_id):
                self.completed.append(lease_id)

            async def cancel(self, lease_id):
                raise AssertionError(f"successful lease must not be cancelled: {lease_id}")

        plugin.rate_limiter = Limiter()
        generated_path = tmp_path / "generated.png"
        image_module.new("RGB", (320, 480), (80, 120, 200)).save(generated_path)
        captured = {}

        class Client:
            async def generate(self, prompt, *, size, quality):
                captured.update(prompt=prompt, size=size, quality=quality)
                return types.SimpleNamespace(
                    content=generated_path.read_bytes(),
                    media_type="image/png",
                    extension=".png",
                )

            async def edit(self, prompt, sources, *, size, quality):
                raise AssertionError(
                    f"first portrait has no references: {prompt} {sources} {size} {quality}"
                )

        plugin._client = lambda: Client()

        class Event:
            def __init__(self):
                self.sent = []

            def get_sender_id(self):
                return "99999"

            def get_sender_name(self):
                return "请求用户"

            def get_group_id(self):
                return "200"

            def get_platform_name(self):
                return "qq"

            def get_messages(self):
                return []

            def is_admin(self):
                return False

            def chain_result(self, parts):
                return parts

            async def send(self, message):
                self.sent.append(message)

        result = await plugin._generate_image(
            Event(),
            "一名青年站在海边，保持用户描述的短发和蓝色外套",
            portrait_user_qqs=("10001",),
        )
        profile = plugin.user_profiles.get("10001")

        assert result is None
        assert "一名青年站在海边" in captured["prompt"]
        assert plugin.rate_limiter.completed == ["portrait-lease"]
        assert plugin.gallery_store.count() == 1
        assert profile is not None
        assert len(profile.image_paths) == 1

    asyncio.run(scenario())


def test_missing_static_character_reference_falls_back_to_user_prompt(
    tmp_path,
) -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {"references": {"strict_references": True}}
    manager = module.CharacterReferenceManager(
        [
            {
                "name": "可可子",
                "reference_images": [str(tmp_path / "missing.png")],
                "prompt_note": "不应自动加入的紫色短发设定",
            }
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )

    resolved = manager.resolve("画可可子站在海边", ["可可子"])

    assert resolved.image_paths == ()
    assert plugin._reference_resolution_error(
        resolved, portrait_user_qqs=()
    ) is None
    assert manager.augment_prompt("画可可子站在海边", resolved) == (
        "画可可子站在海边"
    )


def test_unknown_explicit_character_does_not_block_text_generation(tmp_path) -> None:
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {"references": {"strict_references": True}}
    manager = module.CharacterReferenceManager(
        [], data_dir=tmp_path, plugin_dir=tmp_path
    )
    resolved = manager.resolve("画原创角色阿青", ["阿青"])

    assert resolved.unknown_requested_names == ("阿青",)
    assert plugin._reference_resolution_error(
        resolved, portrait_user_qqs=()
    ) is None


def test_duplicate_profile_alias_requires_numbered_ai_confirmation(tmp_path) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.user_profiles = module.UserProfileStore(tmp_path / "profiles.sqlite3")
        plugin.user_profiles.ensure_profile("10001", display_name="小明")
        plugin.user_profiles.ensure_profile("10002", display_name="小明")

        class Bot:
            async def call_action(self, **kwargs):
                names = {10001: "当前小明甲", 10002: "当前小明乙"}
                return {"card": names[kwargs["user_id"]]}

        class Event:
            bot = Bot()

            def get_sender_id(self):
                return "99999"

            def get_group_id(self):
                return "200"

            def get_messages(self):
                return []

        subjects, error = await plugin._resolve_portrait_subjects(
            Event(), "让小明站在海边", None
        )

        assert subjects == ()
        assert error is not None
        assert "当前没有受理生图任务" in error
        assert "1.（10001）（当前小明甲）" in error
        assert "2.（10002）（当前小明乙）" in error
        assert "portrait_user_qqs" in error

    asyncio.run(scenario())


def test_self_portrait_refreshes_capped_group_alias_and_at_excludes_bot(
    tmp_path,
) -> None:
    async def scenario() -> None:
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.user_profiles = module.UserProfileStore(tmp_path / "profiles.sqlite3")

        class Event:
            def get_sender_id(self):
                return "10001"

            def get_sender_name(self):
                return "特别特别长的群聊昵称" * 4

            def get_group_id(self):
                return "200"

            def get_self_id(self):
                return "99999"

            def get_messages(self):
                return [module.Comp.At(qq="99999"), module.Comp.At(qq="10002")]

        event = Event()
        assert plugin._mentioned_qqs(event) == ("10002",)
        assert plugin._self_portrait_requested("画我在海边散步")
        assert not plugin._self_portrait_requested("帮我画一张我的猫")

        subjects, error = await plugin._resolve_portrait_subjects(
            event, "画我在海边散步", ["self"]
        )
        profile = plugin.user_profiles.get("10001")

        assert error is None
        assert subjects == ("10001", "10002")
        assert profile is not None
        assert len(profile.display_name) == 20
        assert profile.display_name in profile.aliases

    asyncio.run(scenario())


def test_success_archive_updates_single_profile_reference_and_gallery_commands(
    tmp_path,
) -> None:
    async def scenario() -> None:
        image_module = pytest.importorskip("PIL.Image")
        _install_astrbot_stubs()
        sys.modules.pop("astrbot_plugin_img_gener.main", None)
        module = importlib.import_module("astrbot_plugin_img_gener.main")
        plugin = object.__new__(module.GPTImageGeneratorPlugin)
        plugin.gallery_store = module.GalleryStore(tmp_path / "gallery")
        plugin.user_profiles = module.UserProfileStore(tmp_path / "profiles.sqlite3")
        plugin.output_store = module.OutputStore(tmp_path / "outputs")
        plugin.user_profiles.ensure_profile("10001", display_name="画像用户")

        class Event:
            def __init__(self):
                self.stopped = False

            def get_sender_id(self):
                return "99999"

            def get_sender_name(self):
                return "请求用户"

            def get_group_id(self):
                return "200"

            def stop_event(self):
                self.stopped = True

            def plain_result(self, message):
                return message

            def chain_result(self, parts):
                return parts

        event = Event()
        for index in range(4):
            image_path = tmp_path / f"source_{index}.png"
            image_module.new("RGB", (320, 480), (index * 40, 20, 200)).save(
                image_path
            )
            generated = types.SimpleNamespace(
                content=image_path.read_bytes(),
                extension=".png",
                media_type="image/png",
            )
            record, profile_failures = await plugin._archive_generated_image(
                event,
                generated,
                prompt=f"第 {index + 1} 张用户画像",
                size="816x816",
                quality="medium",
                duration_seconds=10 + index,
                job_id=index + 1,
                portrait_user_qqs=("10001",),
            )
            assert record is not None
            assert record.image_id == index + 1
            assert profile_failures == ()

        profile = plugin.user_profiles.get("10001")
        assert profile is not None
        assert [path.stem for path in profile.image_paths] == ["4"]

        summary_results = plugin.history_command(event, "")
        summary = await anext(summary_results)
        assert "当前共保存 4 张成功图片" in summary

        detail_results = plugin.history_command(event, "4")
        detail = await anext(detail_results)
        detail_text = next(part.text for part in detail if hasattr(part, "text"))
        assert "图库编号：4" in detail_text
        assert "请求用户（QQ：99999）" in detail_text
        assert "第 4 张用户画像" in detail_text

        random_results = plugin.history_command(event, "随机 4")
        random_chain = await anext(random_results)
        assert len(random_chain) == 1
        grid_path = random_chain[0].path
        with image_module.open(grid_path) as grid:
            assert grid.size == (1536, 1536)

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
