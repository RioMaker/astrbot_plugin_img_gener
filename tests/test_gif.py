from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest
from astrbot_plugin_img_gener.admin_api import AdminAPI
from astrbot_plugin_img_gener.admin_gallery import AdminError, AdminGallery
from astrbot_plugin_img_gener.contact_sheet import build_gallery_grid
from astrbot_plugin_img_gener.errors import ConfigurationError, ImageAPIError
from astrbot_plugin_img_gener.gallery import GalleryStore
from astrbot_plugin_img_gener.gallery_render import build_personal_gallery
from astrbot_plugin_img_gener.gif_support import GifOptions, first_frame_png
from astrbot_plugin_img_gener.image_client import ImageResponse, OpenAICompatibleImageClient
from astrbot_plugin_img_gener.user_profiles import UserProfileStore
from PIL import Image, ImageDraw
from test_main_background import _install_astrbot_stubs


@pytest.fixture
def gif_content():
    frames = [Image.new("RGB", (64, 64), (255 - index * 10, index * 10, 0)) for index in range(16)]
    output = io.BytesIO()
    frames[0].save(output, "GIF", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return output.getvalue()


def archive_image(store, content, media_type="image/gif"):
    return store.add_image(
        content,
        extension=".gif" if media_type == "image/gif" else ".png",
        media_type=media_type,
        requester_qq="123",
        requester_name="动图用户",
        group_id="456",
        prompt="橘猫眨眼并挥手，无缝循环",
        size="512x512",
        quality="low",
        duration_seconds=2,
    )


@pytest.mark.parametrize("operation", ["generate", "edit", "edit_empty"])
def test_gif_request_wire_format_and_original_bytes(gif_content, monkeypatch, operation):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(gif_content).decode()}]}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw)
    )
    client = OpenAICompatibleImageClient(
        base_url="https://example.test/v1", api_key="test", timeout_seconds=90
    )
    kwargs = dict(size="816x816", quality="high", animation=GifOptions(timeout_seconds=720))
    if operation == "generate":
        result = asyncio.run(client.generate("猫挥手", **kwargs))
    else:
        sources = (
            [("ref.png", first_frame_png(gif_content), "image/png")] if operation == "edit" else []
        )
        result = asyncio.run(client.edit("猫挥手", sources, **kwargs))
    assert result.content == gif_content
    assert result.extension == ".gif" and result.media_type == "image/gif"
    request = requests[0]
    assert request.extensions["timeout"]["read"] == 720
    assert request.extensions["timeout"]["connect"] == 20
    if operation == "edit":
        assert request.url.path == "/v1/images/edits"
        body = request.content.decode("latin1")
        for key, value in GifOptions().request_fields().items():
            assert f'name="{key}"\r\n\r\n{value}\r\n' in body
        assert 'name="image"; filename="ref.png"' in body
        assert 'name="timeout_seconds"' not in body
    else:
        assert request.url.path == "/v1/images/generations"
        body = json.loads(request.content)
        assert GifOptions().request_fields().items() <= body.items()
        assert body["response_format"] == "b64_json" and body["n"] == 1
        assert "timeout_seconds" not in body
    assert client.model == "gpt-image-2"  # GIF must not change the still-image model.
    asyncio.run(client.generate("静态图", size="816x816", quality="low"))
    assert requests[-1].extensions["timeout"]["read"] == 90


@pytest.mark.parametrize("operation", ["generate", "edit"])
def test_gif_timeout_reports_effective_override(gif_content, monkeypatch, operation):
    def timed_out(request):
        assert request.extensions["timeout"]["read"] == 720
        raise httpx.ReadTimeout("test timeout", request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(timed_out), **kw),
    )
    client = OpenAICompatibleImageClient(base_url="https://example.test/v1", api_key="test")
    args = ("猫挥手",)
    if operation == "edit":
        args += ([("ref.png", first_frame_png(gif_content), "image/png")],)
    with pytest.raises(ImageAPIError) as error:
        asyncio.run(
            getattr(client, operation)(
                *args, size="512x512", quality="low", animation=GifOptions(timeout_seconds=720)
            )
        )
    assert "配置上限 720 秒" in error.value.public_message
    assert "limit=720.00s" in error.value.detail


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 300),
        ({"api": {"timeout_seconds": 600}}, 600),
        ({"api": {"timeout_seconds": 600}, "animation": {"timeout_seconds": 0}}, 600),
        ({"api": {"timeout_seconds": 600}, "animation": {"timeout_seconds": 720}}, 720),
    ],
)
def test_gif_timeout_inherits_existing_config_unless_overridden(config, expected):
    assert GifOptions.from_config(config).timeout_seconds == expected


def test_long_gif_reservation_survives_original_timeout_after_reload(tmp_path, monkeypatch):
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    monkeypatch.setattr(module, "get_astrbot_plugin_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(module, "register_admin_api", lambda plugin: None)
    plugin = module.GPTImageGeneratorPlugin(object(), {"animation": {"timeout_seconds": 720}})

    async def scenario():
        first = await plugin.rate_limiter.acquire("first", "group", now=10000)
        assert first.allowed
        # The previous 600-second reservation must not expire during a 720-second GIF request.
        second = await plugin.rate_limiter.acquire("second", "group", now=10700)
        assert not second.allowed and second.code == "GROUP_BUSY"
        await plugin.rate_limiter.cancel(first.lease_id)

    asyncio.run(scenario())


def test_gif_download_url_preserves_media_type_and_bytes(gif_content, monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=gif_content, headers={"content-type": "application/octet-stream"}
                )
            ),
            **kw,
        ),
    )
    client = OpenAICompatibleImageClient(base_url="https://example.test/v1", api_key="test")
    result = asyncio.run(
        client._parse_image_response({"data": [{"url": "https://example.test/out"}]})
    )
    assert result.content == gif_content and result.extension == ".gif"


@pytest.mark.parametrize("operation", ["generate", "edit"])
@pytest.mark.parametrize(
    ("kind", "reason", "message"),
    [
        ("png", "unexpected_format", "PNG"),
        ("single", "single_frame", "只有 1 帧"),
        ("broken", "decode_error", "文件损坏"),
        ("pixels", "pixel_limit", "总像素上限 4000 万"),
    ],
)
def test_gif_failure_explains_returned_file_and_request(
    gif_content, monkeypatch, operation, kind, reason, message
):
    if kind == "png":
        content = first_frame_png(gif_content)
    elif kind == "single":
        single = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(single, "GIF")
        content = single.getvalue()
    elif kind == "pixels":
        # Keep the 16 real frames but declare a 2K canvas: validation must stop before decoding it.
        content = gif_content[:6] + (2048).to_bytes(2, "little") * 2 + gif_content[10:]
    else:
        content = b"GIF89a truncated"
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"data": [{"b64_json": base64.b64encode(content).decode()}]}
                )
            ),
            **kw,
        ),
    )
    client = OpenAICompatibleImageClient(
        base_url="https://example.test/v1", api_key="private-test-key"
    )
    args = ("private-test-prompt",)
    if operation == "edit":
        args += ([("ref.png", first_frame_png(gif_content), "image/png")],)
    with pytest.raises(ImageAPIError) as error:
        asyncio.run(
            getattr(client, operation)(*args, size="512x512", quality="low", animation=GifOptions())
        )
    assert message in error.value.public_message
    detail = error.value.detail
    assert f"reason={reason}" in detail
    assert f"bytes={len(content)}" in detail
    assert "model=gpt-image-2.6 requested_size=512x512 requested_frames=16" in detail
    expected_operation = "参考图生动图" if operation == "edit" else "文生动图"
    assert f"operation={expected_operation}" in detail
    if kind == "pixels":
        assert "format=GIF size=2048x2048 frames=16" in detail
    assert "private-test-key" not in detail and "private-test-prompt" not in detail


@pytest.mark.parametrize(
    "values",
    [
        {"frame_count": 1},
        {"frame_count": 17},
        {"frame_count": True},
        {"frame_duration_ms": 19},
        {"frame_duration_ms": 2001},
        {"frame_duration_ms": 41},
        {"size": "auto"},
        {"quality": "bad"},
        {"model": " "},
        {"timeout_seconds": 1},
        {"timeout_seconds": -1},
        {"timeout_seconds": True},
        {"timeout_seconds": 1000001},
    ],
)
def test_gif_settings_reject_invalid_values(values):
    with pytest.raises(ConfigurationError):
        GifOptions.from_config({"animation": values})


def test_shared_ids_filter_recycle_and_animated_previews(tmp_path, gif_content):
    store = GalleryStore(tmp_path / "gallery")
    profiles = UserProfileStore(tmp_path / "profiles.sqlite3")
    first = archive_image(store, first_frame_png(gif_content), "image/png")
    second = archive_image(store, gif_content)
    third = archive_image(store, first_frame_png(gif_content), "image/png")
    assert [first.image_id, second.image_id, third.image_id] == [1, 2, 3]
    assert GalleryStore(store.root_dir).get(2).image_path.name == "2.gif"
    admin = AdminGallery(store, profiles.database_path)
    for kind, expected in [("all", [3, 2, 1]), ("gif", [2]), ("static", [3, 1])]:
        data = admin.list_images({"media_kind": kind})
        assert [item["image_id"] for item in data["items"]] == expected
    assert admin.list_images({"media_kind": "gif", "requester_qq": "123"})["total"] == 1
    assert admin.list_images({"media_kind": "gif", "search": "无缝循环"})["total"] == 1
    with pytest.raises(AdminError):
        admin.list_images({"media_kind": "gif' OR 1=1 --"})
    api = object.__new__(AdminAPI)
    api.gallery = admin
    for large in (False, True):
        url = api._preview(second, large)["data_url"]
        assert url.startswith("data:image/gif;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == gif_content
        with Image.open(io.BytesIO(decoded)) as animation:
            assert animation.n_frames == 16 and animation.info["loop"] == 0
            for index in range(16):
                animation.seek(index)
                assert animation.info["duration"] == 40
    profiles.add_reference("123", gallery_id=2, image_path=second.image_path)
    admin.change_deleted({"items": [{"image_id": 2, "revision": 0}]})
    assert admin.list_images({"media_kind": "gif"})["total"] == 0
    assert admin.list_images({"media_kind": "gif", "deleted": "1"})["total"] == 1
    assert profiles.get("123").image_paths == ()
    admin.change_deleted({"items": [{"image_id": 2, "revision": 1}]}, restore=True)
    assert store.get(2).image_path.read_bytes() == gif_content
    assert archive_image(store, gif_content).image_id == 4


def test_gallery_uses_first_frame_and_gif_badge(tmp_path, gif_content, monkeypatch):
    store = GalleryStore(tmp_path / "gallery")
    record = archive_image(store, gif_content)
    texts = []
    original = ImageDraw.ImageDraw.text

    def track(self, xy, text, *args, **kwargs):
        texts.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", track)
    personal = build_personal_gallery(
        (record,), user_name="用户", subtitle="动图首帧", command="/画廊"
    )
    random_grid = build_gallery_grid((record.image_path,))
    for output, point in [(personal, (500, 646)), (random_grid, (384, 384))]:
        with Image.open(io.BytesIO(output.content)) as picture:
            assert picture.format == "PNG" and picture.n_frames == 1
            assert picture.getpixel(point) == (255, 0, 0)
    assert texts.count("GIF") == 2
    assert record.image_path.read_bytes() == gif_content


@pytest.fixture
def gif_plugin(tmp_path):
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {"generation": {"enable_evaluation": False}}
    plugin._active_jobs, plugin._background_tasks = {}, set()
    plugin._next_job_id = 0
    plugin.references = module.CharacterReferenceManager([], data_dir=tmp_path, plugin_dir=tmp_path)
    plugin.user_profiles = UserProfileStore(tmp_path / "profiles.sqlite3")
    plugin.gallery_store = GalleryStore(tmp_path / "gallery")
    plugin.output_store = module.OutputStore(tmp_path / "outputs")
    plugin.history_store = module.GenerationHistoryStore(tmp_path / "history.sqlite3")
    plugin.moderator = module.PromptModerator(enabled=False)
    plugin._is_allowed = lambda event: True
    plugin.rate_limiter = module.PersistentRateLimiter(
        tmp_path / "rate.sqlite3",
        module.LimitConfig(
            user_cooldown_seconds=0, group_cooldown_seconds=0, user_max_generations=1
        ),
    )
    return module, plugin


class Event:
    unified_msg_origin = "qq:GroupMessage:456"

    def __init__(self, messages=()):
        self.messages, self.sent, self.stopped = messages, [], False

    def get_sender_id(self):
        return "123"

    def get_group_id(self):
        return "456"

    def get_sender_name(self):
        return "动图用户"

    def get_platform_name(self):
        return "qq"

    def get_messages(self):
        return self.messages

    def get_message_str(self):
        return "猫挥手"

    def is_admin(self):
        return False

    def plain_result(self, value):
        return value

    def chain_result(self, value):
        return value

    def stop_event(self):
        self.stopped = True

    async def send(self, value):
        self.sent.append(value)


def test_gif_command_queue_freezes_options_and_history_returns_original(gif_plugin, gif_content):
    module, plugin = gif_plugin
    plugin.config["animation"] = {"timeout_seconds": 720}
    captured = []

    class Client:
        async def generate(self, prompt, **kwargs):
            captured.append(kwargs)
            return ImageResponse(gif_content, "image/gif")

    plugin._client = lambda: Client()

    async def scenario():
        event = Event()
        result = [item async for item in plugin.gif_command(event, "猫挥手，无缝循环")]
        assert event.stopped and "16 帧 · 40ms/帧" in result[0] and "512x512" in result[0]
        assert next(iter(plugin._active_jobs.values())).source == "/生动图"
        plugin.config["animation"] = {"frame_count": 8, "size": "1024x1024"}
        await asyncio.gather(*plugin._background_tasks)
        assert captured[0]["animation"] == GifOptions(timeout_seconds=720)
        with pytest.raises(FrozenInstanceError):
            captured[0]["animation"].frame_count = 8
        record = plugin.gallery_store.get(1)
        assert record.is_animated and record.image_path.read_bytes() == gif_content
        response = [item async for item in plugin.history_command(event, "1")][0]
        assert any("GIF 动图" in getattr(item, "text", "") for item in response)
        assert Path(response[-1].path).read_bytes() == gif_content
        # The same ordinary image quota is now consumed by this GIF.
        assert not (await plugin.rate_limiter.acquire("qq:123", "qq:456")).allowed
        assert plugin._active_jobs == {}

    asyncio.run(scenario())


def test_gif_reference_is_static_first_frame_and_portrait_can_be_reused(
    gif_plugin,
    gif_content,
    tmp_path,
):
    module, plugin = gif_plugin
    record = archive_image(plugin.gallery_store, gif_content)
    plugin.user_profiles.add_reference(
        "123", gallery_id=record.image_id, image_path=record.image_path
    )
    refs = plugin.user_profiles.character_references(("123",))
    resolved = plugin.references.with_characters(refs).resolve("123", ["123"])
    sources = OpenAICompatibleImageClient.source_from_paths(resolved.image_paths, labels=("123",))
    assert sources[0][0].endswith(".png") and sources[0][2] == "image/png"
    with Image.open(io.BytesIO(sources[0][1])) as image:
        assert image.convert("RGB").getpixel((32, 32)) == (255, 0, 0)
    event = Event([module.Comp.Reply(chain=[module.Comp.Image(str(record.image_path))])])
    quoted = asyncio.run(plugin._capture_quoted_reference_sources(event))
    assert quoted[0][2] == "image/png"


@pytest.mark.parametrize("failure", ["local_review", "remote_review", "upstream"])
def test_gif_failures_do_not_archive_or_consume_generation_quota(gif_plugin, failure):
    module, plugin = gif_plugin
    calls = []
    if failure == "local_review":
        plugin.moderator = module.PromptModerator(enabled=False, blocked_terms=["禁止词"])
        prompt = "禁止词"
    else:
        prompt = "猫挥手"
    if failure == "remote_review":
        plugin.moderator = module.PromptModerator(enabled=True)

        async def reject(*args):
            return '{"allowed": false, "reason": "测试拒绝", "code": "test_rejection"}'

        plugin._remote_review = reject

    class Client:
        async def generate(self, prompt, **kwargs):
            calls.append(kwargs)
            raise ImageAPIError("测试上游故障")

    plugin._client = lambda: Client()

    async def scenario():
        event = Event()
        replies = [item async for item in plugin.gif_command(event, prompt)]
        if failure == "local_review":
            assert "未通过" in replies[0] and not plugin._background_tasks
        await asyncio.gather(*plugin._background_tasks)
        assert bool(calls) == (failure == "upstream")
        assert plugin.gallery_store.count() == 0 and not plugin._active_jobs
        rate = await plugin.rate_limiter.acquire("qq:123", "qq:456")
        assert rate.allowed
        await plugin.rate_limiter.cancel(rate.lease_id)

    asyncio.run(scenario())


def test_gif_generation_with_quoted_reference_uses_edit_and_retains_snapshot(
    gif_plugin,
    gif_content,
    tmp_path,
):
    module, plugin = gif_plugin
    path = tmp_path / "quoted.png"
    reference = first_frame_png(gif_content)
    path.write_bytes(reference)
    calls = []

    class Client:
        async def edit(self, prompt, sources, **kwargs):
            calls.append((sources, kwargs))
            return ImageResponse(gif_content, "image/gif")

        async def generate(self, *args, **kwargs):
            raise AssertionError("a quoted reference must use edits")

    plugin._client = lambda: Client()

    async def scenario():
        event = Event([module.Comp.Reply(chain=[module.Comp.Image(str(path))])])
        await anext(plugin.gif_command(event, "参考图里的猫挥手"))
        path.write_bytes(b"replaced after submission")
        await asyncio.gather(*plugin._background_tasks)
        assert calls[0][0][0][1] == reference
        assert calls[0][1]["animation"] == GifOptions()
        assert plugin.gallery_store.get(1).is_animated

    asyncio.run(scenario())
