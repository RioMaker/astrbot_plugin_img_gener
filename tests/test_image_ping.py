from __future__ import annotations

import asyncio
import importlib
import sys

import httpx
import pytest
from astrbot_plugin_img_gener.image_client import OpenAICompatibleImageClient
from test_main_background import _install_astrbot_stubs


@pytest.fixture
def transport(monkeypatch):
    requests = []
    original = httpx.AsyncClient

    def install(handler):
        async def respond(request):
            requests.append(request)
            return await handler(request)

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
        )

    return requests, install


def client():
    return OpenAICompatibleImageClient(
        base_url="https://images.example/v1", api_key="private-key", user_agent="test-agent"
    )


def test_ping_uses_only_authenticated_get_and_checks_both_models(transport):
    requests, install = transport

    async def respond(request):
        return httpx.Response(200, json={"data": [{"id": "gpt-image-2"}, {"id": "gif-model"}]})

    install(respond)
    result = asyncio.run(client().ping(additional_models=("gif-model",)))
    assert result.status == "ok"
    assert result.models == {"gpt-image-2": True, "gif-model": True}
    assert result.elapsed_ms >= 0
    assert "HTTP 200" in result.report() and "不生成图片" in result.report()
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://images.example/v1/models"
    assert request.method == "GET" and request.content == b""
    assert request.headers["Authorization"] == "Bearer private-key"
    assert request.headers["User-Agent"] == "test-agent"


@pytest.mark.parametrize(
    "code,status,message",
    [
        (401, "error", "鉴权"),
        (403, "error", "权限"),
        (429, "error", "限流"),
        (502, "error", "上游"),
        (400, "error", "拒绝"),
        (404, "warning", "尚未确认"),
        (405, "warning", "尚未确认"),
        (302, "warning", "重定向"),
    ],
)
def test_ping_http_failures_are_diagnostic_and_never_echo_upstream(
    transport, code, status, message
):
    requests, install = transport

    async def respond(request):
        return httpx.Response(
            code,
            headers={"Location": "https://elsewhere.example/private-key"},
            json={"error": {"message": "private-key"}},
        )

    install(respond)
    result = asyncio.run(client().ping())
    assert result.status == status and result.http_status == code
    assert message in result.message
    assert "private-key" not in result.report()
    assert "elsewhere" not in result.report()
    assert len(requests) == 1  # No retry or redirect, especially no fallback generation.


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"data": None},
        {"data": [{}]},
        {"data": ["model"]},
        {"data": [], "error": "secret"},
    ],
)
def test_nonstandard_model_list_cannot_report_success(transport, payload):
    _, install = transport

    async def respond(request):
        return httpx.Response(200, json=payload)

    install(respond)
    result = asyncio.run(client().ping())
    assert result.status == "warning"
    assert result.models == {}
    assert "尚未确认" in result.message


@pytest.mark.parametrize(
    "body",
    [b"<html>login private-key</html>", b"x" * (1024 * 1024 + 1)],
    ids=["html-login", "over-one-mb"],
)
def test_ping_rejects_html_and_oversized_response(transport, body):
    _, install = transport

    async def respond(request):
        return httpx.Response(200, content=body)

    install(respond)
    result = asyncio.run(client().ping())
    assert result.status == "warning" and result.http_status == 200
    assert "private-key" not in result.report()


def test_unlisted_model_is_inconclusive_not_a_generation_failure(transport):
    _, install = transport

    async def respond(request):
        return httpx.Response(200, json={"data": [{"id": "gpt-image-2"}]})

    install(respond)
    result = asyncio.run(client().ping(additional_models=("gif-model",)))
    assert result.status == "warning"
    assert result.models == {"gpt-image-2": True, "gif-model": False}
    assert "不能单独判定" in result.message


@pytest.mark.parametrize(
    "error,message",
    [(httpx.ReadTimeout, "超时"), (httpx.ConnectError, "连接"), (httpx.InvalidURL, "配置")],
)
def test_network_errors_never_echo_credentials(transport, error, message):
    _, install = transport

    async def respond(request):
        raise error("private-key at https://sensitive.example")

    install(respond)
    result = asyncio.run(client().ping())
    assert result.status == "error" and message in result.message
    assert "private-key" not in result.report() and "sensitive.example" not in result.report()


def test_total_deadline_cancels_slow_stream_and_closes_it(transport):
    _, install = transport
    closed = []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"data":['
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def respond(request):
        return httpx.Response(200, stream=SlowStream())

    install(respond)
    result = asyncio.run(client().ping(timeout_seconds=1))
    assert result.status == "error" and "超时" in result.message
    assert closed == [True]


@pytest.fixture
def plugin():
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    instance = object.__new__(module.GPTImageGeneratorPlugin)
    instance.config = {
        "api": {
            "api_key": "private-key",
            "base_url": "https://images.example",
            "ping_timeout_seconds": 3,
        },
        "animation": {"model": "gif-model"},
    }
    instance._ping_running = False
    instance._ping_last_started = None
    return instance


def test_probe_reads_saved_settings_and_does_not_need_generation_stores(
    plugin, transport, monkeypatch
):
    requests, install = transport
    monkeypatch.setenv("PING_TEST_KEY", "resolved-private-key")
    plugin.config["api"]["api_key"] = "$PING_TEST_KEY"

    async def respond(request):
        assert request.extensions["timeout"]["read"] == 3
        return httpx.Response(200, json={"data": [{"id": "gpt-image-2"}, {"id": "gif-model"}]})

    install(respond)
    result = asyncio.run(plugin.ping_image_service())
    assert result.status == "ok"
    assert requests[0].headers["Authorization"] == "Bearer resolved-private-key"
    assert plugin._ping_running is False


@pytest.mark.parametrize(
    "api", [{}, {"api_key": "$UNSET_PING_KEY"}, {"api_key": "x", "base_url": "wrong"}]
)
def test_bad_configuration_fails_without_request(plugin, transport, monkeypatch, api):
    requests, _ = transport
    monkeypatch.delenv("UNSET_PING_KEY", raising=False)
    plugin.config["api"] = api
    result = asyncio.run(plugin.ping_image_service())
    assert result.status == "error"
    assert not requests and plugin._ping_running is False


def test_shared_probe_rejects_overlap_and_cooldown_then_uses_new_config(plugin, transport):
    requests, install = transport

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def respond(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"data": [{"id": "gpt-image-2"}, {"id": "gif-model"}]})

        install(respond)
        first = asyncio.create_task(plugin.ping_image_service())
        await entered.wait()
        plugin.config["api"]["api_key"] = "next-key"
        assert "进行中" in (await plugin.ping_image_service()).message
        release.set()
        assert (await first).status == "ok"
        assert "频繁" in (await plugin.ping_image_service()).message
        assert len(requests) == 1
        plugin._ping_last_started -= 11
        assert (await plugin.ping_image_service()).status == "ok"
        assert requests[0].headers["Authorization"] == "Bearer private-key"
        assert requests[1].headers["Authorization"] == "Bearer next-key"

    asyncio.run(scenario())


def test_cancelled_probe_releases_inflight_guard(plugin, transport):
    _, install = transport

    async def scenario():
        entered = asyncio.Event()

        async def respond(request):
            entered.set()
            await asyncio.Event().wait()

        install(respond)
        task = asyncio.create_task(plugin.ping_image_service())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not plugin._ping_running

    asyncio.run(scenario())
