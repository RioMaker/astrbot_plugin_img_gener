from __future__ import annotations

import asyncio
import base64

import httpx
import pytest
from astrbot_plugin_img_gener.errors import ImageAPIError
from astrbot_plugin_img_gener.image_client import OpenAICompatibleImageClient

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def _client() -> OpenAICompatibleImageClient:
    return OpenAICompatibleImageClient(
        base_url="https://uuapi.cc/v1",
        api_key="test-key",
        max_response_mb=1,
    )


def test_parse_b64_json_response() -> None:
    payload = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
    result = asyncio.run(_client()._parse_image_response(payload))
    assert result.content == PNG
    assert result.media_type == "image/png"


def test_parse_data_url_response() -> None:
    encoded = base64.b64encode(PNG).decode()
    payload = {"data": [{"b64_json": f"data:image/png;base64,{encoded}"}]}
    result = asyncio.run(_client()._parse_image_response(payload))
    assert result.extension == ".png"


def test_rejects_non_image_base64() -> None:
    payload = {"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]}
    with pytest.raises(ImageAPIError):
        asyncio.run(_client()._parse_image_response(payload))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/image.png",
        "http://127.0.0.1/image.png",
        "http://192.168.1.2/image.png",
    ],
)
def test_rejects_unsafe_download_urls(url: str) -> None:
    with pytest.raises(ImageAPIError):
        _client()._validate_download_url(url)


def test_generation_payload_matches_uuapi() -> None:
    client = _client()
    captured = {}

    async def fake_post(path, payload, *, timeout=None):
        captured["path"] = path
        captured["payload"] = payload
        return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}

    client._post_json = fake_post  # type: ignore[method-assign]
    result = asyncio.run(
        client.generate("一只橘猫", size="1024x1024", quality="medium")
    )
    assert result.content == PNG
    assert captured["path"] == "/images/generations"
    assert captured["payload"] == {
        "model": "gpt-image-2",
        "prompt": "一只橘猫",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
        "quality": "medium",
    }


def test_edit_uploads_numbered_reference_files(tmp_path, monkeypatch) -> None:
    client = _client()
    first = tmp_path / "coco.png"
    second = tmp_path / "mushroom.jpg"
    first.write_bytes(PNG)
    second.write_bytes(b"\xff\xd8\xff" + b"payload")
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}

    class FakeClient:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files") or []
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    sources = client.source_from_paths((first, second))
    asyncio.run(
        client.edit("两个角色", sources, size="1024x1024", quality="auto")
    )
    names = [name for _, (name, _, _) in captured["files"]]
    assert names == ["reference_1.png", "reference_2.jpg"]


def test_edit_uploads_single_contact_sheet_source(tmp_path, monkeypatch) -> None:
    client = _client()
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}

    class FakeClient:
        def __init__(self, **kwargs):
            del kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files") or []
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    asyncio.run(
        client.edit(
            "两个角色",
            [("contact_sheet.png", PNG, "image/png")],
            size="1024x1024",
            quality="auto",
        )
    )
    names = [name for _, (name, _, _) in captured["files"]]
    assert names == ["contact_sheet.png"]
