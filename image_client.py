from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .errors import ImageAPIError


@dataclass(frozen=True, slots=True)
class ImageResponse:
    content: bytes
    media_type: str
    revised_prompt: str = ""

    @property
    def extension(self) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/png": ".png",
        }.get(self.media_type, ".png")


class OpenAICompatibleImageClient:
    """Minimal async Images API client tuned for UUAPI compatibility."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gpt-image-2",
        timeout_seconds: float = 300,
        max_response_mb: int = 30,
        user_agent: str = "AstrBot-ImageGenerator/0.1.0",
        edit_image_field: str = "image",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = max(10.0, timeout_seconds)
        self.max_response_bytes = max(1, max_response_mb) * 1024 * 1024
        self.user_agent = user_agent
        self.edit_image_field = edit_image_field if edit_image_field in {"image", "image[]"} else "image"  # noqa: E501

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    def _timeout(self, override: float | None = None) -> httpx.Timeout:
        total = self.timeout_seconds if override is None else max(5.0, override)
        return httpx.Timeout(total, connect=min(20.0, total))

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or error)[:800]
            return str(error)[:800]
        except Exception:
            return response.text[:800]

    async def _post_json(
        self, path: str, payload: dict[str, object], *, timeout: float | None = None
    ) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(
                headers=self._headers(), timeout=self._timeout(timeout), follow_redirects=True
            ) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
        except httpx.TimeoutException as exc:
            raise ImageAPIError("生图服务响应超时，请稍后重试。", detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ImageAPIError("无法连接生图服务，请检查 UUAPI 地址和网络。", detail=str(exc)) from exc  # noqa: E501
        if response.status_code >= 400:
            detail = self._error_detail(response)
            if response.status_code in {401, 403}:
                message = "UUAPI 鉴权失败，请检查 API Key、分组和 User-Agent 设置。"
            elif response.status_code == 429:
                message = "UUAPI 当前限流或额度不足，请稍后重试。"
            elif response.status_code in {400, 422}:
                message = "生图服务拒绝了请求，可能是参数不兼容或内容未通过上游审核。"
            else:
                message = f"生图服务返回错误（HTTP {response.status_code}）。"
            raise ImageAPIError(message, detail=detail)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ImageAPIError("生图服务返回了无法解析的响应。", detail=response.text[:500]) from exc  # noqa: E501
        if not isinstance(data, dict):
            raise ImageAPIError("生图服务返回格式不正确。")
        return data

    async def generate(self, prompt: str, *, size: str, quality: str) -> ImageResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        if quality:
            payload["quality"] = quality
        data = await self._post_json("/images/generations", payload)
        return await self._parse_image_response(data)

    async def edit(
        self,
        prompt: str,
        image_paths: tuple[Path, ...],
        *,
        size: str,
        quality: str,
    ) -> ImageResponse:
        if not image_paths:
            return await self.generate(prompt, size=size, quality=quality)
        form_data: dict[str, str] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        if quality:
            form_data["quality"] = quality
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for path in image_paths:
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append((self.edit_image_field, (path.name, path.read_bytes(), media_type)))
        try:
            async with httpx.AsyncClient(
                headers=self._headers(), timeout=self._timeout(), follow_redirects=True
            ) as client:
                response = await client.post(
                    f"{self.base_url}/images/edits", data=form_data, files=files
                )
        except httpx.TimeoutException as exc:
            raise ImageAPIError("参考图生图响应超时，请稍后重试。", detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ImageAPIError("无法连接参考图生图服务。", detail=str(exc)) from exc
        if response.status_code >= 400:
            raise ImageAPIError(
                "参考图生图请求失败；请检查图片格式、上传字段和上游内容审核。",
                detail=self._error_detail(response),
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ImageAPIError("参考图生图返回了无法解析的响应。") from exc
        if not isinstance(payload, dict):
            raise ImageAPIError("参考图生图返回格式不正确。")
        return await self._parse_image_response(payload)

    async def chat_completion(
        self, model: str, prompt: str, *, timeout_seconds: float = 45
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = await self._post_json(
            "/chat/completions", payload, timeout=timeout_seconds
        )
        try:
            content = data["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageAPIError("审核模型返回格式不正确。") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        return str(content or "")

    async def _parse_image_response(self, payload: dict[str, object]) -> ImageResponse:
        data = payload.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ImageAPIError("生图服务没有返回图片数据。")
        first = data[0]
        revised_prompt = str(first.get("revised_prompt") or "")
        encoded = first.get("b64_json") or first.get("image_base64")
        if encoded:
            content = self._decode_base64(str(encoded))
            return ImageResponse(content, self._detect_media_type(content), revised_prompt)
        url = str(first.get("url") or "").strip()
        if url:
            content, media_type = await self._download_image(url)
            return ImageResponse(content, media_type, revised_prompt)
        raise ImageAPIError("生图服务响应中没有 b64_json 或图片 URL。")

    def _decode_base64(self, value: str) -> bytes:
        encoded = value.strip()
        if encoded.startswith("data:"):
            _, _, encoded = encoded.partition(",")
        if len(encoded) > (self.max_response_bytes * 4 // 3) + 4096:
            raise ImageAPIError("生图服务返回的图片超过大小限制。")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageAPIError("生图服务返回的 Base64 图片无效。") from exc
        if not content or len(content) > self.max_response_bytes:
            raise ImageAPIError("生图服务返回的图片为空或超过大小限制。")
        return content

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ImageAPIError("生图服务返回了不安全的图片地址。")
        hostname = parsed.hostname.casefold()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
            raise ImageAPIError("生图服务返回了本地网络图片地址，已拒绝下载。")
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not ip.is_global:
            raise ImageAPIError("生图服务返回了私有网络图片地址，已拒绝下载。")

    async def _download_image(self, url: str) -> tuple[bytes, str]:
        self._validate_download_url(url)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(min(self.timeout_seconds, 60)), follow_redirects=True
            ) as client, client.stream("GET", url) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.max_response_bytes:
                        raise ImageAPIError("下载的生成图片超过大小限制。")
                    chunks.append(chunk)
                content = b"".join(chunks)
                header_type = response.headers.get("content-type", "").split(";", 1)[0]
        except ImageAPIError:
            raise
        except httpx.HTTPError as exc:
            raise ImageAPIError("无法下载生图服务返回的临时图片。", detail=str(exc)) from exc
        if not content:
            raise ImageAPIError("生图服务返回的临时图片为空。")
        detected = self._detect_media_type(content)
        return content, detected if detected.startswith("image/") else header_type

    @staticmethod
    def _detect_media_type(content: bytes) -> str:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return "image/webp"
        raise ImageAPIError("生图服务返回的内容不是支持的 PNG/JPEG/WebP 图片。")

