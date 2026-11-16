from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import mimetypes
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .errors import ImageAPIError
from .gif_support import GifOptions, first_frame_png, prepare_gif
from .image_ping import ImagePingResult


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
            "image/gif": ".gif",
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
        user_agent: str = "AstrBot-ImageGenerator/0.1.4",
        edit_image_field: str = "image",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = max(10.0, timeout_seconds)
        self.max_response_bytes = max(1, max_response_mb) * 1024 * 1024
        self.user_agent = user_agent
        self.edit_image_field = (
            edit_image_field if edit_image_field in {"image", "image[]"} else "image"
        )  # noqa: E501

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    def _timeout(self, override: float | None = None) -> httpx.Timeout:
        total = self.timeout_seconds if override is None else max(5.0, override)
        return httpx.Timeout(total, connect=min(20.0, total))

    async def ping(
        self, *, timeout_seconds: int = 10, additional_models: tuple[str, ...] = ()
    ) -> ImagePingResult:
        """Probe the authenticated model list without submitting a generation."""
        timeout_seconds = min(60, max(1, timeout_seconds))
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._ping_models(timeout_seconds, additional_models), timeout_seconds
            )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            result = ImagePingResult(
                "error", f"检测超时（上限 {timeout_seconds} 秒），请检查网络或稍后重试。"
            )
        except (httpx.InvalidURL, ValueError):
            result = ImagePingResult("error", "生图 API 地址或请求配置无效，请检查设置。")
        except httpx.HTTPError:
            # URLs, response bodies and exception details can contain credentials.
            result = ImagePingResult(
                "error", "连接生图 API 失败，请检查网络、域名、代理及 TLS 证书。"
            )
        return replace(result, elapsed_ms=round((time.monotonic() - started) * 1000))

    async def _ping_models(
        self, timeout_seconds: int, additional_models: tuple[str, ...]
    ) -> ImagePingResult:
        async with (
            httpx.AsyncClient(
                headers=self._headers(),
                timeout=httpx.Timeout(timeout_seconds, connect=min(5, timeout_seconds)),
                follow_redirects=False,
            ) as client,
            client.stream("GET", f"{self.base_url}/models") as response,
        ):
            status = response.status_code
            if not 200 <= status < 300:
                level = "error"
                if status in {401, 403}:
                    message = "鉴权失败或无访问权限，请检查生图 Key、分组和 User-Agent。"
                elif status == 429:
                    message = "上游限流或额度不足，请稍后重试并检查账户额度。"
                elif status in {404, 405}:
                    level = "warning"
                    message = (
                        "接口已响应，但不支持模型列表检测；"
                        "请核对 API 地址，鉴权与模型权限尚未确认。"
                    )
                elif 300 <= status < 400:
                    level = "warning"
                    message = "接口返回重定向，检测未跟随跳转；请将 API Base URL 配置为最终地址。"
                elif status >= 500:
                    message = "生图 API 上游服务异常，请稍后重试。"
                else:
                    message = "模型列表接口拒绝了请求，请检查 API 地址与网关配置。"
                return ImagePingResult(level, message, http_status=status)

            # Limit the accepted model-list payload, including decoded response bytes.
            content = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                content.extend(chunk)
                if len(content) > 1024 * 1024:
                    return ImagePingResult(
                        "warning", "模型列表响应超过 1MB，无法完成检测。", http_status=status
                    )

        try:
            payload = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            payload = None
        rows = payload.get("data") if isinstance(payload, dict) else None
        if (
            not isinstance(rows, list)
            or payload.get("error")
            or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows)
        ):
            return ImagePingResult(
                "warning",
                "接口已响应，但未返回标准模型列表；"
                "请检查 API 地址或网关兼容性，鉴权与模型权限尚未确认。",
                http_status=status,
            )
        available = {row["id"] for row in rows}
        models = {model: model in available for model in (self.model, *additional_models)}
        listed = all(models.values())
        return ImagePingResult(
            "ok" if listed else "warning",
            "模型列表接口访问成功，当前配置的模型均已列出。"
            if listed
            else "模型列表接口访问成功，但部分配置模型未列出；这不能单独判定实际生图失败。",
            http_status=status,
            models=models,
        )

    @staticmethod
    def _error_detail(response: httpx.Response, *, operation: str = "", path: str = "") -> str:
        try:
            payload = response.json()
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(error, dict):
                upstream = str(error.get("message") or error.get("code") or error)
            else:
                upstream = str(error)
        except Exception:
            upstream = response.text

        parts = []
        if operation:
            parts.append(f"operation={operation}")
        if path:
            parts.append(f"path={path}")
        parts.append(f"http_status={response.status_code}")
        request_id = next(
            (
                response.headers.get(name)
                for name in ("x-request-id", "request-id", "x-trace-id", "cf-ray")
                if response.headers.get(name)
            ),
            "",
        )
        if request_id:
            parts.append(f"request_id={request_id[:160]}")
        parts.append(f"upstream={upstream[:800]}")
        return " ".join(parts)[:1200]

    @classmethod
    def _http_error(cls, response: httpx.Response, *, operation: str, path: str) -> ImageAPIError:
        status = response.status_code
        detail = cls._error_detail(response, operation=operation, path=path)
        if status in {401, 403}:
            message = "UUAPI 鉴权失败，请检查 API Key、分组和 User-Agent 设置。"
        elif status == 429:
            message = "UUAPI 当前限流或额度不足，请稍后重试。"
        elif status in {400, 422}:
            message = f"{operation}拒绝了请求，可能是参数不兼容或内容未通过上游审核。"
        elif status >= 500:
            message = f"{operation}上游服务暂时异常（HTTP {status}），请稍后重试。"
        else:
            message = f"{operation}返回错误（HTTP {status}）。"
        return ImageAPIError(message, detail=detail)

    async def _post_json(
        self,
        path: str,
        payload: dict[str, object],
        *,
        timeout: float | None = None,
        operation: str = "生图服务",
    ) -> dict[str, object]:
        configured_timeout = self.timeout_seconds if timeout is None else max(5.0, timeout)
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                headers=self._headers(),
                timeout=self._timeout(timeout),
                follow_redirects=True,
            ) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - started
            raise ImageAPIError(
                f"{operation}响应超时（已等待约 {elapsed:.0f} 秒，"
                f"配置上限 {configured_timeout:.0f} 秒）。",
                detail=(
                    f"operation={operation} path={path} elapsed={elapsed:.2f}s "
                    f"limit={configured_timeout:.2f}s error={type(exc).__name__}: {exc}"
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise ImageAPIError(
                f"无法连接{operation}，请检查 UUAPI 地址和网络。",
                detail=f"operation={operation} path={path} error={type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise self._http_error(response, operation=operation, path=path)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ImageAPIError(
                f"{operation}返回了无法解析的响应。", detail=response.text[:500]
            ) from exc
        if not isinstance(data, dict):
            raise ImageAPIError(f"{operation}返回格式不正确。")
        return data

    async def generate(
        self, prompt: str, *, size: str, quality: str, animation: GifOptions | None = None
    ) -> ImageResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        if quality:
            payload["quality"] = quality
        if animation is not None:
            payload.update(animation.request_fields())
        data = await self._post_json(
            "/images/generations", payload, timeout=animation.timeout_seconds if animation else None
        )
        result = await self._parse_image_response(data)
        if animation is not None:
            content = await asyncio.to_thread(
                prepare_gif, result.content, options=animation, operation="文生动图"
            )
            result = replace(result, content=content)
        return result

    @staticmethod
    def _safe_reference_label(label: str) -> str:
        value = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", str(label))
        value = re.sub(r"\s+", "_", value).strip(" ._")
        return value[:48] or "character"

    @classmethod
    def source_from_paths(
        cls,
        paths: tuple[Path, ...],
        *,
        labels: tuple[str, ...] = (),
    ) -> list[tuple[str, bytes, str]]:
        sources: list[tuple[str, bytes, str]] = []
        for index, path in enumerate(paths, start=1):
            content = path.read_bytes()
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            suffix = Path(path.name).suffix.lower()
            if content.startswith((b"GIF87a", b"GIF89a")):
                content = first_frame_png(content)
                media_type, suffix = "image/png", ".png"
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                suffix = ".png"
            if index <= len(labels):
                safe_label = cls._safe_reference_label(labels[index - 1])
                filename = f"reference_{index:02d}_{safe_label}{suffix}"
            else:
                # Preserve the legacy generic name for callers that do not
                # provide an explicit image-to-character mapping.
                filename = f"reference_{index}{suffix}"
            sources.append((filename, content, media_type))
        return sources

    async def edit(
        self,
        prompt: str,
        sources: list[tuple[str, bytes, str]],
        *,
        size: str,
        quality: str,
        animation: GifOptions | None = None,
    ) -> ImageResponse:
        """Edit with in-memory image sources: (filename, content, media_type)."""

        if not sources:
            return await self.generate(prompt, size=size, quality=quality, animation=animation)
        form_data: dict[str, str] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        if quality:
            form_data["quality"] = quality
        if animation is not None:
            form_data.update({key: str(value) for key, value in animation.request_fields().items()})
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            (self.edit_image_field, source) for source in sources
        ]
        configured_timeout = animation.timeout_seconds if animation else self.timeout_seconds
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                headers=self._headers(),
                timeout=self._timeout(configured_timeout),
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/images/edits", data=form_data, files=files
                )
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - started
            raise ImageAPIError(
                f"参考图生图响应超时（已等待约 {elapsed:.0f} 秒，"
                f"配置上限 {configured_timeout:.0f} 秒）。",
                detail=(
                    f"operation=参考图生图 path=/images/edits elapsed={elapsed:.2f}s "
                    f"limit={configured_timeout:.2f}s error={type(exc).__name__}: {exc}"
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise ImageAPIError(
                "无法连接参考图生图服务。",
                detail=f"operation=参考图生图 path=/images/edits error={type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise self._http_error(
                response,
                operation="参考图生图",
                path="/images/edits",
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ImageAPIError("参考图生图返回了无法解析的响应。") from exc
        if not isinstance(payload, dict):
            raise ImageAPIError("参考图生图返回格式不正确。")
        result = await self._parse_image_response(payload)
        if animation is not None:
            content = await asyncio.to_thread(
                prepare_gif, result.content, options=animation, operation="参考图生动图"
            )
            result = replace(result, content=content)
        return result

    async def chat_completion(self, model: str, prompt: str, *, timeout_seconds: float = 45) -> str:
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = await self._post_json(
            "/chat/completions",
            payload,
            timeout=timeout_seconds,
            operation="安全审核服务",
        )
        return self._parse_chat_content(data)

    async def chat_completion_vision(
        self,
        model: str,
        prompt: str,
        image_bytes: bytes,
        media_type: str,
        *,
        timeout_seconds: float = 60,
    ) -> str:
        data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        data = await self._post_json(
            "/chat/completions",
            payload,
            timeout=timeout_seconds,
            operation="识图评价服务",
        )
        return self._parse_chat_content(data)

    @staticmethod
    def _parse_chat_content(data: dict[str, object]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageAPIError("模型返回格式不正确。") from exc
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
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout(min(self.timeout_seconds, 60)), follow_redirects=True
                ) as client,
                client.stream("GET", url) as response,
            ):
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
        if content.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        raise ImageAPIError("生图服务返回的内容不是支持的 PNG/JPEG/WebP/GIF 图片。")
