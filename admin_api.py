from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from functools import wraps

from PIL import Image, ImageOps, UnidentifiedImageError

from .admin_config import AdminConfig
from .admin_gallery import AdminError, AdminGallery, serialize

PLUGIN_NAME = "astrbot_plugin_img_gener"


class AdminAPI:
    def __init__(self, plugin, web):
        self.plugin = plugin
        self.web = web
        self.gallery = AdminGallery(plugin.gallery_store, plugin.user_profiles.database_path)
        self.settings = AdminConfig(plugin)
        self.handlers = []
        self.closed = False
        self._image_slots = asyncio.Semaphore(3)

    def register(self):
        routes = (
            ("images", self.images, "GET"),
            ("users", self.users, "GET"),
            ("detail", self.detail, "GET"),
            ("preview", self.preview, "GET"),
            ("download", self.download, "GET"),
            ("images/update", self.update, "POST"),
            ("images/delete", self.delete, "POST"),
            ("images/restore", self.restore, "POST"),
            ("settings", self.get_settings, "GET"),
            ("settings/save", self.save_settings, "POST"),
            ("references/upload", self.upload_reference, "POST"),
        )
        for endpoint, handler, method in routes:
            wrapped = self._guard(handler)
            self.plugin.context.register_web_api(
                f"/{PLUGIN_NAME}/admin/{endpoint}", wrapped, [method], "生图图库管理"
            )
            self.handlers.append(wrapped)

    def close(self):
        self.closed = True
        routes = getattr(self.plugin.context, "registered_web_apis", None)
        if isinstance(routes, list):
            routes[:] = [item for item in routes if item[1] not in self.handlers]

    def _guard(self, handler):
        @wraps(handler)
        async def guarded(*args, **kwargs):
            # Only Dashboard identity supplied by AstrBot is trusted. Never read a
            # caller-provided QQ, username header, token in the URL, or chat admin list.
            if self.closed:
                return self.web.error_response("插件已停用，请刷新页面。", status_code=503)
            if not self.web.request.username:
                return self.web.error_response("请先登录 AstrBot 管理面板。", status_code=401)
            if self.web.request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                return self.web.error_response("请通过 AstrBot 管理面板访问。", status_code=403)
            try:
                return await handler(*args, **kwargs)
            except AdminError as exc:
                return self.web.error_response(str(exc), status_code=exc.status)
            except Exception:
                # Do not echo paths, credentials, raw payloads or exception messages.
                from astrbot.api import logger

                logger.error("[img_gener] admin operation failed: %s", handler.__name__)
                return self.web.error_response("操作失败，请检查服务状态后重试。", status_code=500)

        return guarded

    def response(self, data):
        return self.web.json_response(data, headers={"Cache-Control": "no-store"})

    async def payload(self):
        if (self.web.request.content_type or "").split(";", 1)[0].strip() != "application/json":
            raise AdminError("请使用 JSON 请求。", 415)
        raw = await self.web.request.body()
        if len(raw) > 512 * 1024:
            raise AdminError("请求内容过大。", 413)
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise AdminError("请求须为有效 JSON。") from exc
        if not isinstance(data, dict):
            raise AdminError("请求须为 JSON 对象。")
        return data

    def require_idle(self):
        if self.settings.busy():
            raise AdminError("仍有生图任务运行，请完成后再修改或删除历史。", 409)

    async def images(self):
        return self.response(self.gallery.list_images(self.web.request.query))

    async def users(self):
        return self.response(self.gallery.list_users(self.web.request.query))

    async def detail(self):
        return self.response(serialize(self.gallery.get(self.web.request.query.get("image_id"))))

    def _preview(self, record, large):
        path = self.gallery.image_path(record)
        try:
            with Image.open(path) as original:
                if original.width * original.height > 40_000_000:
                    raise AdminError("图片像素过大，请下载原图查看。")
                original.thumbnail((1400, 1400) if large else (400, 400))
                image = ImageOps.exif_transpose(original).convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=82)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise AdminError("图片损坏或无法读取。", 422) from exc
        return {
            "data_url": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        }

    async def preview(self):
        record = self.gallery.get(self.web.request.query.get("image_id"))
        async with self._image_slots:
            result = await asyncio.to_thread(
                self._preview, record, self.web.request.query.get("large") == "1"
            )
        return self.response(result)

    async def download(self):
        record = self.gallery.get(self.web.request.query.get("image_id"))
        path = self.gallery.image_path(record)
        return self.web.file_response(
            path,
            filename=f"{record.image_id}{path.suffix}",
            content_type=record.media_type,
            headers={"Cache-Control": "no-store"},
        )

    async def update(self):
        data = await self.payload()
        self.require_idle()
        return self.response(self.gallery.update(data))

    async def delete(self):
        data = await self.payload()
        self.require_idle()
        return self.response(self.gallery.change_deleted(data))

    async def restore(self):
        data = await self.payload()
        self.require_idle()
        return self.response(self.gallery.change_deleted(data, restore=True))

    async def get_settings(self):
        return self.response(self.settings.read())

    async def save_settings(self):
        data = await self.payload()
        return self.response(self.settings.save(data))

    async def upload_reference(self):
        files = await self.web.request.files()
        upload = files.get("file")
        if not isinstance(upload, self.web.PluginUploadFile):
            raise AdminError("请选择一张 PNG、JPEG 或 WebP 图片。")
        maximum = (
            min(50, self.plugin._int("references", "max_image_size_mb", default=12, minimum=1))
            * 1024
            * 1024
        )
        content = await upload.read(maximum + 1)
        if not content or len(content) > maximum:
            raise AdminError("参考图为空或超过大小上限。", 413)
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise AdminError("只支持 PNG、JPEG 或 WebP 图片。")
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[image.format]
                if image.width * image.height > 40_000_000:
                    raise AdminError("参考图像素过大。")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise AdminError("参考图损坏或格式不支持。") from exc
        # Generated names only; never use the submitted upload filename as a path.
        folder = self.plugin.data_dir / "admin_references"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid.uuid4().hex}{extension}"
        path.write_bytes(content)
        return self.response({"path": path.relative_to(self.plugin.data_dir).as_posix()})


def register_admin_api(plugin):
    if not callable(getattr(plugin.context, "register_web_api", None)):
        return None
    try:
        from astrbot.api import web
    except ImportError:
        from astrbot.api import logger

        logger.warning("[img_gener] 管理页需要支持 astrbot.api.web 的新版 AstrBot 和 WebUI。")
        return None
    api = AdminAPI(plugin, web)
    api.register()
    return api
