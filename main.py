from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:
    from .character_references import CharacterReferenceManager, ResolvedReferences
    from .config import (
        as_bool,
        as_float,
        as_int,
        as_string_list,
        get_config,
        normalize_base_url,
        resolve_secret,
    )
    from .contact_sheet import build_contact_sheet_from_sources, build_gallery_grid
    from .errors import ConfigurationError, ImageGeneratorError
    from .gallery import GalleryImage, GalleryStore
    from .generation_history import GenerationHistoryRecord, GenerationHistoryStore
    from .image_client import OpenAICompatibleImageClient
    from .moderation import PromptModerator
    from .rate_limiter import LimitConfig, PersistentRateLimiter
    from .storage import OutputStore
    from .user_profiles import UserProfile, UserProfileStore
except ImportError:  # pragma: no cover - direct script-style import fallback.
    from character_references import CharacterReferenceManager, ResolvedReferences
    from config import (
        as_bool,
        as_float,
        as_int,
        as_string_list,
        get_config,
        normalize_base_url,
        resolve_secret,
    )
    from contact_sheet import build_contact_sheet_from_sources, build_gallery_grid
    from errors import ConfigurationError, ImageGeneratorError
    from gallery import GalleryImage, GalleryStore
    from generation_history import GenerationHistoryRecord, GenerationHistoryStore
    from image_client import OpenAICompatibleImageClient
    from moderation import PromptModerator
    from rate_limiter import LimitConfig, PersistentRateLimiter
    from storage import OutputStore
    from user_profiles import UserProfile, UserProfileStore


PLUGIN_NAME = "astrbot_plugin_img_gener"
DEFAULT_START_MESSAGE = (
    "已收到生图请求，正在检查额度并进行安全审核；审核通过后将立即生成，请稍候。"
)
DEFAULT_IMAGE_SIZE = "816x816"
DEFAULT_ALLOWED_SIZES = (
    "auto",
    "816x816",
    "640x1024",
    "1024x640",
    "1024x1024",
    "1024x1536",
    "1536x1024",
)
LEGACY_ALLOWED_SIZES = ("auto", "1024x1024", "1024x1536", "1536x1024")


@dataclass(slots=True)
class ActiveGenerationJob:
    job_id: int
    user_id: str
    group_id: str | None
    source: str
    started_at: float
    size: str = DEFAULT_IMAGE_SIZE
    stage: str = "等待调度"
    finalized: bool = False
    history_persisted: bool = False


class GPTImageGeneratorPlugin(Star):
    """Audited, rate-limited GPT Image 2 tool for AstrBot."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context, config)
        self.config: dict[str, Any] = config or {}
        self.plugin_dir = Path(__file__).resolve().parent
        self.data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)

        limit_config = LimitConfig(
            user_cooldown_seconds=self._int(
                "limits", "user_cooldown_seconds", default=60, minimum=0
            ),
            user_attempt_window_seconds=self._int(
                "limits", "user_attempt_window_seconds", default=600, minimum=1
            ),
            user_max_attempts=self._int(
                "limits", "user_max_attempts", default=3, minimum=0
            ),
            user_generation_window_seconds=self._int(
                "limits", "user_generation_window_seconds", default=3600, minimum=1
            ),
            user_max_generations=self._int(
                "limits", "user_max_generations", default=5, minimum=0
            ),
            group_cooldown_seconds=self._int(
                "limits", "group_cooldown_seconds", default=15, minimum=0
            ),
            group_generation_window_seconds=self._int(
                "limits", "group_generation_window_seconds", default=3600, minimum=1
            ),
            group_max_generations=self._int(
                "limits", "group_max_generations", default=20, minimum=0
            ),
            global_max_concurrent=self._int(
                "limits", "global_max_concurrent", default=2, minimum=1
            ),
            group_max_concurrent=self._int(
                "limits", "group_max_concurrent", default=1, minimum=1
            ),
            reservation_ttl_seconds=max(
                300,
                self._int("api", "timeout_seconds", default=300, minimum=10) * 2,
            ),
        )
        self.rate_limiter = PersistentRateLimiter(
            self.data_dir / "rate_limits.sqlite3", limit_config
        )
        self.history_store = GenerationHistoryStore(
            self.data_dir / "generation_history.sqlite3"
        )
        self._recovered_history_jobs = self.history_store.recover_unfinished()
        self.gallery_store = GalleryStore(self.data_dir / "gallery")
        self.user_profiles = UserProfileStore(
            self.data_dir / "user_profiles.sqlite3"
        )
        self.references = CharacterReferenceManager(
            self._get("references", "characters", default=[]),
            data_dir=self.data_dir,
            plugin_dir=self.plugin_dir,
            max_total_images=self._int(
                "references", "max_reference_images", default=4, minimum=1
            ),
            max_image_size_mb=self._int(
                "references", "max_image_size_mb", default=12, minimum=1
            ),
            per_character_limit=self._int(
                "references", "multi_character_image_limit", default=1, minimum=1
            ),
        )
        self.moderator = PromptModerator(
            enabled=self._bool("safety", "enabled", default=True),
            fail_closed=self._bool("safety", "fail_closed", default=True),
            blocked_terms=as_string_list(
                self._get("safety", "blocked_terms", default=[])
            ),
            custom_policy=str(
                self._get("safety", "custom_policy", default="") or ""
            ),
            max_prompt_chars=self._int(
                "safety", "max_prompt_chars", default=8000, minimum=200
            ),
        )
        self.output_store = OutputStore(
            self.data_dir / "outputs",
            retention_days=self._int(
                "storage", "retention_days", default=7, minimum=0
            ),
        )
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._active_jobs: dict[int, ActiveGenerationJob] = {}
        self._next_job_id = 0

    async def initialize(self) -> None:
        removed = self.output_store.cleanup()
        history_removed = self.history_store.cleanup(
            self._int("storage", "retention_days", default=7, minimum=0)
        )
        logger.info(
            "[img_gener] initialized model=%s base_url=%s references=%s "
            "cleaned=%s history_cleaned=%s recovered=%s",
            self._get("api", "model", default="gpt-image-2"),
            normalize_base_url(self._get("api", "base_url", default="https://uuapi.cc/v1")),
            len(self.references.characters),
            removed,
            history_removed,
            self._recovered_history_jobs,
        )
        logger.info(
            "[img_gener] permanent gallery images=%s",
            self.gallery_store.count(),
        )

    async def terminate(self) -> None:
        tasks = tuple(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_jobs.clear()

    def _get(self, *path: str, default: Any = None) -> Any:
        return get_config(self.config, *path, default=default)

    def _bool(self, *path: str, default: bool = False) -> bool:
        return as_bool(self._get(*path, default=default), default)

    def _int(
        self, *path: str, default: int, minimum: int | None = None
    ) -> int:
        return as_int(self._get(*path, default=default), default, minimum=minimum)

    def _float(
        self, *path: str, default: float, minimum: float | None = None
    ) -> float:
        return as_float(self._get(*path, default=default), default, minimum=minimum)

    def _generation_start_message(self, size: str) -> str:
        message = str(
            self._get(
                "generation",
                "start_message",
                default=DEFAULT_START_MESSAGE,
            )
            or ""
        ).strip()
        return f"{message or DEFAULT_START_MESSAGE}\n本次尺寸：{size}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0.0, seconds)
        if total < 60:
            return f"{total:.1f} 秒"
        minutes, remaining = divmod(total, 60)
        return f"{int(minutes)} 分 {remaining:.1f} 秒"

    @staticmethod
    def _format_window(seconds: int) -> str:
        if seconds >= 3600 and seconds % 3600 == 0:
            return f"{seconds // 3600} 小时"
        if seconds >= 60 and seconds % 60 == 0:
            return f"{seconds // 60} 分钟"
        return f"{seconds} 秒"

    @staticmethod
    def _format_quota(used: int, maximum: int) -> str:
        return f"{used}/不限" if maximum <= 0 else f"{used}/{maximum}"

    @staticmethod
    def _mention(event: AstrMessageEvent) -> Any:
        sender_id = str(event.get_sender_id() or "").strip()
        if not sender_id:
            return None
        at_cls = getattr(Comp, "At", None)
        if at_cls is not None:
            try:
                return at_cls(qq=sender_id)
            except Exception:
                pass
        return Comp.Plain(f"@{sender_id} ")

    def _mention_chain(self, event: AstrMessageEvent, text: str) -> Any:
        parts: list[Any] = []
        mention = self._mention(event)
        if mention is not None:
            parts.append(mention)
        if text:
            parts.append(Comp.Plain(text))
        return event.chain_result(parts)

    def _completion_chain(
        self, event: AstrMessageEvent, text: str, image_path: Path
    ) -> Any:
        parts: list[Any] = []
        mention = self._mention(event)
        if mention is not None:
            parts.append(mention)
        if text:
            parts.append(Comp.Plain(text))
        parts.append(Comp.Image.fromFileSystem(str(image_path)))
        return event.chain_result(parts)

    def _register_job(
        self,
        event: AstrMessageEvent,
        *,
        source: str,
        size: str,
        prompt: str = "",
        quality: str | None = None,
    ) -> tuple[int, float]:
        user_id, group_id = self._identity(event)
        started_at = time.monotonic()
        history_store = getattr(self, "history_store", None)
        job_id = 0
        history_persisted = False
        if history_store is not None:
            try:
                job_id = history_store.create_job(
                    user_id=user_id,
                    group_id=group_id,
                    source=source,
                    prompt=prompt,
                    size=size,
                    quality=str(quality or "auto"),
                )
                history_persisted = job_id > 0
            except Exception:
                logger.exception("[img_gener] failed to create history record")
        if job_id <= 0:
            self._next_job_id = int(getattr(self, "_next_job_id", 0)) + 1
            job_id = self._next_job_id
        else:
            self._next_job_id = max(int(getattr(self, "_next_job_id", 0)), job_id)
        job = ActiveGenerationJob(
            job_id=job_id,
            user_id=user_id,
            group_id=group_id,
            source=source,
            started_at=started_at,
            size=size,
            history_persisted=history_persisted,
        )
        self._active_jobs[job.job_id] = job
        return job.job_id, started_at

    def _record_rejected_attempt(
        self,
        event: AstrMessageEvent,
        *,
        source: str,
        prompt: str,
        size: str,
        quality: str | None,
        stage: str,
        message: str,
    ) -> str:
        history_store = getattr(self, "history_store", None)
        if history_store is None:
            return message
        user_id, group_id = self._identity(event)
        try:
            job_id = history_store.create_job(
                user_id=user_id,
                group_id=group_id,
                source=source,
                prompt=prompt,
                size=size,
                quality=str(quality or "auto"),
            )
            history_store.finish_job(
                job_id,
                status="rejected",
                stage=stage,
                result_message=message,
            )
        except Exception:
            logger.exception("[img_gener] failed to record rejected attempt")
        return message

    def _set_job_stage(self, job_id: int | None, stage: str) -> None:
        if job_id is None or job_id not in self._active_jobs:
            return
        job = self._active_jobs[job_id]
        if job.finalized:
            return
        job.stage = stage
        history_store = getattr(self, "history_store", None)
        if history_store is not None and job.history_persisted:
            try:
                history_store.update_stage(job_id, stage)
            except Exception:
                logger.exception("[img_gener] failed to update history stage job=%s", job_id)

    def _finalize_job(
        self,
        job_id: int | None,
        *,
        status: str,
        stage: str,
        result_message: str = "",
        output_filename: str = "",
    ) -> None:
        if job_id is None:
            return
        job = self._active_jobs.get(job_id)
        if job is not None:
            if job.finalized:
                return
            job.stage = stage
            job.finalized = True
        history_store = getattr(self, "history_store", None)
        if history_store is not None and job is not None and job.history_persisted:
            try:
                history_store.finish_job(
                    job_id,
                    status=status,
                    stage=stage,
                    result_message=result_message,
                    output_filename=output_filename,
                )
            except Exception:
                logger.exception("[img_gener] failed to finalize history job=%s", job_id)

    def _finish_job(self, job_id: int | None) -> None:
        if job_id is not None:
            self._active_jobs.pop(job_id, None)

    def _active_job_snapshot(
        self, user_id: str, group_id: str | None
    ) -> dict[str, Any]:
        jobs = sorted(self._active_jobs.values(), key=lambda job: job.started_at)
        visible = [
            job
            for job in jobs
            if job.user_id == user_id
            or (group_id is not None and job.group_id == group_id)
        ]
        return {
            "global": len(jobs),
            "user": sum(job.user_id == user_id for job in jobs),
            "group": sum(
                group_id is not None and job.group_id == group_id for job in jobs
            ),
            "visible": visible,
        }

    @staticmethod
    def _remaining_text(value: int) -> str:
        return "不限" if value < 0 else f"{value} 次"

    @classmethod
    def _cooldown_text(cls, seconds: int) -> str:
        return "已结束，可立即调用" if seconds <= 0 else f"还需 {cls._format_duration(seconds)}"

    @staticmethod
    def _history_status_label(status: str) -> str:
        return {
            "queued": "等待中",
            "running": "处理中",
            "succeeded": "成功",
            "rejected": "未受理",
            "failed": "失败",
            "delivery_failed": "已生成但发送失败",
            "cancelled": "已取消",
            "interrupted": "已中断",
        }.get(status, status or "未知")

    def _recent_history(
        self, user_id: str, *, limit: int
    ) -> tuple[GenerationHistoryRecord, ...]:
        history_store = getattr(self, "history_store", None)
        if history_store is None:
            return ()
        try:
            return history_store.recent_for_user(
                user_id, limit=max(1, min(20, int(limit)))
            )
        except Exception:
            logger.exception("[img_gener] failed to query generation history")
            return ()

    def _history_lines(
        self, records: tuple[GenerationHistoryRecord, ...]
    ) -> list[str]:
        if not records:
            return ["• 暂无插件已受理的历史生图记录"]
        lines: list[str] = []
        for record in records:
            created = time.strftime("%m-%d %H:%M:%S", time.localtime(record.created_at))
            lines.append(
                f"• #{record.job_id:04d}｜{created}｜"
                f"{self._history_status_label(record.status)}｜{record.stage}｜"
                f"{record.size}｜{record.quality or 'auto'}"
            )
            if record.result_message:
                lines.append(f"  结果：{record.result_message[:240]}")
            if record.prompt:
                lines.append(f"  提示词：{record.prompt[:240]}")
        return lines

    async def _history_report(
        self, event: AstrMessageEvent, *, limit: int = 10
    ) -> str:
        user_id, _group_id = self._identity(event)
        records = self._recent_history(user_id, limit=limit)
        return "\n".join(["🎨 我的生图历史", *self._history_lines(records)])

    async def _status_report(
        self,
        event: AstrMessageEvent,
        *,
        include_history: bool,
        history_limit: int,
    ) -> str:
        user_id, group_id = self._identity(event)
        pool = self._active_job_snapshot(user_id, group_id)
        records = self._recent_history(user_id, limit=history_limit)
        lines = ["🎨 生图真实状态", "", "【结论】"]
        if pool["user"]:
            lines.append(f"• 你的任务确实在队列或处理中：{pool['user']} 个")
        elif records:
            latest = records[0]
            lines.append(
                f"• 当前没有你的任务在队列或处理中；最近任务 #{latest.job_id:04d} "
                f"为“{self._history_status_label(latest.status)}”（{latest.stage}）。"
            )
            if latest.result_message:
                lines.append(f"• 最近结果：{latest.result_message[:240]}")
        else:
            lines.append(
                "• 当前没有你的任务，且插件没有查到已受理记录；"
                "不能声称正在生成，可能是生图工具没有被实际调用或请求在入队前被拒绝。"
            )

        lines.extend(
            [
                "",
                "【队列与进度】",
                f"• 内存任务池：全局 {pool['global']} 个、我的 {pool['user']} 个"
                + (f"、本群 {pool['group']} 个" if group_id else ""),
            ]
        )
        visible_jobs: list[ActiveGenerationJob] = pool["visible"]
        if visible_jobs:
            now = time.monotonic()
            for job in visible_jobs[:8]:
                owner = "我的" if job.user_id == user_id else "本群其他用户"
                elapsed = self._format_duration(now - job.started_at)
                lines.append(
                    f"• #{job.job_id:04d}｜{owner}｜{job.source}｜{job.size}｜"
                    f"{job.stage}｜已用 {elapsed}"
                )
        else:
            lines.append("• 没有可见的等待或处理任务")

        try:
            status = await self.rate_limiter.status(user_id, group_id)
            limits = self.rate_limiter.limits
        except Exception:
            logger.exception("[img_gener] failed to query rate-limit status")
            status = None
            limits = None

        lines.extend(["", "【余量与冷却】"])
        if status is None or limits is None:
            lines.append("• SQLite 限流状态查询失败，当前无法可靠给出余量或冷却时间。")
        else:
            user_attempts = int(status.get("user_attempts", 0))
            user_attempts_remaining = int(
                status.get(
                    "user_attempts_remaining",
                    -1
                    if limits.user_max_attempts <= 0
                    else max(0, limits.user_max_attempts - user_attempts),
                )
            )
            user_generations = int(status.get("user_generations", 0))
            user_pending = int(status.get("user_pending", pool["user"]))
            user_generations_remaining = int(
                status.get(
                    "user_generations_remaining",
                    -1
                    if limits.user_max_generations <= 0
                    else max(
                        0,
                        limits.user_max_generations
                        - user_generations
                        - user_pending,
                    ),
                )
            )
            lines.extend(
                [
                    f"• 我的调用余量：{self._remaining_text(user_attempts_remaining)}"
                    f"（已用 {self._format_quota(user_attempts, limits.user_max_attempts)}；"
                    f"{self._format_window(limits.user_attempt_window_seconds)}窗口）",
                    f"• 我的调用冷却："
                    f"{self._cooldown_text(int(status.get('user_cooldown_remaining', 0)))}",
                    f"• 我的生图余量：{self._remaining_text(user_generations_remaining)}"
                    f"（成功 {user_generations} 次，处理中预留 {user_pending} 次；"
                    f"{self._format_window(limits.user_generation_window_seconds)}窗口）",
                ]
            )
            user_reset = int(status.get("user_generation_reset_after", 0))
            attempt_reset = int(status.get("user_attempt_reset_after", 0))
            if user_generations_remaining == 0 and user_reset > 0:
                lines.append(
                    f"• 我的生图窗口最早约 {self._format_duration(user_reset)} 后释放 1 次"
                )
            if user_attempts_remaining == 0 and attempt_reset > 0:
                lines.append(
                    f"• 我的调用窗口最早约 {self._format_duration(attempt_reset)} 后释放 1 次"
                )

            if group_id:
                group_generations = int(status.get("group_generations", 0))
                group_pending = int(status.get("group_pending", pool["group"]))
                group_generations_remaining = int(
                    status.get(
                        "group_generations_remaining",
                        -1
                        if limits.group_max_generations <= 0
                        else max(
                            0,
                            limits.group_max_generations
                            - group_generations
                            - group_pending,
                        ),
                    )
                )
                lines.extend(
                    [
                        f"• 本群生图余量：{self._remaining_text(group_generations_remaining)}"
                        f"（成功 {group_generations} 次，处理中预留 {group_pending} 次；"
                        f"{self._format_window(limits.group_generation_window_seconds)}窗口）",
                        f"• 本群调用冷却："
                        f"{self._cooldown_text(int(status.get('group_cooldown_remaining', 0)))}",
                    ]
                )
                group_reset = int(status.get("group_generation_reset_after", 0))
                if group_generations_remaining == 0 and group_reset > 0:
                    lines.append(
                        f"• 本群生图窗口最早约 {self._format_duration(group_reset)} 后释放 1 次"
                    )

            lines.append(
                f"• API 并发预留：{int(status.get('global_inflight', 0))}/"
                f"{limits.global_max_concurrent}"
            )
            admin_check = getattr(event, "is_admin", None)
            is_admin = False
            if callable(admin_check):
                try:
                    is_admin = bool(admin_check())
                except Exception:
                    pass
            if is_admin and self._bool(
                "limits", "admins_bypass_limits", default=True
            ):
                lines.append("• 当前管理员配置为跳过用户/群配额，但仍受全局并发限制。")

        gallery_store = getattr(self, "gallery_store", None)
        if gallery_store is not None:
            try:
                lines.extend(
                    ["", "【长期图库】", f"• 已永久归档 {gallery_store.count()} 张成功图片"]
                )
            except Exception:
                logger.exception("[img_gener] failed to count gallery images")
                lines.extend(["", "【长期图库】", "• 图库计数暂时不可用"])
        if include_history:
            lines.extend(["", "【我的最近记录】", *self._history_lines(records)])
        return "\n".join(lines)

    def _client(self) -> OpenAICompatibleImageClient:
        return OpenAICompatibleImageClient(
            base_url=normalize_base_url(
                self._get("api", "base_url", default="https://uuapi.cc/v1")
            ),
            api_key=resolve_secret(self._get("api", "api_key", default="")),
            model=str(self._get("api", "model", default="gpt-image-2") or "gpt-image-2"),
            timeout_seconds=self._float(
                "api", "timeout_seconds", default=300, minimum=10
            ),
            max_response_mb=self._int(
                "api", "max_response_mb", default=30, minimum=1
            ),
            user_agent=str(
                self._get(
                    "api",
                    "user_agent",
                    default="AstrBot-ImageGenerator/0.1.4",
                )
                or "AstrBot-ImageGenerator/0.1.4"
            ),
            edit_image_field=str(
                self._get("api", "edit_image_field", default="image") or "image"
            ),
        )

    def _review_client(self) -> OpenAICompatibleImageClient:
        review_model = str(
            self._get("safety", "review_model", default="") or ""
        ).strip()
        return OpenAICompatibleImageClient(
            base_url=normalize_base_url(
                self._get(
                    "safety",
                    "review_base_url",
                    default="https://uuapi.shop/v1",
                )
            ),
            api_key=resolve_secret(
                self._get("safety", "review_api_key", default="")
            ),
            model=review_model or "review-model-not-configured",
            timeout_seconds=self._float(
                "safety", "review_timeout_seconds", default=45, minimum=5
            ),
            max_response_mb=1,
            user_agent="AstrBot-ImageGenerator/0.1.4",
        )

    @staticmethod
    def _identity(event: AstrMessageEvent) -> tuple[str, str | None]:
        platform = str(event.get_platform_name() or "unknown")
        user_id = f"{platform}:{event.get_sender_id()}"
        raw_group = event.get_group_id()
        group_id = f"{platform}:{raw_group}" if raw_group else None
        return user_id, group_id

    @staticmethod
    def _raw_get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _mentioned_qqs(self, event: AstrMessageEvent) -> tuple[str, ...]:
        """Extract real At targets while excluding the bot itself and @all."""

        values: list[str] = []
        getter = getattr(event, "get_messages", None)
        try:
            chain = getter() if callable(getter) else event.message_obj.message
        except Exception:
            chain = ()
        at_cls = getattr(Comp, "At", None)
        for segment in chain or ():
            is_at = bool(at_cls is not None and isinstance(segment, at_cls))
            is_at = is_at or type(segment).__name__.casefold() == "at"
            if not is_at:
                continue
            raw = next(
                (
                    self._raw_get(segment, key)
                    for key in ("qq", "user_id", "target")
                    if self._raw_get(segment, key) is not None
                ),
                "",
            )
            value = UserProfileStore.normalize_qq(str(raw))
            if value and value.casefold() != "all":
                values.append(value)

        raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
        raw_chain = self._raw_get(raw_message, "message", ())
        if isinstance(raw_chain, list):
            for segment in raw_chain:
                if str(self._raw_get(segment, "type", "")).casefold() != "at":
                    continue
                data = self._raw_get(segment, "data", {})
                value = UserProfileStore.normalize_qq(
                    str(self._raw_get(data, "qq", ""))
                )
                if value and value.casefold() != "all":
                    values.append(value)

        self_ids: set[str] = set()
        self_getter = getattr(event, "get_self_id", None)
        if callable(self_getter):
            try:
                self_ids.add(str(self_getter() or ""))
            except Exception:
                pass
        for source in (
            getattr(event, "message_obj", None),
            raw_message,
            getattr(event, "bot", None),
        ):
            self_ids.add(str(self._raw_get(source, "self_id", "") or ""))
        self_ids.discard("")
        return tuple(
            value for value in dict.fromkeys(values) if value not in self_ids
        )

    async def _group_member_name(
        self, event: AstrMessageEvent, qq: str
    ) -> str:
        """Read a current group card/nickname and cap it to the alias limit."""

        sender_qq = UserProfileStore.normalize_qq(str(event.get_sender_id() or ""))
        if qq == sender_qq:
            getter = getattr(event, "get_sender_name", None)
            if callable(getter):
                try:
                    name = UserProfileStore.normalize_alias(str(getter() or ""))
                    if name:
                        return name
                except Exception:
                    pass
        group_id = str(event.get_group_id() or "").strip()
        client = getattr(event, "bot", None)
        call_action = getattr(client, "call_action", None)
        if group_id and callable(call_action):
            try:
                result = await call_action(
                    action="get_group_member_info",
                    group_id=int(group_id) if group_id.isdigit() else group_id,
                    user_id=int(qq) if qq.isdigit() else qq,
                    no_cache=True,
                )
                data = self._raw_get(result, "data") or result
                name = UserProfileStore.normalize_alias(
                    str(
                        self._raw_get(data, "card")
                        or self._raw_get(data, "nickname")
                        or ""
                    )
                )
                if name:
                    return name
            except Exception as exc:
                logger.warning(
                    "[img_gener] failed to query group member name qq=%s error=%s",
                    qq,
                    str(exc)[:300],
                )
        profile_store = getattr(self, "user_profiles", None)
        if profile_store is not None:
            profile = profile_store.get(qq)
            if profile is not None and profile.display_name:
                return profile.display_name
        return qq

    @staticmethod
    def _event_is_admin(event: AstrMessageEvent) -> bool:
        checker = getattr(event, "is_admin", None)
        try:
            return bool(checker()) if callable(checker) else False
        except Exception:
            return False

    def _portrait_reference_target(
        self, event: AstrMessageEvent, raw_target: str
    ) -> tuple[str | None, str | None]:
        profile_store = getattr(self, "user_profiles", None)
        if profile_store is None:
            return None, "用户画像档案暂时不可用。"
        sender_qq = profile_store.normalize_qq(str(event.get_sender_id() or ""))
        token = str(raw_target or "").strip()
        if not token or token.casefold() in {"self", "me"} or token in {
            "我",
            "自己",
            "本人",
            "我的",
        }:
            target_qq = sender_qq
        else:
            match = re.fullmatch(r"(?:qq:)?(\d{5,20})", token, re.I)
            if match is None:
                return None, "请使用 self、自己或 5～20 位 QQ 号指定用户。"
            target_qq = match.group(1)
        if not target_qq:
            return None, "无法识别当前用户的 QQ 号。"
        if target_qq != sender_qq and not self._event_is_admin(event):
            return None, "只有 AstrBot 管理员可以查看或修改他人的本体参考图。"
        return target_qq, None

    async def _portrait_reference_image(
        self, event: AstrMessageEvent, raw_target: str
    ) -> tuple[str, Path | None]:
        target_qq, error = self._portrait_reference_target(event, raw_target)
        if error or target_qq is None:
            return error or "无法识别目标用户。", None
        try:
            profile = self.user_profiles.get(target_qq)
        except Exception:
            logger.exception(
                "[img_gener] failed to read portrait reference qq=%s", target_qq
            )
            return "用户画像档案暂时不可用，请联系管理员查看日志。", None
        if profile is None or not profile.image_paths:
            return f"QQ {target_qq} 尚未设置本体参考图。", None
        display_name = await self._group_member_name(event, target_qq)
        return (
            f"QQ {target_qq}（{display_name}）当前唯一的本体参考图：",
            profile.image_paths[0],
        )

    @staticmethod
    def _self_portrait_requested(text: str) -> bool:
        compact = " ".join(str(text or "").split())
        patterns = (
            r"(?:画|绘制|生成|制作|做).{0,8}(?:我本人|我的画像|我的肖像|我的形象)",
            r"(?:画|绘制|生成)(?:一下|一张)?我(?:在|穿|站|坐|作为|成为|，|,|$)",
            r"(?:我的|本人)(?:画像|头像|肖像|形象|照片|立绘)",
            r"(?:以我|把我).{0,8}(?:主角|人物|画面|画进去)",
        )
        return any(re.search(pattern, compact) for pattern in patterns)

    async def _portrait_ambiguity_message(
        self,
        event: AstrMessageEvent,
        alias: str,
        candidates: tuple[UserProfile, ...],
    ) -> str:
        lines = [
            f"别名“{alias}”对应多个用户，当前没有受理生图任务。",
            "请先让用户用数字序号确认人物：",
        ]
        group_id = str(event.get_group_id() or "") or None
        for index, profile in enumerate(candidates, start=1):
            display_name = await self._group_member_name(event, profile.qq)
            try:
                self.user_profiles.ensure_profile(
                    profile.qq,
                    display_name=display_name,
                    group_id=group_id,
                )
            except Exception:
                logger.exception(
                    "[img_gener] failed to refresh ambiguous profile qq=%s",
                    profile.qq,
                )
            lines.append(f"{index}.（{profile.qq}）（{display_name}）")
        lines.append(
            "确认后再次调用 generate_image，并把对应 QQ 号放入 "
            "portrait_user_qqs；不要根据重名别名自行猜测。"
        )
        return "\n".join(lines)

    async def _resolve_portrait_subjects(
        self,
        event: AstrMessageEvent,
        text: str,
        requested: list[str] | None,
    ) -> tuple[tuple[str, ...], str | None]:
        profile_store = getattr(self, "user_profiles", None)
        if profile_store is None:
            return (), None
        sender_qq = profile_store.normalize_qq(str(event.get_sender_id() or ""))
        subjects: list[str] = []
        for raw in requested or ():
            token = str(raw or "").strip()
            if token.casefold() in {"self", "me"} or token in {"自己", "本人", "我"}:
                if sender_qq:
                    subjects.append(sender_qq)
                continue
            match = re.fullmatch(r"(?:qq:)?(\d{5,20})", token, re.I)
            if match:
                subjects.append(match.group(1))
                continue
            candidates = profile_store.alias_candidates(token)
            if len(candidates) > 1:
                return (), await self._portrait_ambiguity_message(
                    event, profile_store.normalize_alias(token), candidates
                )
            if len(candidates) == 1:
                subjects.append(candidates[0].qq)
                continue
            return (), f"没有找到用户画像别名“{token}”，请让用户使用 @ 或 QQ 号指定。"

        subjects.extend(self._mentioned_qqs(event))
        if not subjects and sender_qq and self._self_portrait_requested(text):
            subjects.append(sender_qq)

        if not subjects:
            for alias_match in profile_store.match_aliases(text):
                if len(alias_match.candidates) > 1:
                    return (), await self._portrait_ambiguity_message(
                        event, alias_match.alias, alias_match.candidates
                    )
                subjects.append(alias_match.candidates[0].qq)

        resolved = tuple(dict.fromkeys(profile_store.normalize_qq(item) for item in subjects))
        group_id = str(event.get_group_id() or "") or None
        try:
            for qq in resolved:
                display_name = await self._group_member_name(event, qq)
                profile_store.ensure_profile(
                    qq,
                    display_name=display_name,
                    group_id=group_id,
                )
        except Exception:
            logger.exception("[img_gener] failed to prepare user portrait profiles")
            return (), "用户画像档案暂时不可用，当前没有受理生图任务。"
        return resolved, None

    async def _archive_generated_image(
        self,
        event: AstrMessageEvent,
        generated: Any,
        *,
        prompt: str,
        size: str,
        quality: str,
        duration_seconds: float,
        job_id: int | None,
        portrait_user_qqs: tuple[str, ...],
    ) -> tuple[GalleryImage | None, tuple[str, ...]]:
        gallery_store = getattr(self, "gallery_store", None)
        if gallery_store is None:
            return None, portrait_user_qqs
        requester_qq = UserProfileStore.normalize_qq(
            str(event.get_sender_id() or "")
        )
        requester_name = await self._group_member_name(event, requester_qq)
        record = gallery_store.add_image(
            generated.content,
            extension=generated.extension,
            media_type=generated.media_type,
            requester_qq=requester_qq,
            requester_name=requester_name,
            group_id=str(event.get_group_id() or "") or None,
            prompt=prompt,
            size=size,
            quality=quality,
            duration_seconds=duration_seconds,
            job_id=job_id,
        )
        profile_store = getattr(self, "user_profiles", None)
        profile_failures: list[str] = []
        if profile_store is not None:
            for qq in portrait_user_qqs:
                try:
                    profile_store.add_reference(
                        qq,
                        gallery_id=record.image_id,
                        image_path=record.image_path,
                    )
                except Exception:
                    profile_failures.append(qq)
                    logger.exception(
                        "[img_gener] failed to update portrait profile qq=%s gallery=%s",
                        qq,
                        record.image_id,
                    )
        return record, tuple(profile_failures)

    @staticmethod
    def _event_message_text(event: AstrMessageEvent) -> str:
        """Read the user's original text instead of the LLM-rewritten tool args."""

        getter = getattr(event, "get_message_str", None)
        if callable(getter):
            try:
                return str(getter() or "").strip()
            except Exception:
                pass
        return str(getattr(event, "message_str", "") or "").strip()

    def _quoted_image_components(self, event: AstrMessageEvent) -> tuple[Any, ...]:
        """Return Image components embedded in the event's quoted Reply chain."""

        if not self._bool("references", "enable_quoted_images", default=True):
            return ()
        getter = getattr(event, "get_messages", None)
        try:
            chain = getter() if callable(getter) else event.message_obj.message
        except Exception:
            return ()
        reply_cls = getattr(Comp, "Reply", None)
        image_cls = getattr(Comp, "Image", None)
        if reply_cls is None or image_cls is None:
            return ()
        images: list[Any] = []
        for segment in chain or ():
            if not isinstance(segment, reply_cls):
                continue
            for quoted_segment in getattr(segment, "chain", None) or ():
                if isinstance(quoted_segment, image_cls):
                    images.append(quoted_segment)
        return tuple(images)

    def _event_image_components(self, event: AstrMessageEvent) -> tuple[Any, ...]:
        """Return direct and replied Image components for profile replacement."""

        getter = getattr(event, "get_messages", None)
        try:
            chain = getter() if callable(getter) else event.message_obj.message
        except Exception:
            return ()
        image_cls = getattr(Comp, "Image", None)
        reply_cls = getattr(Comp, "Reply", None)
        if image_cls is None:
            return ()
        images: list[Any] = []
        for segment in chain or ():
            if isinstance(segment, image_cls):
                images.append(segment)
                continue
            is_reply = bool(reply_cls is not None and isinstance(segment, reply_cls))
            is_reply = is_reply or type(segment).__name__.casefold() == "reply"
            if not is_reply:
                continue
            for quoted_segment in getattr(segment, "chain", None) or ():
                if isinstance(quoted_segment, image_cls):
                    images.append(quoted_segment)
        return tuple(images)

    @staticmethod
    def _reference_media_type(
        content: bytes, *, source_label: str = "引用消息中的图片"
    ) -> tuple[str, str]:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png", ".png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg", ".jpg"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return "image/webp", ".webp"
        raise ImageGeneratorError(
            f"{source_label}格式不受支持，仅支持 PNG、JPEG 或 WebP。"
        )

    async def _load_image_sources(
        self,
        components: tuple[Any, ...],
        *,
        source_label: str,
        filename_prefix: str,
    ) -> list[tuple[str, bytes, str]]:
        """Resolve AstrBot Image components and validate their bytes."""

        max_bytes = (
            self._int("references", "max_image_size_mb", default=12, minimum=1)
            * 1024
            * 1024
        )
        sources: list[tuple[str, bytes, str]] = []
        for index, component in enumerate(components, start=1):
            converter = getattr(component, "convert_to_file_path", None)
            if not callable(converter):
                raise ImageGeneratorError(
                    f"当前消息平台无法读取{source_label}，请重新发送图片后再试。",
                    detail=f"{filename_prefix} component misses convert_to_file_path",
                )
            try:
                raw_path = await converter()
                path = Path(str(raw_path or "")).resolve()
                if not path.is_file():
                    raise OSError("resolved image path is not a file")
                size = path.stat().st_size
                if size <= 0:
                    raise OSError("resolved image is empty")
                if size > max_bytes:
                    raise ImageGeneratorError(
                        f"{source_label}超过单张大小上限，请压缩后重新发送。"
                    )
                content = await asyncio.to_thread(path.read_bytes)
            except ImageGeneratorError:
                raise
            except Exception as exc:
                raise ImageGeneratorError(
                    f"{source_label}已失效或无法读取，请重新发送图片后再试。",
                    detail=f"{filename_prefix}={index} error={type(exc).__name__}: {exc}",
                ) from exc
            media_type, suffix = self._reference_media_type(
                content, source_label=source_label
            )
            sources.append((f"{filename_prefix}_{index}{suffix}", content, media_type))
        return sources

    async def _load_quoted_image_sources(
        self, components: tuple[Any, ...]
    ) -> list[tuple[str, bytes, str]]:
        return await self._load_image_sources(
            components,
            source_label="引用消息中的图片",
            filename_prefix="quoted",
        )

    @staticmethod
    def _augment_quoted_reference_prompt(
        prompt: str,
        *,
        start_index: int,
        count: int,
        use_sheet: bool,
        source_names: tuple[str, ...] = (),
    ) -> str:
        if count <= 0:
            return prompt
        lines = [prompt.rstrip(), "", "用户引用图片映射："]
        for offset in range(count):
            ref_index = start_index + offset
            ref_id = f"REF-{ref_index:02d}"
            if use_sheet:
                location = f"拼合图第 {ref_index} 格"
            elif offset < len(source_names):
                location = f"文件 {source_names[offset]}"
            else:
                location = f"上传的第 {ref_index} 张参考图"
            lines.append(
                f"- {ref_id}（{location}）= 用户引用的历史聊天图片"
            )
        lines.append(
            "- 按上方中文生图提示词使用这些 REF；若作为人物参考且提示词未明确要求改变，"
            "尽量保持引用图片中的人物原样，不要自行补充或重新设计其外观。"
        )
        return "\n".join(lines)

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _effective_tool_prompt(
        self, prompt: str, original_message: str
    ) -> tuple[str, str]:
        """Use the AI-authored prompt; keep raw user text only for reference matching.

        The original chat message can contain conversational instructions or other
        context that is unsuitable as an image prompt, so it must never replace
        the prompt prepared by the tool-calling model.
        """
        tool_prompt = str(prompt or "").strip()
        original = str(original_message or "").strip()
        return tool_prompt, "\n".join(item for item in (original, tool_prompt) if item)

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        if not self._bool("access", "enabled", default=True):
            return False
        if self._bool("access", "admin_only", default=False) and not event.is_admin():
            return False
        allowed = set(
            as_string_list(self._get("access", "allowed_user_ids", default=[]))
        )
        if allowed and str(event.get_sender_id()) not in allowed:
            return False
        denied_groups = set(
            as_string_list(self._get("access", "denied_group_ids", default=[]))
        )
        group_id = event.get_group_id()
        return not (group_id and str(group_id) in denied_groups)

    def _normalize_size(self, size: str | None) -> str:
        configured_default = str(
            self._get("generation", "default_size", default=DEFAULT_IMAGE_SIZE)
            or DEFAULT_IMAGE_SIZE
        ).strip().lower().replace("×", "x")
        value = str(size or "").strip().lower().replace("×", "x")
        aliases = {
            "": configured_default,
            "小图": "816x816",
            "小方图": "816x816",
            "small": "816x816",
            "小竖图": "640x1024",
            "小横图": "1024x640",
            "方图": "1024x1024",
            "正方形": "1024x1024",
            "square": "1024x1024",
            "竖图": "1024x1536",
            "portrait": "1024x1536",
            "横图": "1536x1024",
            "landscape": "1536x1024",
        }
        configured_default = aliases.get(configured_default, configured_default)
        value = aliases.get(value, value)
        allowed = as_string_list(
            self._get(
                "generation",
                "allowed_sizes",
                default=DEFAULT_ALLOWED_SIZES,
            )
        )
        if allowed == list(LEGACY_ALLOWED_SIZES):
            allowed = list(DEFAULT_ALLOWED_SIZES)
        if allowed and value not in allowed and value != configured_default:
            raise ConfigurationError(
                "不支持该图片尺寸，可选：" + "、".join(allowed[:12])
            )
        return value

    def _normalize_quality(self, quality: str | None) -> str:
        value = str(
            quality
            or self._get("generation", "default_quality", default="auto")
            or "auto"
        ).strip().lower()
        if value not in {"auto", "low", "medium", "high"}:
            raise ConfigurationError("图片质量只支持 auto、low、medium 或 high。")
        return value

    async def _remote_review(self, _event: AstrMessageEvent, prompt: str) -> str:
        timeout = self._float("safety", "review_timeout_seconds", default=45, minimum=5)
        review_model = str(self._get("safety", "review_model", default="") or "").strip()
        if not review_model:
            raise ConfigurationError(
                "启用 LLM 安全审核时必须配置独立的审核模型 ID。"
            )
        try:
            return await self._review_client().chat_completion(
                review_model, prompt, timeout_seconds=timeout
            )
        except ImageGeneratorError as exc:
            logger.warning("[img_gener] safety review failed: %s", exc.detail[:800])
            raise

    def _log_review(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        allowed: bool,
        code: str,
        references: ResolvedReferences | None,
    ) -> None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "[img_gener] review allowed=%s code=%s prompt_sha256=%s sender=%s group=%s refs=%s",
            allowed,
            code,
            digest,
            event.get_sender_id(),
            event.get_group_id() or "private",
            ",".join(references.names) if references and references.names else "none",
        )
        if self._bool("safety", "log_raw_prompt", default=False):
            logger.info("[img_gener] reviewed raw prompt=%s", prompt[:8000])

    async def _preflight_local_review(
        self, event: AstrMessageEvent, prompt: str
    ) -> str | None:
        """Reject local hard-rule matches before announcing or queuing a task."""

        review = self.moderator.local_review(prompt)
        if review.allowed:
            return None

        user_id, group_id = self._identity(event)
        bypass_limits = bool(
            event.is_admin()
            and self._bool("limits", "admins_bypass_limits", default=True)
        )
        rate = await self.rate_limiter.acquire(
            user_id, group_id, bypass_limits=bypass_limits
        )
        if not rate.allowed:
            return rate.message
        await self.rate_limiter.cancel(rate.lease_id)
        self._log_review(
            event,
            prompt,
            allowed=False,
            code=review.code,
            references=None,
        )
        return f"请求未通过安全审核（{review.code}）：{review.reason}"

    async def _run_background_generation(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        size: str,
        quality: str | None,
        characters: list[str] | None,
        portrait_user_qqs: tuple[str, ...] = (),
        reference_prompt: str = "",
        job_id: int,
        started_at: float,
    ) -> None:
        try:
            message = await self._generate_image(
                event,
                prompt,
                size=size,
                quality=quality,
                characters=characters,
                portrait_user_qqs=portrait_user_qqs,
                reference_prompt=reference_prompt,
                started_at=started_at,
                job_id=job_id,
            )
            job = self._active_jobs.get(job_id)
            if message and (job is None or not job.finalized):
                self._finalize_job(
                    job_id,
                    status="rejected",
                    stage="未进入生成",
                    result_message=message,
                )
            elif not message and (job is None or not job.finalized):
                self._finalize_job(
                    job_id,
                    status="succeeded",
                    stage="已完成",
                    result_message="图片已生成并发送。",
                )
            if message:
                await event.send(event.plain_result(message))
        except asyncio.CancelledError:
            self._finalize_job(
                job_id,
                status="cancelled",
                stage="任务已取消",
                result_message="后台任务被取消。",
            )
            raise
        except Exception as exc:
            logger.exception("[img_gener] background generation failed: %s", exc)
            self._finalize_job(
                job_id,
                status="failed",
                stage="后台任务异常",
                result_message="后台生图任务异常终止，请联系管理员查看日志。",
            )
            try:
                await event.send(
                    event.plain_result(
                        "后台生图任务异常终止，请联系管理员查看日志。"
                    )
                )
            except Exception:
                logger.exception(
                    "[img_gener] failed to send background task error"
                )
        finally:
            self._finish_job(job_id)

    def _start_background_generation(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        size: str,
        quality: str | None,
        characters: list[str] | None,
        portrait_user_qqs: tuple[str, ...] = (),
        reference_prompt: str = "",
        source: str,
    ) -> int:
        job_id, started_at = self._register_job(
            event,
            source=source,
            size=size,
            prompt=prompt,
            quality=quality,
        )
        try:
            task = asyncio.create_task(
                self._run_background_generation(
                    event,
                    prompt,
                    size=size,
                    quality=quality,
                    characters=characters,
                    portrait_user_qqs=portrait_user_qqs,
                    reference_prompt=reference_prompt,
                    job_id=job_id,
                    started_at=started_at,
                ),
                name=f"{PLUGIN_NAME}:generate_image",
            )
        except Exception:
            self._finalize_job(
                job_id,
                status="failed",
                stage="调度失败",
                result_message="后台任务创建失败。",
            )
            self._finish_job(job_id)
            raise
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return job_id

    async def _start_notice(self, event: AstrMessageEvent, size: str) -> str:
        user_id, group_id = self._identity(event)
        pool = self._active_job_snapshot(user_id, group_id)
        try:
            status = await self.rate_limiter.status(user_id, group_id)
            inflight = int(status.get("global_inflight", 0))
            global_max = int(self.rate_limiter.limits.global_max_concurrent)
            concurrency = f"全局并发：{inflight}/{global_max}"
        except Exception:
            concurrency = "全局并发：暂不可用"
        expected = self._int("generation", "expected_minutes", default=3, minimum=1)
        queue_line = f"当前队列：全局 {pool['global']} 个、你的 {pool['user']} 个"
        if group_id:
            queue_line += f"、本群 {pool['group']} 个"
        lines = [
            self._generation_start_message(size),
            queue_line,
            concurrency,
            (
                f"预计生成约 {expected} 分钟，完成后会单独发送图片和简短评价；"
                "可用 /生图状态 查看进度与已用时间。"
            ),
        ]
        return "\n".join(lines)

    async def _evaluate_image(
        self,
        prompt: str,
        image_content: bytes,
        media_type: str,
    ) -> str | None:
        if not self._bool("generation", "enable_evaluation", default=True):
            return None
        vision_model = str(self._get("safety", "vision_model", default="") or "").strip()
        if not vision_model:
            logger.info("[img_gener] vision model not configured, skipping evaluation")
            return None
        timeout = self._float("safety", "review_timeout_seconds", default=45, minimum=5)
        client = self._review_client()
        vision_prompt = (
            "请用 1~3 句话简短评价下面这张 AI 生成的图片："
            "描述画面内容与风格，并判断是否贴合用户需求，语气自然、口语化。\n"
            f"实际生图提示词：{prompt}"
        )
        try:
            text = await client.chat_completion_vision(
                vision_model,
                vision_prompt,
                image_content,
                media_type,
                timeout_seconds=timeout,
            )
            return text.strip()[:500] or None
        except ImageGeneratorError as exc:
            logger.warning(
                "[img_gener] vision evaluation failed, falling back to prompt-only: %s",
                exc.detail[:400],
            )
        fallback_prompt = (
            "（识图失败，无法读取图片）请仅根据下面的生成提示词，"
            "用 1~3 句话简短评价这张预期生成的画面，语气自然。\n"
            f"提示词：{prompt}"
        )
        try:
            text = await client.chat_completion(
                vision_model, fallback_prompt, timeout_seconds=timeout
            )
            return text.strip()[:500] or None
        except ImageGeneratorError as exc:
            logger.warning(
                "[img_gener] prompt-only evaluation failed: %s", exc.detail[:400]
            )
            return None

    def _reference_resolution_error(
        self,
        resolved: ResolvedReferences,
        *,
        portrait_user_qqs: tuple[str, ...],
    ) -> str | None:
        if not resolved.image_paths or not self._bool(
            "references", "strict_references", default=True
        ):
            return None

        portrait_ids = set(portrait_user_qqs)
        strict_missing = tuple(
            name
            for name in resolved.missing_characters
            if name not in portrait_ids
        )
        if strict_missing:
            return "人物参考图缺失或不可用：" + "、".join(strict_missing)
        if resolved.capped_characters:
            return (
                "同时指定的人物超过了单次参考图上限，以下人物未能携带参考图："
                + "、".join(resolved.capped_characters)
                + "。请减少同框人物数量，或在插件设置中调大「单次最多携带参考图」。"
            )
        return None

    async def _generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        size: str | None = None,
        quality: str | None = None,
        characters: list[str] | None = None,
        portrait_user_qqs: tuple[str, ...] = (),
        reference_prompt: str = "",
        started_at: float | None = None,
        job_id: int | None = None,
    ) -> str | None:
        started_at = time.monotonic() if started_at is None else started_at
        self._set_job_stage(job_id, "检查权限与配额")
        if not self._is_allowed(event):
            return "生图功能未启用，或当前用户/群没有使用权限。"

        raw_prompt = str(prompt or "").strip()
        if not raw_prompt:
            return "请提供需要生成的画面描述。"
        try:
            selected_size = self._normalize_size(size)
            selected_quality = self._normalize_quality(quality)
        except ImageGeneratorError as exc:
            return exc.public_message

        quoted_components = self._quoted_image_components(event)
        max_quoted_images = self._int(
            "references", "max_quoted_images", default=2, minimum=1
        )
        if len(quoted_components) > max_quoted_images:
            return (
                f"单次最多使用 {max_quoted_images} 张引用消息图片，"
                "请减少引用图片数量后重试。"
            )
        profile_store = getattr(self, "user_profiles", None)
        profile_references = (
            profile_store.character_references(portrait_user_qqs)
            if profile_store is not None and portrait_user_qqs
            else ()
        )
        reference_manager = self.references.with_characters(profile_references)
        total_reference_limit = reference_manager.max_total_images
        if len(quoted_components) > total_reference_limit:
            return (
                f"引用消息中有 {len(quoted_components)} 张图片，超过单次参考图上限 "
                f"{total_reference_limit} 张。"
            )
        requested_characters = [*list(characters or ()), *portrait_user_qqs]
        resolved = reference_manager.resolve(
            "\n".join(item for item in (reference_prompt, raw_prompt) if item),
            requested_characters or None,
            max_total_images=total_reference_limit - len(quoted_components),
        )
        reference_error = self._reference_resolution_error(
            resolved,
            portrait_user_qqs=portrait_user_qqs,
        )
        if reference_error:
            return reference_error

        user_id, group_id = self._identity(event)
        bypass_limits = bool(
            event.is_admin()
            and self._bool("limits", "admins_bypass_limits", default=True)
        )
        rate = await self.rate_limiter.acquire(
            user_id, group_id, bypass_limits=bypass_limits
        )
        if not rate.allowed:
            return rate.message

        lease_active = True
        try:
            self._set_job_stage(job_id, "读取引用图片")
            raw_quoted_sources = await self._load_quoted_image_sources(
                quoted_components
            )
            configured_sources = OpenAICompatibleImageClient.source_from_paths(
                resolved.image_paths,
                labels=resolved.image_character_names,
            )
            quoted_sources = [
                (
                    f"reference_{len(configured_sources) + offset:02d}_"
                    f"user_quote_{offset:02d}{Path(name).suffix or '.png'}",
                    content,
                    media_type,
                )
                for offset, (name, content, media_type) in enumerate(
                    raw_quoted_sources, start=1
                )
            ]
            all_sources = [*configured_sources, *quoted_sources]
            reference_mode = str(
                self._get("references", "reference_mode", default="auto") or "auto"
            ).strip().lower()
            use_sheet = (
                reference_mode == "sheet"
                or (
                    reference_mode == "auto"
                    and (len(resolved.characters) > 1 or bool(quoted_sources))
                )
            ) and len(all_sources) > 1
            sheet = None
            if use_sheet:
                sheet_labels = [
                    f"REF-{index:02d} | {character_name}"
                    for index, character_name in enumerate(
                        resolved.image_character_names, start=1
                    )
                ]
                sheet_labels.extend(
                    f"REF-{len(configured_sources) + index:02d} | 用户引用图 {index}"
                    for index in range(1, len(quoted_sources) + 1)
                )
                try:
                    sheet = build_contact_sheet_from_sources(
                        all_sources, labels=sheet_labels
                    )
                except ImageGeneratorError:
                    logger.warning(
                        "[img_gener] contact sheet build failed, "
                        "falling back to separate reference uploads"
                    )
                    use_sheet = False
            sources = (
                [("contact_sheet_REF_map.png", sheet.content, sheet.media_type)]
                if use_sheet and sheet is not None
                else all_sources
            )
            logger.info(
                "[img_gener] reference upload mode=%s files=%s configured=%s quoted=%s",
                "sheet" if use_sheet else "multi",
                ",".join(name for name, _, _ in sources) or "none",
                len(configured_sources),
                len(quoted_sources),
            )

            effective_prompt = CharacterReferenceManager.augment_prompt(
                raw_prompt,
                resolved,
                use_sheet=use_sheet,
                reference_filenames=tuple(
                    name for name, _, _ in configured_sources
                ),
            )
            effective_prompt = self._augment_quoted_reference_prompt(
                effective_prompt,
                start_index=len(configured_sources) + 1,
                count=len(quoted_sources),
                use_sheet=use_sheet,
                source_names=tuple(name for name, _, _ in quoted_sources),
            )
            prompt_prefix = str(
                self._get("generation", "prompt_prefix", default="") or ""
            ).strip()
            if prompt_prefix:
                effective_prompt = f"{prompt_prefix}\n{effective_prompt}"

            self._set_job_stage(job_id, "安全审核")
            review = await self.moderator.review(
                effective_prompt,
                lambda review_prompt: self._remote_review(event, review_prompt),
            )
            self._log_review(
                event,
                effective_prompt,
                allowed=review.allowed,
                code=review.code,
                references=resolved,
            )
            if not review.allowed:
                return f"请求未通过安全审核（{review.code}）：{review.reason}"
            self._set_job_stage(job_id, "生成图片")

            client = self._client()
            if sources:
                generated = await client.edit(
                    effective_prompt,
                    sources,
                    size=selected_size,
                    quality=selected_quality,
                )
            else:
                generated = await client.generate(
                    effective_prompt,
                    size=selected_size,
                    quality=selected_quality,
                )

            await self.rate_limiter.complete(rate.lease_id)
            lease_active = False
            output_path = self.output_store.save(generated)
            duration_seconds = time.monotonic() - started_at
            self._set_job_stage(job_id, "归档长期图库")
            gallery_record: GalleryImage | None = None
            profile_failures: tuple[str, ...] = ()
            archive_failed = False
            try:
                gallery_record, profile_failures = await self._archive_generated_image(
                    event,
                    generated,
                    prompt=raw_prompt,
                    size=selected_size,
                    quality=selected_quality,
                    duration_seconds=duration_seconds,
                    job_id=job_id,
                    portrait_user_qqs=portrait_user_qqs,
                )
            except Exception:
                archive_failed = True
                logger.exception(
                    "[img_gener] failed to archive generated image job=%s", job_id
                )

            self._set_job_stage(job_id, "发送图片")
            reference_parts = []
            if resolved.names:
                reference_parts.append("人物参考：" + "、".join(resolved.names))
            if quoted_sources:
                reference_parts.append(f"引用图片：{len(quoted_sources)} 张")
            reference_text = (
                "；已携带" + "；".join(reference_parts) if reference_parts else ""
            )
            archive_parts = []
            if gallery_record is not None:
                archive_parts.append(f"图库编号：{gallery_record.image_id}")
                updated_profiles = tuple(
                    qq for qq in portrait_user_qqs if qq not in set(profile_failures)
                )
                if updated_profiles:
                    archive_parts.append(
                        "已更新用户画像：" + "、".join(updated_profiles)
                    )
                if profile_failures:
                    archive_parts.append(
                        "画像档案更新失败：" + "、".join(profile_failures)
                    )
            elif archive_failed:
                archive_parts.append("图库归档失败，请联系管理员")
            archive_text = "；" + "；".join(archive_parts) if archive_parts else ""
            duration = self._format_duration(duration_seconds)
            caption = (
                f" 图片已生成（{selected_size}，{selected_quality}）"
                f"{reference_text}{archive_text}。总用时：{duration}。\n"
                f"提示词：{raw_prompt}"
            )
            try:
                await event.send(self._completion_chain(event, caption, output_path))
            except Exception as exc:
                gallery_hint = (
                    f"，已归档为图库编号 {gallery_record.image_id}"
                    if gallery_record is not None
                    else ""
                )
                message = (
                    f"图片已经生成并计入配额{gallery_hint}，但聊天平台发送失败，"
                    "请联系管理员查看日志。"
                )
                logger.warning(
                    "[img_gener] generated but failed to send image path=%s error=%s",
                    output_path,
                    str(exc)[:500],
                )
                self._finalize_job(
                    job_id,
                    status="delivery_failed",
                    stage="图片发送失败",
                    result_message=message,
                    output_filename=output_path.name,
                )
                return message
            self._set_job_stage(job_id, "生成评价")
            evaluation = await self._evaluate_image(
                raw_prompt, generated.content, generated.media_type
            )
            if evaluation:
                try:
                    await event.send(self._mention_chain(event, f" {evaluation}"))
                except Exception as exc:
                    logger.warning(
                        "[img_gener] failed to send evaluation: %s", str(exc)[:500]
                    )
            self._finalize_job(
                job_id,
                status="succeeded",
                stage="已完成",
                result_message=(
                    "图片已生成并发送。"
                    + (
                        f" 图库编号：{gallery_record.image_id}。"
                        if gallery_record is not None
                        else " 图库归档失败。" if archive_failed else ""
                    )
                ),
                output_filename=output_path.name,
            )
            return None
        except ImageGeneratorError as exc:
            active_jobs = getattr(self, "_active_jobs", {})
            job = active_jobs.get(job_id) if job_id is not None else None
            failed_stage = job.stage if job else "未知阶段"
            logger.warning(
                "[img_gener] generation failed job=%s stage=%s public=%s detail=%s",
                job_id if job_id is not None else "-",
                failed_stage,
                exc.public_message,
                exc.detail[:1200],
            )
            self._finalize_job(
                job_id,
                status="failed",
                stage=f"{failed_stage}失败",
                result_message=exc.public_message,
            )
            return exc.public_message
        except Exception as exc:
            logger.exception("[img_gener] unexpected generation error: %s", exc)
            message = "生图过程中发生内部错误，请联系管理员查看日志。"
            active_jobs = getattr(self, "_active_jobs", {})
            job = active_jobs.get(job_id) if job_id is not None else None
            self._finalize_job(
                job_id,
                status="failed",
                stage=f"{job.stage if job else '未知阶段'}失败",
                result_message=message,
            )
            return message
        finally:
            if lease_active:
                await self.rate_limiter.cancel(rate.lease_id)

    @filter.llm_tool(name="get_image_generation_status")
    async def get_image_generation_status(
        self,
        event: AstrMessageEvent,
        include_history: bool = True,
        history_limit: str = "5",
    ) -> str:
        """Query authoritative image task progress, history, quotas, and cooldowns.

        Always call this tool when the user asks whether an image task is really
        queued or still generating, why it disappeared, whether the last request
        failed, recent image-generation history, remaining personal/group quota,
        or personal/group cooldown. Never infer status from prior conversation.
        If this tool reports no active task, explicitly tell the user it is not
        currently generating. This read-only tool does not submit a new image.

        Args:
            include_history(boolean): Include the current user's recent accepted task records; default true.
            history_limit(string): Number of the current user's recent records to return as numeric text, from 1 to 20; default "5".
        """  # noqa: E501
        return await self._status_report(
            event,
            include_history=as_bool(include_history, True),
            history_limit=min(20, as_int(history_limit, 5, minimum=1)),
        )

    @filter.llm_tool(name="get_user_portrait_reference")
    async def get_user_portrait_reference(
        self,
        event: AstrMessageEvent,
        target_user_qq: str = "self",
    ) -> str:
        """Send the authoritative canonical portrait reference for one QQ user.

        Call this read-only tool when the user asks to see, inspect, or confirm
        their currently stored personal portrait reference. Ordinary users may
        retrieve only their own image by using "self". AstrBot administrators
        may provide an explicit QQ number to retrieve another user's image.
        Never guess that an image exists from chat history; relay the exact result.

        Args:
            target_user_qq(string): "self" for the requester, or an explicit QQ
                number when the requester is an AstrBot administrator.
        """
        message, image_path = await self._portrait_reference_image(
            event, target_user_qq
        )
        if image_path is None:
            return message
        await event.send(self._completion_chain(event, message, image_path))
        return "已向用户发送档案中当前唯一的本体参考图。"

    @filter.llm_tool(name="generate_image")
    async def generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str | None = None,
        quality: str | None = None,
        characters: list[str] | None = None,
        portrait_user_qqs: list[str] | None = None,
    ) -> str:
        """Submit one image generation request to GPT Image 2 and return immediately.

        Use this tool only when the user clearly asks to create an image. First
        rewrite the user's request into a complete, directly usable CHINESE image
        prompt. Do not copy the conversational message verbatim and do not translate
        the prompt to English. Remove request wrappers such as “帮我画一张”, while
        preserving requested subjects, composition, actions, style, lighting,
        colors, text, and constraints. The plugin performs safety review and rate
        limiting. Configured character names such as 可可子 or 菌菌 are detected
        automatically and their stored reference images are attached.
        Images contained in a quoted/replied historical message are also attached
        automatically; describe how those images should guide the result in prompt.

        For a portrait of the requester, pass "self" in portrait_user_qqs. For a
        user explicitly mentioned with @, pass that QQ number; the plugin also reads
        At components directly. Each QQ portrait profile uses only its one current
        canonical reference image. If an alias belongs to multiple QQ users, do not
        guess: return the numbered QQ/name candidates to the user, ask them
        to reply with a number, then call this tool again with the confirmed QQ.

        IMPORTANT: generation runs in the BACKGROUND and typically takes about 3
        minutes. When this tool returns, the image is NOT ready yet — the request
        has only been accepted. Tell the user it is generating and ask them to
        wait; do NOT claim the image is finished. The plugin will later send the
        image, the prompt, and a short evaluation as separate messages.
        This instruction applies only when the tool explicitly returns an accepted
        request with queue information. If it returns rejection, no active task,
        or numbered identity candidates, relay that result and do NOT say generating.

        If the user asks about progress, queue state, failures, history, remaining
        quota, or cooldown instead of requesting a new image, do not guess and do
        not call this generation tool. Call get_image_generation_status instead.

        Args:
            prompt(string): AI-prepared complete Chinese image prompt; never the raw conversational request.
            size(string): Optional allowed output size; use 816x816 for a small fast square draft.
            quality(string): Optional quality: auto, low, medium, or high.
            characters(array): Optional configured character names; omit to auto-detect names and aliases from prompt.
            portrait_user_qqs(array): QQ portrait subjects; use ["self"] for the
                requester or explicit QQ numbers after @/alias confirmation.
        """  # noqa: E501
        if not self._is_allowed(event):
            return "生图功能未启用，或当前用户/群没有使用权限。"
        original_message = self._event_message_text(event)
        effective_prompt, reference_prompt = self._effective_tool_prompt(
            prompt, original_message
        )
        if not effective_prompt:
            return "请提供需要生成的画面描述。"
        if not self._contains_cjk(effective_prompt):
            message = (
                "请先将用户需求整理成完整、可直接生图的中文提示词，再调用 "
                "generate_image；不要传入英文提示词，也不要直接复制用户原话。"
            )
            return self._record_rejected_attempt(
                event,
                source="LLM",
                prompt=effective_prompt,
                size=str(size or DEFAULT_IMAGE_SIZE),
                quality=quality,
                stage="中文提示词校验失败",
                message=message,
            )
        try:
            selected_size = self._normalize_size(size)
        except ImageGeneratorError as exc:
            return self._record_rejected_attempt(
                event,
                source="LLM",
                prompt=effective_prompt,
                size=str(size or DEFAULT_IMAGE_SIZE),
                quality=quality,
                stage="参数校验失败",
                message=exc.public_message,
            )
        portrait_subjects, portrait_error = await self._resolve_portrait_subjects(
            event,
            "\n".join(item for item in (reference_prompt, effective_prompt) if item),
            portrait_user_qqs,
        )
        if portrait_error:
            return self._record_rejected_attempt(
                event,
                source="LLM",
                prompt=effective_prompt,
                size=selected_size,
                quality=quality,
                stage="人物身份待确认",
                message=portrait_error,
            )
        local_rejection = await self._preflight_local_review(event, effective_prompt)
        if local_rejection:
            return self._record_rejected_attempt(
                event,
                source="LLM",
                prompt=effective_prompt,
                size=selected_size,
                quality=quality,
                stage="入队前审核或限流拒绝",
                message=local_rejection,
            )
        self._start_background_generation(
            event,
            effective_prompt,
            size=selected_size,
            quality=quality,
            characters=characters,
            portrait_user_qqs=portrait_subjects,
            reference_prompt=reference_prompt,
            source="LLM",
        )
        return await self._start_notice(event, selected_size)

    @filter.command("生图", alias={"绘图", "draw"})
    async def draw_command(self, event: AstrMessageEvent, prompt: GreedyStr):
        """直接测试 GPT Image 2 生图；正常聊天也可由 LLM 自动调用。"""

        event.stop_event()
        raw_prompt = str(prompt or "").strip()
        try:
            selected_size = self._normalize_size(None)
        except ImageGeneratorError as exc:
            message = self._record_rejected_attempt(
                event,
                source="/生图",
                prompt=raw_prompt,
                size=DEFAULT_IMAGE_SIZE,
                quality=None,
                stage="参数校验失败",
                message=exc.public_message,
            )
            yield event.plain_result(message)
            return
        if not self._is_allowed(event):
            yield event.plain_result(
                "生图功能未启用，或当前用户/群没有使用权限。"
            )
            return
        if not raw_prompt:
            yield event.plain_result("请提供需要生成的画面描述。")
            return
        portrait_subjects, portrait_error = await self._resolve_portrait_subjects(
            event, raw_prompt, None
        )
        if portrait_error:
            message = self._record_rejected_attempt(
                event,
                source="/生图",
                prompt=raw_prompt,
                size=selected_size,
                quality=None,
                stage="人物身份待确认",
                message=portrait_error,
            )
            yield event.plain_result(message)
            return
        local_rejection = await self._preflight_local_review(event, raw_prompt)
        if local_rejection:
            message = self._record_rejected_attempt(
                event,
                source="/生图",
                prompt=raw_prompt,
                size=selected_size,
                quality=None,
                stage="入队前审核或限流拒绝",
                message=local_rejection,
            )
            yield event.plain_result(message)
            return
        self._start_background_generation(
            event,
            raw_prompt,
            size=selected_size,
            quality=None,
            characters=None,
            portrait_user_qqs=portrait_subjects,
            source="/生图",
        )
        yield event.plain_result(await self._start_notice(event, selected_size))

    @filter.command("生图参考图", alias={"本体参考图", "画像参考图"})
    async def portrait_reference_command(
        self, event: AstrMessageEvent, query: GreedyStr
    ):
        """查看或替换 QQ 用户当前唯一的本体参考图。"""

        event.stop_event()
        tail = str(query or "").strip()
        parts = tail.split()
        action = parts[0].casefold() if parts else ""

        if action in {"列表", "list"}:
            if len(parts) != 1:
                yield event.plain_result("用法：/生图参考图 列表")
                return
            if not self._event_is_admin(event):
                yield event.plain_result("只有 AstrBot 管理员可以查看画像档案列表。")
                return
            profile_store = getattr(self, "user_profiles", None)
            if profile_store is None:
                yield event.plain_result("用户画像档案暂时不可用。")
                return
            try:
                profiles = profile_store.all_profiles()
            except Exception:
                logger.exception("[img_gener] failed to list user portrait profiles")
                yield event.plain_result("用户画像档案暂时不可用，请联系管理员查看日志。")
                return
            if not profiles:
                yield event.plain_result("尚未建立任何 QQ 用户画像档案。")
                return
            lines = ["QQ 用户画像档案："]
            for index, profile in enumerate(profiles, start=1):
                name = profile.display_name or "未记录群名"
                state = "已设置" if profile.image_paths else "未设置"
                lines.append(f"{index}.（{profile.qq}）（{name}）— {state}")
            lines.extend(
                (
                    "查看图片：/生图参考图 <QQ号>",
                    "替换图片：发送或引用一张图片并使用 /生图参考图 设置 <QQ号>",
                )
            )
            yield event.plain_result("\n".join(lines))
            return

        setting = action in {"设置", "替换", "set"}
        if setting and len(parts) > 2:
            yield event.plain_result(
                "用法：发送或引用一张图片并使用 /生图参考图 设置 [QQ号]"
            )
            return
        if not setting and len(parts) > 1:
            yield event.plain_result("用法：/生图参考图 [QQ号]")
            return
        if setting:
            raw_target = parts[1] if len(parts) == 2 else "self"
        else:
            raw_target = parts[0] if parts else "self"

        if not setting:
            message, image_path = await self._portrait_reference_image(
                event, raw_target
            )
            if image_path is None:
                yield event.plain_result(message)
            else:
                yield self._completion_chain(event, message, image_path)
            return

        target_qq, error = self._portrait_reference_target(event, raw_target)
        if error or target_qq is None:
            yield event.plain_result(error or "无法识别目标用户。")
            return
        components = self._event_image_components(event)
        if len(components) != 1:
            yield event.plain_result(
                "请在当前消息发送或引用恰好 1 张图片，再使用 "
                "/生图参考图 设置 [QQ号]。"
            )
            return
        try:
            sources = await self._load_image_sources(
                components,
                source_label="用于设置个人参考图的图片",
                filename_prefix="profile",
            )
            filename, content, _media_type = sources[0]
            display_name = await self._group_member_name(event, target_qq)
            self.user_profiles.ensure_profile(
                target_qq,
                display_name=display_name,
                group_id=str(event.get_group_id() or "") or None,
            )
            image_path = self.user_profiles.set_manual_reference(
                target_qq,
                content=content,
                extension=Path(filename).suffix,
            )
        except ImageGeneratorError as exc:
            yield event.plain_result(exc.public_message)
            return
        except Exception:
            logger.exception(
                "[img_gener] failed to replace portrait reference qq=%s",
                target_qq,
            )
            yield event.plain_result("替换本体参考图失败，请联系管理员查看日志。")
            return
        yield self._completion_chain(
            event,
            f"已替换 QQ {target_qq} 当前唯一的本体参考图：",
            image_path,
        )

    @filter.command("生图人物", alias={"生图角色"})
    async def character_list_command(self, event: AstrMessageEvent):
        """列出已配置的人物参考图及其可用状态。"""

        event.stop_event()
        if not self.references.characters:
            yield event.plain_result("尚未在插件设置中配置人物参考图。")
            return
        lines = ["已配置人物参考："]
        for character in self.references.characters:
            resolved = self.references.resolve(character.name, [character.name])
            usable = sum(1 for path in resolved.image_paths if path in character.image_paths)
            aliases = "、".join(character.aliases) or "无"
            lines.append(
                f"- {character.name}（别名：{aliases}；可用参考图：{usable} 张）"
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("生图状态", alias={"生图配额"})
    async def status_command(self, event: AstrMessageEvent):
        """查看真实队列、进度、最近结果、余量和冷却。"""

        event.stop_event()
        report = await self._status_report(
            event,
            include_history=True,
            history_limit=3,
        )
        yield event.plain_result(report)

    @filter.command("生图历史", alias={"生图记录"})
    async def history_command(self, event: AstrMessageEvent, query: GreedyStr):
        """查看永久图库总量、指定编号详情，或随机图片拼图。"""

        event.stop_event()
        gallery_store = getattr(self, "gallery_store", None)
        if gallery_store is None:
            yield event.plain_result("长期图库尚未初始化。")
            return
        tail = str(query or "").strip()
        if not tail:
            try:
                total = gallery_store.count()
            except Exception:
                logger.exception("[img_gener] failed to count gallery")
                yield event.plain_result("长期图库暂时不可用，请联系管理员查看日志。")
                return
            yield event.plain_result(
                "🎨 永久生图图库\n"
                f"当前共保存 {total} 张成功图片。\n"
                "图片编号按成功归档顺序自增：1、2、3……\n"
                "查看详情：/生图历史 <图片编号>\n"
                "随机单图：/生图历史 随机\n"
                "随机拼图：/生图历史 随机 <数量>（2～9）"
            )
            return

        parts = tail.split()
        if parts[0].casefold() in {"随机", "random"}:
            if len(parts) > 2:
                yield event.plain_result("用法：/生图历史 随机 [1～9]")
                return
            try:
                requested_count = int(parts[1]) if len(parts) == 2 else 1
            except ValueError:
                yield event.plain_result("随机图片数量必须是 1～9 的整数。")
                return
            if not 1 <= requested_count <= 9:
                yield event.plain_result("随机图片数量必须在 1～9 之间。")
                return
            try:
                records = tuple(
                    record
                    for record in gallery_store.random(requested_count)
                    if record.image_path.is_file()
                )
            except Exception:
                logger.exception("[img_gener] failed to query random gallery images")
                yield event.plain_result("随机读取图库失败，请联系管理员查看日志。")
                return
            if not records:
                yield event.plain_result("图库中还没有可读取的成功图片。")
                return
            if requested_count == 1:
                yield event.chain_result(
                    [Comp.Image.fromFileSystem(str(records[0].image_path))]
                )
                return
            try:
                grid = await asyncio.to_thread(
                    build_gallery_grid,
                    tuple(record.image_path for record in records),
                )
                grid_path = self.output_store.save_bytes(grid.content, ".png")
            except ImageGeneratorError as exc:
                yield event.plain_result(exc.public_message)
                return
            except Exception:
                logger.exception("[img_gener] failed to build random gallery grid")
                yield event.plain_result("随机图库拼图保存失败，请联系管理员查看日志。")
                return
            yield event.chain_result([Comp.Image.fromFileSystem(str(grid_path))])
            return

        if len(parts) != 1 or not parts[0].isdigit():
            yield event.plain_result(
                "用法：/生图历史、/生图历史 <图片编号>、"
                "/生图历史 随机 [1～9]"
            )
            return
        try:
            record = gallery_store.get(int(parts[0]))
        except Exception:
            logger.exception("[img_gener] failed to query gallery image")
            yield event.plain_result("查询图库失败，请联系管理员查看日志。")
            return
        if record is None:
            yield event.plain_result(f"图库中不存在编号 {parts[0]}。")
            return
        if not record.image_path.is_file():
            yield event.plain_result(
                f"图库编号 {record.image_id} 的图片文件缺失，请联系管理员检查长期存储。"
            )
            return
        created = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(record.created_at)
        )
        requester = record.requester_name or "未知昵称"
        group_text = record.group_id or "私聊"
        details = (
            f" 图库编号：{record.image_id}\n"
            f"请求人：{requester}（QQ：{record.requester_qq}）\n"
            f"会话：{group_text}\n"
            f"时间：{created}\n"
            f"用时：{self._format_duration(record.duration_seconds)}\n"
            f"规格：{record.size}，{record.quality}\n"
            f"提示词：{record.prompt}"
        )
        yield self._completion_chain(event, details, record.image_path)
