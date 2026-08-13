from __future__ import annotations

import asyncio
import hashlib
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
    from .errors import ConfigurationError, ImageGeneratorError
    from .image_client import OpenAICompatibleImageClient
    from .moderation import PromptModerator
    from .rate_limiter import LimitConfig, PersistentRateLimiter
    from .storage import OutputStore
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
    from errors import ConfigurationError, ImageGeneratorError
    from image_client import OpenAICompatibleImageClient
    from moderation import PromptModerator
    from rate_limiter import LimitConfig, PersistentRateLimiter
    from storage import OutputStore


PLUGIN_NAME = "astrbot_plugin_img_gener"
DEFAULT_START_MESSAGE = (
    "已收到生图请求，正在检查额度并进行安全审核；审核通过后将立即生成，请稍候。"
)


@dataclass(slots=True)
class ActiveGenerationJob:
    job_id: int
    user_id: str
    group_id: str | None
    source: str
    started_at: float
    stage: str = "等待调度"


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
        logger.info(
            "[img_gener] initialized model=%s base_url=%s references=%s cleaned=%s",
            self._get("api", "model", default="gpt-image-2"),
            normalize_base_url(self._get("api", "base_url", default="https://uuapi.cc/v1")),
            len(self.references.characters),
            removed,
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

    def _generation_start_message(self) -> str:
        message = str(
            self._get(
                "generation",
                "start_message",
                default=DEFAULT_START_MESSAGE,
            )
            or ""
        ).strip()
        return message or DEFAULT_START_MESSAGE

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

    def _register_job(
        self, event: AstrMessageEvent, *, source: str
    ) -> tuple[int, float]:
        self._next_job_id += 1
        user_id, group_id = self._identity(event)
        started_at = time.monotonic()
        job = ActiveGenerationJob(
            job_id=self._next_job_id,
            user_id=user_id,
            group_id=group_id,
            source=source,
            started_at=started_at,
        )
        self._active_jobs[job.job_id] = job
        return job.job_id, started_at

    def _set_job_stage(self, job_id: int | None, stage: str) -> None:
        if job_id is not None and job_id in self._active_jobs:
            self._active_jobs[job_id].stage = stage

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
        value = str(size or "").strip().lower().replace("×", "x")
        aliases = {
            "": str(self._get("generation", "default_size", default="1024x1024")),
            "方图": "1024x1024",
            "正方形": "1024x1024",
            "square": "1024x1024",
            "竖图": "1024x1536",
            "portrait": "1024x1536",
            "横图": "1536x1024",
            "landscape": "1536x1024",
        }
        value = aliases.get(value, value)
        allowed = as_string_list(
            self._get(
                "generation",
                "allowed_sizes",
                default=["auto", "1024x1024", "1024x1536", "1536x1024"],
            )
        )
        if allowed and value not in allowed:
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
        references: ResolvedReferences,
    ) -> None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "[img_gener] review allowed=%s code=%s prompt_sha256=%s sender=%s group=%s refs=%s",
            allowed,
            code,
            digest,
            event.get_sender_id(),
            event.get_group_id() or "private",
            ",".join(references.names) or "none",
        )
        if self._bool("safety", "log_raw_prompt", default=False):
            logger.info("[img_gener] reviewed raw prompt=%s", prompt[:8000])

    async def _run_background_generation(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        size: str | None,
        quality: str | None,
        characters: list[str] | None,
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
                started_at=started_at,
                job_id=job_id,
            )
            await event.send(event.plain_result(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[img_gener] background generation failed: %s", exc)
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
        size: str | None,
        quality: str | None,
        characters: list[str] | None,
        source: str,
    ) -> None:
        job_id, started_at = self._register_job(event, source=source)
        try:
            task = asyncio.create_task(
                self._run_background_generation(
                    event,
                    prompt,
                    size=size,
                    quality=quality,
                    characters=characters,
                    job_id=job_id,
                    started_at=started_at,
                ),
                name=f"{PLUGIN_NAME}:generate_image",
            )
        except Exception:
            self._finish_job(job_id)
            raise
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        size: str | None = None,
        quality: str | None = None,
        characters: list[str] | None = None,
        started_at: float | None = None,
        job_id: int | None = None,
    ) -> str:
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

        resolved = self.references.resolve(raw_prompt, characters)
        if resolved.unknown_requested_names and self._bool(
            "references", "strict_references", default=True
        ):
            return "未配置这些人物参考：" + "、".join(resolved.unknown_requested_names)
        if resolved.missing_characters and self._bool(
            "references", "strict_references", default=True
        ):
            return "人物参考图缺失或不可用：" + "、".join(resolved.missing_characters)

        prompt_prefix = str(
            self._get("generation", "prompt_prefix", default="") or ""
        ).strip()
        effective_prompt = CharacterReferenceManager.augment_prompt(
            raw_prompt, resolved
        )
        if prompt_prefix:
            effective_prompt = f"{prompt_prefix}\n{effective_prompt}"

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
        self._set_job_stage(job_id, "安全审核")
        try:
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
            if resolved.image_paths:
                generated = await client.edit(
                    effective_prompt,
                    resolved.image_paths,
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
            self._set_job_stage(job_id, "发送图片")
            try:
                await event.send(
                    event.chain_result([Comp.Image.fromFileSystem(str(output_path))])
                )
            except Exception as exc:
                logger.warning(
                    "[img_gener] generated but failed to send image path=%s error=%s",
                    output_path,
                    str(exc)[:500],
                )
                return "图片已经生成并计入配额，但聊天平台发送失败，请联系管理员查看日志。"
            reference_text = (
                "；已携带人物参考：" + "、".join(resolved.names)
                if resolved.names
                else ""
            )
            duration = self._format_duration(time.monotonic() - started_at)
            return (
                f"图片已通过审核并发送（{selected_size}，{selected_quality}）"
                f"{reference_text}。总用时：{duration}。"
            )
        except ImageGeneratorError as exc:
            logger.warning("[img_gener] generation failed: %s", exc.detail[:800])
            return exc.public_message
        except Exception as exc:
            logger.exception("[img_gener] unexpected generation error: %s", exc)
            return "生图过程中发生内部错误，请联系管理员查看日志。"
        finally:
            if lease_active:
                await self.rate_limiter.cancel(rate.lease_id)

    @filter.llm_tool(name="generate_image")
    async def generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        size: str | None = None,
        quality: str | None = None,
        characters: list[str] | None = None,
    ) -> str:
        """Generate and send one policy-compliant image with GPT Image 2.

        Use this tool only when the user clearly asks to create an image. Keep the
        user's visual intent in prompt. The plugin performs safety review and rate
        limiting. Configured character names such as 可可子 or 菌菌 are detected
        automatically and their stored reference images are attached.

        Args:
            prompt(string): Complete visual description for the requested image.
            size(string): Optional output size: auto, 1024x1024, 1024x1536, or 1536x1024.
            quality(string): Optional quality: auto, low, medium, or high.
            characters(array): Optional configured character names; omit to auto-detect names and aliases from prompt.
        """  # noqa: E501
        if not self._is_allowed(event):
            return "生图功能未启用，或当前用户/群没有使用权限。"
        if not str(prompt or "").strip():
            return "请提供需要生成的画面描述。"
        self._start_background_generation(
            event,
            prompt,
            size=size,
            quality=quality,
            characters=characters,
            source="LLM",
        )
        return self._generation_start_message()

    @filter.command("生图", alias={"绘图", "draw"})
    async def draw_command(self, event: AstrMessageEvent, prompt: GreedyStr):
        """直接测试 GPT Image 2 生图；正常聊天也可由 LLM 自动调用。"""

        event.stop_event()
        raw_prompt = str(prompt or "").strip()
        if not self._is_allowed(event):
            yield event.plain_result(
                "生图功能未启用，或当前用户/群没有使用权限。"
            )
            return
        if not raw_prompt:
            yield event.plain_result("请提供需要生成的画面描述。")
            return
        self._start_background_generation(
            event,
            raw_prompt,
            size=None,
            quality=None,
            characters=None,
            source="/生图",
        )
        yield event.plain_result(self._generation_start_message())

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
        """查看任务池、当前用户与群聊配额以及基础配置状态。"""

        event.stop_event()
        user_id, group_id = self._identity(event)
        status = await self.rate_limiter.status(user_id, group_id)
        pool = self._active_job_snapshot(user_id, group_id)
        limits = self.rate_limiter.limits
        api_key_configured = bool(
            str(self._get("api", "api_key", default="") or "").strip()
        )
        review_key_configured = bool(
            str(self._get("safety", "review_api_key", default="") or "").strip()
        )
        review_model = str(
            self._get("safety", "review_model", default="") or "未配置"
        )
        fail_mode = "故障即拒绝" if self.moderator.fail_closed else "故障时放行"
        safety_text = (
            f"独立 LLM 审核（{fail_mode}）"
            if self.moderator.enabled
            else "仅本地硬规则"
        )
        lines = [
            "🎨 生图服务状态",
            "",
            "【服务配置】",
            f"• 生图模型：{self._get('api', 'model', default='gpt-image-2')}",
            f"• 生图接口：{self._get('api', 'base_url', default='https://uuapi.cc/v1')}",
            f"• 生图密钥：{'已配置' if api_key_configured else '未配置'}",
            f"• 安全审核：{safety_text}",
            f"• 审核模型：{review_model}",
            f"• 审核密钥：{'已配置' if review_key_configured else '未配置'}",
            f"• 人物参考：{len(self.references.characters)} 个",
            "",
            "【任务池 / 队列】",
            f"• 全局任务：{pool['global']} 个",
            f"• 并发槽占用：{status['global_inflight']}/{limits.global_max_concurrent}",
            f"• 我的任务：{pool['user']} 个",
        ]
        if group_id:
            lines.append(
                f"• 本群任务：{pool['group']}/{limits.group_max_concurrent}"
            )

        visible_jobs: list[ActiveGenerationJob] = pool["visible"]
        if visible_jobs:
            lines.extend(["", "【当前任务】"])
            now = time.monotonic()
            for job in visible_jobs[:8]:
                owner = "我的" if job.user_id == user_id else "本群"
                elapsed = self._format_duration(now - job.started_at)
                lines.append(
                    f"• #{job.job_id:04d}｜{owner}｜{job.source}｜"
                    f"{job.stage}｜已用 {elapsed}"
                )
            if len(visible_jobs) > 8:
                lines.append(f"• 另有 {len(visible_jobs) - 8} 个任务未展开")
        else:
            lines.append("• 当前用户/群暂无等待或处理中的任务")

        user_attempts = self._format_quota(
            status["user_attempts"], limits.user_max_attempts
        )
        user_generations = self._format_quota(
            status["user_generations"], limits.user_max_generations
        )
        group_generations = self._format_quota(
            status["group_generations"], limits.group_max_generations
        )
        lines.extend(
            [
                "",
                "【我的配额】",
                f"• 调用次数：{user_attempts}"
                f" · {self._format_window(limits.user_attempt_window_seconds)}窗口",
                f"• 成功生图：{user_generations}"
                f" · {self._format_window(limits.user_generation_window_seconds)}窗口",
            ]
        )
        if group_id:
            lines.extend(
                [
                    "",
                    "【本群配额】",
                    f"• 成功生图：{group_generations}"
                    f" · {self._format_window(limits.group_generation_window_seconds)}窗口",
                ]
            )
        yield event.plain_result("\n".join(lines))

