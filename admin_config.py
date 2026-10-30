from __future__ import annotations

import copy
import hashlib
import json
import re
from urllib.parse import urlsplit

from .admin_gallery import AdminError
from .character_references import CharacterReferenceManager
from .moderation import PromptModerator
from .rate_limiter import LimitConfig

SECRET_FIELDS = {"api.api_key", "safety.review_api_key"}
ZERO_ALLOWED = {
    "user_cooldown_seconds",
    "user_max_attempts",
    "user_max_generations",
    "group_cooldown_seconds",
    "group_max_generations",
    "retention_days",
}


class AdminConfig:
    def __init__(self, plugin):
        self.plugin = plugin
        self.schema = json.loads((plugin.plugin_dir / "_conf_schema.json").read_text("utf-8"))

    def revision(self) -> str:
        # Use a per-process salt so this token cannot be used to guess secret values offline.
        import secrets

        if not hasattr(self, "_salt"):
            self._salt = secrets.token_bytes(32)
        raw = json.dumps(dict(self.plugin.config), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(self._salt + raw.encode()).hexdigest()

    def read(self) -> dict:
        values, secrets_set = {}, {}
        for group, section in self.schema.items():
            values[group] = {}
            for key, spec in section["items"].items():
                value = self.plugin.config.get(group, {}).get(key, spec.get("default"))
                path = f"{group}.{key}"
                if path in SECRET_FIELDS:
                    secrets_set[path] = bool(value)
                    value = ""
                values[group][key] = copy.deepcopy(value)
        return {
            "schema": self.schema,
            "values": values,
            "secrets_set": secrets_set,
            "revision": self.revision(),
            "busy": self.busy(),
        }

    def busy(self) -> bool:
        return bool(
            self.plugin._active_jobs
            or any(not task.done() for task in self.plugin._background_tasks)
        )

    def _validate(self, value, spec: dict, path: str):
        kind = spec["type"]
        if kind == "bool":
            valid = isinstance(value, bool)
        elif kind == "int":
            minimum = 0 if path.rsplit(".", 1)[-1] in ZERO_ALLOWED else 1
            minimum = {
                "api.timeout_seconds": 10,
                "safety.review_timeout_seconds": 5,
                "safety.max_prompt_chars": 200,
            }.get(path, minimum)
            valid = type(value) is int and minimum <= value <= 1000000
        elif kind in {"string", "text"}:
            valid = isinstance(value, str) and len(value) <= 32000 and "\x00" not in value
        elif kind in {"list", "file"}:
            valid = (
                isinstance(value, list)
                and len(value) <= 500
                and all(
                    isinstance(item, str) and len(item) <= 8000 and "\x00" not in item
                    for item in value
                )
            )
        elif kind == "template_list":
            valid = isinstance(value, list) and len(value) <= 100
            if valid:
                for entry in value:
                    if isinstance(entry, str) and "=" in entry and len(entry) <= 8000:
                        continue  # Existing legacy name|alias=path entries remain editable.
                    if not isinstance(entry, dict):
                        raise AdminError(f"{path}：人物必须为对象。")
                    template = entry.get("__template_key", "character")
                    if not isinstance(template, str) or template not in spec["templates"]:
                        raise AdminError(f"{path}：未知人物模板。")
                    fields = spec["templates"][template]["items"]
                    if set(entry) - (set(fields) | {"__template_key"}):
                        raise AdminError(f"{path}：存在未知人物字段。")
                    for key, item in entry.items():
                        if key != "__template_key":
                            self._validate(item, fields[key], f"{path}.{key}")
        else:
            valid = False
        if not valid:
            raise AdminError(f"{path}：类型或取值范围不正确。")
        if spec.get("options") and value not in spec["options"]:
            raise AdminError(f"{path}：请选择允许的选项。")
        if path.endswith("base_url"):
            try:
                parsed = urlsplit(value)
                _ = parsed.port  # Validate malformed port and IPv6 syntax as well.
            except ValueError as exc:
                raise AdminError(f"{path}：API 地址格式不正确。") from exc
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise AdminError(f"{path}：请填写不带密码、查询参数的 HTTP(S) 地址。")
        if path == "generation.allowed_sizes" and any(
            size != "auto" and not re.fullmatch(r"[1-9]\d{1,4}x[1-9]\d{1,4}", size)
            for size in value
        ):
            raise AdminError("允许尺寸须为 auto 或 宽x高。")

    def save(self, payload: dict) -> dict:
        if self.busy():
            raise AdminError("仍有生图任务运行，请完成后再保存设置。", 409)
        if payload.get("revision") != self.revision():
            raise AdminError("配置已发生变化，请重新读取后再保存。", 409)
        patch = payload.get("changes")
        if not isinstance(patch, dict) or set(patch) - self.schema.keys():
            raise AdminError("配置字段无效。")
        candidate = copy.deepcopy(dict(self.plugin.config))
        for group, changes in patch.items():
            fields = self.schema[group]["items"]
            if not isinstance(changes, dict) or set(changes) - fields.keys():
                raise AdminError(f"{group}：存在未知配置字段。")
            for key, value in changes.items():
                self._validate(value, fields[key], f"{group}.{key}")
                candidate.setdefault(group, {})[key] = copy.deepcopy(value)
        # Fill defaults for runtime construction without removing legacy saved fields.
        values = {
            group: {
                key: candidate.get(group, {}).get(key, spec.get("default"))
                for key, spec in section["items"].items()
            }
            for group, section in self.schema.items()
        }
        limits = LimitConfig(
            **{
                key: values["limits"][key]
                for key in LimitConfig.__dataclass_fields__
                if key != "reservation_ttl_seconds"
            },
            reservation_ttl_seconds=max(300, int(values["api"]["timeout_seconds"]) * 2),
        )
        refs = values["references"]
        references = CharacterReferenceManager(
            refs["characters"],
            data_dir=self.plugin.data_dir,
            plugin_dir=self.plugin.plugin_dir,
            max_total_images=refs["max_reference_images"],
            max_image_size_mb=refs["max_image_size_mb"],
            per_character_limit=refs["multi_character_image_limit"],
        )
        safety = values["safety"]
        moderator = PromptModerator(
            **{
                key: safety[key]
                for key in (
                    "enabled",
                    "fail_closed",
                    "blocked_terms",
                    "custom_policy",
                    "max_prompt_chars",
                )
            }
        )
        config = self.plugin.config
        save = getattr(config, "save_config", None)
        if not callable(save):
            raise AdminError("当前配置不支持持久化，请从 AstrBot 插件配置页保存。", 503)
        before = copy.deepcopy(dict(config))
        try:
            config.clear()
            config.update(candidate)
            save()
        except Exception as exc:
            config.clear()
            config.update(before)
            raise AdminError("配置写入失败，运行配置未变更，请检查存储权限。", 500) from exc
        # This synchronous section contains no await: tasks cannot observe a partial swap.
        self.plugin.rate_limiter.limits = limits
        self.plugin.references = references
        self.plugin.moderator = moderator
        self.plugin.output_store.retention_days = values["storage"]["retention_days"]
        return self.read()
