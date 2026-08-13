from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from .errors import ConfigurationError


def get_config(config: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    """Read a nested configuration value without mutating AstrBotConfig."""

    current: Any = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "开启"}
    return bool(value)


def as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def as_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = value.replace("，", ",").split(",")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def normalize_base_url(value: Any) -> str:
    base_url = str(value or "https://uuapi.cc/v1").strip().rstrip("/")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigurationError("生图 API 地址必须以 http:// 或 https:// 开头。")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def resolve_secret(value: Any) -> str:
    """Resolve a literal secret or an environment reference such as $UUAPI_KEY."""

    secret = str(value or "").strip()
    if secret.startswith("$") and len(secret) > 1:
        env_name = secret[1:]
        secret = str(os.environ.get(env_name, "")).strip()
        if not secret:
            raise ConfigurationError(f"环境变量 {env_name} 未设置，无法连接生图服务。")
    if not secret:
        raise ConfigurationError("尚未配置 UUAPI API Key。")
    return secret

