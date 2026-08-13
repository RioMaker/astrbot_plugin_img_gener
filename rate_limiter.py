from __future__ import annotations

import asyncio
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LimitConfig:
    user_cooldown_seconds: int = 60
    user_attempt_window_seconds: int = 600
    user_max_attempts: int = 3
    user_generation_window_seconds: int = 3600
    user_max_generations: int = 5
    group_cooldown_seconds: int = 15
    group_generation_window_seconds: int = 3600
    group_max_generations: int = 20
    global_max_concurrent: int = 2
    group_max_concurrent: int = 1
    reservation_ttl_seconds: int = 900


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    lease_id: str = ""
    code: str = ""
    retry_after: int = 0
    message: str = ""


class PersistentRateLimiter:
    """SQLite-backed user/group rate limiting with in-flight reservations.

    Attempts are recorded as soon as a request is admitted. Generation quotas are
    only charged after the image API succeeds, while reservations stop concurrent
    calls from overshooting those quotas.
    """

    def __init__(self, database_path: Path, limits: LimitConfig) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.limits = limits
        self._lock = asyncio.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rate_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rate_events_lookup
                    ON rate_events(scope_type, scope_id, event_type, created_at);
                CREATE TABLE IF NOT EXISTS generation_reservations (
                    lease_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    group_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_generation_reservations_user
                    ON generation_reservations(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_generation_reservations_group
                    ON generation_reservations(group_id, created_at);
                """
            )

    @staticmethod
    def _count_events(
        connection: sqlite3.Connection,
        scope_type: str,
        scope_id: str,
        event_type: str,
        since: float,
    ) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM rate_events
            WHERE scope_type = ? AND scope_id = ? AND event_type = ? AND created_at >= ?
            """,
            (scope_type, scope_id, event_type, since),
        ).fetchone()
        return int(row["count"] if row else 0)

    @staticmethod
    def _latest_event(
        connection: sqlite3.Connection,
        scope_type: str,
        scope_id: str,
        event_type: str,
    ) -> float | None:
        row = connection.execute(
            """
            SELECT MAX(created_at) AS created_at
            FROM rate_events
            WHERE scope_type = ? AND scope_id = ? AND event_type = ?
            """,
            (scope_type, scope_id, event_type),
        ).fetchone()
        if not row or row["created_at"] is None:
            return None
        return float(row["created_at"])

    @staticmethod
    def _count_reservations(
        connection: sqlite3.Connection, column: str | None = None, value: str = ""
    ) -> int:
        if column is None:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM generation_reservations"
            ).fetchone()
        else:
            if column not in {"user_id", "group_id"}:
                raise ValueError("invalid reservation column")
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM generation_reservations WHERE {column} = ?",
                (value,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def _cleanup(self, connection: sqlite3.Connection, now: float) -> None:
        longest_window = max(
            self.limits.user_attempt_window_seconds,
            self.limits.user_generation_window_seconds,
            self.limits.group_generation_window_seconds,
            self.limits.user_cooldown_seconds,
            self.limits.group_cooldown_seconds,
            3600,
        )
        connection.execute(
            "DELETE FROM rate_events WHERE created_at < ?",
            (now - longest_window - 60,),
        )
        connection.execute(
            "DELETE FROM generation_reservations WHERE created_at < ?",
            (now - self.limits.reservation_ttl_seconds,),
        )

    @staticmethod
    def _rejection(code: str, retry_after: float, message: str) -> RateLimitResult:
        return RateLimitResult(
            allowed=False,
            code=code,
            retry_after=max(1, math.ceil(retry_after)),
            message=message,
        )

    async def acquire(
        self,
        user_id: str,
        group_id: str | None,
        *,
        bypass_limits: bool = False,
        now: float | None = None,
    ) -> RateLimitResult:
        current = time.time() if now is None else float(now)
        async with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._cleanup(connection, current)

                global_inflight = self._count_reservations(connection)
                if global_inflight >= max(1, self.limits.global_max_concurrent):
                    connection.rollback()
                    return self._rejection(
                        "GLOBAL_BUSY", 10, "当前生图任务较多，请稍后再试。"
                    )

                if not bypass_limits:
                    user_pending = self._count_reservations(
                        connection, "user_id", user_id
                    )
                    user_generated = self._count_events(
                        connection,
                        "user",
                        user_id,
                        "generation",
                        current - self.limits.user_generation_window_seconds,
                    )
                    if (
                        self.limits.user_max_generations > 0
                        and user_generated + user_pending
                        >= self.limits.user_max_generations
                    ):
                        connection.rollback()
                        return self._rejection(
                            "USER_GENERATION_LIMIT",
                            self.limits.user_generation_window_seconds,
                            "你在当前计费窗口内的生图次数已用完，请稍后再试。",
                        )

                    if group_id:
                        group_pending = self._count_reservations(
                            connection, "group_id", group_id
                        )
                        if group_pending >= max(1, self.limits.group_max_concurrent):
                            connection.rollback()
                            return self._rejection(
                                "GROUP_BUSY", 10, "本群已有生图任务正在执行，请稍后再试。"
                            )
                        group_generated = self._count_events(
                            connection,
                            "group",
                            group_id,
                            "generation",
                            current - self.limits.group_generation_window_seconds,
                        )
                        if (
                            self.limits.group_max_generations > 0
                            and group_generated + group_pending
                            >= self.limits.group_max_generations
                        ):
                            connection.rollback()
                            return self._rejection(
                                "GROUP_GENERATION_LIMIT",
                                self.limits.group_generation_window_seconds,
                                "本群在当前计费窗口内的生图次数已用完。",
                            )

                    attempts = self._count_events(
                        connection,
                        "user",
                        user_id,
                        "attempt",
                        current - self.limits.user_attempt_window_seconds,
                    )
                    if (
                        self.limits.user_max_attempts > 0
                        and attempts >= self.limits.user_max_attempts
                    ):
                        connection.rollback()
                        return self._rejection(
                            "USER_ATTEMPT_LIMIT",
                            self.limits.user_attempt_window_seconds,
                            "你的生图调用过于频繁，请稍后再试。",
                        )

                    latest_user = self._latest_event(
                        connection, "user", user_id, "attempt"
                    )
                    if latest_user is not None:
                        remaining = (
                            latest_user
                            + self.limits.user_cooldown_seconds
                            - current
                        )
                        if remaining > 0:
                            connection.rollback()
                            return self._rejection(
                                "USER_COOLDOWN",
                                remaining,
                                f"你的生图调用太快了，请等待 {math.ceil(remaining)} 秒。",
                            )

                    if group_id:
                        latest_group = self._latest_event(
                            connection, "group", group_id, "attempt"
                        )
                        if latest_group is not None:
                            remaining = (
                                latest_group
                                + self.limits.group_cooldown_seconds
                                - current
                            )
                            if remaining > 0:
                                connection.rollback()
                                return self._rejection(
                                    "GROUP_COOLDOWN",
                                    remaining,
                                    f"本群刚刚调用过生图，请等待 {math.ceil(remaining)} 秒。",
                                )

                    connection.execute(
                        """
                        INSERT INTO rate_events(scope_type, scope_id, event_type, created_at)
                        VALUES('user', ?, 'attempt', ?)
                        """,
                        (user_id, current),
                    )
                    if group_id:
                        connection.execute(
                            """
                            INSERT INTO rate_events(scope_type, scope_id, event_type, created_at)
                            VALUES('group', ?, 'attempt', ?)
                            """,
                            (group_id, current),
                        )

                lease_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO generation_reservations(lease_id, user_id, group_id, created_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    (lease_id, user_id, group_id, current),
                )
                connection.commit()
                return RateLimitResult(allowed=True, lease_id=lease_id)
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def complete(self, lease_id: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        async with self._lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT user_id, group_id FROM generation_reservations
                    WHERE lease_id = ?
                    """,
                    (lease_id,),
                ).fetchone()
                if not row:
                    connection.rollback()
                    return
                connection.execute(
                    "DELETE FROM generation_reservations WHERE lease_id = ?", (lease_id,)
                )
                connection.execute(
                    """
                    INSERT INTO rate_events(scope_type, scope_id, event_type, created_at)
                    VALUES('user', ?, 'generation', ?)
                    """,
                    (str(row["user_id"]), current),
                )
                if row["group_id"]:
                    connection.execute(
                        """
                        INSERT INTO rate_events(scope_type, scope_id, event_type, created_at)
                        VALUES('group', ?, 'generation', ?)
                        """,
                        (str(row["group_id"]), current),
                    )
                connection.commit()

    async def cancel(self, lease_id: str) -> None:
        if not lease_id:
            return
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM generation_reservations WHERE lease_id = ?", (lease_id,)
                )

    async def status(
        self, user_id: str, group_id: str | None, *, now: float | None = None
    ) -> dict[str, int]:
        current = time.time() if now is None else float(now)
        async with self._lock:
            with self._connect() as connection:
                return {
                    "user_attempts": self._count_events(
                        connection,
                        "user",
                        user_id,
                        "attempt",
                        current - self.limits.user_attempt_window_seconds,
                    ),
                    "user_generations": self._count_events(
                        connection,
                        "user",
                        user_id,
                        "generation",
                        current - self.limits.user_generation_window_seconds,
                    ),
                    "group_generations": (
                        self._count_events(
                            connection,
                            "group",
                            group_id,
                            "generation",
                            current - self.limits.group_generation_window_seconds,
                        )
                        if group_id
                        else 0
                    ),
                    "global_inflight": self._count_reservations(connection),
                }

