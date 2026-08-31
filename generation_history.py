from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


TERMINAL_STATUSES = {
    "succeeded",
    "rejected",
    "failed",
    "delivery_failed",
    "cancelled",
    "interrupted",
}


@dataclass(frozen=True, slots=True)
class GenerationHistoryRecord:
    job_id: int
    user_id: str
    group_id: str | None
    source: str
    prompt: str
    size: str
    quality: str
    status: str
    stage: str
    created_at: float
    updated_at: float
    finished_at: float | None
    result_message: str
    output_filename: str


class GenerationHistoryStore:
    """SQLite task ledger used by commands and Agent status tools."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
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
                CREATE TABLE IF NOT EXISTS generation_history (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    group_id TEXT,
                    source TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    size TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL,
                    result_message TEXT NOT NULL DEFAULT '',
                    output_filename TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_generation_history_user
                    ON generation_history(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_generation_history_group
                    ON generation_history(group_id, created_at DESC);
                """
            )

    @staticmethod
    def _compact(value: str, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    def create_job(
        self,
        *,
        user_id: str,
        group_id: str | None,
        source: str,
        prompt: str,
        size: str,
        quality: str,
        now: float | None = None,
    ) -> int:
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO generation_history(
                    user_id, group_id, source, prompt, size, quality,
                    status, stage, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, 'queued', '等待调度', ?, ?)
                """,
                (
                    user_id,
                    group_id,
                    self._compact(source, 40),
                    self._compact(prompt, 2000),
                    self._compact(size, 32),
                    self._compact(quality, 24),
                    current,
                    current,
                ),
            )
            return int(cursor.lastrowid)

    def update_stage(
        self, job_id: int, stage: str, *, now: float | None = None
    ) -> None:
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE generation_history
                SET status = 'running', stage = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (self._compact(stage, 80), current, int(job_id)),
            )

    def finish_job(
        self,
        job_id: int,
        *,
        status: str,
        stage: str,
        result_message: str = "",
        output_filename: str = "",
        now: float | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE generation_history
                SET status = ?, stage = ?, updated_at = ?, finished_at = ?,
                    result_message = ?, output_filename = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    self._compact(stage, 80),
                    current,
                    current,
                    self._compact(result_message, 1000),
                    self._compact(output_filename, 255),
                    int(job_id),
                ),
            )

    def recover_unfinished(self, *, now: float | None = None) -> int:
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_history
                SET status = 'interrupted', stage = '插件重启后中断',
                    updated_at = ?, finished_at = ?,
                    result_message = '任务因插件重启或进程中断，当前已不在执行队列中。'
                WHERE status IN ('queued', 'running')
                """,
                (current, current),
            )
            return int(cursor.rowcount)

    def recent_for_user(
        self, user_id: str, *, limit: int = 5
    ) -> tuple[GenerationHistoryRecord, ...]:
        bounded_limit = max(1, min(20, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generation_history
                WHERE user_id = ?
                ORDER BY job_id DESC
                LIMIT ?
                """,
                (user_id, bounded_limit),
            ).fetchall()
        return tuple(GenerationHistoryRecord(**dict(row)) for row in rows)

    def cleanup(self, retention_days: int, *, now: float | None = None) -> int:
        if retention_days <= 0:
            return 0
        current = time.time() if now is None else float(now)
        cutoff = current - retention_days * 86400
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM generation_history WHERE created_at < ?",
                (cutoff,),
            )
            return int(cursor.rowcount)
