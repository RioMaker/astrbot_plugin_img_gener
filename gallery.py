from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .gallery_selection import MAX_GALLERY_IMAGES, MAX_SQLITE_INTEGER

ALLOWED_GALLERY_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True, slots=True)
class GalleryImage:
    image_id: int
    job_id: int | None
    requester_qq: str
    requester_name: str
    group_id: str | None
    prompt: str
    size: str
    quality: str
    media_type: str
    image_path: Path
    created_at: float
    duration_seconds: float


class GalleryStore:
    """Permanent successful-image archive under AstrBot's plugin data directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.image_dir = root_dir / "images"
        self.database_path = root_dir / "gallery.sqlite3"
        self.image_dir.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS gallery_images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    requester_qq TEXT NOT NULL,
                    requester_name TEXT NOT NULL,
                    group_id TEXT,
                    prompt TEXT NOT NULL,
                    size TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    duration_seconds REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_gallery_images_job
                    ON gallery_images(job_id);
                CREATE INDEX IF NOT EXISTS idx_gallery_images_requester
                    ON gallery_images(requester_qq, image_id DESC);
                """
            )

    @staticmethod
    def _compact(value: str, limit: int) -> str:
        return " ".join(str(value or "").replace("\x00", "").split())[:limit]

    @staticmethod
    def _suffix(extension: str, media_type: str) -> str:
        suffix = str(extension or "").strip().lower()
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix in ALLOWED_GALLERY_SUFFIXES:
            return suffix
        return {
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }.get(str(media_type).lower(), ".png")

    def _record(self, row: sqlite3.Row | None) -> GalleryImage | None:
        if row is None:
            return None
        return GalleryImage(
            image_id=int(row["image_id"]),
            job_id=int(row["job_id"]) if row["job_id"] is not None else None,
            requester_qq=str(row["requester_qq"]),
            requester_name=str(row["requester_name"]),
            group_id=str(row["group_id"]) if row["group_id"] else None,
            prompt=str(row["prompt"]),
            size=str(row["size"]),
            quality=str(row["quality"]),
            media_type=str(row["media_type"]),
            image_path=(self.image_dir / str(row["image_filename"])).resolve(),
            created_at=float(row["created_at"]),
            duration_seconds=float(row["duration_seconds"]),
        )

    def add_image(
        self,
        content: bytes,
        *,
        extension: str,
        media_type: str,
        requester_qq: str,
        requester_name: str,
        group_id: str | None,
        prompt: str,
        size: str,
        quality: str,
        duration_seconds: float,
        job_id: int | None = None,
        now: float | None = None,
    ) -> GalleryImage:
        if not content:
            raise ValueError("gallery image content is empty")
        current = time.time() if now is None else float(now)
        suffix = self._suffix(extension, media_type)
        connection = self._connect()
        temporary_path: Path | None = None
        final_path: Path | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO gallery_images(
                    job_id, requester_qq, requester_name, group_id, prompt,
                    size, quality, media_type, image_filename, created_at,
                    duration_seconds
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)
                """,
                (
                    int(job_id) if job_id is not None else None,
                    self._compact(requester_qq, 64),
                    self._compact(requester_name, 80),
                    self._compact(group_id or "", 64) or None,
                    self._compact(prompt, 8000),
                    self._compact(size, 32),
                    self._compact(quality, 24),
                    self._compact(media_type, 64),
                    current,
                    max(0.0, float(duration_seconds)),
                ),
            )
            image_id = int(cursor.lastrowid)
            final_path = self.image_dir / f"{image_id}{suffix}"
            temporary_path = self.image_dir / f".{image_id}.{uuid.uuid4().hex}.tmp"
            temporary_path.write_bytes(content)
            temporary_path.replace(final_path)
            connection.execute(
                "UPDATE gallery_images SET image_filename = ? WHERE image_id = ?",
                (final_path.name, image_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            for path in (temporary_path, final_path):
                if path is not None:
                    path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        record = self.get(image_id)
        if record is None:  # pragma: no cover - defensive SQLite consistency check.
            raise RuntimeError("gallery record disappeared after commit")
        return record

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM gallery_images"
            ).fetchone()
        return int(row["count"] if row else 0)

    def get(self, image_id: int) -> GalleryImage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM gallery_images WHERE image_id = ?",
                (int(image_id),),
            ).fetchone()
        return self._record(row)

    def recent_for_requester(
        self, requester_qq: str, *, offset: int = 0, limit: int = MAX_GALLERY_IMAGES
    ) -> tuple[int, tuple[GalleryImage, ...]]:
        """Read a person's archive count and page from one consistent snapshot."""

        offset = int(offset)
        if not 0 <= offset <= MAX_SQLITE_INTEGER:
            raise ValueError("gallery offset must be a nonnegative SQLite integer")
        bounded = max(1, min(MAX_GALLERY_IMAGES, int(limit)))
        requester = self._compact(requester_qq, 64)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN")
            count_row = connection.execute(
                "SELECT COUNT(*) AS count FROM gallery_images WHERE requester_qq = ?",
                (requester,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT * FROM gallery_images
                WHERE requester_qq = ?
                ORDER BY image_id DESC LIMIT ? OFFSET ?
                """,
                (requester, bounded, offset),
            ).fetchall()
        total = int(count_row["count"] if count_row else 0)
        records = tuple(record for row in rows if (record := self._record(row)) is not None)
        return total, records

    def get_many(self, image_ids: tuple[int, ...]) -> tuple[GalleryImage, ...]:
        """Read global archive IDs in caller order, preserving repeated IDs."""

        if len(image_ids) > MAX_GALLERY_IMAGES:
            raise ValueError(f"at most {MAX_GALLERY_IMAGES} gallery IDs may be queried")
        if not image_ids:
            return ()
        selected = tuple(int(image_id) for image_id in image_ids)
        if any(not 1 <= image_id <= MAX_SQLITE_INTEGER for image_id in selected):
            raise ValueError("gallery IDs must be positive SQLite integers")
        unique_ids = tuple(dict.fromkeys(selected))
        placeholders = ",".join("?" for _ in unique_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM gallery_images WHERE image_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        records = {
            record.image_id: record for row in rows if (record := self._record(row)) is not None
        }
        return tuple(records[image_id] for image_id in selected if image_id in records)

    def random(self, limit: int = 1) -> tuple[GalleryImage, ...]:
        bounded = max(1, min(9, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM gallery_images ORDER BY RANDOM() LIMIT ?",
                (bounded,),
            ).fetchall()
        return tuple(record for row in rows if (record := self._record(row)) is not None)
