from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from .character_references import CharacterReference
except ImportError:  # pragma: no cover - direct script-style import fallback.
    from character_references import CharacterReference


MAX_PROFILE_ALIASES_LENGTH = 20
MAX_PROFILE_IMAGES = 3


@dataclass(frozen=True, slots=True)
class UserProfile:
    qq: str
    display_name: str
    aliases: tuple[str, ...]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class AliasMatch:
    alias: str
    candidates: tuple[UserProfile, ...]


class UserProfileStore:
    """QQ-keyed portrait profiles with bounded aliases and reference candidates."""

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
                CREATE TABLE IF NOT EXISTS user_profiles (
                    qq TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_profile_aliases (
                    qq TEXT NOT NULL,
                    alias TEXT NOT NULL COLLATE NOCASE
                        CHECK(length(alias) <= 20),
                    group_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(qq, alias),
                    FOREIGN KEY(qq) REFERENCES user_profiles(qq) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_user_profile_alias_lookup
                    ON user_profile_aliases(alias COLLATE NOCASE);
                CREATE TABLE IF NOT EXISTS user_profile_images (
                    qq TEXT NOT NULL,
                    gallery_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(qq, gallery_id),
                    FOREIGN KEY(qq) REFERENCES user_profiles(qq) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_user_profile_images_recent
                    ON user_profile_images(qq, created_at DESC, gallery_id DESC);
                """
            )

    @staticmethod
    def normalize_qq(value: str) -> str:
        qq = str(value or "").strip()
        if qq.lower().startswith("qq:"):
            qq = qq[3:].strip()
        return qq[:64]

    @staticmethod
    def normalize_alias(value: str) -> str:
        return " ".join(str(value or "").replace("\x00", "").split())[
            :MAX_PROFILE_ALIASES_LENGTH
        ]

    def ensure_profile(
        self,
        qq: str,
        *,
        display_name: str = "",
        group_id: str | None = None,
        now: float | None = None,
    ) -> UserProfile:
        normalized_qq = self.normalize_qq(qq)
        if not normalized_qq:
            raise ValueError("QQ number is required")
        alias = self.normalize_alias(display_name)
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT INTO user_profiles(qq, display_name, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(qq) DO UPDATE SET
                    display_name = CASE
                        WHEN excluded.display_name <> '' THEN excluded.display_name
                        ELSE user_profiles.display_name
                    END,
                    updated_at = excluded.updated_at
                """,
                (normalized_qq, alias, current, current),
            )
            if alias and alias.casefold() != normalized_qq.casefold():
                connection.execute(
                    """
                    INSERT INTO user_profile_aliases(
                        qq, alias, group_id, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(qq, alias) DO UPDATE SET
                        group_id = excluded.group_id,
                        updated_at = excluded.updated_at
                    """,
                    (normalized_qq, alias, str(group_id or "") or None, current, current),
                )
        profile = self.get(normalized_qq)
        if profile is None:  # pragma: no cover
            raise RuntimeError("user profile disappeared after commit")
        return profile

    def add_reference(
        self,
        qq: str,
        *,
        gallery_id: int,
        image_path: Path,
        now: float | None = None,
    ) -> tuple[int, ...]:
        normalized_qq = self.normalize_qq(qq)
        current = time.time() if now is None else float(now)
        self.ensure_profile(normalized_qq, now=current)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profile_images(qq, gallery_id, image_path, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(qq, gallery_id) DO UPDATE SET
                    image_path = excluded.image_path,
                    created_at = excluded.created_at
                """,
                (normalized_qq, int(gallery_id), str(image_path.resolve()), current),
            )
            rows = connection.execute(
                """
                SELECT gallery_id FROM user_profile_images
                WHERE qq = ?
                ORDER BY created_at DESC, gallery_id DESC
                """,
                (normalized_qq,),
            ).fetchall()
            removed = tuple(int(row["gallery_id"]) for row in rows[MAX_PROFILE_IMAGES:])
            if removed:
                placeholders = ",".join("?" for _ in removed)
                connection.execute(
                    f"DELETE FROM user_profile_images WHERE qq = ? "
                    f"AND gallery_id IN ({placeholders})",
                    (normalized_qq, *removed),
                )
        return removed

    def get(self, qq: str) -> UserProfile | None:
        normalized_qq = self.normalize_qq(qq)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT qq, display_name FROM user_profiles WHERE qq = ?",
                (normalized_qq,),
            ).fetchone()
            if row is None:
                return None
            aliases = connection.execute(
                """
                SELECT alias FROM user_profile_aliases
                WHERE qq = ? ORDER BY updated_at DESC, alias COLLATE NOCASE
                """,
                (normalized_qq,),
            ).fetchall()
            images = connection.execute(
                """
                SELECT image_path FROM user_profile_images
                WHERE qq = ? ORDER BY created_at DESC, gallery_id DESC
                LIMIT ?
                """,
                (normalized_qq, MAX_PROFILE_IMAGES),
            ).fetchall()
        paths = tuple(
            path
            for image in images
            if (path := Path(str(image["image_path"]))).is_file()
        )
        return UserProfile(
            qq=str(row["qq"]),
            display_name=str(row["display_name"]),
            aliases=tuple(str(item["alias"]) for item in aliases),
            image_paths=paths,
        )

    def profiles(self, qqs: list[str] | tuple[str, ...]) -> tuple[UserProfile, ...]:
        result: list[UserProfile] = []
        seen: set[str] = set()
        for qq in qqs:
            normalized = self.normalize_qq(qq)
            if not normalized or normalized in seen:
                continue
            profile = self.get(normalized)
            if profile is not None:
                result.append(profile)
                seen.add(normalized)
        return tuple(result)

    @staticmethod
    def _contains(text: str, alias: str) -> bool:
        if alias.isascii() and re.fullmatch(r"[A-Za-z0-9_\- ]+", alias):
            return bool(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                    text,
                    re.I,
                )
            )
        return alias in text

    def match_aliases(self, text: str) -> tuple[AliasMatch, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT alias FROM user_profile_aliases
                ORDER BY length(alias) DESC, alias COLLATE NOCASE
                """
            ).fetchall()
        matches: list[AliasMatch] = []
        for row in rows:
            alias = str(row["alias"])
            if len(alias) < 2 or not self._contains(text, alias):
                continue
            with self._connect() as connection:
                candidates = connection.execute(
                    """
                    SELECT qq FROM user_profile_aliases
                    WHERE alias = ? COLLATE NOCASE ORDER BY qq
                    """,
                    (alias,),
                ).fetchall()
            profiles = self.profiles(tuple(str(item["qq"]) for item in candidates))
            if profiles:
                matches.append(AliasMatch(alias=alias, candidates=profiles))
        return tuple(matches)

    def alias_candidates(self, alias: str) -> tuple[UserProfile, ...]:
        normalized = self.normalize_alias(alias)
        if not normalized:
            return ()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT qq FROM user_profile_aliases
                WHERE alias = ? COLLATE NOCASE ORDER BY qq
                """,
                (normalized,),
            ).fetchall()
        return self.profiles(tuple(str(row["qq"]) for row in rows))

    def character_references(
        self, qqs: list[str] | tuple[str, ...]
    ) -> tuple[CharacterReference, ...]:
        return tuple(
            CharacterReference(
                name=profile.qq,
                aliases=profile.aliases,
                image_paths=profile.image_paths,
            )
            for profile in self.profiles(qqs)
        )
