"""Bounded, parameterized archive management; no model or quota calls."""

from __future__ import annotations

import time
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .gallery import GalleryImage, GalleryStore


class AdminError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def integer(value: Any, name: str, minimum: int = 1, maximum: int = 2**53 - 1) -> int:
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit():
        raise AdminError(f"{name}必须是整数。")
    result = int(value)
    if not minimum <= result <= maximum:
        raise AdminError(f"{name}超出允许范围（{minimum}～{maximum}）。")
    return result


def text_value(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise AdminError(f"{name}必须是最多 {limit} 字的文本。")
    result = value.strip()
    if required and not result:
        raise AdminError(f"{name}不能为空。")
    return result


def serialize(record: GalleryImage) -> dict:
    result = asdict(record)
    result.pop("image_path")
    # Local paths are never returned to the browser.
    return result


class AdminGallery:
    def __init__(self, gallery: GalleryStore, profile_database: Path):
        self.gallery = gallery
        self.profile_database = profile_database

    def list_images(self, query: Any) -> dict:
        page = integer(query.get("page", "1"), "页码", maximum=1000000)
        limit = integer(query.get("limit", "24"), "每页数量", maximum=60)
        order = query.get("order", "desc")
        if order not in {"asc", "desc"}:
            raise AdminError("排序只能是 asc 或 desc。")
        deleted = query.get("deleted", "0")
        if deleted not in {"0", "1"}:
            raise AdminError("回收站参数无效。")
        where = ["deleted_at IS NOT NULL" if deleted == "1" else "deleted_at IS NULL"]
        args: list[Any] = []
        for field, length in (("requester_qq", 64), ("group_id", 64)):
            value = text_value(query.get(field, ""), field, length)
            if value:
                where.append(f"{field} = ?")
                args.append(value)
        if query.get("image_id"):
            where.append("image_id = ?")
            args.append(integer(query.get("image_id"), "图库编号"))
        search = text_value(query.get("search", ""), "搜索词", 200)
        if search:
            where.append(
                "(instr(prompt, ?) > 0 OR instr(requester_name, ?) > 0 OR instr(admin_note, ?) > 0)"
            )
            args.extend([search] * 3)
        for field, operator in (("date_from", ">="), ("date_to", "<")):
            value = query.get(field, "")
            if value:
                try:
                    date = datetime.strptime(value, "%Y-%m-%d").replace(
                        tzinfo=timezone(timedelta(hours=8))
                    )
                except (ValueError, TypeError) as exc:
                    raise AdminError("日期格式必须为 YYYY-MM-DD。") from exc
                if field == "date_to":
                    date += timedelta(days=1)
                where.append(f"created_at {operator} ?")
                args.append(date.timestamp())
        condition = " AND ".join(where)
        with closing(self.gallery._connect()) as db, db:
            db.execute("BEGIN")
            total = db.execute(
                f"SELECT COUNT(*) FROM gallery_images WHERE {condition}", args
            ).fetchone()[0]
            rows = db.execute(
                f"SELECT * FROM gallery_images WHERE {condition} "
                f"ORDER BY image_id {order.upper()} LIMIT ? OFFSET ?",
                (*args, limit, (page - 1) * limit),
            ).fetchall()
            stats = dict(
                db.execute(
                    "SELECT COUNT(*) AS total, COUNT(DISTINCT requester_qq) AS users, "
                    "MAX(image_id) AS latest_id FROM gallery_images WHERE deleted_at IS NULL"
                ).fetchone()
            )
            stats["deleted"] = db.execute(
                "SELECT COUNT(*) FROM gallery_images WHERE deleted_at IS NOT NULL"
            ).fetchone()[0]
        return {
            "items": [serialize(self.gallery._record(row)) for row in rows],
            "total": total,
            "page": page,
            "limit": limit,
            "stats": stats,
        }

    def list_users(self, query: Any) -> dict:
        page = integer(query.get("page", "1"), "页码", maximum=1000000)
        limit = integer(query.get("limit", "24"), "每页数量", maximum=60)
        search = text_value(query.get("search", ""), "搜索词", 200)
        # Search historical names too; grouping always uses the immutable platform ID.
        condition = (
            "deleted_at IS NULL AND (instr(requester_qq, ?) > 0 OR "
            "requester_qq IN (SELECT requester_qq FROM gallery_images "
            "WHERE deleted_at IS NULL AND instr(requester_name, ?) > 0))"
        )
        with closing(self.gallery._connect()) as db, db:
            db.execute("BEGIN")
            total = db.execute(
                f"SELECT COUNT(DISTINCT requester_qq) FROM gallery_images WHERE {condition}",
                (search, search),
            ).fetchone()[0]
            rows = db.execute(
                "SELECT summary.*, latest.requester_name FROM ("
                "SELECT requester_qq, COUNT(*) AS count, MAX(image_id) AS latest_id, "
                "MAX(created_at) AS latest_at, COUNT(DISTINCT group_id) AS group_count "
                f"FROM gallery_images WHERE {condition} GROUP BY requester_qq"
                ") summary JOIN gallery_images latest ON latest.image_id = summary.latest_id "
                "ORDER BY summary.latest_id DESC LIMIT ? OFFSET ?",
                (search, search, limit, (page - 1) * limit),
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total, "page": page, "limit": limit}

    def get(self, image_id: Any) -> GalleryImage:
        record = self.gallery.get(integer(image_id, "图库编号"), include_deleted=True)
        if record is None:
            raise AdminError("该图库编号不存在。", 404)
        return record

    def image_path(self, record: GalleryImage) -> Path:
        path = record.image_path.resolve()
        if path.parent != self.gallery.image_dir.resolve():
            raise AdminError("图片路径超出图库目录。", 403)
        if not path.is_file():
            raise AdminError("原图文件缺失，仍可管理这条记录。", 404)
        return path

    def update(self, payload: dict) -> dict:
        image_id = integer(payload.get("image_id"), "图库编号")
        revision = integer(payload.get("revision"), "记录版本", minimum=0)
        fields = payload.get("fields")
        allowed = {"requester_name": 80, "prompt": 8000, "admin_note": 2000}
        if not isinstance(fields, dict) or not fields or set(fields) - allowed.keys():
            raise AdminError("仅允许编辑显示昵称、提示词和管理备注。")
        values = {
            key: text_value(value, key, allowed[key], required=key == "prompt")
            for key, value in fields.items()
        }
        with closing(self.gallery._connect()) as db, db:
            cursor = db.execute(
                "UPDATE gallery_images SET "
                + ", ".join(f"{key} = ?" for key in values)
                + ", revision = revision + 1 WHERE image_id = ? AND revision = ? "
                "AND deleted_at IS NULL",
                (*values.values(), image_id, revision),
            )
            if not cursor.rowcount:
                raise AdminError("记录已被修改或删除，请刷新后重试。", 409)
        return serialize(self.get(image_id))

    def change_deleted(self, payload: dict, *, restore: bool = False) -> dict:
        entries = payload.get("items")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 60:
            raise AdminError("一次请选择 1～60 条记录。")
        selected = []
        for item in entries:
            if not isinstance(item, dict):
                raise AdminError("记录格式无效。")
            selected.append(
                (
                    integer(item.get("image_id"), "图库编号"),
                    integer(item.get("revision"), "记录版本", minimum=0),
                )
            )
        if len({item[0] for item in selected}) != len(selected):
            raise AdminError("图库编号不能重复。")
        with closing(self.gallery._connect()) as db, db:
            db.execute("ATTACH DATABASE ? AS profiles", (str(self.profile_database),))
            db.execute("BEGIN IMMEDIATE")
            for image_id, revision in selected:
                cursor = db.execute(
                    "UPDATE gallery_images SET deleted_at = ?, revision = revision + 1 "
                    "WHERE image_id = ? AND revision = ? AND deleted_at IS "
                    + ("NOT NULL" if restore else "NULL"),
                    (None if restore else time.time(), image_id, revision),
                )
                if not cursor.rowcount:
                    raise AdminError(f"#{image_id} 已发生变化，请刷新后重试；本批次未修改。", 409)
                if not restore:
                    db.execute(
                        "DELETE FROM profiles.user_profile_images WHERE gallery_id = ?", (image_id,)
                    )
        return {"changed": len(selected)}
