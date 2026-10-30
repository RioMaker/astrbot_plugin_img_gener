from __future__ import annotations

import asyncio
import copy
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot_plugin_img_gener.admin_api import AdminAPI
from astrbot_plugin_img_gener.admin_config import AdminConfig
from astrbot_plugin_img_gener.admin_gallery import AdminError, AdminGallery
from astrbot_plugin_img_gener.gallery import GalleryStore
from astrbot_plugin_img_gener.user_profiles import UserProfileStore
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def add(store, qq="123", name="用户甲", now=1_700_000_000):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 60), "blue").save(buffer, "PNG")
    return store.add_image(
        buffer.getvalue(),
        extension=".png",
        media_type="image/png",
        requester_qq=qq,
        requester_name=name,
        group_id="456",
        prompt="雨夜车站 100%",
        size="816x816",
        quality="auto",
        duration_seconds=10,
        job_id=1,
        now=now,
    )


@pytest.fixture
def archive(tmp_path):
    gallery = GalleryStore(tmp_path / "gallery")
    profiles = UserProfileStore(tmp_path / "profiles.sqlite3")
    admin = AdminGallery(gallery, profiles.database_path)
    return gallery, profiles, admin


def test_old_database_migrates_without_losing_images_or_ids(tmp_path):
    root = tmp_path / "gallery"
    store = GalleryStore(root)
    record = add(store)
    with sqlite3.connect(store.database_path) as db:
        for field in ("admin_note", "revision", "deleted_at"):
            db.execute(f"ALTER TABLE gallery_images DROP COLUMN {field}")
    reopened = GalleryStore(root)
    assert reopened.get(record.image_id).image_path.read_bytes() == record.image_path.read_bytes()
    assert reopened.get(record.image_id).revision == 0
    assert add(reopened).image_id == record.image_id + 1


def test_filtered_pagination_sort_and_user_grouping(archive):
    store, _, admin = archive
    first = add(store, name="旧昵称")
    second = add(store, qq="789", name="其他人")
    third = add(store, name="新昵称")
    assert [x["image_id"] for x in admin.list_images({"limit": "2"})["items"]] == [3, 2]
    assert admin.list_images({"limit": "2", "page": "2"})["items"][0]["image_id"] == 1
    assert [
        x["image_id"] for x in admin.list_images({"order": "asc", "requester_qq": "123"})["items"]
    ] == [first.image_id, third.image_id]
    assert admin.list_images({"image_id": str(second.image_id)})["total"] == 1
    assert admin.list_images({"search": "%"})["total"] == 3
    assert admin.list_images({"search": "' OR 1=1 --"})["total"] == 0
    assert admin.list_images({"date_from": "2023-11-15", "date_to": "2023-11-15"})["total"] == 3
    assert admin.list_images({"date_to": "2023-11-14"})["total"] == 0
    users = admin.list_users({"search": "旧昵称"})
    assert users["total"] == 1
    assert users["items"][0]["count"] == 2
    assert users["items"][0]["requester_name"] == "新昵称"
    assert "image_path" not in admin.list_images({})["items"][0]


@pytest.mark.parametrize(
    "query",
    [
        {"page": "0"},
        {"limit": "61"},
        {"order": "desc; DROP TABLE gallery_images"},
        {"image_id": "9223372036854775808"},
        {"page": True},
        {"deleted": "yes"},
        {"date_from": "yesterday"},
        {"search": "x" * 201},
    ],
)
def test_invalid_queries_are_rejected(archive, query):
    with pytest.raises(AdminError):
        archive[2].list_images(query)


def test_edit_conflicts_and_immutable_ownership(archive):
    store, _, admin = archive
    record = add(store)
    result = admin.update(
        {
            "image_id": record.image_id,
            "revision": 0,
            "fields": {"prompt": "修改后的提示词", "admin_note": "备注"},
        }
    )
    assert result["revision"] == 1
    assert store.get(record.image_id).prompt == "修改后的提示词"
    with pytest.raises(AdminError, match="已被修改"):
        admin.update({"image_id": record.image_id, "revision": 0, "fields": {"prompt": "旧页面"}})
    for field in ("requester_qq", "image_path", "created_at", "job_id"):
        with pytest.raises(AdminError):
            admin.update({"image_id": record.image_id, "revision": 1, "fields": {field: "bad"}})


def test_recycle_hides_all_chat_queries_detaches_reference_and_restores_id(archive):
    store, profiles, admin = archive
    record = add(store)
    profiles.add_reference("123", gallery_id=record.image_id, image_path=record.image_path)
    payload = {"items": [{"image_id": record.image_id, "revision": 0}]}
    admin.change_deleted(payload)
    assert store.get(record.image_id) is None
    assert store.count() == 0
    assert store.random() == ()
    assert store.get_many((record.image_id,)) == ()
    assert store.recent_for_requester("123") == (0, ())
    assert profiles.get("123").image_paths == ()
    assert record.image_path.is_file()
    assert admin.list_images({"deleted": "1"})["total"] == 1
    assert admin.list_users({})["total"] == 0
    payload["items"][0]["revision"] = 1
    admin.change_deleted(payload, restore=True)
    assert store.get(record.image_id).revision == 2
    assert profiles.get("123").image_paths == ()
    assert add(store).image_id == record.image_id + 1


def test_conflicting_batch_rolls_back_images_and_portrait_links(archive):
    store, profiles, admin = archive
    record = add(store)
    profiles.add_reference("123", gallery_id=record.image_id, image_path=record.image_path)
    with pytest.raises(AdminError):
        admin.change_deleted(
            {
                "items": [
                    {"image_id": record.image_id, "revision": 0},
                    {"image_id": 999, "revision": 0},
                ]
            }
        )
    assert store.count() == 1
    assert profiles.get("123").image_paths == (record.image_path,)


def test_preview_path_cannot_escape_gallery_and_missing_files_stay_manageable(archive, tmp_path):
    store, _, admin = archive
    record = add(store)
    outside = tmp_path / "private.txt"
    outside.write_text("secret", encoding="utf-8")
    with store._connect() as db:
        db.execute("UPDATE gallery_images SET image_filename = ?", (str(outside),))
    with pytest.raises(AdminError, match="超出"):
        admin.image_path(admin.get(record.image_id))
    record.image_path.unlink()
    with store._connect() as db:
        db.execute("UPDATE gallery_images SET image_filename = 'missing.png'")
    with pytest.raises(AdminError, match="缺失"):
        admin.image_path(admin.get(record.image_id))
    admin.change_deleted({"items": [{"image_id": record.image_id, "revision": 0}]})
    assert outside.read_text("utf-8") == "secret"


class Config(dict):
    saved = None
    fail = False

    def save_config(self):
        if self.fail:
            raise OSError("secret-that-must-not-leak")
        self.saved = copy.deepcopy(dict(self))


@pytest.fixture
def plugin(archive, tmp_path):
    store, profiles, _ = archive
    config = Config(
        {
            "api": {"api_key": "private-generation-secret"},
            "safety": {"review_api_key": "$REVIEW_SECRET"},
        }
    )
    routes = []
    context = SimpleNamespace(
        registered_web_apis=routes, register_web_api=lambda *args: routes.append(args)
    )
    return SimpleNamespace(
        config=config,
        plugin_dir=ROOT,
        data_dir=tmp_path,
        gallery_store=store,
        user_profiles=profiles,
        context=context,
        _active_jobs={},
        _background_tasks=set(),
        rate_limiter=SimpleNamespace(limits=None),
        references=None,
        moderator=None,
        output_store=SimpleNamespace(retention_days=7),
        _int=lambda *a, **kw: kw["default"],
    )


def test_settings_mask_secrets_preserve_unmodified_keys_and_apply_runtime(plugin):
    settings = AdminConfig(plugin)
    data = settings.read()
    assert "private-generation-secret" not in json.dumps(data)
    assert "$REVIEW_SECRET" not in json.dumps(data)
    assert data["secrets_set"]["api.api_key"]
    result = settings.save(
        {
            "revision": data["revision"],
            "changes": {
                "limits": {"user_max_generations": 13},
                "storage": {"retention_days": 30},
                "safety": {"blocked_terms": ["不允许的词"]},
            },
        }
    )
    assert plugin.config.saved["api"]["api_key"] == "private-generation-secret"
    assert plugin.rate_limiter.limits.user_max_generations == 13
    assert plugin.output_store.retention_days == 30
    assert plugin.moderator is not None
    assert result["revision"] != data["revision"]
    with pytest.raises(AdminError, match="已发生变化"):
        settings.save({"revision": data["revision"], "changes": {"api": {"model": "x"}}})


@pytest.mark.parametrize(
    "changes",
    [
        {"limits": {"global_max_concurrent": 0}},
        {"limits": {"user_max_attempts": True}},
        {"api": {"model": 123}},
        {"api": {"unknown": "x"}},
        {"unknown": {}},
        {"api": {"base_url": "file:///etc/passwd"}},
        {"api": {"base_url": "https://name:password@host/v1"}},
        {"generation": {"default_quality": "bad"}},
        {"generation": {"allowed_sizes": [123]}},
        {"references": {"characters": [{"__template_key": "../bad"}]}},
        {"safety": {"blocked_terms": "bad"}},
    ],
)
def test_settings_validation_is_atomic(plugin, changes):
    settings = AdminConfig(plugin)
    before = copy.deepcopy(dict(plugin.config))
    with pytest.raises(AdminError):
        settings.save({"revision": settings.revision(), "changes": changes})
    assert dict(plugin.config) == before
    assert plugin.config.saved is None


def test_settings_busy_write_failure_and_explicit_secret_clear(plugin):
    settings = AdminConfig(plugin)
    revision = settings.revision()
    plugin._active_jobs[1] = object()
    with pytest.raises(AdminError, match="仍有生图"):
        settings.save({"revision": revision, "changes": {}})
    plugin._active_jobs.clear()
    plugin.config.fail = True
    with pytest.raises(AdminError, match="写入失败"):
        settings.save({"revision": revision, "changes": {"api": {"api_key": "new-secret"}}})
    assert settings.revision() == revision
    assert plugin.config["api"]["api_key"] == "private-generation-secret"
    plugin.config.fail = False
    settings.save({"revision": revision, "changes": {"api": {"api_key": ""}}})
    assert not settings.read()["secrets_set"]["api.api_key"]


class Upload:
    def __init__(self, content):
        self.content = content

    async def read(self, size):
        return self.content[:size]


class Request:
    username = "admin"
    headers = {}
    content_type = "application/json"
    query = {}
    content = b"{}"
    upload = None

    async def body(self):
        return self.content

    async def files(self):
        return {"file": self.upload}


@pytest.fixture
def web_api(plugin):
    request = Request()
    web = SimpleNamespace(
        request=request,
        PluginUploadFile=Upload,
        json_response=lambda data, **kw: {"data": data, "status": kw.get("status_code", 200)},
        error_response=lambda message, **kw: {"error": message, "status": kw["status_code"]},
        file_response=lambda path, **kw: {"file": path, "status": 200},
    )
    api = AdminAPI(plugin, web)
    api.register()
    return api, request


def call(api, endpoint, method="GET"):
    handler = next(
        row[1]
        for row in api.plugin.context.registered_web_apis
        if row[0].endswith(f"/admin/{endpoint}") and row[2] == [method]
    )
    return asyncio.run(handler())


def test_every_endpoint_requires_dashboard_identity_and_is_removed_on_close(web_api):
    api, request = web_api
    request.username = None
    for _, handler, _, _ in api.plugin.context.registered_web_apis:
        assert asyncio.run(handler())["status"] == 401
    request.username = "admin"
    handler = api.plugin.context.registered_web_apis[0][1]
    api.close()
    assert api.plugin.context.registered_web_apis == []
    assert asyncio.run(handler())["status"] == 503


def test_api_rejects_cross_site_and_non_json_mutations(web_api):
    api, request = web_api
    request.headers = {"Sec-Fetch-Site": "cross-site"}
    assert call(api, "settings/save", "POST")["status"] == 403
    request.headers = {}
    request.content_type = "text/plain"
    assert call(api, "settings/save", "POST")["status"] == 415
    assert api.plugin.config.saved is None


def test_api_crud_preview_upload_and_bad_payloads(web_api):
    api, request = web_api
    record = add(api.plugin.gallery_store)
    request.query = {"image_id": str(record.image_id)}
    assert call(api, "detail")["data"]["image_id"] == record.image_id
    assert call(api, "preview")["data"]["data_url"].startswith("data:image/jpeg;base64,")
    assert call(api, "download")["file"] == record.image_path
    request.content = b"[]"
    assert call(api, "images/delete", "POST")["status"] == 400
    request.content = b"x" * (512 * 1024 + 1)
    assert call(api, "images/update", "POST")["status"] == 413
    request.content = json.dumps({"items": [{"image_id": record.image_id, "revision": 0}]}).encode()
    api.plugin._active_jobs[1] = object()
    assert call(api, "images/delete", "POST")["status"] == 409
    api.plugin._active_jobs.clear()
    assert call(api, "images/delete", "POST")["status"] == 200
    request.upload = Upload(b"not-an-image")
    assert call(api, "references/upload", "POST")["status"] == 400
    request.upload = Upload(record.image_path.read_bytes())
    result = call(api, "references/upload", "POST")["data"]
    assert result["path"].startswith("admin_references/")
    assert (api.plugin.data_dir / result["path"]).is_file()
