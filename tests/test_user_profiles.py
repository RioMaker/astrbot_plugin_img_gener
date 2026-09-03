from __future__ import annotations

import sqlite3

from astrbot_plugin_img_gener.user_profiles import (
    MAX_PROFILE_ALIASES_LENGTH,
    UserProfileStore,
)


def test_profile_alias_is_capped_and_collisions_keep_qq_identity(tmp_path) -> None:
    store = UserProfileStore(tmp_path / "profiles.sqlite3")
    long_name = "很长的群聊名称" * 5
    first = store.ensure_profile("10001", display_name=long_name, group_id="200")
    store.ensure_profile("10001", display_name="小明", group_id="200")
    store.ensure_profile("10002", display_name="小明", group_id="200")

    candidates = store.alias_candidates("小明")

    assert len(first.display_name) == MAX_PROFILE_ALIASES_LENGTH
    assert [profile.qq for profile in candidates] == ["10001", "10002"]
    assert all(profile.qq != profile.display_name for profile in candidates)


def test_profile_keeps_only_latest_reference(tmp_path) -> None:
    store = UserProfileStore(tmp_path / "profiles.sqlite3")
    store.ensure_profile("10001", display_name="小明", group_id="200")
    paths = []
    removed = ()
    for gallery_id in range(1, 5):
        path = tmp_path / f"{gallery_id}.png"
        path.write_bytes(b"png")
        paths.append(path.resolve())
        removed = store.add_reference(
            "10001",
            gallery_id=gallery_id,
            image_path=path,
            now=100 + gallery_id,
        )

    profile = store.get("10001")
    references = store.character_references(("10001",))

    assert removed == (3,)
    assert profile is not None
    assert profile.image_paths == (paths[3],)
    assert references[0].name == "10001"
    assert references[0].aliases == ("小明",)
    assert references[0].image_paths == profile.image_paths


def test_manual_reference_replaces_previous_file_and_lists_profiles(tmp_path) -> None:
    store = UserProfileStore(tmp_path / "profiles.sqlite3")
    store.ensure_profile("10001", display_name="小明", group_id="200")

    first = store.set_manual_reference(
        "10001",
        content=b"\xff\xd8\xfffirst",
        extension=".jpg",
        now=100,
    )
    second = store.set_manual_reference(
        "10001",
        content=b"\x89PNG\r\n\x1a\nsecond",
        extension="png",
        now=101,
    )
    profile = store.get("10001")

    assert not first.exists()
    assert second.is_file()
    assert second.read_bytes().endswith(b"second")
    assert profile is not None
    assert profile.image_paths == (second,)
    assert [item.qq for item in store.all_profiles()] == ["10001"]


def test_initialization_migrates_legacy_candidates_to_one_reference(tmp_path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    store = UserProfileStore(database_path)
    store.ensure_profile("10001", display_name="小明")
    paths = []
    for gallery_id in range(1, 4):
        path = tmp_path / f"legacy_{gallery_id}.png"
        path.write_bytes(b"png")
        paths.append(path.resolve())

    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO user_profile_images(qq, gallery_id, image_path, created_at)
            VALUES('10001', ?, ?, ?)
            """,
            [
                (gallery_id, str(path), 100 + gallery_id)
                for gallery_id, path in enumerate(paths, start=1)
            ],
        )

    migrated = UserProfileStore(database_path).get("10001")

    assert migrated is not None
    assert migrated.image_paths == (paths[2],)
