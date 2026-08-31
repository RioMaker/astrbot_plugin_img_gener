from __future__ import annotations

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


def test_profile_keeps_only_three_recent_reference_candidates(tmp_path) -> None:
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

    assert removed == (1,)
    assert profile is not None
    assert profile.image_paths == (paths[3], paths[2], paths[1])
    assert references[0].name == "10001"
    assert references[0].aliases == ("小明",)
    assert references[0].image_paths == profile.image_paths
