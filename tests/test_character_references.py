from __future__ import annotations

from astrbot_plugin_img_gener.character_references import CharacterReferenceManager


def _fake_png(path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)


def test_auto_detects_names_and_aliases(tmp_path) -> None:
    coco = tmp_path / "coco.png"
    mushroom = tmp_path / "mushroom.png"
    _fake_png(coco)
    _fake_png(mushroom)
    manager = CharacterReferenceManager(
        [
            {
                "enabled": True,
                "name": "可可子",
                "aliases": ["可可", "Coco"],
                "reference_images": [str(coco)],
                "prompt_note": "紫色短发",
            },
            {
                "enabled": True,
                "name": "菌菌",
                "aliases": [],
                "reference_images": [str(mushroom)],
            },
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )
    resolved = manager.resolve("画可可和菌菌在咖啡店聊天")
    assert resolved.names == ("可可子", "菌菌")
    assert resolved.image_paths == (coco.resolve(), mushroom.resolve())
    augmented = manager.augment_prompt("原始提示", resolved)
    assert "必须完整出现以下 2 个角色" in augmented
    assert "缺一不可" in augmented
    assert "第 1 张参考图 对应角色「可可子」" in augmented
    assert "第 2 张参考图 对应角色「菌菌」" in augmented


def test_multi_character_defaults_to_one_image_per_character(tmp_path) -> None:
    coco_a = tmp_path / "coco_a.png"
    coco_b = tmp_path / "coco_b.png"
    mushroom_a = tmp_path / "mushroom_a.png"
    mushroom_b = tmp_path / "mushroom_b.png"
    for path in (coco_a, coco_b, mushroom_a, mushroom_b):
        _fake_png(path)
    manager = CharacterReferenceManager(
        [
            {
                "enabled": True,
                "name": "可可子",
                "reference_images": [str(coco_a), str(coco_b)],
            },
            {
                "enabled": True,
                "name": "菌菌",
                "reference_images": [str(mushroom_a), str(mushroom_b)],
            },
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
        max_total_images=4,
    )
    resolved = manager.resolve("可可子和菌菌同框")
    assert resolved.image_paths == (coco_a.resolve(), mushroom_a.resolve())
    augmented = manager.augment_prompt("原始提示", resolved)
    assert "第 1 张参考图 对应角色「可可子」" in augmented
    assert "第 2 张参考图 对应角色「菌菌」" in augmented


def test_single_character_still_carries_multiple_images(tmp_path) -> None:
    images = [tmp_path / f"coco_{index}.png" for index in range(3)]
    for path in images:
        _fake_png(path)
    manager = CharacterReferenceManager(
        [
            {
                "enabled": True,
                "name": "可可子",
                "reference_images": [str(path) for path in images],
            }
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
        max_total_images=4,
    )
    resolved = manager.resolve("画可可子")
    assert resolved.image_paths == tuple(path.resolve() for path in images)
    augmented = manager.augment_prompt("原始提示", resolved)
    assert "第 1-3 张参考图" in augmented


def test_sheet_mode_designates_cells_by_grid_order(tmp_path) -> None:
    coco = tmp_path / "coco.png"
    mushroom = tmp_path / "mushroom.png"
    _fake_png(coco)
    _fake_png(mushroom)
    manager = CharacterReferenceManager(
        [
            {"enabled": True, "name": "可可子", "reference_images": [str(coco)]},
            {
                "enabled": True,
                "name": "菌菌",
                "reference_images": [str(mushroom)],
                "prompt_note": "绿色头发",
            },
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )
    resolved = manager.resolve("可可子和菌菌同框")
    augmented = manager.augment_prompt("原始提示", resolved, use_sheet=True)
    assert "拼合参考图" in augmented
    assert "第 1 格（左）" in augmented
    assert "第 2 格（右）" in augmented
    assert "对应角色「可可子」" in augmented
    assert "对应角色「菌菌」" in augmented
    assert "绿色头发" in augmented


def test_two_character_multi_mode_adds_side_by_side_hint(tmp_path) -> None:
    coco = tmp_path / "coco.png"
    mushroom = tmp_path / "mushroom.png"
    _fake_png(coco)
    _fake_png(mushroom)
    manager = CharacterReferenceManager(
        [
            {"enabled": True, "name": "可可子", "reference_images": [str(coco)]},
            {"enabled": True, "name": "菌菌", "reference_images": [str(mushroom)]},
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )
    resolved = manager.resolve("可可子和菌菌同框")
    augmented = manager.augment_prompt("原始提示", resolved)
    assert "左右并排" in augmented


def test_characters_beyond_global_cap_are_reported(tmp_path) -> None:
    entries = []
    for index, name in enumerate(["甲", "乙", "丙", "丁", "戊"]):
        image = tmp_path / f"{index}.png"
        _fake_png(image)
        entries.append(
            {"enabled": True, "name": name, "reference_images": [str(image)]}
        )
    manager = CharacterReferenceManager(
        entries, data_dir=tmp_path, plugin_dir=tmp_path, max_total_images=4
    )
    resolved = manager.resolve("合影", ["甲", "乙", "丙", "丁", "戊"])
    assert resolved.capped_characters == ("戊",)
    assert len(resolved.image_paths) == 4
    assert "甲" in manager.augment_prompt("原始提示", resolved)


def test_missing_image_is_reported(tmp_path) -> None:
    manager = CharacterReferenceManager(
        [
            {
                "enabled": True,
                "name": "可可子",
                "reference_images": [str(tmp_path / "missing.png")],
            }
        ],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )
    resolved = manager.resolve("画可可子")
    assert resolved.missing_characters == ("可可子",)
    assert resolved.image_paths == ()


def test_unknown_explicit_character_is_reported(tmp_path) -> None:
    manager = CharacterReferenceManager([], data_dir=tmp_path, plugin_dir=tmp_path)
    resolved = manager.resolve("画角色", ["不存在的人物"])
    assert resolved.unknown_requested_names == ("不存在的人物",)


def test_legacy_character_syntax_is_supported(tmp_path) -> None:
    image = tmp_path / "coco.png"
    _fake_png(image)
    manager = CharacterReferenceManager(
        [f"可可子|可可={image}"], data_dir=tmp_path, plugin_dir=tmp_path
    )
    resolved = manager.resolve("可可在公园")
    assert resolved.names == ("可可子",)
