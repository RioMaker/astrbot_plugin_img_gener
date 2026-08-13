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
    assert "参考图 1 对应角色“可可子”" in augmented
    assert "参考图 2 对应角色“菌菌”" in augmented


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
