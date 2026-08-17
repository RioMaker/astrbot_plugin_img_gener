"""Rewrite _conf_schema.json character templates: per-character slot templates.

Each slot template gets its own upload folder in AstrBot (files/... per template
key), so each character's reference images are managed independently in WebUI.
The legacy "character" template is kept for backwards compatibility.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "_conf_schema.json"

SLOT_COUNT = 12

CHARACTER_ITEMS = {
    "enabled": {
        "description": "启用",
        "type": "bool",
        "default": True,
    },
    "name": {
        "description": "人物名称",
        "type": "string",
        "default": "",
        "hint": "例如：可可子",
    },
    "aliases": {
        "description": "别名",
        "type": "list",
        "default": [],
        "items": {"type": "string"},
        "hint": "例如：可可、Coco。过短的单字别名不会自动匹配。",
    },
    "reference_images": {
        "description": "参考图片",
        "type": "file",
        "default": [],
        "file_types": ["png", "jpg", "jpeg", "webp"],
        "hint": (
            "本角色位拥有独立的图片库，其他角色的图片不会出现在这里。"
            "建议清晰正脸或角色设定图；可上传多张。多个角色同框时默认每人只取第 1 张。"
        ),
    },
    "prompt_note": {
        "description": "人物补充设定",
        "type": "text",
        "default": "",
        "hint": "例如固定发色、服装或不要改变的特征。",
    },
}

SLOT_HINT = (
    "独立的角色图库：每个角色位对应一个专属图片文件夹，互不可见。"
    "新增角色时请选择一个尚未使用的角色位。"
)
LEGACY_HINT = (
    "旧版兼容模板：此模板下的所有条目共用同一个图片库。"
    "新增角色请改用上方带编号的角色位。"
)


def slot_template(slot: str) -> dict:
    return {
        "name": f"角色位 {slot}",
        "hint": SLOT_HINT,
        "display_item": "name",
        "hide_hint_in_list": True,
        "items": CHARACTER_ITEMS,
    }


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    characters = schema["references"]["items"]["characters"]
    characters["hint"] = (
        "每个「角色位」拥有独立的图片库，角色之间互不可见。"
        "用户提示词提到人物名或别名时，插件自动改走 /v1/images/edits 并携带对应图片。"
    )
    templates = {}
    for index in range(1, SLOT_COUNT + 1):
        templates[f"c{index:02d}"] = slot_template(f"{index:02d}")
    templates["character"] = {
        "name": "人物参考（旧版共享图库）",
        "hint": LEGACY_HINT,
        "display_item": "name",
        "hide_hint_in_list": True,
        "items": CHARACTER_ITEMS,
    }
    characters["templates"] = templates
    SCHEMA_PATH.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"rewrote {SCHEMA_PATH.name} with {SLOT_COUNT + 1} templates")


if __name__ == "__main__":
    main()
