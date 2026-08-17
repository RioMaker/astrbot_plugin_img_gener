from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_schema_is_valid_json() -> None:
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["api"]["items"]["model"]["default"] == "gpt-image-2"
    assert schema["references"]["items"]["characters"]["type"] == "template_list"

    assert schema["api"]["items"]["base_url"]["default"] == (
        "https://uuapi.cc/v1"
    )
    assert schema["safety"]["items"]["review_base_url"]["default"] == (
        "https://uuapi.shop/v1"
    )
    assert "review_api_key" in schema["safety"]["items"]
    assert "start_message" in schema["generation"]["items"]


def test_character_templates_have_isolated_slots() -> None:
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    templates = schema["references"]["items"]["characters"]["templates"]
    slot_keys = [key for key in templates if key != "character"]
    assert len(slot_keys) >= 12
    # Each slot keeps its own upload folder in AstrBot (folder derived from the
    # template key), so every character manages its images independently.
    for key in slot_keys:
        assert templates[key]["items"]["reference_images"]["type"] == "file"
        assert templates[key]["items"]["name"]["type"] == "string"
    # Legacy shared template stays for existing installs.
    assert templates["character"]["items"]["reference_images"]["type"] == "file"


def test_metadata_and_requirements_exist() -> None:
    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "name: astrbot_plugin_img_gener" in metadata
    assert "httpx" in requirements


def test_local_api_keys_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "uuapi.key" in ignore
    assert "*.key" in ignore


def test_llm_tool_uses_background_task() -> None:
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "asyncio.create_task(" in source
    assert "_generation_start_message" in source
    assert "总用时" in source
