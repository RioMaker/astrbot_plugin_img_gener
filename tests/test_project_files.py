from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_schema_is_valid_json() -> None:
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["api"]["items"]["model"]["default"] == "gpt-image-2"
    assert schema["references"]["items"]["characters"]["type"] == "template_list"


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
    assert "生图任务已受理" in source
