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
