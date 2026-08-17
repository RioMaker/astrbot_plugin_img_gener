from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / ".pytest_tmp"


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-backed tmp_path: the sandbox rejects writes inside dirs made by
    tempfile.mkdtemp, so build one with os.makedirs instead."""

    _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    path = _WORKSPACE_TMP / f"test-{uuid.uuid4().hex}"
    os.makedirs(path, exist_ok=True)
    yield path
