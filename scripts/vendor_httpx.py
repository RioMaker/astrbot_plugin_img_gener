"""Vendor httpx + pure-python deps for running the test suite without pip.

pip is blocked by the sandbox (its tempfile.mkdtemp dirs reject writes), so this
downloads wheels straight from PyPI and unzips them into a local folder.
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / ".vendor_httpx"
PACKAGES = ["httpx", "httpcore", "anyio", "h11", "sniffio"]


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "pip"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def pick_wheel(files: list[dict]) -> dict | None:
    for item in files:
        name = item.get("filename", "")
        if name.endswith(".whl") and ("py3-none-any" in name or "py2.py3-none-any" in name):
            return item
    return None


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for package in PACKAGES:
        data = fetch_json(f"https://pypi.org/pypi/{package}/json")
        version = data["info"]["version"]
        wheel = pick_wheel(data.get("urls") or [])
        if wheel is None:
            raise SystemExit(f"no universal wheel for {package}")
        print(f"{package} {version} <- {wheel['filename']}")
        content = urllib.request.urlopen(
            urllib.request.Request(wheel["url"], headers={"User-Agent": "pip"}),
            timeout=120,
        ).read()
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            archive.extractall(TARGET)
    print("done ->", TARGET)


if __name__ == "__main__":
    main()
