from __future__ import annotations

import time
import uuid
from pathlib import Path

from .image_client import ImageResponse


class OutputStore:
    def __init__(self, output_dir: Path, retention_days: int = 7) -> None:
        self.output_dir = output_dir
        self.retention_days = max(0, retention_days)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image: ImageResponse) -> Path:
        path = self.output_dir / f"{int(time.time())}_{uuid.uuid4().hex}{image.extension}"
        path.write_bytes(image.content)
        return path

    def cleanup(self, *, now: float | None = None) -> int:
        if self.retention_days <= 0:
            return 0
        current = time.time() if now is None else now
        cutoff = current - self.retention_days * 86400
        removed = 0
        for path in self.output_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed
