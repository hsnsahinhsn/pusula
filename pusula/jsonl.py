"""Small JSONL helpers: append-safe writing and resumable reading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Example


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave a truncated last line; skip it so the stage can resume.
                continue


def read_examples(path: str | Path) -> list[Example]:
    return [Example(**row) for row in read_jsonl(path)]


def done_ids(path: str | Path) -> set[str]:
    return {row["id"] for row in read_jsonl(path) if "id" in row}


class JsonlWriter:
    """Appends one JSON object per line and flushes, so a killed stage loses at most one line."""

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open(mode, encoding="utf-8")

    def write(self, obj) -> None:
        if isinstance(obj, Example):
            obj = obj.model_dump(exclude_none=True)
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_examples(path: str | Path, examples: Iterable[Example]) -> int:
    n = 0
    with JsonlWriter(path, mode="w") as w:
        for ex in examples:
            w.write(ex)
            n += 1
    return n
