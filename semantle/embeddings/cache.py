"""埋め込みのキャッシュ（メモリ＋SQLite）。

同じ単語を何度も計算しないために、バックエンドを包んで使う。
SQLite は書き込みがアトミックなので、途中で落ちてもファイルが壊れにくい。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from semantle.embeddings.base import EmbeddingBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model_key TEXT NOT NULL,
    text      TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    PRIMARY KEY (model_key, text)
)
"""


class CachedBackend(EmbeddingBackend):
    def __init__(self, inner: EmbeddingBackend, db_path: str | Path | None) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache_key = inner.cache_key
        self.db_path = Path(db_path) if db_path is not None else None
        self._mem: dict[str, np.ndarray] = {}
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.execute(_SCHEMA)

    # ---- SQLite ----
    def _connect(self) -> sqlite3.Connection:
        assert self.db_path is not None
        con = sqlite3.connect(self.db_path, timeout=10)
        return con

    def _load_from_disk(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self.db_path is None or not texts:
            return {}
        found: dict[str, np.ndarray] = {}
        with self._connect() as con:
            for text in texts:
                row = con.execute(
                    "SELECT dim, vec FROM embeddings WHERE model_key=? AND text=?",
                    (self.cache_key, text),
                ).fetchone()
                if row is not None:
                    dim, blob = row
                    found[text] = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
        return found

    def _store_to_disk(self, items: dict[str, np.ndarray]) -> None:
        if self.db_path is None or not items:
            return
        with self._connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO embeddings (model_key, text, dim, vec) VALUES (?, ?, ?, ?)",
                [
                    (self.cache_key, text, int(vec.shape[0]), np.asarray(vec, dtype=np.float32).tobytes())
                    for text, vec in items.items()
                ],
            )

    # ---- API ----
    def warm_up(self) -> None:
        self.inner.warm_up()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        unique = list(dict.fromkeys(texts))
        missing = [t for t in unique if t not in self._mem]
        if missing:
            self._mem.update(self._load_from_disk(missing))
            missing = [t for t in missing if t not in self._mem]
        if missing:
            vecs = self.inner.embed(missing)
            if vecs.shape[0] != len(missing):
                raise RuntimeError("バックエンドが返した行数が入力数と一致しません")
            fresh = {t: vecs[i] for i, t in enumerate(missing)}
            self._mem.update(fresh)
            self._store_to_disk(fresh)
        return np.stack([self._mem[t] for t in texts]).astype(np.float32)

    def cached_count(self) -> int:
        if self.db_path is None:
            return len(self._mem)
        with self._connect() as con:
            (n,) = con.execute("SELECT COUNT(*) FROM embeddings WHERE model_key=?", (self.cache_key,)).fetchone()
        return int(n)
