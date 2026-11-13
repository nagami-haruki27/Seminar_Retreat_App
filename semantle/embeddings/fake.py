"""テスト用の決定的な埋め込みバックエンド。

* 指定した単語についてはお題との類似度が狙った値になるよう 2 次元ベクトルを作る
* それ以外の単語は文字列のハッシュから決まる単位ベクトル（毎回同じ）

モデルを読まずに判定の流れを通しで確認するために使う。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from semantle.embeddings.base import EmbeddingBackend, l2_normalize


class FakeBackend(EmbeddingBackend):
    name = "テスト用（モック）"
    cache_key = "fake|"

    def __init__(self, sims: Mapping[str, float] | None = None, dim: int = 8) -> None:
        """``sims`` は {テキスト: お題との目標コサイン類似度}。お題自身は 1.0 を入れる。

        お題は「sims の中で値が 1.0 のテキスト」として扱う。
        """
        self.sims = dict(sims or {})
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        if text in self.sims:
            s = max(-1.0, min(1.0, float(self.sims[text])))
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = s
            v[1] = math.sqrt(max(0.0, 1.0 - s * s))
            return v
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        v = rng.normal(size=self.dim).astype(np.float32)
        # 軸0成分を小さめにして「無関係な語」は類似度が低くなるようにする
        v[0] = abs(v[0]) * 0.1
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        self.calls.append(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))


class FailingBackend(EmbeddingBackend):
    """常に失敗するバックエンド（エラー表示のテスト用）。"""

    name = "失敗するバックエンド"
    cache_key = "failing|"

    def __init__(self, message: str = "テスト用の失敗", kind: str = "load") -> None:
        self.message = message
        self.kind = kind

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from semantle.embeddings.base import EmbeddingError

        raise EmbeddingError(self.message, kind=self.kind)
