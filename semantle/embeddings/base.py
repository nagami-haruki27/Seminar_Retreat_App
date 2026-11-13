"""埋め込みバックエンドの共通インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EmbeddingError(RuntimeError):
    """埋め込みの計算に失敗した（モデル未ダウンロード、読み込み失敗など）。

    ``kind`` は UI でメッセージを出し分けるための分類:
    "not_found" / "load" / "compute" / "other"
    """

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


class EmbeddingBackend(ABC):
    """テキスト列を (n, d) の L2 正規化済み float32 行列に変換する。"""

    #: 設定画面などに出す表示名
    name: str = "backend"
    #: キャッシュのキーに使う一意な識別子（モデルID＋接頭辞設定を含めること）
    cache_key: str = "backend"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """``texts`` の順番どおりに埋め込みを返す。空リストなら shape (0, d) 相当の空配列。"""

    def warm_up(self) -> None:
        """モデルの読み込みなど、初回の重い処理を前倒しで行う（任意）。"""
        return


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"2次元配列が必要です: shape={arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
