"""sentence-transformers によるローカルモデルのバックエンド。"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from semantle.embeddings.base import EmbeddingBackend, EmbeddingError, l2_normalize
from semantle.embeddings.registry import ModelSpec


def set_offline_mode(offline: bool) -> None:
    """Hugging Face Hub への問い合わせを止める。

    会場の Wi-Fi が不安定だと、ダウンロード済みでも更新確認で固まることがあるため、
    アプリは既定でオフラインにする。ダウンロードスクリプトだけがオンラインで動く。
    """
    value = "1" if offline else "0"
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ["TRANSFORMERS_OFFLINE"] = value


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 32) -> None:
        self.spec = spec
        self.name = spec.label
        self.cache_key = spec.cache_key
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    def warm_up(self) -> None:
        self._get_model()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - 環境依存
            raise EmbeddingError(f"sentence-transformers が入っていません: {e}", kind="load") from e
        try:
            # CPU では bfloat16 のままだと大幅に遅いので float32 で読む（Qwen3 が bf16 で配布されている）
            model_kwargs: dict[str, Any] = {}
            if self.device == "cpu":
                import torch

                model_kwargs["dtype"] = torch.float32
            self._model = SentenceTransformer(self.spec.model_id, device=self.device, model_kwargs=model_kwargs)
        except OSError as e:
            # ファイルが無い / オフラインで取得できない
            raise EmbeddingError(
                f"モデル {self.spec.model_id} を読み込めませんでした。"
                f"`python scripts/download_models.py --models {self.spec.key}` を実行してください。\n{e}",
                kind="not_found",
            ) from e
        except Exception as e:
            raise EmbeddingError(f"モデル {self.spec.model_id} の読み込みに失敗: {e}", kind="load") from e
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._get_model()
        formatted = [self.spec.format(t) for t in texts]
        try:
            vecs = model.encode(
                formatted,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            raise EmbeddingError(f"埋め込みの計算に失敗: {e}", kind="compute") from e
        return l2_normalize(np.asarray(vecs, dtype=np.float32))
