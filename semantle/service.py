"""埋め込み → 補正 → 判定の一連の流れ。UI から呼ぶ唯一の入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from semantle.calibration import CalibrationMode
from semantle.embeddings.base import EmbeddingBackend
from semantle.judge import (
    AnswerResult,
    ReferenceStats,
    RoundSettings,
    TeamAnswer,
    filter_reference_words,
    prepare_entries,
    rank_entries,
    score_entries,
    texts_to_embed,
)
from semantle.normalize import clean_for_embedding


@dataclass
class RoundOutcome:
    results: list[AnswerResult]
    reference: ReferenceStats
    topic: str


class JudgeService:
    def __init__(self, backend: EmbeddingBackend, reference_words: Sequence[str]) -> None:
        self.backend = backend
        self.reference_words = list(reference_words)

    def topic_reference_sims(self, topic: str) -> tuple[np.ndarray, ReferenceStats]:
        """お題と基準語彙の類似度（お題と重なる語は除く）。"""
        topic_clean = clean_for_embedding(topic)
        kept, dropped = filter_reference_words(self.reference_words, topic_clean)
        if not kept:
            raise ValueError("基準語彙が空です（data/reference_words.txt を確認してください）")
        vecs = self.backend.embed([topic_clean, *kept])
        sims = vecs[1:] @ vecs[0]
        stats = ReferenceStats(
            count=len(kept), mean=float(sims.mean()), minimum=float(sims.min()), maximum=float(sims.max()),
            excluded=dropped,
        )
        return sims, stats

    def similarities(self, topic: str, texts: Sequence[str]) -> dict[str, float]:
        topic_clean = clean_for_embedding(topic)
        texts = list(texts)
        if not texts:
            return {}
        vecs = self.backend.embed([topic_clean, *texts])
        sims = vecs[1:] @ vecs[0]
        return {t: float(s) for t, s in zip(texts, sims)}

    def play_round(self, topic: str, answers: Sequence[TeamAnswer], settings: RoundSettings) -> RoundOutcome:
        """1ラウンド分の判定。埋め込み失敗は EmbeddingError として呼び出し側へ伝える。"""
        topic_clean = clean_for_embedding(topic)
        entries = prepare_entries(answers, topic_clean, settings)
        texts = texts_to_embed(entries)
        # 補正なしでも基準語彙の統計は表示に使うので毎回計算する（キャッシュ済みなら一瞬）
        ref_sims, stats = self.topic_reference_sims(topic_clean)
        raw_by_text = self.similarities(topic_clean, texts)
        score_entries(entries, raw_by_text, ref_sims, settings.calibration)
        rank_entries(entries, settings)
        return RoundOutcome(results=entries, reference=stats, topic=topic_clean)

    def score_table(self, topic: str, words: Sequence[str]) -> list[dict[str, float | str]]:
        """比較スクリプト用: 各単語の raw / A / B を返す。"""
        topic_clean = clean_for_embedding(topic)
        ref_sims, _ = self.topic_reference_sims(topic_clean)
        cleaned = [clean_for_embedding(w) for w in words]
        raw = self.similarities(topic_clean, cleaned)
        from semantle.calibration import calibrate

        rows: list[dict[str, float | str]] = []
        for w in cleaned:
            r = raw[w]
            rows.append(
                {
                    "word": w,
                    "raw": r,
                    "linear": calibrate(r, ref_sims, CalibrationMode.LINEAR),
                    "percentile": calibrate(r, ref_sims, CalibrationMode.PERCENTILE),
                }
            )
        return rows
