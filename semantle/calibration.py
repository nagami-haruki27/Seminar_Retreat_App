"""スコアの補正（キャリブレーション）。

* RAW      … 生のコサイン類似度をそのまま使う
* LINEAR   … 方式A。基準語彙とお題の平均類似度を 0、お題そのものを 1 とする線形変換
* PERCENTILE … 方式B。基準語彙の類似度分布の中での位置（0〜1、隣接する語の間は線形補間）
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class CalibrationMode(str, Enum):
    RAW = "raw"
    LINEAR = "linear"
    PERCENTILE = "percentile"

    @property
    def label(self) -> str:
        return {
            CalibrationMode.RAW: "補正なし（生の類似度）",
            CalibrationMode.LINEAR: "方式A：平均を0、お題を1にする線形補正",
            CalibrationMode.PERCENTILE: "方式B：基準語彙の中での順位（パーセンタイル）",
        }[self]


class CalibrationError(ValueError):
    """補正に必要な前提が満たされていない（基準語彙が空、など）。"""


_EPS = 1e-6


def linear_score(raw: float, reference_sims: np.ndarray) -> float:
    """方式A: (raw - μ) / (1 - μ)。負の値も返す（切り詰めない）。"""
    ref = np.asarray(reference_sims, dtype=np.float64)
    if ref.size == 0:
        raise CalibrationError("基準語彙が空です")
    mu = float(ref.mean())
    denom = 1.0 - mu
    if denom < _EPS:
        raise CalibrationError("基準語彙の平均類似度が 1 に近すぎて補正できません")
    return (float(raw) - mu) / denom


def percentile_score(raw: float, reference_sims: np.ndarray) -> float:
    """方式B: 基準語彙の類似度を昇順に並べ、raw がどの位置にあるかを 0〜1 で返す。

    最小値以下なら 0、最大値以上なら 1。間は隣接する語の間で線形補間する。
    """
    ref = np.sort(np.asarray(reference_sims, dtype=np.float64))
    n = ref.size
    if n == 0:
        raise CalibrationError("基準語彙が空です")
    if n == 1:
        return 0.0 if raw < ref[0] else 1.0
    x = float(raw)
    if x <= ref[0]:
        return 0.0
    if x >= ref[-1]:
        return 1.0
    # ref[i] <= x < ref[i+1] となる i
    i = int(np.searchsorted(ref, x, side="right")) - 1
    lo, hi = ref[i], ref[i + 1]
    frac = 0.0 if hi - lo < _EPS else (x - lo) / (hi - lo)
    return (i + frac) / (n - 1)


def calibrate(raw: float, reference_sims: np.ndarray, mode: CalibrationMode) -> float:
    if mode == CalibrationMode.RAW:
        return float(raw)
    if mode == CalibrationMode.LINEAR:
        return linear_score(raw, reference_sims)
    if mode == CalibrationMode.PERCENTILE:
        return percentile_score(raw, reference_sims)
    raise ValueError(f"未知の補正方式: {mode}")
