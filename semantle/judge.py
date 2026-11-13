"""判定ロジック（順位付け・得点）。埋め込みには依存しない純粋な処理。

流れ:
1. ``prepare_entries``  … 入力を整え、未回答・反則を決め、同じ単語をまとめる
2. ``score_entries``    … 生の類似度と補正スコアを埋める（類似度は呼び出し側が計算）
3. ``rank_entries``     … 順位と得点を付ける（司会判断の反則後の再集計にも使う）
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from semantle.calibration import CalibrationMode, calibrate
from semantle.normalize import FoulKind, check_foul, clean_for_embedding, normalize_key

_TOL = 1e-9


class Rule(str, Enum):
    CLOSEST = "closest"  # ルール①: お題に一番近い単語
    TARGET = "target"  # ルール②: 目標値に一番近いスコア

    @property
    def label(self) -> str:
        return {Rule.CLOSEST: "① お題に一番近い単語", Rule.TARGET: "② 目標値に一番近いスコア"}[self]


class Status(str, Enum):
    OK = "ok"
    FOUL = "foul"
    EMPTY = "empty"


@dataclass(frozen=True)
class RoundSettings:
    rule: Rule = Rule.CLOSEST
    target: float = 0.5
    calibration: CalibrationMode = CalibrationMode.LINEAR
    points: tuple[int, ...] = (3, 2, 1)
    strict_contained: bool = False

    def __post_init__(self) -> None:
        if any(p < 0 for p in self.points):
            raise ValueError("得点は 0 以上にしてください")


@dataclass(frozen=True)
class TeamAnswer:
    team_id: str
    team_name: str
    word: str


@dataclass
class AnswerResult:
    team_id: str
    team_name: str
    word: str  # 表示用（整形済み）
    embed_text: str  # モデルに渡した文字列（同じ単語のチームは代表の表記に揃う）
    status: Status = Status.OK
    foul: FoulKind | None = None
    raw: float | None = None
    score: float | None = None
    rank: int | None = None
    points: int = 0

    @property
    def is_valid(self) -> bool:
        return self.status == Status.OK

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["foul"] = self.foul.value if self.foul else None
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AnswerResult:
        return cls(
            team_id=str(d["team_id"]),
            team_name=str(d["team_name"]),
            word=str(d.get("word", "")),
            embed_text=str(d.get("embed_text", d.get("word", ""))),
            status=Status(d.get("status", "ok")),
            foul=FoulKind(d["foul"]) if d.get("foul") else None,
            raw=None if d.get("raw") is None else float(d["raw"]),
            score=None if d.get("score") is None else float(d["score"]),
            rank=None if d.get("rank") is None else int(d["rank"]),
            points=int(d.get("points", 0)),
        )


@dataclass
class ReferenceStats:
    """そのラウンドで使った基準語彙の類似度の要約（表示用）。"""

    count: int
    mean: float
    minimum: float
    maximum: float
    excluded: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 1. 準備


def prepare_entries(answers: Sequence[TeamAnswer], topic: str, settings: RoundSettings) -> list[AnswerResult]:
    """入力を整形し、未回答・反則・同一単語のまとめを行う。順番は入力順を保つ。"""
    topic_clean = clean_for_embedding(topic)
    if not topic_clean:
        raise ValueError("お題が空です")

    representative: dict[str, str] = {}  # 正規化キー -> 代表表記
    entries: list[AnswerResult] = []
    for ans in answers:
        word = clean_for_embedding(ans.word)
        if not word:
            entries.append(AnswerResult(ans.team_id, ans.team_name, "", "", status=Status.EMPTY))
            continue
        key = normalize_key(word)
        rep = representative.setdefault(key, word)
        foul = check_foul(word, topic_clean, strict_contained=settings.strict_contained)
        if foul is not None:
            entries.append(AnswerResult(ans.team_id, ans.team_name, word, rep, status=Status.FOUL, foul=foul))
        else:
            entries.append(AnswerResult(ans.team_id, ans.team_name, word, rep))
    return entries


def texts_to_embed(entries: Sequence[AnswerResult]) -> list[str]:
    """埋め込みが必要な（有効な）単語の一覧。重複なし、出現順。"""
    return list(dict.fromkeys(e.embed_text for e in entries if e.is_valid))


def filter_reference_words(reference_words: Sequence[str], topic: str) -> tuple[list[str], list[str]]:
    """お題と同じ・お題を含む・お題に含まれる基準語を除く。(残った語, 除いた語)"""
    t = normalize_key(topic)
    kept: list[str] = []
    dropped: list[str] = []
    for w in reference_words:
        k = normalize_key(w)
        if not k or k == t or t in k or k in t:
            dropped.append(w)
        else:
            kept.append(w)
    return kept, dropped


# ---------------------------------------------------------------- 2. スコア


def score_entries(
    entries: Sequence[AnswerResult],
    raw_by_text: Mapping[str, float],
    reference_sims: np.ndarray,
    mode: CalibrationMode,
) -> None:
    """有効な回答に raw と score を書き込む（その場で更新）。"""
    for e in entries:
        if not e.is_valid:
            e.raw = None
            e.score = None
            continue
        if e.embed_text not in raw_by_text:
            raise KeyError(f"類似度が計算されていません: {e.embed_text!r}")
        e.raw = float(raw_by_text[e.embed_text])
        e.score = float(calibrate(e.raw, reference_sims, mode))


# ---------------------------------------------------------------- 3. 順位


def _sort_key(e: AnswerResult, settings: RoundSettings) -> tuple[float, ...]:
    assert e.score is not None and e.raw is not None
    if settings.rule == Rule.CLOSEST:
        return (-round(e.score, 9), -round(e.raw, 9))
    return (round(abs(e.score - settings.target), 9),)


def rank_entries(entries: Sequence[AnswerResult], settings: RoundSettings) -> None:
    """順位（1 始まり、同点は同順位で次を飛ばす）と得点を書き込む。

    反則・未回答は順位なし・0 点。有効な回答が 1 つでもあれば、それは 1 位になる。
    """
    for e in entries:
        e.rank = None
        e.points = 0
    valid = [e for e in entries if e.is_valid]
    if not valid:
        return
    for e in valid:
        if e.score is None or e.raw is None:
            raise ValueError(f"スコアが未計算です: {e.team_name}")
    ordered = sorted(valid, key=lambda e: _sort_key(e, settings))
    prev_key: tuple[float, ...] | None = None
    rank = 0
    for pos, e in enumerate(ordered, start=1):
        key = _sort_key(e, settings)
        if prev_key is None or any(abs(a - b) > _TOL for a, b in zip(key, prev_key)):
            rank = pos
        prev_key = key
        e.rank = rank
        e.points = settings.points[rank - 1] if rank - 1 < len(settings.points) else 0


def display_order(entries: Sequence[AnswerResult]) -> list[AnswerResult]:
    """表示用の並び: 順位の良い順 → 反則 → 未回答。同順位内は入力順。"""
    valid = sorted((e for e in entries if e.is_valid), key=lambda e: (e.rank or 0))
    fouls = [e for e in entries if e.status == Status.FOUL]
    empties = [e for e in entries if e.status == Status.EMPTY]
    return valid + fouls + empties


def reveal_order(entries: Sequence[AnswerResult]) -> list[AnswerResult]:
    """演出用の並び: 未回答 → 反則 → 下位から上位へ（1位が最後）。"""
    return list(reversed(display_order(entries)))


def set_manual_foul(entries: Sequence[AnswerResult], team_id: str, foul: bool, settings: RoundSettings) -> bool:
    """司会判断で反則にする／取り消す。該当チームが無い・未回答なら False。"""
    for e in entries:
        if e.team_id != team_id or e.status == Status.EMPTY:
            continue
        if foul:
            if e.status == Status.FOUL:
                return False
            e.status = Status.FOUL
            e.foul = FoulKind.MANUAL
        else:
            if e.foul != FoulKind.MANUAL:
                return False
            e.status = Status.OK
            e.foul = None
        rank_entries(entries, settings)
        return True
    return False


# ---------------------------------------------------------------- まとめ


def judge_round(
    answers: Sequence[TeamAnswer],
    topic: str,
    raw_by_text: Mapping[str, float],
    reference_sims: np.ndarray,
    settings: RoundSettings,
) -> list[AnswerResult]:
    """類似度が既に分かっているときの一括判定（テストやスクリプト用）。"""
    entries = prepare_entries(answers, topic, settings)
    score_entries(entries, raw_by_text, reference_sims, settings.calibration)
    rank_entries(entries, settings)
    return entries
