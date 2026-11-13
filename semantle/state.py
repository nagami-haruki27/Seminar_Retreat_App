"""ゲームの状態（チーム・ラウンド履歴・手動修正・設定）と、JSON 保存／復元、CSV 出力。

累計得点は「ラウンド履歴の得点 ＋ 手動修正の合計」から毎回計算する。
Undo は履歴の末尾を外すだけなので、状態の整合性が崩れない。
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from semantle.calibration import CalibrationMode
from semantle.embeddings.registry import DEFAULT_MODEL_KEY
from semantle.judge import AnswerResult, RoundSettings, Rule, rank_entries, set_manual_foul

STATE_VERSION = 1


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Team:
    id: str
    name: str


@dataclass
class GameSettings:
    model_key: str = DEFAULT_MODEL_KEY
    rule: str = Rule.CLOSEST.value
    target: float = 0.5
    calibration: str = CalibrationMode.LINEAR.value
    points: list[int] = field(default_factory=lambda: [3, 2, 1])
    strict_contained: bool = False
    reveal_enabled: bool = True
    reveal_auto: bool = False
    reveal_interval_sec: float = 3.0
    timer_sec: int = 0  # 0 なら使わない

    def round_settings(self) -> RoundSettings:
        return RoundSettings(
            rule=Rule(self.rule),
            target=float(self.target),
            calibration=CalibrationMode(self.calibration),
            points=tuple(int(p) for p in self.points),
            strict_contained=bool(self.strict_contained),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameSettings:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        s = cls(**known)
        # 不正な値が入っていても落ちないように軽く検証する
        if s.rule not in {r.value for r in Rule}:
            s.rule = Rule.CLOSEST.value
        if s.calibration not in {c.value for c in CalibrationMode}:
            s.calibration = CalibrationMode.LINEAR.value
        s.points = [max(0, int(p)) for p in s.points] or [3, 2, 1]
        return s


@dataclass
class RoundRecord:
    round_no: int
    timestamp: str
    topic: str
    rule: str
    target: float
    calibration: str
    points: list[int]
    model_key: str
    model_name: str
    results: list[AnswerResult]
    reference_mean: float | None = None
    reference_count: int = 0

    def round_settings(self, strict_contained: bool = False) -> RoundSettings:
        return RoundSettings(
            rule=Rule(self.rule),
            target=self.target,
            calibration=CalibrationMode(self.calibration),
            points=tuple(self.points),
            strict_contained=strict_contained,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoundRecord:
        return cls(
            round_no=int(d["round_no"]),
            timestamp=str(d.get("timestamp", "")),
            topic=str(d["topic"]),
            rule=str(d.get("rule", Rule.CLOSEST.value)),
            target=float(d.get("target", 0.5)),
            calibration=str(d.get("calibration", CalibrationMode.LINEAR.value)),
            points=[int(p) for p in d.get("points", [3, 2, 1])],
            model_key=str(d.get("model_key", "")),
            model_name=str(d.get("model_name", "")),
            results=[AnswerResult.from_dict(r) for r in d.get("results", [])],
            reference_mean=None if d.get("reference_mean") is None else float(d["reference_mean"]),
            reference_count=int(d.get("reference_count", 0)),
        )


@dataclass
class Adjustment:
    id: str
    timestamp: str
    team_id: str
    delta: int
    note: str = ""


@dataclass
class GameState:
    version: int = STATE_VERSION
    created_at: str = field(default_factory=_now)
    teams: list[Team] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)
    settings: GameSettings = field(default_factory=GameSettings)

    # ---- 生成 ----
    @classmethod
    def new_game(cls, team_names: Sequence[str], settings: GameSettings | None = None) -> GameState:
        names = [n.strip() or f"チーム{i + 1}" for i, n in enumerate(team_names)]
        if len(names) < 1:
            raise ValueError("チームは 1 つ以上必要です")
        teams = [Team(id=f"t{i + 1}", name=n) for i, n in enumerate(names)]
        return cls(teams=teams, settings=settings or GameSettings())

    # ---- チーム ----
    def team_by_id(self, team_id: str) -> Team:
        for t in self.teams:
            if t.id == team_id:
                return t
        raise KeyError(f"チームが見つかりません: {team_id}")

    def rename_team(self, team_id: str, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("チーム名が空です")
        self.team_by_id(team_id).name = new_name
        # 履歴の表示名も揃える
        for r in self.rounds:
            for res in r.results:
                if res.team_id == team_id:
                    res.team_name = new_name

    def add_team(self, name: str) -> Team:
        name = name.strip() or f"チーム{len(self.teams) + 1}"
        existing = {t.id for t in self.teams}
        n = len(self.teams) + 1
        while f"t{n}" in existing:
            n += 1
        team = Team(id=f"t{n}", name=name)
        self.teams.append(team)
        return team

    # ---- ラウンド ----
    def next_round_no(self) -> int:
        return (self.rounds[-1].round_no + 1) if self.rounds else 1

    def add_round(self, record: RoundRecord) -> None:
        self.rounds.append(record)

    def undo_last_round(self) -> RoundRecord | None:
        if not self.rounds:
            return None
        return self.rounds.pop()

    def set_manual_foul(self, round_no: int, team_id: str, foul: bool) -> bool:
        for r in self.rounds:
            if r.round_no == round_no:
                return set_manual_foul(r.results, team_id, foul, r.round_settings())
        return False

    def rerank_round(self, round_no: int) -> None:
        for r in self.rounds:
            if r.round_no == round_no:
                rank_entries(r.results, r.round_settings())

    # ---- 手動修正 ----
    def add_adjustment(self, team_id: str, delta: int, note: str = "") -> Adjustment:
        self.team_by_id(team_id)
        adj = Adjustment(id=_new_id(), timestamp=_now(), team_id=team_id, delta=int(delta), note=note.strip())
        self.adjustments.append(adj)
        return adj

    def remove_adjustment(self, adj_id: str) -> bool:
        before = len(self.adjustments)
        self.adjustments = [a for a in self.adjustments if a.id != adj_id]
        return len(self.adjustments) != before

    # ---- 集計 ----
    def totals(self, *, exclude_round_no: int | None = None) -> dict[str, int]:
        totals = {t.id: 0 for t in self.teams}
        for r in self.rounds:
            if r.round_no == exclude_round_no:
                continue
            for res in r.results:
                if res.team_id in totals:
                    totals[res.team_id] += res.points
        for a in self.adjustments:
            if a.team_id in totals:
                totals[a.team_id] += a.delta
        return totals

    def leaderboard(self, *, exclude_round_no: int | None = None) -> list[tuple[int, Team, int]]:
        """(順位, チーム, 累計) の一覧。累計の降順、同点は同順位。"""
        totals = self.totals(exclude_round_no=exclude_round_no)
        ordered = sorted(self.teams, key=lambda t: (-totals[t.id], self.teams.index(t)))
        out: list[tuple[int, Team, int]] = []
        rank = 0
        prev: int | None = None
        for pos, t in enumerate(ordered, start=1):
            if prev is None or totals[t.id] != prev:
                rank = pos
            prev = totals[t.id]
            out.append((rank, t, totals[t.id]))
        return out

    # ---- 直列化 ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "teams": [asdict(t) for t in self.teams],
            "rounds": [r.to_dict() for r in self.rounds],
            "adjustments": [asdict(a) for a in self.adjustments],
            "settings": asdict(self.settings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameState:
        if not isinstance(d, dict):
            raise ValueError("保存データの形式が不正です")
        version = int(d.get("version", 1))
        if version > STATE_VERSION:
            raise ValueError(f"保存ファイルのバージョン {version} はこのアプリより新しいです")
        state = cls(
            version=STATE_VERSION,
            created_at=str(d.get("created_at", _now())),
            teams=[Team(id=str(t["id"]), name=str(t["name"])) for t in d.get("teams", [])],
            rounds=[RoundRecord.from_dict(r) for r in d.get("rounds", [])],
            adjustments=[
                Adjustment(
                    id=str(a["id"]), timestamp=str(a.get("timestamp", "")), team_id=str(a["team_id"]),
                    delta=int(a["delta"]), note=str(a.get("note", "")),
                )
                for a in d.get("adjustments", [])
            ],
            settings=GameSettings.from_dict(d.get("settings", {})),
        )
        if len({t.id for t in state.teams}) != len(state.teams):
            raise ValueError("チームIDが重複しています")
        return state


# ---------------------------------------------------------------- 保存・復元


def save_state(state: GameState, path: str | Path) -> None:
    """一時ファイルに書いてから置き換える（途中で落ちても元のファイルが残る）。1世代の .bak を保つ。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        os.replace(path, bak)
    os.replace(tmp, path)


class StateLoadError(RuntimeError):
    pass


def load_state(path: str | Path) -> GameState | None:
    """保存ファイルを読む。無ければ None。壊れていれば .bak を試し、それも駄目なら StateLoadError。"""
    path = Path(path)
    candidates = [path, path.with_suffix(path.suffix + ".bak")]
    errors: list[str] = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                return GameState.from_dict(json.load(f))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            errors.append(f"{p.name}: {e}")
    if errors:
        raise StateLoadError("保存ファイルを読めませんでした: " + " / ".join(errors))
    return None


# ---------------------------------------------------------------- CSV

CSV_COLUMNS = [
    "round", "timestamp", "topic", "rule", "target", "calibration", "model",
    "team", "word", "status", "foul", "raw", "score", "rank", "points", "note",
]


def export_csv(state: GameState) -> str:
    """全ラウンドの結果（1 行 = 1 チーム × 1 ラウンド）と手動修正を CSV 文字列にする。"""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in state.rounds:
        for res in r.results:
            w.writerow(
                {
                    "round": r.round_no, "timestamp": r.timestamp, "topic": r.topic, "rule": r.rule,
                    "target": r.target if r.rule == Rule.TARGET.value else "", "calibration": r.calibration,
                    "model": r.model_name or r.model_key, "team": res.team_name, "word": res.word,
                    "status": res.status.value, "foul": res.foul.label if res.foul else "",
                    "raw": "" if res.raw is None else f"{res.raw:.6f}",
                    "score": "" if res.score is None else f"{res.score:.6f}",
                    "rank": "" if res.rank is None else res.rank, "points": res.points, "note": "",
                }
            )
    for a in state.adjustments:
        try:
            name = state.team_by_id(a.team_id).name
        except KeyError:
            name = a.team_id
        w.writerow(
            {
                "round": "adjustment", "timestamp": a.timestamp, "topic": "", "rule": "", "target": "",
                "calibration": "", "model": "", "team": name, "word": "", "status": "", "foul": "",
                "raw": "", "score": "", "rank": "", "points": a.delta, "note": a.note,
            }
        )
    for rank, team, total in state.leaderboard():
        w.writerow(
            {
                "round": "total", "timestamp": "", "topic": "", "rule": "", "target": "", "calibration": "",
                "model": "", "team": team.name, "word": "", "status": "", "foul": "", "raw": "", "score": "",
                "rank": rank, "points": total, "note": "",
            }
        )
    return buf.getvalue()


def write_csv(state: GameState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(export_csv(state), encoding="utf-8-sig")  # Excel で開けるように BOM 付き
    os.replace(tmp, path)
