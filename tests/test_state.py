import json
from pathlib import Path

import numpy as np
import pytest

from semantle.judge import TeamAnswer, judge_round
from semantle.state import (
    GameSettings,
    GameState,
    RoundRecord,
    StateLoadError,
    export_csv,
    load_state,
    save_state,
    write_csv,
)

REF = np.array([0.1, 0.2, 0.3, 0.4])


def make_round(state: GameState, topic: str, words: dict[str, str], raw: dict[str, float]) -> RoundRecord:
    answers = [TeamAnswer(t.id, t.name, words.get(t.id, "")) for t in state.teams]
    settings = state.settings.round_settings()
    results = judge_round(answers, topic, raw, REF, settings)
    return RoundRecord(
        round_no=state.next_round_no(), timestamp="2026-11-10T10:00:00", topic=topic, rule=settings.rule.value,
        target=settings.target, calibration=settings.calibration.value, points=list(settings.points),
        model_key="fake", model_name="テスト", results=results, reference_mean=0.25, reference_count=4,
    )


@pytest.fixture
def state() -> GameState:
    return GameState.new_game(["赤", "青", "緑", "黄"])


class TestScoring:
    def test_points_accumulate(self, state: GameState) -> None:
        state.add_round(make_round(state, "お題1", {"t1": "a", "t2": "b", "t3": "c"}, {"a": 0.9, "b": 0.8, "c": 0.7}))
        state.add_round(make_round(state, "お題2", {"t1": "a", "t2": "b", "t4": "d"}, {"a": 0.5, "b": 0.9, "d": 0.7}))
        assert state.totals() == {"t1": 3 + 1, "t2": 2 + 3, "t3": 1, "t4": 2}

    def test_leaderboard_ties(self, state: GameState) -> None:
        state.add_adjustment("t1", 5)
        state.add_adjustment("t2", 5)
        board = state.leaderboard()
        assert [(rank, t.id, total) for rank, t, total in board] == [(1, "t1", 5), (1, "t2", 5), (3, "t3", 0), (3, "t4", 0)]

    def test_undo(self, state: GameState) -> None:
        state.add_round(make_round(state, "お題1", {"t1": "a"}, {"a": 0.9}))
        state.add_round(make_round(state, "お題2", {"t2": "b"}, {"b": 0.9}))
        undone = state.undo_last_round()
        assert undone is not None and undone.topic == "お題2"
        assert state.totals() == {"t1": 3, "t2": 0, "t3": 0, "t4": 0}
        assert state.next_round_no() == 2
        state.undo_last_round()
        assert state.undo_last_round() is None
        assert state.totals() == {"t1": 0, "t2": 0, "t3": 0, "t4": 0}

    def test_manual_adjustment_and_removal(self, state: GameState) -> None:
        adj = state.add_adjustment("t3", -2, "減点")
        assert state.totals()["t3"] == -2
        assert state.remove_adjustment(adj.id)
        assert not state.remove_adjustment(adj.id)
        assert state.totals()["t3"] == 0

    def test_adjustment_unknown_team(self, state: GameState) -> None:
        with pytest.raises(KeyError):
            state.add_adjustment("nope", 1)

    def test_manual_foul_reranks_round(self, state: GameState) -> None:
        state.add_round(make_round(state, "お題", {"t1": "a", "t2": "b"}, {"a": 0.9, "b": 0.8}))
        assert state.set_manual_foul(1, "t1", True)
        assert state.totals() == {"t1": 0, "t2": 3, "t3": 0, "t4": 0}
        assert not state.set_manual_foul(99, "t1", True)

    def test_exclude_round_for_reveal(self, state: GameState) -> None:
        state.add_round(make_round(state, "お題", {"t1": "a"}, {"a": 0.9}))
        assert state.totals(exclude_round_no=1)["t1"] == 0
        assert state.totals()["t1"] == 3


class TestTeams:
    def test_rename_updates_history(self, state: GameState) -> None:
        state.add_round(make_round(state, "お題", {"t1": "a"}, {"a": 0.9}))
        state.rename_team("t1", "レッド")
        assert state.team_by_id("t1").name == "レッド"
        assert state.rounds[0].results[0].team_name == "レッド"
        with pytest.raises(ValueError):
            state.rename_team("t1", "  ")

    def test_add_team(self, state: GameState) -> None:
        t = state.add_team("紫")
        assert t.id == "t5" and state.totals()["t5"] == 0

    def test_new_game_defaults_names(self) -> None:
        s = GameState.new_game(["", "B"])
        assert [t.name for t in s.teams] == ["チーム1", "B"]
        with pytest.raises(ValueError):
            GameState.new_game([])


class TestPersistence:
    def test_roundtrip(self, state: GameState, tmp_path: Path) -> None:
        state.settings = GameSettings(rule="target", target=0.3, calibration="percentile", points=[5, 3, 1], timer_sec=60)
        state.add_round(make_round(state, "お題", {"t1": "a", "t2": "お題", "t3": ""}, {"a": 0.9}))
        state.add_adjustment("t4", 2, "ボーナス")
        state.set_manual_foul(1, "t1", True)
        path = tmp_path / "s" / "game.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded is not None
        assert loaded.to_dict() == state.to_dict()
        assert loaded.totals() == state.totals()
        assert loaded.settings.round_settings() == state.settings.round_settings()

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_state(tmp_path / "nothing.json") is None

    def test_backup_is_kept_and_used(self, state: GameState, tmp_path: Path) -> None:
        path = tmp_path / "game.json"
        save_state(state, path)
        state.add_adjustment("t1", 1)
        save_state(state, path)
        assert path.with_suffix(".json.bak").exists()
        path.write_text("{ broken json", encoding="utf-8")
        recovered = load_state(path)
        assert recovered is not None
        assert recovered.totals()["t1"] == 0  # 1世代前

    def test_both_broken_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "game.json"
        path.write_text("{", encoding="utf-8")
        path.with_suffix(".json.bak").write_text("[]", encoding="utf-8")
        with pytest.raises(StateLoadError):
            load_state(path)

    def test_newer_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "game.json"
        path.write_text(json.dumps({"version": 99, "teams": []}), encoding="utf-8")
        with pytest.raises(StateLoadError):
            load_state(path)

    def test_settings_from_dict_tolerates_garbage(self) -> None:
        s = GameSettings.from_dict({"rule": "nope", "calibration": "???", "points": [], "unknown": 1})
        assert s.rule == "closest" and s.calibration == "linear" and s.points == [3, 2, 1]

    def test_no_tmp_left_behind(self, state: GameState, tmp_path: Path) -> None:
        path = tmp_path / "game.json"
        save_state(state, path)
        assert not path.with_suffix(".json.tmp").exists()


class TestCsv:
    def test_export_contains_rounds_adjustments_totals(self, state: GameState, tmp_path: Path) -> None:
        state.add_round(make_round(state, "お題", {"t1": "a", "t2": "お題"}, {"a": 0.9}))
        state.add_adjustment("t3", 2, "ボーナス")
        text = export_csv(state)
        lines = text.splitlines()
        assert lines[0].startswith("round,timestamp,topic")
        assert sum(1 for line in lines if line.startswith("1,")) == 4
        assert any(line.startswith("adjustment,") and "ボーナス" in line for line in lines)
        assert sum(1 for line in lines if line.startswith("total,")) == 4
        assert "お題と同じ" in text
        out = tmp_path / "r.csv"
        write_csv(state, out)
        assert out.read_bytes().startswith(b"\xef\xbb\xbf")  # Excel 用 BOM

    def test_export_empty_state(self, state: GameState) -> None:
        text = export_csv(state)
        assert text.count("\n") == 1 + len(state.teams)
