import numpy as np
import pytest

from semantle.calibration import CalibrationMode
from semantle.judge import (
    AnswerResult,
    RoundSettings,
    Rule,
    Status,
    TeamAnswer,
    display_order,
    filter_reference_words,
    judge_round,
    prepare_entries,
    rank_entries,
    reveal_order,
    set_manual_foul,
    texts_to_embed,
)
from semantle.normalize import FoulKind

REF = np.array([0.1, 0.2, 0.3, 0.4])  # μ = 0.25


def ans(*words: str) -> list[TeamAnswer]:
    return [TeamAnswer(f"t{i + 1}", f"チーム{i + 1}", w) for i, w in enumerate(words)]


def by_team(results: list[AnswerResult]) -> dict[str, AnswerResult]:
    return {r.team_id: r for r in results}


class TestRule1:
    def test_ranking_and_points(self) -> None:
        raw = {"ノイズ": 0.8, "カレー": 0.3, "画像生成": 0.7, "猫": 0.4}
        res = by_team(judge_round(ans("ノイズ", "カレー", "画像生成", "猫"), "拡散モデル", raw, REF, RoundSettings()))
        assert [res[t].rank for t in ("t1", "t2", "t3", "t4")] == [1, 4, 2, 3]
        assert [res[t].points for t in ("t1", "t2", "t3", "t4")] == [3, 0, 2, 1]

    def test_calibrated_score_and_raw_both_present(self) -> None:
        raw = {"ノイズ": 0.8}
        r = judge_round(ans("ノイズ"), "拡散モデル", raw, REF, RoundSettings(calibration=CalibrationMode.LINEAR))[0]
        assert r.raw == pytest.approx(0.8)
        assert r.score == pytest.approx((0.8 - 0.25) / 0.75)

    def test_calibration_does_not_change_order_for_linear(self) -> None:
        raw = {"a": 0.9, "b": 0.5, "c": 0.7}
        for mode in (CalibrationMode.RAW, CalibrationMode.LINEAR):
            res = by_team(judge_round(ans("a", "b", "c"), "お題", raw, REF, RoundSettings(calibration=mode)))
            assert [res[t].rank for t in ("t1", "t2", "t3")] == [1, 3, 2]

    def test_percentile_saturation_breaks_ties_by_raw(self) -> None:
        # 方式Bでは基準語彙の最大値以上が全て 1.0 になるが、生の類似度で順位を決める
        raw = {"a": 0.9, "b": 0.95, "c": 0.5}
        res = by_team(
            judge_round(ans("a", "b", "c"), "お題", raw, REF, RoundSettings(calibration=CalibrationMode.PERCENTILE))
        )
        assert res["t1"].score == res["t2"].score == 1.0
        assert [res[t].rank for t in ("t1", "t2", "t3")] == [2, 1, 3]


class TestRule2:
    def test_closest_to_target_wins(self) -> None:
        raw = {"a": 0.9, "b": 0.55, "c": 0.3}
        settings = RoundSettings(rule=Rule.TARGET, target=0.5, calibration=CalibrationMode.RAW)
        res = by_team(judge_round(ans("a", "b", "c"), "お題", raw, REF, settings))
        assert [res[t].rank for t in ("t1", "t2", "t3")] == [3, 1, 2]
        assert res["t2"].points == 3

    def test_equidistant_above_and_below_tie(self) -> None:
        raw = {"a": 0.6, "b": 0.4}
        settings = RoundSettings(rule=Rule.TARGET, target=0.5, calibration=CalibrationMode.RAW)
        res = by_team(judge_round(ans("a", "b"), "お題", raw, REF, settings))
        assert res["t1"].rank == res["t2"].rank == 1
        assert res["t1"].points == res["t2"].points == 3

    def test_target_uses_calibrated_score(self) -> None:
        # 補正後 0.5 になる raw は 0.25 + 0.5*0.75 = 0.625
        raw = {"a": 0.625, "b": 0.5}
        settings = RoundSettings(rule=Rule.TARGET, target=0.5, calibration=CalibrationMode.LINEAR)
        res = by_team(judge_round(ans("a", "b"), "お題", raw, REF, settings))
        assert res["t1"].rank == 1


class TestTies:
    def test_same_word_same_rank_competition_numbering(self) -> None:
        raw = {"ノイズ": 0.8, "猫": 0.4}
        res = by_team(judge_round(ans("ノイズ", "ノイズ", "猫"), "拡散モデル", raw, REF, RoundSettings()))
        assert res["t1"].rank == res["t2"].rank == 1
        assert res["t3"].rank == 3
        assert [res[t].points for t in ("t1", "t2", "t3")] == [3, 3, 1]

    def test_same_word_with_variants_shares_representative(self) -> None:
        raw = {"ノイズ": 0.8}  # 代表表記は最初のチームのもの
        entries = prepare_entries(ans("ノイズ", "のいず", " ﾉｲｽﾞ "), "拡散モデル", RoundSettings())
        assert [e.embed_text for e in entries] == ["ノイズ", "ノイズ", "ノイズ"]
        assert [e.word for e in entries] == ["ノイズ", "のいず", "ノイズ"]
        assert texts_to_embed(entries) == ["ノイズ"]
        res = judge_round(ans("ノイズ", "のいず", " ﾉｲｽﾞ "), "拡散モデル", raw, REF, RoundSettings())
        assert all(r.rank == 1 for r in res)

    def test_ties_in_score_same_rank(self) -> None:
        raw = {"a": 0.8, "b": 0.8, "c": 0.8}
        res = judge_round(ans("a", "b", "c"), "お題", raw, REF, RoundSettings())
        assert all(r.rank == 1 and r.points == 3 for r in res)

    def test_custom_points_table(self) -> None:
        raw = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
        res = by_team(judge_round(ans("a", "b", "c", "d"), "お題", raw, REF, RoundSettings(points=(5, 1))))
        assert [res[t].points for t in ("t1", "t2", "t3", "t4")] == [5, 1, 0, 0]

    def test_negative_points_rejected(self) -> None:
        with pytest.raises(ValueError):
            RoundSettings(points=(3, -1))


class TestFoulsAndEdges:
    def test_foul_has_no_rank_and_zero_points(self) -> None:
        raw = {"ノイズ": 0.8}
        res = by_team(judge_round(ans("拡散モデル", "ノイズ", "潜在拡散モデル"), "拡散モデル", raw, REF, RoundSettings()))
        assert res["t1"].status == Status.FOUL and res["t1"].foul == FoulKind.SAME
        assert res["t3"].status == Status.FOUL and res["t3"].foul == FoulKind.CONTAINS
        assert res["t1"].rank is None and res["t1"].points == 0
        assert res["t2"].rank == 1 and res["t2"].points == 3

    def test_foul_words_are_not_embedded(self) -> None:
        entries = prepare_entries(ans("拡散モデル", "ノイズ"), "拡散モデル", RoundSettings())
        assert texts_to_embed(entries) == ["ノイズ"]

    def test_strict_contained_setting(self) -> None:
        entries = prepare_entries(ans("拡散"), "拡散モデル", RoundSettings(strict_contained=True))
        assert entries[0].foul == FoulKind.CONTAINED

    def test_all_fouls(self) -> None:
        res = judge_round(ans("拡散モデル", "拡散モデル", "潜在拡散モデル"), "拡散モデル", {}, REF, RoundSettings())
        assert all(r.rank is None and r.points == 0 for r in res)

    def test_empty_answers(self) -> None:
        raw = {"ノイズ": 0.8}
        res = by_team(judge_round(ans("", "ノイズ", "   "), "拡散モデル", raw, REF, RoundSettings()))
        assert res["t1"].status == Status.EMPTY and res["t3"].status == Status.EMPTY
        assert res["t1"].rank is None and res["t1"].points == 0
        assert res["t2"].rank == 1

    def test_only_one_team_answered(self) -> None:
        raw = {"ノイズ": 0.8}
        res = by_team(judge_round(ans("", "ノイズ", ""), "拡散モデル", raw, REF, RoundSettings()))
        assert res["t2"].rank == 1 and res["t2"].points == 3

    def test_nobody_answered(self) -> None:
        res = judge_round(ans("", ""), "拡散モデル", {}, REF, RoundSettings())
        assert all(r.status == Status.EMPTY for r in res)

    def test_empty_topic_raises(self) -> None:
        with pytest.raises(ValueError):
            prepare_entries(ans("a"), "   ", RoundSettings())

    def test_missing_similarity_raises(self) -> None:
        with pytest.raises(KeyError):
            judge_round(ans("a"), "お題", {}, REF, RoundSettings())


class TestManualFoul:
    def test_manual_foul_and_undo(self) -> None:
        raw = {"a": 0.9, "b": 0.8}
        settings = RoundSettings()
        res = judge_round(ans("a", "b"), "お題", raw, REF, settings)
        assert set_manual_foul(res, "t1", True, settings)
        r = by_team(res)
        assert r["t1"].status == Status.FOUL and r["t1"].foul == FoulKind.MANUAL and r["t1"].points == 0
        assert r["t2"].rank == 1 and r["t2"].points == 3
        assert set_manual_foul(res, "t1", False, settings)
        r = by_team(res)
        assert r["t1"].rank == 1 and r["t2"].rank == 2

    def test_cannot_undo_automatic_foul(self) -> None:
        res = judge_round(ans("お題", "b"), "お題", {"b": 0.5}, REF, RoundSettings())
        assert not set_manual_foul(res, "t1", False, RoundSettings())

    def test_unknown_team_or_empty(self) -> None:
        res = judge_round(ans("", "b"), "お題", {"b": 0.5}, REF, RoundSettings())
        assert not set_manual_foul(res, "t1", True, RoundSettings())
        assert not set_manual_foul(res, "zzz", True, RoundSettings())


class TestOrdering:
    def test_display_and_reveal_order(self) -> None:
        raw = {"a": 0.9, "b": 0.3}
        res = judge_round(ans("a", "お題", "b", ""), "お題", raw, REF, RoundSettings())
        assert [r.team_id for r in display_order(res)] == ["t1", "t3", "t2", "t4"]
        assert [r.team_id for r in reveal_order(res)] == ["t4", "t2", "t3", "t1"]

    def test_rank_entries_requires_scores(self) -> None:
        e = AnswerResult("t1", "a", "x", "x")
        with pytest.raises(ValueError):
            rank_entries([e], RoundSettings())


def test_filter_reference_words() -> None:
    kept, dropped = filter_reference_words(["冬", "冬休み", "りんご", "真冬", ""], "冬")
    assert kept == ["りんご"]
    assert dropped == ["冬", "冬休み", "真冬", ""]


def test_answer_result_roundtrip() -> None:
    r = judge_round(ans("お題"), "お題", {}, REF, RoundSettings())[0]
    assert AnswerResult.from_dict(r.to_dict()) == r
    r2 = judge_round(ans("a"), "お題", {"a": 0.5}, REF, RoundSettings())[0]
    assert AnswerResult.from_dict(r2.to_dict()) == r2
