from pathlib import Path

import numpy as np
import pytest

from semantle.calibration import CalibrationMode
from semantle.embeddings.base import EmbeddingError, l2_normalize
from semantle.embeddings.cache import CachedBackend
from semantle.embeddings.fake import FailingBackend, FakeBackend
from semantle.embeddings.registry import MODEL_SPECS, get_spec
from semantle.judge import RoundSettings, Status, TeamAnswer
from semantle.reference import parse_reference_words
from semantle.service import JudgeService

REF_WORDS = ["りんご", "自転車", "机", "猫", "冬"]


def answers(*words: str) -> list[TeamAnswer]:
    return [TeamAnswer(f"t{i + 1}", f"チーム{i + 1}", w) for i, w in enumerate(words)]


class TestFakeBackend:
    def test_controlled_similarity(self) -> None:
        be = FakeBackend({"お題": 1.0, "近い": 0.8, "遠い": 0.2})
        v = be.embed(["お題", "近い", "遠い"])
        assert v.shape == (3, 8)
        assert v[1] @ v[0] == pytest.approx(0.8)
        assert v[2] @ v[0] == pytest.approx(0.2)

    def test_deterministic_and_normalized(self) -> None:
        be = FakeBackend()
        a = be.embed(["x", "y"])
        b = be.embed(["x", "y"])
        assert np.allclose(a, b)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)

    def test_empty(self) -> None:
        assert FakeBackend().embed([]).shape[0] == 0


class TestCachedBackend:
    def test_inner_called_once_per_text(self, tmp_path: Path) -> None:
        inner = FakeBackend({"a": 0.5})
        be = CachedBackend(inner, tmp_path / "c" / "emb.sqlite")
        v1 = be.embed(["a", "b", "a"])
        v2 = be.embed(["b", "a", "c"])
        assert v1.shape == (3, 8) and v2.shape == (3, 8)
        assert np.allclose(v1[0], v2[1])
        flattened = [t for call in inner.calls for t in call]
        assert sorted(flattened) == ["a", "b", "c"]
        assert be.cached_count() == 3

    def test_disk_cache_survives_new_instance(self, tmp_path: Path) -> None:
        db = tmp_path / "emb.sqlite"
        first = CachedBackend(FakeBackend({"a": 0.5}), db)
        v = first.embed(["a", "b"])
        inner2 = FakeBackend()  # 違う値を返すが、キャッシュがあれば呼ばれない
        second = CachedBackend(inner2, db)
        v2 = second.embed(["a", "b"])
        assert np.allclose(v, v2)
        assert inner2.calls == []

    def test_cache_key_separates_models(self, tmp_path: Path) -> None:
        db = tmp_path / "emb.sqlite"
        a = FakeBackend({"a": 0.9})
        a.cache_key = "model-a|"
        b = FakeBackend({"a": 0.1})
        b.cache_key = "model-b|"
        va = CachedBackend(a, db).embed(["a"])
        vb = CachedBackend(b, db).embed(["a"])
        assert not np.allclose(va, vb)

    def test_memory_only(self) -> None:
        be = CachedBackend(FakeBackend(), None)
        be.embed(["a"])
        assert be.cached_count() == 1

    def test_empty_input(self, tmp_path: Path) -> None:
        assert CachedBackend(FakeBackend(), tmp_path / "e.sqlite").embed([]).shape[0] == 0


class TestJudgeService:
    def test_play_round_end_to_end(self) -> None:
        be = FakeBackend({"拡散モデル": 1.0, "ノイズ": 0.8, "カレー": 0.1})
        svc = JudgeService(be, REF_WORDS)
        out = svc.play_round(" 拡散モデル ", answers("ノイズ", "カレー", "拡散モデル", ""), RoundSettings())
        by = {r.team_id: r for r in out.results}
        assert out.topic == "拡散モデル"
        assert by["t1"].rank == 1 and by["t2"].rank == 2
        assert by["t3"].status == Status.FOUL and by["t4"].status == Status.EMPTY
        assert by["t1"].raw == pytest.approx(0.8)
        assert out.reference.count == len(REF_WORDS)
        # 補正A: (raw - μ)/(1 - μ)
        mu = out.reference.mean
        assert by["t1"].score == pytest.approx((0.8 - mu) / (1 - mu))

    def test_reference_words_overlapping_topic_are_excluded(self) -> None:
        svc = JudgeService(FakeBackend({"冬": 1.0}), REF_WORDS)
        out = svc.play_round("冬", answers("雪"), RoundSettings())
        assert out.reference.count == len(REF_WORDS) - 1
        assert out.reference.excluded == ["冬"]

    def test_all_reference_words_excluded_raises(self) -> None:
        svc = JudgeService(FakeBackend(), ["冬"])
        with pytest.raises(ValueError):
            svc.play_round("冬", answers("雪"), RoundSettings())

    def test_backend_failure_propagates(self) -> None:
        svc = JudgeService(FailingBackend("落ちた", kind="load"), REF_WORDS)
        with pytest.raises(EmbeddingError) as ei:
            svc.play_round("お題", answers("a"), RoundSettings())
        assert ei.value.kind == "load"

    def test_no_valid_answers_skips_embedding_of_words(self) -> None:
        be = FakeBackend()
        svc = JudgeService(be, REF_WORDS)
        out = svc.play_round("お題", answers("お題", ""), RoundSettings())
        assert all(not r.is_valid for r in out.results)
        assert all("お題" not in call[1:] for call in be.calls)

    def test_score_table(self) -> None:
        svc = JudgeService(FakeBackend({"お題": 1.0, "近い": 0.9, "遠い": 0.0}), REF_WORDS)
        rows = svc.score_table("お題", ["近い", "遠い"])
        assert rows[0]["raw"] == pytest.approx(0.9)
        assert rows[0]["percentile"] == 1.0
        assert rows[1]["percentile"] == 0.0
        assert rows[0]["linear"] > rows[1]["linear"]

    def test_calibration_modes_via_service(self) -> None:
        be = FakeBackend({"お題": 1.0, "a": 0.9})
        for mode in CalibrationMode:
            out = JudgeService(be, REF_WORDS).play_round("お題", answers("a"), RoundSettings(calibration=mode))
            assert out.results[0].raw == pytest.approx(0.9)
            assert out.results[0].score is not None


def test_parse_reference_words() -> None:
    text = "# コメント\nりんご\n\n猫  # 行末コメント\nネコ\nねこ\n  自転車  \n"
    assert parse_reference_words(text) == ["りんご", "猫", "ネコ", "自転車"]


def test_bundled_reference_words_file_is_sane() -> None:
    from semantle.paths import reference_words_file
    from semantle.reference import load_reference_words

    words = load_reference_words(reference_words_file())
    assert 80 <= len(words) <= 200
    assert len(set(words)) == len(words)


def test_registry() -> None:
    assert get_spec("ruri-v3-310m").model_id == "cl-nagoya/ruri-v3-310m"
    assert get_spec("me5-large").format("猫") == "query: 猫"
    assert get_spec("ruri-v3-310m").format("猫") == "猫"
    assert get_spec("qwen3-0.6b").format("猫").startswith("Instruct: ")
    with pytest.raises(KeyError):
        get_spec("nope")
    assert len({s.cache_key for s in MODEL_SPECS.values()}) == len(MODEL_SPECS)


def test_l2_normalize_handles_zero_vector() -> None:
    v = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(v[1], [0.6, 0.8]) and np.allclose(v[0], 0)
