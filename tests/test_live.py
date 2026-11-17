"""共有状態（スライドの進行）と投影用スライドの HTML。Streamlit は使わない。"""

from __future__ import annotations

from semantle.judge import RoundSettings, TeamAnswer, judge_round
from semantle.live import LiveSession, get_live, reset_live
from semantle.slides import render_slide
from semantle.state import GameSettings, GameState, RoundRecord, load_state, save_state

import numpy as np

REF_TOP = [
    {"word": "りんご", "raw": 0.61, "score": 0.42},
    {"word": "机", "raw": 0.55, "score": 0.33},
]


def make_live(n_rounds: int = 0, reference: bool = True, **settings: object) -> LiveSession:
    live = LiveSession()
    live.game = GameState.new_game(["赤", "青", "緑"], GameSettings(**settings))
    for _ in range(n_rounds):
        add_round(live, reference=reference)
    return live


def add_round(live: LiveSession, reference: bool = True) -> RoundRecord:
    game = live.game
    assert game is not None
    answers = [TeamAnswer(t.id, t.name, w) for t, w in zip(game.teams, ["ノイズ", "カレー", "お題"])]
    results = judge_round(
        answers, "お題", {"ノイズ": 0.8, "カレー": 0.2}, np.array([0.1, 0.3]), RoundSettings()
    )
    rec = RoundRecord(
        round_no=game.next_round_no(), timestamp="", topic="お題", rule="closest", target=0.5,
        calibration="linear", points=[3, 2, 1], model_key="fake", model_name="fake", results=results,
        reference_mean=0.2, reference_count=100, reference_top=list(REF_TOP) if reference else [],
    )
    game.add_round(rec)
    return rec


class TestFlow:
    def test_before_any_round(self) -> None:
        live = make_live()
        assert live.available() == {"title", "rules", "standings"}
        assert live.next_action() == ("rules", "ルール説明へ ▶")
        assert live.go_next() and live.pres.slide == "rules"
        assert live.next_action() is None  # お題を出すまで先へ進めない
        live.publish_topic("お題")
        assert live.pres.slide == "topic" and "topic" in live.available()
        assert live.next_action() is None
        assert live.go_prev() and live.pres.slide == "rules"
        assert live.go_next() and live.pres.slide == "topic"

    def test_cannot_show_unavailable_slide(self) -> None:
        live = make_live()
        assert not live.show("result") and live.pres.slide == "title"
        assert not LiveSession().show("standings")

    def test_reveal_then_reference_then_standings(self) -> None:
        live = make_live()
        live.publish_topic("お題")
        live.pres.timer_end = 123.0
        add_round(live)
        live.start_reveal(enabled=True, now=0.0)
        assert live.pres.slide == "result" and live.pres.topic == "" and live.pres.timer_end is None
        assert live.reveal_progress() == (0, 3) and live.pending_round_no() == 1
        assert live.next_action() == ("reveal", "次を発表 ▶（0 / 3）")
        for expected in (1, 2, 3):
            assert live.go_next()
            assert live.reveal_progress() == (expected, 3)
        assert live.reveal_complete() and live.pending_round_no() is None
        assert live.next_action() == ("reference", "基準単語へ ▶")
        assert live.go_next() and live.pres.slide == "reference"
        assert live.go_next() and live.pres.slide == "standings"
        assert not live.go_next()
        assert live.go_prev() and live.pres.slide == "reference"
        assert live.go_prev() and live.pres.slide == "result"

    def test_reveal_disabled_shows_everything(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=False)
        assert live.reveal_complete()

    def test_reference_slide_skipped_when_off_or_missing(self) -> None:
        off = make_live(1, show_reference_slide=False)
        off.start_reveal(enabled=False)
        assert off.next_action() == ("standings", "累計得点へ ▶")
        assert "reference" in off.available()
        assert off.go_next() and off.prev_target() == "result"
        old = make_live(1, reference=False)  # 以前の保存ファイルには基準単語の記録が無い
        old.start_reveal(enabled=False)
        assert "reference" not in old.available()
        assert old.next_action() == ("standings", "累計得点へ ▶")

    def test_restart_treats_last_round_as_revealed(self) -> None:
        live = make_live(1)
        assert live.reveal_progress() == (3, 3)

    def test_auto_reveal_tick(self) -> None:
        live = make_live(1, reveal_auto=True, reveal_interval_sec=3.0)
        live.start_reveal(enabled=True, now=100.0)
        assert not live.tick(now=101.0)
        assert live.tick(now=103.0) and live.reveal_progress() == (1, 3)
        assert not live.tick(now=103.5)  # 2 つの画面から呼ばれても二重には進まない
        live.show("standings")
        assert not live.tick(now=200.0)

    def test_tick_does_nothing_without_auto(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=True, now=0.0)
        assert not live.tick(now=999.0) and live.reveal_progress() == (0, 3)

    def test_undo_leaves_result_slide(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=True)
        assert live.game is not None
        live.game.undo_last_round()
        live.round_removed()
        assert live.pres.slide == "standings"

    def test_get_live_is_shared_per_file(self, tmp_path) -> None:
        a = get_live(tmp_path / "a.json")
        assert get_live(tmp_path / "a.json") is a
        assert get_live(tmp_path / "b.json") is not a
        reset_live()
        assert get_live(tmp_path / "a.json") is not a


class TestSlides:
    def test_title_without_game(self) -> None:
        assert "準備中" in render_slide(LiveSession())

    def test_hidden_rows_do_not_leak_words(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=True)
        html = render_slide(live)
        assert "ノイズ" not in html and "カレー" not in html and "🏆" not in html
        assert html.count("？？？") == 3
        live.go_next()  # 反則のチームから
        html = render_slide(live)
        assert "反則" in html and "カレー" not in html and html.count("？？？") == 2
        live.advance_reveal(None)
        html = render_slide(live)
        assert "🏆 1位：赤「ノイズ」" in html and "？？？" not in html

    def test_reference_slide_hides_winner_until_revealed(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=True)
        live.show("reference")
        html = render_slide(live)
        assert "基準単語の最高スコア" in html and "0.420" in html and "りんご" in html
        assert "ノイズ" not in html
        live.advance_reveal(None)
        assert "ノイズ" in render_slide(live)

    def test_standings_exclude_unrevealed_round(self) -> None:
        live = make_live(1)
        live.start_reveal(enabled=True)
        live.show("standings")
        assert "3点" not in render_slide(live)
        live.advance_reveal(None)
        assert "3点" in render_slide(live)

    def test_topic_slide_timer(self) -> None:
        live = make_live(timer_sec=90)
        live.publish_topic("拡散モデル")
        html = render_slide(live, now=0.0)
        assert "拡散モデル" in html and "第1問" in html and "制限時間 01:30" in html
        live.pres.timer_end = 75.0
        assert "01:15" in render_slide(live, now=0.0)
        assert "時間です" in render_slide(live, now=80.0)

    def test_rules_slide_follows_settings(self) -> None:
        live = make_live(rule="target", target=0.4, points=[5, 0, 0], strict_contained=True)
        live.show("rules")
        html = render_slide(live)
        assert "目標値 0.40" in html and "1位 5点" in html and "2位" not in html and "お題に含まれる単語" in html

    def test_escapes_and_single_line(self) -> None:
        live = make_live()
        assert live.game is not None
        live.game.rename_team("t1", "<script>x</script>")
        live.publish_topic("a & b")
        for slide in ("title", "topic", "standings"):
            live.show(slide)
            html = render_slide(live)
            assert "<script>" not in html and "\n" not in html
        live.show("topic")
        assert "a &amp; b" in render_slide(live)

    def test_unavailable_slide_falls_back(self) -> None:
        live = make_live()
        live.pres.slide = "result"
        assert "累計得点" in render_slide(live)


def test_reference_top_survives_save_and_load(tmp_path) -> None:
    live = make_live(1)
    assert live.game is not None
    save_state(live.game, tmp_path / "s.json")
    loaded = load_state(tmp_path / "s.json")
    assert loaded is not None
    assert loaded.rounds[0].reference_top == REF_TOP
    assert loaded.settings.show_reference_slide is True


def test_old_save_without_reference_top_loads() -> None:
    live = make_live(1)
    assert live.game is not None
    d = live.game.to_dict()
    del d["rounds"][0]["reference_top"]
    del d["settings"]["show_reference_slide"]
    loaded = GameState.from_dict(d)
    assert loaded.rounds[0].reference_top == [] and loaded.settings.show_reference_slide is True
