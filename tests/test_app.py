"""Streamlit AppTest による画面の通しテスト。モデルは読まず FakeBackend を使う。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from semantle.live import reset_live
from semantle.paths import results_csv_file, state_file

SIMS = {"拡散モデル": 1.0, "ノイズ": 0.8, "画像生成": 0.7, "カレー": 0.2, "猫": 0.3}


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(parts)


def start_game(at: AppTest, n: int = 4) -> AppTest:
    at.run()
    assert not at.exception
    at.number_input(key="setup_n_teams").set_value(n).run()
    at.button(key="btn_start").click().run()
    assert not at.exception
    return at


def play(at: AppTest, topic: str, words: dict[str, str]) -> AppTest:
    at.text_input(key="topic").input(topic)
    for team_id, word in words.items():
        at.text_input(key=f"ans_{team_id}").input(word)
    at.run()
    at.button(key="btn_judge").click().run()
    assert not at.exception, at.exception
    return at


def projector(app_path: str) -> AppTest:
    """投影用画面（?view=screen）を別ウィンドウで開いた状態。"""
    screen = AppTest.from_file(app_path, default_timeout=60)
    screen.query_params["view"] = "screen"
    screen.run()
    assert not screen.exception, screen.exception
    return screen


def slide_text(screen: AppTest) -> str:
    screen.run()
    assert not screen.exception, screen.exception
    return all_text(screen)


@pytest.fixture
def at(app_path: str, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("SEMANTLE_FAKE_SIMS", json.dumps(SIMS, ensure_ascii=False))
    return AppTest.from_file(app_path, default_timeout=60)


class TestSetup:
    def test_first_run_shows_setup(self, at: AppTest) -> None:
        at.run()
        assert not at.exception
        assert at.button(key="btn_start")
        assert not state_file().exists()

    def test_start_creates_state_file(self, at: AppTest) -> None:
        start_game(at, 3)
        assert state_file().exists()
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert len(data["teams"]) == 3


class TestRound:
    def test_judge_without_reveal(self, at: AppTest) -> None:
        start_game(at)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー", "t3": "拡散モデル", "t4": ""})
        text = all_text(at)
        assert "🏆 1位：チーム1「ノイズ」" in text
        assert "反則：お題と同じ" in text
        assert "（未回答）" in text
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert len(data["rounds"]) == 1
        by = {r["team_id"]: r for r in data["rounds"][0]["results"]}
        assert by["t1"]["rank"] == 1 and by["t1"]["points"] == 3
        assert by["t2"]["rank"] == 2 and by["t2"]["points"] == 2
        assert by["t3"]["status"] == "foul" and by["t4"]["status"] == "empty"
        assert results_csv_file().exists()
        # 累計
        assert "3点" in text

    def test_reveal_step_by_step(self, at: AppTest, app_path: str) -> None:
        start_game(at)
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー", "t3": "画像生成", "t4": "猫"})
        # 司会者用: 発表前でも全部見える
        text = all_text(at)
        assert "発表済み 0 / 4" in text
        assert "🏆 1位：チーム1「ノイズ」" in text and "🙈 未発表" in text
        # 投影用: まだ何も出ていない
        shown = slide_text(screen)
        assert "結果発表　0 / 4" in shown
        assert "ノイズ" not in shown and "カレー" not in shown and "🏆" not in shown
        at.button(key="btn_slide_next").click().run()
        assert "発表済み 1 / 4" in all_text(at)
        shown = slide_text(screen)
        assert "カレー" in shown and "ノイズ" not in shown
        at.button(key="btn_reveal_all").click().run()
        shown = slide_text(screen)
        assert "🏆 1位：チーム1「ノイズ」" in shown
        assert "🙈 未発表" not in all_text(at)
        assert not at.exception

    def test_rule2_target(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        at.sidebar.radio(key="set_rule").set_value("target").run()
        at.sidebar.radio(key="set_calib").set_value("raw").run()
        at.sidebar.slider(key="set_target").set_value(0.25).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})  # 0.8 vs 0.2 → カレーが 0.25 に近い
        assert "🏆 1位：チーム2「カレー」" in all_text(at)
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["rounds"][0]["rule"] == "target" and data["rounds"][0]["target"] == 0.25
        assert data["settings"]["rule"] == "target"

    def test_judge_with_no_answers_warns(self, at: AppTest) -> None:
        start_game(at, 2)
        at.text_input(key="topic").input("拡散モデル").run()
        at.button(key="btn_judge").click().run()
        assert not at.exception
        assert any("入力されていません" in w.value for w in at.warning)
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["rounds"] == []

    def test_result_screen_then_next_round(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        # 判定後は入力欄が消え、結果だけになる
        assert not [t for t in at.text_input if t.key == "topic"]
        assert "🏆 1位：チーム1「ノイズ」" in all_text(at)
        assert at.button(key="btn_next_round")
        at.button(key="btn_next_round").click().run()
        assert not at.exception
        # 入力画面に戻り、入力は空。前回のまとめは残る
        assert at.text_input(key="topic").value == ""
        assert at.text_input(key="ans_t1").value == ""
        text = all_text(at)
        assert "前回の結果" in text and "🏆 1位：チーム1「ノイズ」" in text
        # 結果をもう一度見る
        at.button(key="btn_show_result").click().run()
        assert not [t for t in at.text_input if t.key == "topic"]
        assert at.button(key="mf_on_t1")

    def test_back_to_input_keeps_inputs(self, at: AppTest) -> None:
        start_game(at, 2)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        assert "発表済み 0 / 2" in all_text(at)
        at.button(key="btn_back_input").click().run()
        assert not at.exception
        assert at.text_input(key="ans_t1").value == "ノイズ"
        text = all_text(at)
        assert "発表済み" not in text and "発表が終わるまで" not in text  # 途中で戻ったら発表は終わった扱い
        assert "3点" in text

    def test_judge_disabled_without_topic(self, at: AppTest) -> None:
        start_game(at, 2)
        assert at.button(key="btn_judge").disabled

    def test_all_fouls(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "拡散モデル", "t2": "潜在拡散モデル"})
        text = all_text(at)
        assert "有効な回答なし" in text
        assert not at.exception

    def test_backend_failure_shows_error_not_crash(self, app_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEMANTLE_FAKE_BACKEND", "0")
        monkeypatch.setenv("SEMANTLE_ONLINE", "0")
        monkeypatch.setenv("HF_HOME", str(Path(state_file()).parent / "empty_hf"))  # 何もないキャッシュ
        at = AppTest.from_file(app_path, default_timeout=120)
        start_game(at, 2)
        at.sidebar.selectbox(key="set_model").set_value("ruri-v3-30m").run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        assert not at.exception
        assert at.error, "モデルが無い場合はエラー表示になるはず"
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["rounds"] == []
        assert at.text_input(key="ans_t1").value == "ノイズ"  # 入力は残る


class TestHostTools:
    def test_undo(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        at.sidebar.checkbox(key="undo_confirm").check().run()
        at.sidebar.button(key="btn_undo").click().run()
        assert not at.exception
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["rounds"] == []
        assert "0点" in all_text(at)

    def test_manual_adjustment(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.selectbox(key="adj_team").set_value("t2")
        at.sidebar.number_input(key="adj_delta").set_value(5)
        at.sidebar.text_input(key="adj_note").input("盛り上げ賞")
        at.run()
        at.sidebar.button(key="btn_adjust").click().run()
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["adjustments"][0]["team_id"] == "t2" and data["adjustments"][0]["delta"] == 5
        assert "5点" in all_text(at)
        at.sidebar.button(key=f"del_adj_{data['adjustments'][0]['id']}").click().run()
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["adjustments"] == []

    def test_rename_team(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.text_input(key="rename_t1").input("レッド").run()
        assert not at.exception
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["teams"][0]["name"] == "レッド"
        assert at.text_input(key="ans_t1").label == "レッド"

    def test_manual_foul_after_judgement(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        at.button(key="mf_on_t1").click().run()
        assert not at.exception
        data = json.loads(state_file().read_text(encoding="utf-8"))
        by = {r["team_id"]: r for r in data["rounds"][0]["results"]}
        assert by["t1"]["foul"] == "manual" and by["t1"]["points"] == 0
        assert by["t2"]["rank"] == 1 and by["t2"]["points"] == 3
        at.button(key="mf_off_t1").click().run()
        data = json.loads(state_file().read_text(encoding="utf-8"))
        by = {r["team_id"]: r for r in data["rounds"][0]["results"]}
        assert by["t1"]["rank"] == 1

    def test_points_setting_changes_awards(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        at.sidebar.number_input(key="set_pts_0").set_value(10).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        data = json.loads(state_file().read_text(encoding="utf-8"))
        assert data["rounds"][0]["results"][0]["points"] == 10

    def test_new_game_archives_file(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.checkbox(key="newgame_confirm").check().run()
        at.sidebar.button(key="btn_newgame").click().run()
        assert not at.exception
        assert at.button(key="btn_start")
        archived = list(state_file().parent.glob("game_state_*.json"))
        assert len(archived) == 1


class TestResume:
    def test_restart_restores_state(self, at: AppTest, app_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        # 別プロセス相当: 共有状態を捨て、新しい AppTest で起動し直す
        reset_live()
        at2 = AppTest.from_file(app_path, default_timeout=60)
        at2.run()
        assert not at2.exception
        text = all_text(at2)
        assert "第1ラウンド：お題「拡散モデル」" in text
        assert "🏆 1位：チーム1「ノイズ」" in text  # 再起動後は演出なしで全部表示
        assert "3点" in text
        assert at2.sidebar.toggle(key="set_reveal").value is False  # 設定も復元

    def test_broken_file_offers_new_game(self, at: AppTest) -> None:
        state_file().parent.mkdir(parents=True, exist_ok=True)
        state_file().write_text("{ broken", encoding="utf-8")
        at.run()
        assert not at.exception
        assert at.error and "読めません" in at.error[0].value
        at.button(key="btn_start").click().run()
        assert not at.exception
        assert len(list(state_file().parent.glob("game_state_*.json"))) == 1
        assert json.loads(state_file().read_text(encoding="utf-8"))["rounds"] == []


class TestTimer:
    def test_timer_renders(self, at: AppTest) -> None:
        start_game(at, 2)
        at.sidebar.number_input(key="set_timer").set_value(60).run()
        assert "01:00" in all_text(at)
        at.button(key="btn_timer_start").click().run()
        assert not at.exception
        at.button(key="btn_timer_stop").click().run()
        assert not at.exception


class TestProjector:
    """投影用画面は司会者用画面の操作に追従し、入力中の単語や操作ボタンを出さない。"""

    def test_title_before_and_after_start(self, at: AppTest, app_path: str) -> None:
        screen = projector(app_path)
        assert "準備中" in all_text(screen)
        assert not screen.button and not screen.text_input
        start_game(at, 2)
        text = slide_text(screen)
        assert "準備中" not in text and "チーム1" in text and "チーム2" in text
        assert not screen.button and not screen.text_input

    def test_slide_navigation(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        screen = projector(app_path)
        assert at.button(key="btn_slide_topic").disabled and at.button(key="btn_slide_result").disabled
        at.button(key="btn_slide_next").click().run()
        assert "ルール説明" in slide_text(screen)
        at.button(key="btn_slide_prev").click().run()
        assert "チーム対抗" in slide_text(screen)
        at.button(key="btn_slide_standings").click().run()
        assert "累計得点" in slide_text(screen)

    def test_topic_is_projected_but_answers_are_not(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        screen = projector(app_path)
        at.text_input(key="topic").input("拡散モデル")
        at.text_input(key="ans_t1").input("ノイズ")
        at.run()
        assert "拡散モデル" not in slide_text(screen)  # 入力しただけでは出ない
        at.button(key="btn_publish_topic").click().run()
        text = slide_text(screen)
        assert "お題発表" in text and "拡散モデル" in text and "第1問" in text
        assert "ノイズ" not in text

    def test_timer_shows_on_topic_slide(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        screen = projector(app_path)
        at.sidebar.number_input(key="set_timer").set_value(90).run()
        at.text_input(key="topic").input("拡散モデル").run()
        at.button(key="btn_publish_topic").click().run()
        assert "制限時間 01:30" in slide_text(screen)
        at.button(key="btn_timer_start").click().run()
        assert "⏱ 01:" in slide_text(screen)

    def test_full_round_flow_with_reference_slide(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        assert "🏆 1位：チーム1「ノイズ」" in slide_text(screen)
        # 司会者用画面に基準単語の最高スコアが出る
        data = json.loads(state_file().read_text(encoding="utf-8"))
        top = data["rounds"][0]["reference_top"]
        assert len(top) == 10 and top[0]["score"] >= top[1]["score"]
        assert "基準単語との比較" in all_text(at)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["基準単語の最高スコア"] == f"{top[0]['score']:.3f}"
        # 次へ → 基準単語 → 累計得点
        at.button(key="btn_slide_next").click().run()
        text = slide_text(screen)
        assert "基準単語の最高スコア" in text and top[0]["word"] in text and "ノイズ" in text
        at.button(key="btn_slide_next").click().run()
        text = slide_text(screen)
        assert "累計得点" in text and "3点" in text and "1 問終了" in text
        assert at.button(key="btn_slide_next").disabled

    def test_reference_slide_can_be_skipped(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        at.sidebar.toggle(key="set_ref_slide").set_value(False).run()
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        at.button(key="btn_slide_next").click().run()
        assert "累計得点" in slide_text(screen)
        assert not at.button(key="btn_slide_reference").disabled  # 直接映すことはできる

    def test_standings_hold_points_until_revealed(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        at.button(key="btn_slide_standings").click().run()
        text = slide_text(screen)
        assert "3点" not in text and "0 問終了" in text
        at.button(key="btn_slide_result").click().run()
        at.button(key="btn_reveal_all").click().run()
        at.button(key="btn_slide_standings").click().run()
        assert "3点" in slide_text(screen)

    def test_next_round_shows_standings_and_undo_leaves_result(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "ノイズ", "t2": "カレー"})
        at.button(key="btn_next_round").click().run()
        assert "累計得点" in slide_text(screen)
        at.button(key="btn_slide_result").click().run()
        at.sidebar.checkbox(key="undo_confirm").check().run()
        at.sidebar.button(key="btn_undo").click().run()
        text = slide_text(screen)
        assert "累計得点" in text and "ノイズ" not in text

    def test_html_in_names_is_escaped(self, at: AppTest, app_path: str) -> None:
        start_game(at, 2)
        at.sidebar.toggle(key="set_reveal").set_value(False).run()
        screen = projector(app_path)
        play(at, "拡散モデル", {"t1": "<b>ノイズ</b>", "t2": "カレー"})
        text = slide_text(screen)
        assert "&lt;b&gt;ノイズ&lt;/b&gt;" in text and "<b>ノイズ" not in text
