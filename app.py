"""人力Semantle — 画面（Streamlit）。ロジックは semantle/ パッケージにある。

起動:  streamlit run app.py
環境変数:
  SEMANTLE_FAKE_BACKEND=1   モデルを読まずテスト用のモック埋め込みを使う（AppTest 用）
  SEMANTLE_FAKE_SIMS='{"単語": 類似度, ...}'  モックが返すお題との類似度（任意）
  SEMANTLE_SAVE_DIR / SEMANTLE_CACHE_DIR      保存先の変更
  SEMANTLE_ONLINE=1          Hugging Face Hub への接続を許可する（既定はオフライン）
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from semantle.calibration import CalibrationMode
from semantle.embeddings.base import EmbeddingBackend, EmbeddingError
from semantle.embeddings.cache import CachedBackend
from semantle.embeddings.registry import MODEL_SPECS, get_spec
from semantle.judge import AnswerResult, Rule, Status, TeamAnswer, display_order
from semantle.paths import cache_db_file, reference_words_file, results_csv_file, state_file
from semantle.reference import load_reference_words
from semantle.service import JudgeService
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

st.set_page_config(page_title="人力Semantle", page_icon="🧠", layout="wide")

MAX_TEAMS = 8

# ====================================================================== 見た目


def inject_css(scale: float) -> None:
    st.markdown(
        f"""
<style>
html {{ font-size: {int(16 * scale)}px; }}
h1 {{ font-size: 2.6rem !important; }}
h2 {{ font-size: 2.0rem !important; }}
h3 {{ font-size: 1.6rem !important; }}
.stTextInput input {{ font-size: 1.5rem !important; padding: 0.5rem 0.75rem !important; }}
.stTextInput label p, .stNumberInput label p, .stSelectbox label p, .stRadio label p {{ font-size: 1.1rem !important; }}
.stButton button {{ font-size: 1.3rem !important; padding: 0.5rem 1.2rem !important; }}
div[data-testid="stMarkdownContainer"] p {{ font-size: 1.15rem; }}
.sem-row {{ display:flex; align-items:center; gap:1rem; padding:0.6rem 1rem; margin:0.35rem 0;
           border-radius:0.8rem; border:2px solid rgba(128,128,128,0.35); font-size:1.7rem; }}
.sem-row .rank {{ font-size:2.2rem; font-weight:800; min-width:3.5rem; text-align:center; }}
.sem-row .team {{ font-weight:700; min-width:8rem; }}
.sem-row .word {{ font-size:2.0rem; font-weight:800; flex:1; }}
.sem-row .score {{ font-variant-numeric: tabular-nums; font-weight:700; }}
.sem-row .pts {{ font-size:1.4rem; font-weight:800; min-width:4rem; text-align:right; }}
.sem-row.first {{ border-color:#d4a017; background:rgba(212,160,23,0.12); }}
.sem-row.foul {{ border-color:#c0392b; background:rgba(192,57,43,0.10); }}
.sem-row.empty {{ opacity:0.55; }}
.sem-banner {{ font-size:2.6rem; font-weight:900; text-align:center; padding:1rem;
              border-radius:1rem; background:rgba(212,160,23,0.18); margin:0.5rem 0 1rem 0; }}
.sem-board {{ font-size:1.5rem; }}
.sem-board .row {{ display:flex; gap:0.8rem; padding:0.35rem 0.5rem; border-bottom:1px solid rgba(128,128,128,0.3);
                  font-variant-numeric: tabular-nums; }}
.sem-board .rank {{ font-weight:800; min-width:2.5rem; }}
.sem-board .name {{ flex:1; font-weight:600; }}
.sem-board .pts {{ font-weight:800; }}
.sem-timer {{ font-size:3.2rem; font-weight:900; text-align:center; font-variant-numeric: tabular-nums; }}
.sem-timer.over {{ color:#c0392b; }}
.sem-last {{ font-size:1.2rem; padding:0.6rem 1rem; border-radius:0.8rem; border:1px solid rgba(128,128,128,0.35); }}
</style>
""",
        unsafe_allow_html=True,
    )


# ====================================================================== バックエンド


def fake_mode() -> bool:
    return os.environ.get("SEMANTLE_FAKE_BACKEND") == "1"


@st.cache_resource(show_spinner=False)
def get_backend(model_key: str) -> EmbeddingBackend:
    if fake_mode():
        from semantle.embeddings.fake import FakeBackend

        sims = json.loads(os.environ.get("SEMANTLE_FAKE_SIMS", "{}"))
        return FakeBackend(sims)
    from semantle.embeddings.local_st import SentenceTransformerBackend, set_offline_mode

    set_offline_mode(os.environ.get("SEMANTLE_ONLINE") != "1")
    return CachedBackend(SentenceTransformerBackend(get_spec(model_key)), cache_db_file())


def load_backend_with_feedback(model_key: str) -> EmbeddingBackend | None:
    """モデルを読み込む。失敗しても画面は止めず、メッセージを出して None を返す。"""
    label = MODEL_SPECS[model_key].label if model_key in MODEL_SPECS else model_key
    try:
        with st.spinner(f"{label} を読み込み中…（初回は数十秒かかります）"):
            backend = get_backend(model_key)
            backend.warm_up()
        st.session_state.setdefault("loaded_models", set()).add(model_key)
        return backend
    except EmbeddingError as e:
        get_backend.clear()
        st.error(f"モデルを読み込めませんでした。\n\n{e}")
        if e.kind == "not_found":
            st.info("ネットのある場所で `python scripts/download_models.py` を実行してからもう一度試してください。")
        return None
    except Exception as e:  # noqa: BLE001 - 画面を止めない
        get_backend.clear()
        st.error(f"モデルの読み込みで予期しないエラー: {e}")
        return None


# ====================================================================== 状態


def persist(state: GameState) -> None:
    try:
        save_state(state, state_file())
        write_csv(state, results_csv_file())
    except OSError as e:
        st.error(f"保存に失敗しました（ゲームは続けられます）: {e}")


def settings_keys() -> dict[str, str]:
    return {
        "model_key": "set_model",
        "rule": "set_rule",
        "target": "set_target",
        "calibration": "set_calib",
        "strict_contained": "set_strict",
        "reveal_enabled": "set_reveal",
        "reveal_auto": "set_reveal_auto",
        "reveal_interval_sec": "set_reveal_interval",
        "timer_sec": "set_timer",
    }


def sync_settings_to_widgets(settings: GameSettings) -> None:
    for attr, key in settings_keys().items():
        st.session_state[key] = getattr(settings, attr)
    pts = list(settings.points) + [0, 0, 0]
    for i in range(3):
        st.session_state[f"set_pts_{i}"] = int(pts[i])


def request_reset(key: str) -> None:
    """ウィジェットの値は描画後に変更できないので、次の実行の冒頭で False に戻す。"""
    st.session_state.setdefault("_reset_keys", []).append(key)


def apply_requested_resets() -> None:
    for key in st.session_state.pop("_reset_keys", []):
        st.session_state[key] = False


def init_state() -> None:
    apply_requested_resets()
    if "game" in st.session_state:
        return
    st.session_state["game"] = None
    st.session_state["load_error"] = None
    try:
        state = load_state(state_file())
    except StateLoadError as e:
        st.session_state["load_error"] = str(e)
        state = None
    st.session_state["game"] = state
    if state is not None:
        sync_settings_to_widgets(state.settings)
    else:
        sync_settings_to_widgets(GameSettings())


def archive_state_file() -> None:
    path = state_file()
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.rename(path.with_name(f"game_state_{stamp}.json"))


def current_reveal(state: GameState) -> tuple[int, int]:
    """(表示済みの件数, 全件数)。演出が終わっていれば両者が等しい。"""
    if not state.rounds:
        return 0, 0
    last = state.rounds[-1]
    n = len(last.results)
    rv = st.session_state.get("reveal")
    if not rv or rv.get("round_no") != last.round_no:
        return n, n
    return min(int(rv.get("shown", n)), n), n


def start_reveal(round_no: int, total: int, enabled: bool) -> None:
    st.session_state["reveal"] = {
        "round_no": round_no,
        "shown": 0 if enabled else total,
        "last_advance": time.time(),
    }


def advance_reveal(state: GameState, step: int | None = 1) -> None:
    shown, n = current_reveal(state)
    rv = st.session_state.get("reveal") or {}
    rv["round_no"] = state.rounds[-1].round_no
    rv["shown"] = n if step is None else min(n, shown + step)
    rv["last_advance"] = time.time()
    st.session_state["reveal"] = rv


# ====================================================================== 部品


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def result_row_html(r: AnswerResult) -> str:
    """1チーム分の結果行。細かい数値（生の類似度など）は出さず、順位・単語・スコア・得点だけ。"""
    if r.status == Status.EMPTY:
        return f'<div class="sem-row empty"><span class="rank">—</span><span class="team">{r.team_name}</span><span class="word">（未回答）</span></div>'
    if r.status == Status.FOUL:
        label = r.foul.label if r.foul else "反則"
        return (
            f'<div class="sem-row foul"><span class="rank">🚫</span><span class="team">{r.team_name}</span>'
            f'<span class="word">{r.word}</span><span class="score">反則：{label}</span><span class="pts">0点</span></div>'
        )
    cls = "sem-row first" if r.rank == 1 else "sem-row"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank or 0, f"{r.rank}位")
    return (
        f'<div class="{cls}"><span class="rank">{medal}</span><span class="team">{r.team_name}</span>'
        f'<span class="word">{r.word}</span><span class="score">{fmt(r.score)}</span><span class="pts">+{r.points}点</span></div>'
    )


def winner_banner_html(results: list[AnswerResult]) -> str:
    winners = [r for r in results if r.rank == 1]
    if not winners:
        return '<div class="sem-banner">😱 有効な回答なし</div>'
    names = "・".join(f"{w.team_name}「{w.word}」" for w in winners)
    return f'<div class="sem-banner">🏆 1位：{names}</div>'


def details_table(results: list[AnswerResult]) -> pd.DataFrame:
    rows = []
    for r in display_order(results):
        rows.append(
            {
                "チーム": r.team_name, "単語": r.word or "（未回答）", "順位": "" if r.rank is None else r.rank,
                "スコア": fmt(r.score), "生の類似度": fmt(r.raw), "反則": r.foul.label if r.foul else "", "得点": r.points,
            }
        )
    return pd.DataFrame(rows)


def score_chart(results: list[AnswerResult], rule: Rule, target: float, calibrated: bool) -> alt.Chart | None:
    valid = [r for r in results if r.is_valid and r.score is not None]
    if not valid:
        return None
    df = pd.DataFrame(
        {
            "label": [f"{r.team_name}「{r.word}」" for r in valid],
            "score": [r.score for r in valid],
            "raw": [r.raw for r in valid],
            "rank": [r.rank for r in valid],
        }
    )
    order = [row.label for row in sorted(df.itertuples(), key=lambda t: t.rank)]
    lo = min(0.0, float(df["score"].min()))
    hi = max(1.0, float(df["score"].max()))
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=20, labelLimit=400)),
        x=alt.X(
            "score:Q",
            title="補正スコア" if calibrated else "コサイン類似度",
            scale=alt.Scale(domain=[lo, hi]),
            axis=alt.Axis(labelFontSize=16, titleFontSize=16),
        ),
    )
    bars = base.mark_bar(size=26, cornerRadiusEnd=4, color="#4C78A8").encode(
        tooltip=[alt.Tooltip("label:N", title="チーム"), alt.Tooltip("score:Q", format=".3f", title="スコア"),
                 alt.Tooltip("raw:Q", format=".3f", title="生の類似度")]
    )
    labels = base.mark_text(align="left", dx=6, fontSize=18).encode(text=alt.Text("score:Q", format=".3f"))
    layers: list[alt.Chart] = [bars, labels]
    if rule == Rule.TARGET:
        rule_df = pd.DataFrame({"x": [target]})
        layers.append(
            alt.Chart(rule_df).mark_rule(color="#C0392B", strokeDash=[8, 6], size=3).encode(x="x:Q")
        )
        layers.append(
            alt.Chart(rule_df).mark_text(text=f"目標 {target:.2f}", color="#C0392B", dy=-8, fontSize=16, align="left", dx=4)
            .encode(x="x:Q", y=alt.value(0))
        )
    return alt.layer(*layers).properties(height=max(160, 56 * len(valid)))


def leaderboard_html(state: GameState, exclude_round_no: int | None) -> str:
    rows = state.leaderboard(exclude_round_no=exclude_round_no)
    html = ['<div class="sem-board">']
    for rank, team, total in rows:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
        html.append(f'<div class="row"><span class="rank">{medal}</span><span class="name">{team.name}</span><span class="pts">{total}点</span></div>')
    html.append("</div>")
    return "".join(html)


# ====================================================================== サイドバー


def sidebar(state: GameState | None) -> GameSettings:
    with st.sidebar:
        st.header("⚙️ 設定")
        st.slider("文字の大きさ", 0.8, 1.8, 1.15, 0.05, key="font_scale")

        model_key = st.selectbox(
            "埋め込みモデル", list(MODEL_SPECS), format_func=lambda k: MODEL_SPECS[k].label, key="set_model"
        )
        loaded = model_key in st.session_state.get("loaded_models", set())
        col_a, col_b = st.columns([1, 1])
        col_a.caption("✅ 読み込み済み" if loaded else "⏳ 未読み込み")
        if col_b.button("モデルを読み込む", key="btn_load_model", disabled=loaded):
            load_backend_with_feedback(model_key)
            st.rerun()
        st.caption(MODEL_SPECS[model_key].notes)

        st.radio("ルール", [r.value for r in Rule], format_func=lambda v: Rule(v).label, key="set_rule")
        st.slider("目標値（ルール②）", -0.5, 1.0, step=0.05, key="set_target",
                  disabled=st.session_state["set_rule"] != Rule.TARGET.value)
        st.radio(
            "スコアの補正", [c.value for c in CalibrationMode],
            format_func=lambda v: CalibrationMode(v).label, key="set_calib",
        )
        st.markdown("**得点（1位 / 2位 / 3位）**")
        p1, p2, p3 = st.columns(3)
        p1.number_input("1位", 0, 20, key="set_pts_0", label_visibility="collapsed")
        p2.number_input("2位", 0, 20, key="set_pts_1", label_visibility="collapsed")
        p3.number_input("3位", 0, 20, key="set_pts_2", label_visibility="collapsed")
        st.toggle("お題に含まれる単語も反則にする（例: 拡散モデル→「拡散」）", key="set_strict")

        st.markdown("**演出**")
        st.toggle("下位から1チームずつ発表する", key="set_reveal")
        st.toggle("自動で次を表示する", key="set_reveal_auto", disabled=not st.session_state["set_reveal"])
        st.slider("自動表示の間隔（秒）", 1.0, 10.0, step=0.5, key="set_reveal_interval",
                  disabled=not (st.session_state["set_reveal"] and st.session_state["set_reveal_auto"]))
        st.number_input("制限時間（秒、0で使わない）", 0, 600, step=10, key="set_timer")

        settings = GameSettings(
            model_key=st.session_state["set_model"],
            rule=st.session_state["set_rule"],
            target=float(st.session_state["set_target"]),
            calibration=st.session_state["set_calib"],
            points=[int(st.session_state[f"set_pts_{i}"]) for i in range(3)],
            strict_contained=bool(st.session_state["set_strict"]),
            reveal_enabled=bool(st.session_state["set_reveal"]),
            reveal_auto=bool(st.session_state["set_reveal_auto"]),
            reveal_interval_sec=float(st.session_state["set_reveal_interval"]),
            timer_sec=int(st.session_state["set_timer"]),
        )

        if state is not None:
            if settings != state.settings:
                state.settings = settings
                persist(state)
            host_tools(state)
    return settings


def host_tools(state: GameState) -> None:
    st.divider()
    st.header("🎙️ 司会ツール")

    with st.expander("👥 チーム名の変更", expanded=False):
        for team in state.teams:
            new_name = st.text_input(f"{team.id}", value=team.name, key=f"rename_{team.id}")
            if new_name.strip() and new_name.strip() != team.name:
                state.rename_team(team.id, new_name)
                persist(state)
                st.rerun()
        if len(state.teams) < MAX_TEAMS and st.button("チームを追加", key="btn_add_team"):
            state.add_team(f"チーム{len(state.teams) + 1}")
            persist(state)
            st.rerun()

    with st.expander("↩️ 直前のラウンドを取り消す（Undo）", expanded=False):
        if state.rounds:
            last = state.rounds[-1]
            st.caption(f"第{last.round_no}ラウンド「{last.topic}」を取り消します")
            if st.checkbox("取り消してよい", key="undo_confirm"):
                if st.button("取り消す", key="btn_undo", type="primary"):
                    state.undo_last_round()
                    st.session_state.pop("reveal", None)
                    st.session_state["screen"] = "input"
                    persist(state)
                    request_reset("undo_confirm")
                    st.toast(f"第{last.round_no}ラウンドを取り消しました")
                    st.rerun()
        else:
            st.caption("まだラウンドがありません")

    with st.expander("✏️ 得点の手動修正", expanded=False):
        team_id = st.selectbox(
            "チーム", [t.id for t in state.teams], format_func=lambda i: state.team_by_id(i).name, key="adj_team"
        )
        delta = st.number_input("増減（マイナス可）", -50, 50, 1, key="adj_delta")
        note = st.text_input("メモ", key="adj_note", placeholder="例: 盛り上げ賞")
        if st.button("反映する", key="btn_adjust"):
            state.add_adjustment(team_id, int(delta), note)
            persist(state)
            st.rerun()
        if state.adjustments:
            st.caption("修正の履歴")
            for adj in state.adjustments:
                c1, c2 = st.columns([4, 1])
                c1.write(f"{state.team_by_id(adj.team_id).name}  {adj.delta:+d}点  {adj.note}")
                if c2.button("削除", key=f"del_adj_{adj.id}"):
                    state.remove_adjustment(adj.id)
                    persist(state)
                    st.rerun()

    with st.expander("💾 保存・出力", expanded=False):
        st.caption(f"自動保存先: {state_file()}")
        st.download_button(
            "全ラウンドの結果を CSV でダウンロード",
            data=export_csv(state).encode("utf-8-sig"),
            file_name=f"semantle_results_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            key="btn_csv",
        )
        st.caption(f"（{results_csv_file()} にも毎ラウンド自動で書き出しています）")

    with st.expander("🆕 新しいゲームを始める", expanded=False):
        st.caption("今の得点・履歴はファイル名に日時を付けて残し、まっさらな状態にします。")
        if st.checkbox("本当に新しいゲームを始める", key="newgame_confirm"):
            if st.button("新しいゲームへ", key="btn_newgame", type="primary"):
                archive_state_file()
                st.session_state["game"] = None
                st.session_state.pop("reveal", None)
                st.session_state["screen"] = "input"
                request_reset("newgame_confirm")
                st.rerun()


# ====================================================================== 画面


def setup_screen(settings: GameSettings) -> None:
    st.title("🧠 人力Semantle")
    if st.session_state.get("load_error"):
        st.error(f"前回の保存ファイルを読めませんでした（壊れている可能性があります）。\n\n{st.session_state['load_error']}")
        st.caption("ファイルはそのまま残します。新しいゲームを始めると日時付きの名前で退避されます。")
    st.subheader("チームを決めて始めましょう")
    n = st.number_input("チーム数", 1, MAX_TEAMS, 4, key="setup_n_teams")
    names: list[str] = []
    cols = st.columns(int(n))
    for i in range(int(n)):
        names.append(cols[i].text_input(f"チーム{i + 1}", value=f"チーム{i + 1}", key=f"setup_name_{i}"))
    if st.button("🎮 ゲーム開始", type="primary", key="btn_start"):
        if st.session_state.get("load_error"):
            archive_state_file()
            st.session_state["load_error"] = None
        state = GameState.new_game(names, settings)
        st.session_state["game"] = state
        persist(state)
        st.rerun()


def timer_area(timer_sec: int) -> None:
    if timer_sec <= 0:
        return
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("⏱ スタート", key="btn_timer_start"):
        st.session_state["timer_end"] = time.time() + timer_sec
    if c2.button("⏹ 停止", key="btn_timer_stop"):
        st.session_state.pop("timer_end", None)
    end = st.session_state.get("timer_end")

    @st.fragment(run_every=1.0 if end else None)
    def show() -> None:
        end_t = st.session_state.get("timer_end")
        if not end_t:
            st.markdown(f'<div class="sem-timer">{timer_sec // 60:02d}:{timer_sec % 60:02d}</div>', unsafe_allow_html=True)
            return
        remain = int(round(end_t - time.time()))
        if remain <= 0:
            st.markdown('<div class="sem-timer over">⏰ 時間です！</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sem-timer">{remain // 60:02d}:{remain % 60:02d}</div>', unsafe_allow_html=True)

    with c3:
        show()


def input_area(state: GameState, settings: GameSettings) -> None:
    if st.session_state.pop("clear_inputs", False):
        st.session_state["topic"] = ""
        for team in state.teams:
            st.session_state[f"ans_{team.id}"] = ""
    # 結果画面の間は入力欄が無いので値が消える。判定時に控えたものを戻す
    for key, value in (st.session_state.pop("restore_inputs", None) or {}).items():
        st.session_state[key] = value

    st.text_input("📝 お題", key="topic", placeholder="例：拡散モデル")
    cols = st.columns(len(state.teams))
    for col, team in zip(cols, state.teams):
        col.text_input(team.name, key=f"ans_{team.id}", placeholder="単語")

    timer_area(settings.timer_sec)

    b1, b2 = st.columns([3, 1])
    topic = st.session_state.get("topic", "")
    if b1.button("🎯 判定！", type="primary", key="btn_judge", disabled=not topic.strip()):
        run_judgement(state, settings)
    if b2.button("🧹 入力をクリア", key="btn_clear"):
        st.session_state["clear_inputs"] = True
        st.rerun()


def run_judgement(state: GameState, settings: GameSettings) -> None:
    topic = st.session_state.get("topic", "")
    answers = [TeamAnswer(t.id, t.name, st.session_state.get(f"ans_{t.id}", "")) for t in state.teams]
    if not any(a.word.strip() for a in answers):
        st.warning("どのチームも単語が入力されていません。")
        return
    backend = load_backend_with_feedback(settings.model_key)
    if backend is None:
        return
    try:
        words = load_reference_words(reference_words_file())
        service = JudgeService(backend, words)
        with st.spinner("判定中…"):
            outcome = service.play_round(topic, answers, settings.round_settings())
    except EmbeddingError as e:
        st.error(f"埋め込みの計算に失敗しました。入力はそのまま残っています。\n\n{e}")
        return
    except (ValueError, FileNotFoundError) as e:
        st.warning(str(e))
        return
    except Exception as e:  # noqa: BLE001 - 画面を止めない
        st.error(f"判定中に予期しないエラーが起きました。入力はそのまま残っています。\n\n{e}")
        st.code(traceback.format_exc())
        return
    record = RoundRecord(
        round_no=state.next_round_no(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        topic=outcome.topic,
        rule=settings.rule,
        target=settings.target,
        calibration=settings.calibration,
        points=list(settings.points),
        model_key=settings.model_key,
        model_name=backend.name,
        results=outcome.results,
        reference_mean=outcome.reference.mean,
        reference_count=outcome.reference.count,
    )
    state.add_round(record)
    persist(state)
    start_reveal(record.round_no, len(record.results), settings.reveal_enabled)
    st.session_state["timer_end"] = None
    st.session_state["last_inputs"] = {"topic": topic, **{f"ans_{a.team_id}": a.word for a in answers}}
    st.session_state["screen"] = "result"
    st.rerun()


def go_to_input(state: GameState, clear: bool) -> None:
    """結果画面から入力画面へ。発表の途中なら全部表示したことにして累計を確定させる。"""
    if state.rounds:
        advance_reveal(state, None)
    if clear:
        st.session_state["clear_inputs"] = True
    else:
        st.session_state["restore_inputs"] = dict(st.session_state.get("last_inputs") or {})
    st.session_state["screen"] = "input"
    st.rerun()


def last_round_summary(state: GameState) -> None:
    """入力画面の下に出す、前回の結果の 1 行まとめ。"""
    last = state.rounds[-1]
    shown, n = current_reveal(state)
    line = f"前回の結果　第{last.round_no}ラウンド：お題「{last.topic}」"
    if shown >= n:
        winners = [r for r in last.results if r.rank == 1]
        if winners:
            line += "　🏆 1位：" + "・".join(f"{w.team_name}「{w.word}」" for w in winners)
        else:
            line += "　😱 有効な回答なし"
    else:
        line += "　（発表中）"
    c1, c2 = st.columns([4, 1])
    c1.markdown(f'<div class="sem-last">{line}</div>', unsafe_allow_html=True)
    if c2.button("結果をもう一度見る", key="btn_show_result"):
        st.session_state["screen"] = "result"
        st.rerun()


def history_expander(state: GameState) -> None:
    with st.expander("📜 これまでのラウンド", expanded=False):
        rows = []
        for rec in state.rounds:
            for r in display_order(rec.results):
                rows.append(
                    {
                        "R": rec.round_no, "お題": rec.topic, "チーム": r.team_name, "単語": r.word or "（未回答）",
                        "順位": "" if r.rank is None else r.rank, "スコア": fmt(r.score), "生の類似度": fmt(r.raw),
                        "反則": r.foul.label if r.foul else "", "得点": r.points,
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def input_screen(state: GameState, settings: GameSettings) -> None:
    input_area(state, settings)
    if not state.rounds:
        st.info("お題と各チームの単語を入力して「判定！」を押してください。")
        return
    st.divider()
    last_round_summary(state)
    history_expander(state)


def result_screen(state: GameState, settings: GameSettings) -> None:
    """判定直後の画面。このラウンドの結果だけを大きく出す。"""
    last = state.rounds[-1]
    rule = Rule(last.rule)
    calibrated = last.calibration != CalibrationMode.RAW.value
    shown, n = current_reveal(state)
    complete = shown >= n

    st.markdown(f"## 第{last.round_no}ラウンド：お題「{last.topic}」")
    sub = f"{rule.label}"
    if rule == Rule.TARGET:
        sub += f"（目標 {last.target:.2f}）"
    sub += f"　/　{CalibrationMode(last.calibration).label}　/　{last.model_name}"
    st.caption(sub)

    if not complete:
        c1, c2, c3 = st.columns([1, 1, 3])
        if c1.button("次を表示 ▶", key="btn_reveal_next", type="primary"):
            advance_reveal(state, 1)
            st.rerun()
        if c2.button("全部表示", key="btn_reveal_all"):
            advance_reveal(state, None)
            st.rerun()
        c3.caption(f"発表中… {shown} / {n}")

    auto = settings.reveal_auto and settings.reveal_enabled and not complete

    @st.fragment(run_every=settings.reveal_interval_sec if auto else None)
    def body() -> None:
        shown_now, total = current_reveal(state)
        if auto and shown_now < total:
            rv = st.session_state.get("reveal") or {}
            if time.time() - float(rv.get("last_advance", 0)) >= settings.reveal_interval_sec - 0.05:
                advance_reveal(state, 1)
                shown_now, total = current_reveal(state)
            if shown_now >= total:
                st.rerun()
        ordered = display_order(last.results)
        visible = ordered[len(ordered) - shown_now:] if shown_now < len(ordered) else ordered
        if shown_now >= total:
            st.markdown(winner_banner_html(last.results), unsafe_allow_html=True)
        for r in visible:
            st.markdown(result_row_html(r), unsafe_allow_html=True)

    body()

    st.write("")
    b1, b2, _ = st.columns([2, 2, 3])
    if b1.button("➡️ 次のラウンドへ", key="btn_next_round", type="primary"):
        go_to_input(state, clear=True)
    if b2.button("↩️ 入力に戻る（入力を残す）", key="btn_back_input"):
        go_to_input(state, clear=False)

    if not complete:
        return

    with st.expander("📊 グラフと詳しい数値", expanded=False):
        chart = score_chart(display_order(last.results), rule, last.target, calibrated)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.dataframe(details_table(last.results), hide_index=True, width="stretch")
        if last.reference_count:
            st.caption(
                f"基準語彙 {last.reference_count} 語とお題の平均類似度: {fmt(last.reference_mean)}"
                + ("（補正スコア 0 に相当）" if last.calibration == CalibrationMode.LINEAR.value else "")
            )

    with st.expander("🎙️ 司会判断（反則にする／取り消す）", expanded=False):
        for r in last.results:
            if r.status == Status.EMPTY:
                continue
            c1, c2 = st.columns([3, 1])
            c1.write(f"{r.team_name}「{r.word}」" + ("　🚫 " + r.foul.label if r.foul else ""))
            if r.status == Status.OK:
                if c2.button("反則にする", key=f"mf_on_{r.team_id}"):
                    state.set_manual_foul(last.round_no, r.team_id, True)
                    persist(state)
                    st.rerun()
            elif r.foul and r.foul.value == "manual":
                if c2.button("取り消す", key=f"mf_off_{r.team_id}"):
                    state.set_manual_foul(last.round_no, r.team_id, False)
                    persist(state)
                    st.rerun()


def leaderboard_area(state: GameState) -> None:
    shown, n = current_reveal(state)
    exclude = state.rounds[-1].round_no if (state.rounds and shown < n) else None
    st.markdown("### 🏅 累計得点" + ("（発表中）" if exclude else ""))
    st.markdown(leaderboard_html(state, exclude), unsafe_allow_html=True)
    st.caption(f"{len(state.rounds)} ラウンド終了")


def game_screen(state: GameState, settings: GameSettings) -> None:
    st.title("🧠 人力Semantle")
    screen = st.session_state.get("screen", "input")
    if screen == "result" and not state.rounds:
        screen = "input"
        st.session_state["screen"] = screen
    main, side = st.columns([3, 1])
    with main:
        if screen == "result":
            result_screen(state, settings)
        else:
            input_screen(state, settings)
    with side:
        leaderboard_area(state)


def main() -> None:
    init_state()
    state: GameState | None = st.session_state["game"]
    settings = sidebar(state)
    inject_css(float(st.session_state.get("font_scale", 1.15)))
    if state is None:
        setup_screen(settings)
    else:
        game_screen(state, settings)


main()
