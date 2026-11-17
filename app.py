"""人力Semantle — 画面（Streamlit）。ロジックは semantle/ パッケージにある。

起動:  streamlit run app.py
画面:
  http://localhost:8501/               司会者用（入力・操作・すべての数値）
  http://localhost:8501/?view=screen   投影用（スライドだけ。司会者用画面の操作に追従する）
環境変数:
  SEMANTLE_FAKE_BACKEND=1   モデルを読まずテスト用のモック埋め込みを使う（AppTest 用）
  SEMANTLE_FAKE_SIMS='{"単語": 類似度, ...}'  モックが返すお題との類似度（任意）
  SEMANTLE_SAVE_DIR / SEMANTLE_CACHE_DIR      保存先の変更
  SEMANTLE_ONLINE=1          Hugging Face Hub への接続を許可する（既定はオフライン）
"""

from __future__ import annotations

import html
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
from semantle.live import SLIDES, LiveSession, get_live
from semantle.normalize import clean_for_embedding
from semantle.paths import cache_db_file, reference_words_file, results_csv_file, state_file
from semantle.reference import load_reference_words
from semantle.service import JudgeService
from semantle.slides import PROJECTOR_CSS, render_slide, score_name, winner_text
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

PROJECTOR = st.query_params.get("view") == "screen"

st.set_page_config(
    page_title="人力Semantle（投影用）" if PROJECTOR else "人力Semantle（司会者用）", page_icon="🧠", layout="wide"
)

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
.sem-row .tag {{ font-size:1.0rem; font-weight:700; padding:0.1rem 0.6rem; border-radius:1rem;
                background:rgba(128,128,128,0.2); white-space:nowrap; }}
.sem-row.pending {{ border-style:dashed; }}
.sem-now {{ font-size:1.5rem; font-weight:800; }}
.sem-now small {{ font-size:1.05rem; font-weight:600; opacity:0.7; }}
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
        "show_reference_slide": "set_ref_slide",
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


def init_state() -> LiveSession:
    """共有状態を用意する。保存ファイルはプロセスで 1 回だけ読み、ウィンドウごとに設定欄を合わせる。"""
    apply_requested_resets()
    live = get_live(state_file())
    with live.lock:
        if not live.loaded:
            live.loaded = True
            try:
                live.game = load_state(state_file())
            except StateLoadError as e:
                live.load_error = str(e)
    if not st.session_state.get("_synced"):
        st.session_state["_synced"] = True
        sync_settings_to_widgets(live.game.settings if live.game is not None else GameSettings())
        st.session_state["proj_scale"] = float(live.pres.font_scale)
    return live


def archive_state_file() -> None:
    path = state_file()
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.rename(path.with_name(f"game_state_{stamp}.json"))


# ====================================================================== 部品


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def result_row_html(r: AnswerResult, pending: bool = False) -> str:
    """司会者用の 1 チーム分の結果行。``pending`` は投影側でまだ発表していない行。"""
    tag = '<span class="tag">🙈 未発表</span>' if pending else ""
    extra = " pending" if pending else ""
    team = f'<span class="team">{esc(r.team_name)}</span>'
    if r.status == Status.EMPTY:
        return f'<div class="sem-row empty{extra}"><span class="rank">—</span>{team}<span class="word">（未回答）</span>{tag}</div>'
    if r.status == Status.FOUL:
        label = r.foul.label if r.foul else "反則"
        return (
            f'<div class="sem-row foul{extra}"><span class="rank">🚫</span>{team}'
            f'<span class="word">{esc(r.word)}</span>{tag}<span class="score">反則：{esc(label)}</span><span class="pts">0点</span></div>'
        )
    cls = "sem-row first" if r.rank == 1 else "sem-row"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank or 0, f"{r.rank}位")
    return (
        f'<div class="{cls}{extra}"><span class="rank">{medal}</span>{team}'
        f'<span class="word">{esc(r.word)}</span>{tag}<span class="score">{fmt(r.score)}</span><span class="pts">+{r.points}点</span></div>'
    )


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


def leaderboard_html(state: GameState) -> str:
    rows = state.leaderboard()
    parts = ['<div class="sem-board">']
    for rank, team, total in rows:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
        parts.append(f'<div class="row"><span class="rank">{medal}</span><span class="name">{esc(team.name)}</span><span class="pts">{total}点</span></div>')
    parts.append("</div>")
    return "".join(parts)


# ====================================================================== サイドバー


def sidebar(live: LiveSession) -> GameSettings:
    state = live.game
    with st.sidebar:
        st.header("⚙️ 設定")
        st.slider("この画面の文字の大きさ", 0.7, 1.6, 1.0, 0.05, key="font_scale")
        st.slider("投影画面の文字の大きさ", 0.7, 1.4, step=0.05, key="proj_scale")
        live.pres.font_scale = float(st.session_state["proj_scale"])

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
        st.toggle("結果のあとに「基準単語」スライドを挟む", key="set_ref_slide")
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
            show_reference_slide=bool(st.session_state["set_ref_slide"]),
        )

        if state is not None:
            if settings != state.settings:
                state.settings = settings
                persist(state)
            host_tools(live, state)
    return settings


def host_tools(live: LiveSession, state: GameState) -> None:
    st.divider()
    st.header("🎙️ 司会ツール")

    with st.expander("👥 チーム名の変更", expanded=False):
        for team in state.teams:
            new_name = st.text_input(f"{team.id}", value=team.name, key=f"rename_{team.id}")
            if new_name.strip() and new_name.strip() != team.name:
                with live.lock:
                    state.rename_team(team.id, new_name)
                persist(state)
                st.rerun()
        if len(state.teams) < MAX_TEAMS and st.button("チームを追加", key="btn_add_team"):
            with live.lock:
                state.add_team(f"チーム{len(state.teams) + 1}")
            persist(state)
            st.rerun()

    with st.expander("↩️ 直前のラウンドを取り消す（Undo）", expanded=False):
        if state.rounds:
            last = state.rounds[-1]
            st.caption(f"第{last.round_no}ラウンド「{last.topic}」を取り消します")
            if st.checkbox("取り消してよい", key="undo_confirm"):
                if st.button("取り消す", key="btn_undo", type="primary"):
                    with live.lock:
                        state.undo_last_round()
                        live.round_removed()
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
            with live.lock:
                state.add_adjustment(team_id, int(delta), note)
            persist(state)
            st.rerun()
        if state.adjustments:
            st.caption("修正の履歴")
            for adj in state.adjustments:
                c1, c2 = st.columns([4, 1])
                c1.write(f"{state.team_by_id(adj.team_id).name}  {adj.delta:+d}点  {adj.note}")
                if c2.button("削除", key=f"del_adj_{adj.id}"):
                    with live.lock:
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
                with live.lock:
                    live.game = None
                    live.reset_presentation()
                st.session_state["screen"] = "input"
                request_reset("newgame_confirm")
                st.rerun()


# ====================================================================== 投影の操作


def open_projector_link() -> None:
    st.markdown(
        '<a href="?view=screen" target="_blank">📺 投影用の画面を開く ↗</a>'
        "<br><small>開いたウィンドウをプロジェクター側へ移して F11</small>",
        unsafe_allow_html=True,
    )


def control_panel(live: LiveSession) -> None:
    """投影用画面に何を映すかを決めるリモコン。"""
    pres = live.pres
    shown, n = live.reveal_progress()
    revealing = pres.slide == "result" and shown < n
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        note = f"　<small>発表済み {shown} / {n}</small>" if revealing else ""
        c1.markdown(f'<div class="sem-now">📺 投影中：{SLIDES[pres.slide]}{note}</div>', unsafe_allow_html=True)
        with c2:
            open_projector_link()

        prev_target, action = live.prev_target(), live.next_action()
        b1, b2, b3 = st.columns([1, 2, 1])
        if b1.button("◀ 前へ", key="btn_slide_prev", disabled=prev_target is None, width="stretch"):
            live.go_prev()
            st.rerun()
        label = action[1] if action else "次へ ▶"
        if b2.button(label, key="btn_slide_next", type="primary", disabled=action is None, width="stretch"):
            live.go_next()
            st.rerun()
        if b3.button("全部発表", key="btn_reveal_all", disabled=not revealing, width="stretch"):
            live.advance_reveal(None)
            st.rerun()

        available = live.available()
        for col, (slide, name) in zip(st.columns(len(SLIDES)), SLIDES.items()):
            current = slide == pres.slide
            if col.button(("● " if current else "") + name, key=f"btn_slide_{slide}",
                          disabled=current or slide not in available, width="stretch"):
                live.show(slide)
                st.rerun()

    # 自動発表の間は、投影側の進み具合に合わせてこの画面も更新する
    rendered = live.signature()

    @st.fragment(run_every=1.0 if live.auto_reveal_active() else None)
    def follow() -> None:
        live.tick()
        if live.signature() != rendered:
            st.rerun()

    follow()


# ====================================================================== 画面


def setup_screen(live: LiveSession, settings: GameSettings) -> None:
    st.title("🧠 人力Semantle")
    open_projector_link()
    if live.load_error:
        st.error(f"前回の保存ファイルを読めませんでした（壊れている可能性があります）。\n\n{live.load_error}")
        st.caption("ファイルはそのまま残します。新しいゲームを始めると日時付きの名前で退避されます。")
    st.subheader("チームを決めて始めましょう")
    n = st.number_input("チーム数", 1, MAX_TEAMS, 4, key="setup_n_teams")
    names: list[str] = []
    cols = st.columns(int(n))
    for i in range(int(n)):
        names.append(cols[i].text_input(f"チーム{i + 1}", value=f"チーム{i + 1}", key=f"setup_name_{i}"))
    if st.button("🎮 ゲーム開始", type="primary", key="btn_start"):
        if live.load_error:
            archive_state_file()
            live.load_error = None
        state = GameState.new_game(names, settings)
        with live.lock:
            live.game = state
            live.reset_presentation()
        persist(state)
        st.rerun()


def timer_area(live: LiveSession, timer_sec: int) -> None:
    if timer_sec <= 0:
        return
    pres = live.pres
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("⏱ スタート", key="btn_timer_start"):
        pres.timer_total = timer_sec
        pres.timer_end = time.time() + timer_sec
    if c2.button("⏹ 停止", key="btn_timer_stop"):
        pres.timer_end = None

    @st.fragment(run_every=1.0 if pres.timer_end else None)
    def show() -> None:
        end_t = live.pres.timer_end
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


def input_area(live: LiveSession, state: GameState, settings: GameSettings) -> None:
    if st.session_state.pop("clear_inputs", False):
        st.session_state["topic"] = ""
        for team in state.teams:
            st.session_state[f"ans_{team.id}"] = ""
    # 結果画面の間は入力欄が無いので値が消える。判定時に控えたものを戻す
    for key, value in (st.session_state.pop("restore_inputs", None) or {}).items():
        st.session_state[key] = value

    st.markdown(f"### 第{state.next_round_no()}問")
    t1, t2 = st.columns([2, 1], vertical_alignment="bottom")
    t1.text_input("📝 お題", key="topic", placeholder="例：拡散モデル")
    topic = st.session_state.get("topic", "")
    topic_clean = clean_for_embedding(topic)
    if t2.button("📢 お題を投影", key="btn_publish_topic", disabled=not topic_clean, width="stretch"):
        live.publish_topic(topic_clean)
        st.rerun()
    if live.pres.topic:
        st.caption(f"投影中のお題：「{live.pres.topic}」")
    else:
        st.caption("お題はまだ投影していません（「お題を投影」を押すとスライドに出ます）")

    st.markdown("**各チームの単語**（この画面だけ。判定するまで投影されません）")
    cols = st.columns(len(state.teams))
    for col, team in zip(cols, state.teams):
        col.text_input(team.name, key=f"ans_{team.id}", placeholder="単語")

    timer_area(live, settings.timer_sec)

    b1, b2 = st.columns([3, 1])
    if b1.button("🎯 判定！", type="primary", key="btn_judge", disabled=not topic.strip()):
        run_judgement(live, state, settings)
    if b2.button("🧹 入力をクリア", key="btn_clear"):
        st.session_state["clear_inputs"] = True
        st.rerun()


def run_judgement(live: LiveSession, state: GameState, settings: GameSettings) -> None:
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
        reference_top=outcome.reference_top,
    )
    with live.lock:
        state.add_round(record)
        live.start_reveal(settings.reveal_enabled)  # 投影は結果スライドへ
    persist(state)
    st.session_state["last_inputs"] = {"topic": topic, **{f"ans_{a.team_id}": a.word for a in answers}}
    st.session_state["screen"] = "result"
    st.rerun()


def go_to_input(live: LiveSession, clear: bool) -> None:
    """結果画面から入力画面へ。発表の途中なら全部表示したことにして累計を確定させる。"""
    with live.lock:
        if live.last_round is not None:
            live.advance_reveal(None)
        if clear:
            live.show("standings")  # 次のお題を出すまでは累計得点を映しておく
    if clear:
        st.session_state["clear_inputs"] = True
    else:
        st.session_state["restore_inputs"] = dict(st.session_state.get("last_inputs") or {})
    st.session_state["screen"] = "input"
    st.rerun()


def last_round_summary(state: GameState) -> None:
    """入力画面の下に出す、前回の結果の 1 行まとめ。"""
    last = state.rounds[-1]
    line = f"前回の結果　第{last.round_no}ラウンド：お題「{last.topic}」　{winner_text(last.results)}"
    c1, c2 = st.columns([4, 1])
    c1.markdown(f'<div class="sem-last">{esc(line)}</div>', unsafe_allow_html=True)
    if c2.button("結果をもう一度見る", key="btn_show_result"):
        st.session_state["screen"] = "result"
        st.rerun()


def history_expander(state: GameState) -> None:
    with st.expander("📜 これまでのラウンド", expanded=False):
        rows = []
        for rec in state.rounds:
            best = rec.reference_top[0] if rec.reference_top else None
            for r in display_order(rec.results):
                rows.append(
                    {
                        "R": rec.round_no, "お題": rec.topic, "チーム": r.team_name, "単語": r.word or "（未回答）",
                        "順位": "" if r.rank is None else r.rank, "スコア": fmt(r.score), "生の類似度": fmt(r.raw),
                        "反則": r.foul.label if r.foul else "", "得点": r.points,
                        "基準単語の最高": f"{best['word']} {fmt(best['score'])}" if best else "",
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def input_screen(live: LiveSession, state: GameState, settings: GameSettings) -> None:
    input_area(live, state, settings)
    if not state.rounds:
        st.info("お題を入力して「お題を投影」→ 各チームの単語を入力して「判定！」を押してください。")
        return
    st.divider()
    last_round_summary(state)
    history_expander(state)


def reference_panel(last: RoundRecord) -> None:
    """基準単語（ふつうの単語 100 語）がこのお題でどこまで高いスコアになったか。"""
    st.markdown("#### 📚 基準単語との比較")
    if not last.reference_top:
        st.caption("このラウンドには基準単語の記録がありません（以前のバージョンで判定したラウンドです）。")
        return
    name = score_name(last.calibration)
    best = last.reference_top[0]
    valid = [r for r in last.results if r.is_valid and r.score is not None]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"基準単語の最高{name}", fmt(best["score"]), f"「{best['word']}」", delta_color="off")
    if valid:
        top_team = max(valid, key=lambda r: r.score or 0.0)
        m2.metric(f"チームの最高{name}", fmt(top_team.score), f"{top_team.team_name}「{top_team.word}」", delta_color="off")
        beat = sum(1 for r in valid if (r.score or 0.0) > float(best["score"]))
        m3.metric("基準単語の最高を超えたチーム", f"{beat} / {len(valid)}")
    rows = [
        {"順位": i, "基準単語": w["word"], name: fmt(w["score"]), "生の類似度": fmt(w["raw"])}
        for i, w in enumerate(last.reference_top, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"基準語彙 {last.reference_count} 語とお題の平均類似度: {fmt(last.reference_mean)}"
        + ("（補正スコア 0 に相当）" if last.calibration == CalibrationMode.LINEAR.value else "")
    )


def result_screen(live: LiveSession, state: GameState) -> None:
    """判定後の司会者用画面。発表前でも全チームの結果と数値が見える。"""
    last = state.rounds[-1]
    rule = Rule(last.rule)
    calibrated = last.calibration != CalibrationMode.RAW.value
    shown, n = live.reveal_progress()

    st.markdown(f"## 第{last.round_no}ラウンド：お題「{last.topic}」")
    sub = f"{rule.label}"
    if rule == Rule.TARGET:
        sub += f"（目標 {last.target:.2f}）"
    sub += f"　/　{CalibrationMode(last.calibration).label}　/　{last.model_name}"
    st.caption(sub)

    st.markdown(f'<div class="sem-banner">{esc(winner_text(last.results))}</div>', unsafe_allow_html=True)
    if shown < n:
        st.caption(f"投影側は発表中です（{shown} / {n}）。🙈 の行はまだ映っていません。上の「次を発表 ▶」で進めます。")
    ordered = display_order(last.results)
    hidden = len(ordered) - shown if shown < n else 0
    for i, r in enumerate(ordered):
        st.markdown(result_row_html(r, pending=i < hidden), unsafe_allow_html=True)

    st.write("")
    b1, b2, _ = st.columns([2, 2, 3])
    if b1.button("➡️ 次のラウンドへ", key="btn_next_round", type="primary"):
        go_to_input(live, clear=True)
    if b2.button("↩️ 入力に戻る（入力を残す）", key="btn_back_input"):
        go_to_input(live, clear=False)

    reference_panel(last)

    with st.expander("📊 グラフと詳しい数値", expanded=False):
        chart = score_chart(ordered, rule, last.target, calibrated)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.dataframe(details_table(last.results), hide_index=True, width="stretch")

    with st.expander("🎙️ 司会判断（反則にする／取り消す）", expanded=False):
        for r in last.results:
            if r.status == Status.EMPTY:
                continue
            c1, c2 = st.columns([3, 1])
            c1.write(f"{r.team_name}「{r.word}」" + ("　🚫 " + r.foul.label if r.foul else ""))
            if r.status == Status.OK:
                if c2.button("反則にする", key=f"mf_on_{r.team_id}"):
                    with live.lock:
                        state.set_manual_foul(last.round_no, r.team_id, True)
                    persist(state)
                    st.rerun()
            elif r.foul and r.foul.value == "manual":
                if c2.button("取り消す", key=f"mf_off_{r.team_id}"):
                    with live.lock:
                        state.set_manual_foul(last.round_no, r.team_id, False)
                    persist(state)
                    st.rerun()


def leaderboard_area(live: LiveSession, state: GameState) -> None:
    st.markdown("### 🏅 累計得点")
    st.markdown(leaderboard_html(state), unsafe_allow_html=True)
    st.caption(f"{len(state.rounds)} ラウンド終了")
    if live.pending_round_no() is not None:
        st.caption("（投影側は発表が終わるまで、このラウンドの得点を足しません）")


def game_screen(live: LiveSession, state: GameState, settings: GameSettings) -> None:
    st.markdown("### 🎙️ 人力Semantle　司会者用")
    control_panel(live)
    screen = st.session_state.get("screen", "input")
    if screen == "result" and not state.rounds:
        screen = "input"
        st.session_state["screen"] = screen
    main, side = st.columns([3, 1])
    with main:
        if screen == "result":
            result_screen(live, state)
        else:
            input_screen(live, state, settings)
    with side:
        leaderboard_area(live, state)


def projector_screen(live: LiveSession) -> None:
    """投影用。操作するものは何も置かず、共有状態を 1 秒ごとに読んでスライドを描く。"""
    st.markdown(PROJECTOR_CSS, unsafe_allow_html=True)

    @st.fragment(run_every=1.0)
    def slide() -> None:
        live.tick()
        with live.lock:
            content = render_slide(live)
        st.markdown(content, unsafe_allow_html=True)

    slide()


def main() -> None:
    live = init_state()
    if PROJECTOR:
        projector_screen(live)
        return
    settings = sidebar(live)
    inject_css(float(st.session_state.get("font_scale", 1.0)))
    if live.game is None:
        setup_screen(live, settings)
    else:
        game_screen(live, live.game, settings)


main()
