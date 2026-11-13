"""人力Semantle — チーム対抗・埋め込み類似度ゲーム

起動:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from core import BASELINE_WORDS, adjusted_scores, judge

st.set_page_config(page_title="人力Semantle", page_icon="🧠", layout="wide")

# ---------------- サイドバー（司会用の設定） ----------------
st.sidebar.header("⚙️ 設定")
model_name = st.sidebar.selectbox(
    "埋め込みモデル",
    ["intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"],
    help="2024年のM2クイズと同じME5。PCが非力ならsmall。",
)
mode = st.sidebar.radio("ルール", ["① 最も似ている単語", "② 目標値ぴったり"])
target = None
if mode.startswith("②"):
    target = st.sidebar.slider("目標値", 0.0, 1.0, 0.5, 0.05)
use_adjusted = st.sidebar.toggle(
    "補正スコアを使う（推奨）",
    value=True,
    help="E5は異方性のため無関係な単語でも0.7〜0.8程度になる。"
    "無関係な単語の平均を0、お題そのものを1に引き直したスコアで判定する。",
)
n_teams = st.sidebar.number_input("チーム数", 2, 8, 4)
team_names = [st.sidebar.text_input(f"チーム{i+1}の名前", f"チーム{i+1}") for i in range(n_teams)]
points_rule = [3, 2, 1]  # 1位・2位・3位の得点

if st.sidebar.button("🔄 得点をリセット"):
    st.session_state.clear()


@st.cache_resource(show_spinner="モデルを読み込み中…（初回のみ）")
def load_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode(texts):
    model = load_model(model_name)
    # E5は "query: " プレフィックスを付けて使う
    return model.encode([f"query: {t}" for t in texts], normalize_embeddings=True)


@st.cache_data(show_spinner=False)
def baseline_vectors(name):
    return encode(BASELINE_WORDS)


st.session_state.setdefault("total", {})
st.session_state.setdefault("history", [])

# ---------------- メイン画面 ----------------
st.title("🧠 人力Semantle")
rule_text = "お題に**一番近い**単語を言ったチームの勝ち" if target is None else f"類似度が **{target:.2f}** に一番近いチームの勝ち"
st.markdown(f"### ルール：{rule_text}")

theme = st.text_input("📝 お題", placeholder="例：拡散モデル")
cols = st.columns(n_teams)
answers = [c.text_input(name, key=f"ans_{i}") for i, (c, name) in enumerate(zip(cols, team_names))]

if st.button("🎯 判定！", type="primary", width="stretch", disabled=not theme):
    filled = [(n, a.strip()) for n, a in zip(team_names, answers) if a.strip()]
    if not filled:
        st.warning("回答が入力されていません")
        st.stop()
    names, words = zip(*filled)
    vecs = encode([theme, *words])
    raw = vecs[1:] @ vecs[0]
    base = baseline_vectors(model_name) @ vecs[0]
    scores = adjusted_scores(raw, base) if use_adjusted else raw
    result = judge(names, words, raw, scores, theme, target)

    # 得点加算（反則以外の上位3チーム）
    valid = result[result["反則"] == ""]
    for rank, team in enumerate(valid["チーム"].tolist()[: len(points_rule)]):
        st.session_state["total"][team] = st.session_state["total"].get(team, 0) + points_rule[rank]
    st.session_state["history"].append({"お題": theme, **{r.チーム: f"{r.単語}（{r.スコア:.3f}）" for r in result.itertuples()}})

    st.balloons()
    st.markdown(f"## 🏆 勝者：{valid.iloc[0]['チーム']}「{valid.iloc[0]['単語']}」" if len(valid) else "## 全チーム反則…")
    show = result[["順位", "チーム", "単語", "スコア", "生の類似度", "反則"]]
    st.dataframe(show.style.format({"スコア": "{:.3f}", "生の類似度": "{:.3f}"}), hide_index=True, width="stretch")
    st.bar_chart(result.set_index("チーム")["スコア"])

# ---------------- 累計 ----------------
if st.session_state["total"]:
    st.divider()
    st.subheader("📊 累計得点")
    tot = pd.Series(st.session_state["total"]).sort_values(ascending=False)
    st.bar_chart(tot)
    with st.expander("これまでの結果"):
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
