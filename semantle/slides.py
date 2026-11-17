"""投影用スライドの HTML。1 枚のスライドを 1 つの HTML 文字列として返す（Streamlit には依存しない）。

* 大きさはすべて画面の高さ・幅に対する比率（em）で決めるので、どの解像度でも同じ見た目になる
* 司会者だけが知っている情報（未発表の単語、入力中の単語）は出さない
* st.markdown に渡すので、HTML は改行・行頭の空白を含めない
"""

from __future__ import annotations

import html
import time

from semantle.calibration import CalibrationMode
from semantle.judge import AnswerResult, Rule, Status, display_order
from semantle.live import LiveSession
from semantle.state import GameSettings, GameState, RoundRecord

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# 投影用画面では Streamlit のヘッダーやサイドバーを消し、スライドを画面いっぱいに敷く
PROJECTOR_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"], footer { display:none !important; }
.stApp { background:#f6f3ea; }
[data-testid="stElementContainer"] { opacity:1 !important; transition:none !important; }
.sl { --ink:#1c2333; --mut:#667085; --acc:#b8860b; --navy:#24406e; --red:#c0392b; --line:rgba(28,35,51,.18);
  position:fixed; inset:0; z-index:999999; box-sizing:border-box; overflow:hidden;
  display:flex; flex-direction:column; padding:2.2em 3.4em 1.4em 3.4em;
  background:#f6f3ea; color:var(--ink); line-height:1.25;
  font-family:"Yu Gothic UI","Yu Gothic","Meiryo","Hiragino Sans",sans-serif;
  font-size:calc(min(2.2vh, 1.24vw) * var(--scale, 1)); }
.sl * { box-sizing:border-box; }
.sl .head { display:flex; align-items:center; gap:1em; padding-bottom:.7em; border-bottom:.18em solid var(--navy); }
.sl .chip { background:var(--navy); color:#fff; font-size:1.5em; font-weight:800; padding:.15em .8em; border-radius:.5em; white-space:nowrap; }
.sl .htitle { font-size:2.6em; font-weight:900; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl .hsub { font-size:1.2em; color:var(--mut); white-space:nowrap; }
.sl .body { flex:1; min-height:0; display:flex; flex-direction:column; justify-content:center; gap:.55em; padding:.9em 0; }
.sl .foot { display:flex; justify-content:space-between; font-size:1em; color:var(--mut); }
.sl .center { align-items:center; text-align:center; }
.sl .giant { font-size:7em; font-weight:900; line-height:1.15; overflow-wrap:anywhere; }
.sl .big { font-size:4.4em; font-weight:900; }
.sl .lead { font-size:2em; color:var(--mut); font-weight:700; }
.sl .note { font-size:1.5em; color:var(--mut); }
.sl .chips { display:flex; flex-wrap:wrap; justify-content:center; gap:.7em; margin-top:1em; }
.sl .tchip { font-size:1.7em; font-weight:700; padding:.2em .9em; border-radius:2em; border:.1em solid var(--navy); background:#fff; }
.sl .timer { font-size:4.2em; font-weight:900; font-variant-numeric:tabular-nums; margin-top:.2em; }
.sl .timer.over { color:var(--red); }
.sl .banner { font-size:2.3em; font-weight:900; text-align:center; padding:.25em .6em; border-radius:.4em;
  background:rgba(184,134,11,.2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl .row { flex:1 1 0; min-height:0; max-height:4.6em; display:flex; align-items:center; gap:1.1em; padding:0 1.3em;
  border-radius:.7em; border:.12em solid var(--line); background:#fff; }
.sl .row .rank { font-size:2em; font-weight:800; width:2.2em; text-align:center; white-space:nowrap; }
.sl .row .team { font-size:1.5em; font-weight:700; width:9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl .row .word { font-size:2.1em; font-weight:800; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl .row .score { font-size:1.8em; font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }
.sl .row .pts { font-size:1.6em; font-weight:800; width:3.6em; text-align:right; color:var(--acc); white-space:nowrap; }
.sl .row.first { border-color:var(--acc); background:#fbf1d3; }
.sl .row.foul { border-color:var(--red); background:#fbe9e6; }
.sl .row.foul .score { color:var(--red); font-size:1.4em; }
.sl .row.empty { opacity:.55; }
.sl .row.hid { background:transparent; border-style:dashed; color:var(--mut); }
.sl .row.you { border-color:var(--navy); background:#e8eef8; }
.sl .stat { display:flex; align-items:baseline; justify-content:center; gap:.8em; margin-bottom:.3em; }
.sl .stat .label { font-size:1.6em; color:var(--mut); font-weight:700; }
.sl .stat .num { font-size:4.6em; font-weight:900; font-variant-numeric:tabular-nums; color:var(--navy); }
.sl .stat .who { font-size:2.6em; font-weight:900; }
.sl ul.rules { margin:0; padding-left:1.2em; font-size:2.1em; font-weight:700; display:flex; flex-direction:column; gap:.55em; }
.sl ul.rules li small { font-size:.75em; color:var(--mut); font-weight:600; }
</style>
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def score_name(calibration: str) -> str:
    return "類似度" if calibration == CalibrationMode.RAW.value else "スコア"


def rule_line(rule: str, target: float) -> str:
    if rule == Rule.TARGET.value:
        return f"スコアが目標値 {target:.2f} に一番近いチームの勝ち"
    return "お題に意味が一番近い単語を出したチームの勝ち"


def winner_text(results: list[AnswerResult]) -> str:
    winners = [r for r in results if r.rank == 1]
    if not winners:
        return "😱 有効な回答なし"
    return "🏆 1位：" + "・".join(f"{w.team_name}「{w.word}」" for w in winners)


def _wrap(head: str, body: str, foot_right: str, scale: float, body_class: str = "") -> str:
    cls = f"body {body_class}".strip()
    return (
        f'<div class="sl" style="--scale:{scale:.2f}">{head}<div class="{cls}">{body}</div>'
        f'<div class="foot"><span>🧠 人力Semantle</span><span>{foot_right}</span></div></div>'
    )


def _head(chip: str, title: str, sub: str = "") -> str:
    chip_html = f'<span class="chip">{esc(chip)}</span>' if chip else ""
    sub_html = f'<span class="hsub">{esc(sub)}</span>' if sub else ""
    return f'<div class="head">{chip_html}<span class="htitle">{esc(title)}</span>{sub_html}</div>'


# ---------------------------------------------------------------- 各スライド


def title_slide(game: GameState | None, scale: float) -> str:
    body = '<div class="giant">🧠 人力Semantle</div><div class="lead">意味の近さで勝負する、チーム対抗ワードバトル</div>'
    if game is None:
        body += '<div class="note">準備中です。しばらくお待ちください</div>'
    else:
        chips = "".join(f'<span class="tchip">{esc(t.name)}</span>' for t in game.teams)
        body += f'<div class="chips">{chips}</div>'
    return _wrap("", body, "", scale, "center")


def rules_slide(settings: GameSettings, scale: float) -> str:
    pts = " / ".join(f"{i + 1}位 {p}点" for i, p in enumerate(settings.points) if p > 0) or "得点なし"
    foul = "お題と同じ単語・お題を含む単語は反則で 0 点"
    if settings.strict_contained:
        foul = "お題と同じ単語・お題を含む単語・お題に含まれる単語は反則で 0 点"
    items = [
        "司会が「お題」を出します",
        "各チームで相談して、単語を 1 つ決めます",
        f"{rule_line(settings.rule, settings.target)}<br><small>近さは AI（埋め込みモデル）が数値で判定します</small>",
        f"得点：{esc(pts)}",
        esc(foul),
    ]
    body = '<ul class="rules">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    return _wrap(_head("", "ルール説明"), body, "", scale)


def _timer_html(live: LiveSession, settings: GameSettings, now: float) -> str:
    end = live.pres.timer_end
    if end:
        remain = int(round(end - now))
        if remain <= 0:
            return '<div class="timer over">⏰ 時間です！</div>'
        return f'<div class="timer">⏱ {remain // 60:02d}:{remain % 60:02d}</div>'
    if settings.timer_sec > 0:
        return f'<div class="note">制限時間 {settings.timer_sec // 60:02d}:{settings.timer_sec % 60:02d}</div>'
    return ""


def topic_slide(live: LiveSession, game: GameState, scale: float, now: float) -> str:
    s = game.settings
    body = (
        f'<div class="lead">お題</div><div class="giant">{esc(live.pres.topic)}</div>'
        f'<div class="note">{esc(rule_line(s.rule, s.target))}</div>{_timer_html(live, s, now)}'
    )
    no = game.next_round_no()
    return _wrap(_head(f"第{no}問", "お題発表"), body, f"第{no}問", scale, "center")


def _result_row(r: AnswerResult) -> str:
    team = f'<span class="team">{esc(r.team_name)}</span>'
    if r.status == Status.EMPTY:
        return f'<div class="row empty"><span class="rank">—</span>{team}<span class="word">（未回答）</span></div>'
    if r.status == Status.FOUL:
        label = r.foul.label if r.foul else "反則"
        return (
            f'<div class="row foul"><span class="rank">🚫</span>{team}<span class="word">{esc(r.word)}</span>'
            f'<span class="score">反則：{esc(label)}</span><span class="pts">0点</span></div>'
        )
    cls = "row first" if r.rank == 1 else "row"
    medal = MEDALS.get(r.rank or 0, f"{r.rank}位")
    return (
        f'<div class="{cls}"><span class="rank">{medal}</span>{team}<span class="word">{esc(r.word)}</span>'
        f'<span class="score">{fmt(r.score)}</span><span class="pts">+{r.points}点</span></div>'
    )


_HIDDEN_ROW = '<div class="row hid"><span class="rank">？</span><span class="word">？？？</span></div>'


def result_slide(live: LiveSession, last: RoundRecord, scale: float) -> str:
    shown, n = live.reveal_progress()
    ordered = display_order(last.results)
    hidden = len(ordered) - shown if shown < n else 0
    rows = [_HIDDEN_ROW] * hidden + [_result_row(r) for r in ordered[hidden:]]
    if shown >= n:
        banner = f'<div class="banner">{esc(winner_text(last.results))}</div>'
    else:
        banner = f'<div class="banner">結果発表　{shown} / {n}</div>'
    sub = f"目標 {last.target:.2f}" if last.rule == Rule.TARGET.value else ""
    head = _head(f"第{last.round_no}問", f"お題「{last.topic}」", sub)
    return _wrap(head, banner + "".join(rows), f"第{last.round_no}問　結果発表", scale)


REFERENCE_SLIDE_ROWS = 5


def reference_slide(live: LiveSession, last: RoundRecord, scale: float) -> str:
    top = last.reference_top
    name = score_name(last.calibration)
    best = top[0]
    stat = (
        f'<div class="stat"><span class="label">基準単語の最高{name}</span>'
        f'<span class="num">{fmt(best["score"])}</span><span class="who">「{esc(best["word"])}」</span></div>'
    )
    rows = "".join(
        f'<div class="row"><span class="rank">{i}</span><span class="word">{esc(w["word"])}</span>'
        f'<span class="score">{fmt(w["score"])}</span></div>'
        for i, w in enumerate(top[:REFERENCE_SLIDE_ROWS], start=1)
    )
    # 1位の回答は発表が終わるまで出さない
    winners = [r for r in last.results if r.rank == 1]
    if live.reveal_complete() and winners:
        w = winners[0]
        names = "・".join(r.team_name for r in winners)
        rows += (
            f'<div class="row you"><span class="rank">🏆</span><span class="team">{esc(names)}</span>'
            f'<span class="word">{esc(w.word)}</span><span class="score">{fmt(w.score)}</span></div>'
        )
    sub = f"ふつうの単語 {last.reference_count} 語とお題の近さ"
    head = _head(f"第{last.round_no}問", f"お題「{last.topic}」に近い基準単語", sub)
    return _wrap(head, stat + rows, f"第{last.round_no}問　基準単語", scale)


def standings_slide(live: LiveSession, game: GameState, scale: float) -> str:
    pending = live.pending_round_no()
    rows = []
    for rank, team, total in game.leaderboard(exclude_round_no=pending):
        cls = "row first" if rank == 1 and total > 0 else "row"
        rows.append(
            f'<div class="{cls}"><span class="rank">{MEDALS.get(rank, f"{rank}位")}</span>'
            f'<span class="word">{esc(team.name)}</span><span class="score">{total}点</span></div>'
        )
    done = len(game.rounds) - (1 if pending is not None else 0)
    return _wrap(_head("", "🏅 累計得点", f"{done} 問終了"), "".join(rows), "累計得点", scale)


def render_slide(live: LiveSession, now: float | None = None) -> str:
    """いま映すべきスライド。映せない状態（ラウンドが無いのに結果、など）は近いものに読み替える。"""
    now = time.time() if now is None else now
    game, pres = live.game, live.pres
    scale = pres.font_scale
    slide = pres.slide if pres.slide in live.available() else ("standings" if game is not None else "title")
    last = live.last_round
    if game is None or slide == "title":
        return title_slide(game, scale)
    if slide == "rules":
        return rules_slide(game.settings, scale)
    if slide == "topic":
        return topic_slide(live, game, scale, now)
    if slide == "result" and last is not None:
        return result_slide(live, last, scale)
    if slide == "reference" and last is not None:
        return reference_slide(live, last, scale)
    return standings_slide(live, game, scale)
