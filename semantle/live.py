"""司会者用画面と投影用画面で共有する「いまの状態」。

Streamlit はブラウザのウィンドウごとにセッションが分かれるので、ゲームの状態と
「いま何のスライドを映しているか」をプロセス内の 1 か所に置き、両方の画面から参照する。
司会者用画面が書き換え、投影用画面は 1 秒ごとに読み直して描画する。

スライドの進行（「次へ」で何が起きるか）もここで決める。Streamlit には依存しない。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from semantle.state import GameState, RoundRecord

# スライドの種類（並び順 = 進行順）と表示名
SLIDES: dict[str, str] = {
    "title": "タイトル",
    "rules": "ルール説明",
    "topic": "お題",
    "result": "結果発表",
    "reference": "基準単語",
    "standings": "累計得点",
}


@dataclass
class Presentation:
    slide: str = "title"
    topic: str = ""  # 投影中のお題（判定前）。空なら「お題」スライドは出せない
    reveal_round_no: int | None = None  # 発表の演出をしているラウンド
    reveal_shown: int = 0  # 発表済みのチーム数
    last_advance: float = 0.0
    timer_end: float | None = None
    timer_total: int = 0
    font_scale: float = 1.0


class LiveSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.game: GameState | None = None
        self.load_error: str | None = None
        self.loaded = False  # 保存ファイルの読み込みを試したか
        self.pres = Presentation()

    # ---- 参照 ----
    @property
    def last_round(self) -> RoundRecord | None:
        return self.game.rounds[-1] if self.game and self.game.rounds else None

    def reveal_progress(self) -> tuple[int, int]:
        """(発表済みの件数, 全件数)。演出の対象でないラウンドは全部発表済みとして扱う。"""
        last = self.last_round
        if last is None:
            return 0, 0
        n = len(last.results)
        if self.pres.reveal_round_no != last.round_no:
            return n, n
        return max(0, min(self.pres.reveal_shown, n)), n

    def reveal_complete(self) -> bool:
        shown, n = self.reveal_progress()
        return shown >= n

    def pending_round_no(self) -> int | None:
        """発表が終わっていないラウンド（累計得点にまだ入れない）。"""
        last = self.last_round
        return last.round_no if last is not None and not self.reveal_complete() else None

    def has_reference(self) -> bool:
        last = self.last_round
        return bool(last and last.reference_top)

    def available(self) -> set[str]:
        """いま映せるスライド。"""
        out = {"title", "rules"}
        if self.game is None:
            return out
        out.add("standings")
        if self.pres.topic:
            out.add("topic")
        if self.last_round is not None:
            out.add("result")
            if self.has_reference():
                out.add("reference")
        return out

    def signature(self) -> tuple[str, int]:
        return self.pres.slide, self.pres.reveal_shown

    # ---- 操作 ----
    def show(self, slide: str) -> bool:
        with self.lock:
            if slide not in self.available():
                return False
            self.pres.slide = slide
            return True

    def publish_topic(self, topic: str) -> None:
        with self.lock:
            self.pres.topic = topic
            self.pres.slide = "topic"

    def start_reveal(self, enabled: bool, now: float | None = None) -> None:
        """判定直後に呼ぶ。結果スライドに切り替え、演出ありなら 0 件目から始める。"""
        with self.lock:
            last = self.last_round
            if last is None:
                return
            self.pres.reveal_round_no = last.round_no
            self.pres.reveal_shown = 0 if enabled else len(last.results)
            self.pres.last_advance = time.time() if now is None else now
            self.pres.topic = ""
            self.pres.timer_end = None
            self.pres.slide = "result"

    def advance_reveal(self, step: int | None = 1, now: float | None = None) -> None:
        """``step`` 件ぶん発表を進める。None なら全部。"""
        with self.lock:
            last = self.last_round
            if last is None:
                return
            shown, n = self.reveal_progress()
            self.pres.reveal_round_no = last.round_no
            self.pres.reveal_shown = n if step is None else min(n, shown + step)
            self.pres.last_advance = time.time() if now is None else now

    def round_removed(self) -> None:
        """Undo のあと。無くなったラウンドのスライドを映し続けない。"""
        with self.lock:
            self.pres.reveal_round_no = None
            self.pres.reveal_shown = 0
            if self.pres.slide not in self.available() or self.pres.slide in ("result", "reference"):
                self.pres.slide = "standings"

    def reset_presentation(self) -> None:
        with self.lock:
            self.pres = Presentation(font_scale=self.pres.font_scale)

    # ---- 進行 ----
    def _after_result(self) -> str:
        use_ref = self.game is not None and self.game.settings.show_reference_slide and self.has_reference()
        return "reference" if use_ref else "standings"

    def next_action(self) -> tuple[str, str] | None:
        """「次へ」で起きること。(種類, ボタンの文言)。進めないときは None。

        種類は "reveal"（次のチームを発表）か、移動先のスライド名。
        """
        slide = self.pres.slide
        if slide == "title":
            return "rules", "ルール説明へ ▶"
        if slide == "rules":
            return ("topic", "お題へ ▶") if self.pres.topic else None
        if slide == "result" and self.last_round is not None:
            shown, n = self.reveal_progress()
            if shown < n:
                return "reveal", f"次を発表 ▶（{shown} / {n}）"
            target = self._after_result()
            return target, f"{SLIDES[target]}へ ▶"
        if slide == "reference":
            return "standings", "累計得点へ ▶"
        return None

    def go_next(self) -> bool:
        with self.lock:
            action = self.next_action()
            if action is None:
                return False
            if action[0] == "reveal":
                self.advance_reveal(1)
            else:
                self.pres.slide = action[0]
            return True

    def prev_target(self) -> str | None:
        slide = self.pres.slide
        if slide == "rules":
            return "title"
        if slide == "topic":
            return "rules"
        if slide == "reference":
            return "result"
        if slide == "standings" and self.last_round is not None:
            return "reference" if self._after_result() == "reference" else "result"
        return None

    def go_prev(self) -> bool:
        with self.lock:
            target = self.prev_target()
            if target is None:
                return False
            self.pres.slide = target
            return True

    def auto_reveal_active(self) -> bool:
        s = self.game.settings if self.game else None
        return bool(
            s and s.reveal_enabled and s.reveal_auto and self.pres.slide == "result" and not self.reveal_complete()
        )

    def tick(self, now: float | None = None) -> bool:
        """自動発表を進める。どちらの画面から何度呼んでもよい。進めたら True。"""
        with self.lock:
            if not self.auto_reveal_active():
                return False
            now = time.time() if now is None else now
            assert self.game is not None
            if now - self.pres.last_advance < self.game.settings.reveal_interval_sec - 0.05:
                return False
            self.advance_reveal(1, now=now)
            return True


_SESSIONS: dict[str, LiveSession] = {}
_SESSIONS_LOCK = threading.Lock()


def get_live(path: str | Path) -> LiveSession:
    """保存ファイルごとに 1 つの共有状態を返す（無ければ作る）。"""
    key = str(Path(path).resolve())
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            _SESSIONS[key] = LiveSession()
        return _SESSIONS[key]


def reset_live() -> None:
    """共有状態を捨てる（アプリの再起動に相当。テスト用）。"""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()
