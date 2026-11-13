"""表記ゆれの正規化と反則判定。

2種類の正規化を区別する:

* ``clean_for_embedding`` … モデルに渡す文字列。全角・半角と前後の空白だけを
  整え、大文字・小文字やカタカナはそのまま残す（BERT と bert は別物として扱う）。
* ``normalize_key`` … 「同じ単語か」「お題を含むか」を判定するためのキー。
  全角・半角、前後の空白、大文字・小文字、カタカナ・ひらがなを吸収する。
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

_WS_RE = re.compile(r"\s+")


def clean_for_embedding(text: str) -> str:
    """モデルに渡す形に整える（NFKC、前後の空白除去、連続空白の圧縮）。"""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def katakana_to_hiragana(text: str) -> str:
    """全角カタカナ（ァ..ヶ、ヴ含む）をひらがなに変換する。長音「ー」はそのまま。"""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize_key(text: str) -> str:
    """反則・同一判定用のキー。空文字なら空文字を返す。"""
    text = clean_for_embedding(text)
    text = text.casefold()
    text = katakana_to_hiragana(text)
    return text


class FoulKind(str, Enum):
    SAME = "same"  # お題と同じ
    CONTAINS = "contains"  # お題を含む
    CONTAINED = "contained"  # お題に含まれる（設定で有効化）
    MANUAL = "manual"  # 司会判断

    @property
    def label(self) -> str:
        return _FOUL_LABELS[self]


_FOUL_LABELS = {
    FoulKind.SAME: "お題と同じ",
    FoulKind.CONTAINS: "お題を含む",
    FoulKind.CONTAINED: "お題に含まれる",
    FoulKind.MANUAL: "司会判断",
}


def check_foul(answer: str, topic: str, *, strict_contained: bool = False) -> FoulKind | None:
    """反則なら種類を返し、問題なければ None を返す。

    ``strict_contained`` を True にすると「お題に含まれる単語」
    （お題「拡散モデル」に対する「拡散」）も反則にする。
    空の回答は反則ではない（未回答として別扱い）。
    """
    a = normalize_key(answer)
    t = normalize_key(topic)
    if not a or not t:
        return None
    if a == t:
        return FoulKind.SAME
    if t in a:
        return FoulKind.CONTAINS
    if strict_contained and a in t:
        return FoulKind.CONTAINED
    return None
