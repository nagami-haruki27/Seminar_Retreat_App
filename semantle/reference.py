"""基準語彙（補正に使う一般的な単語）の読み込み。"""

from __future__ import annotations

from pathlib import Path

from semantle.normalize import clean_for_embedding, normalize_key


def parse_reference_words(text: str) -> list[str]:
    """1行1語。空行と # で始まる行は無視。重複（表記ゆれ込み）は最初のものだけ残す。"""
    words: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        word = clean_for_embedding(line)
        if not word:
            continue
        key = normalize_key(word)
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
    return words


def load_reference_words(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"基準語彙ファイルがありません: {path}")
    return parse_reference_words(path.read_text(encoding="utf-8"))
