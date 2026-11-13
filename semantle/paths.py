"""ファイルの置き場所。環境変数で上書きできる（テストでは一時フォルダに向ける）。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def save_dir() -> Path:
    return Path(os.environ.get("SEMANTLE_SAVE_DIR", PROJECT_ROOT / "save"))


def cache_dir() -> Path:
    return Path(os.environ.get("SEMANTLE_CACHE_DIR", PROJECT_ROOT / "cache"))


def state_file() -> Path:
    return save_dir() / "game_state.json"


def results_csv_file() -> Path:
    return save_dir() / "results.csv"


def cache_db_file() -> Path:
    return cache_dir() / "embeddings.sqlite"


def reference_words_file() -> Path:
    return Path(os.environ.get("SEMANTLE_REFERENCE_WORDS", DATA_DIR / "reference_words.txt"))
