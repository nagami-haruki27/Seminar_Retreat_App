from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """保存先とキャッシュを一時フォルダに向け、モデルを読まないモックモードにする。"""
    monkeypatch.setenv("SEMANTLE_SAVE_DIR", str(tmp_path / "save"))
    monkeypatch.setenv("SEMANTLE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SEMANTLE_FAKE_BACKEND", "1")
    monkeypatch.delenv("SEMANTLE_FAKE_SIMS", raising=False)
    monkeypatch.setenv("SEMANTLE_REFERENCE_WORDS", str(ROOT / "data" / "reference_words.txt"))
    return tmp_path


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    yield
    try:
        import streamlit as st

        st.cache_resource.clear()
        st.cache_data.clear()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def app_path() -> str:
    return str(ROOT / "app.py")


