"""利用できる埋め込みモデルの定義。

モデルID・接頭辞は 2026-11-09 に Hugging Face のモデルカードで確認したもの。
単語同士の「意味の近さ」を測る対称タスクとしての使い方に合わせている。

* Ruri v3   … 接頭辞なし（空文字が「意味の符号化」用。"トピック: " は分類・クラスタリング用）
* Qwen3     … "Instruct: {指示}\\nQuery:{文}" 形式。カードは検索用途しか書いていないため、
              対称タスクでは両側に同じ指示を付ける（scripts/compare_models.py で指示なしとも比較できる）
* mE5       … 対称タスクは両側に "query: "（モデルカードの FAQ に明記）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    prefix: str = ""
    notes: str = ""
    approx_size_gb: float = 0.0
    min_transformers: str = ""

    @property
    def cache_key(self) -> str:
        # 接頭辞を変えたらキャッシュも別扱いになるように含める
        return f"{self.model_id}|{self.prefix}"

    def format(self, text: str) -> str:
        return f"{self.prefix}{text}"


QWEN3_STS_INSTRUCTION = "Retrieve semantically similar text."

MODEL_SPECS: dict[str, ModelSpec] = {
    "ruri-v3-310m": ModelSpec(
        key="ruri-v3-310m",
        label="Ruri v3 310M（日本語・推奨）",
        model_id="cl-nagoya/ruri-v3-310m",
        prefix="",
        notes="日本語特化。接頭辞なしで意味の符号化。",
        approx_size_gb=1.3,
        min_transformers="4.48.0",
    ),
    "ruri-v3-30m": ModelSpec(
        key="ruri-v3-30m",
        label="Ruri v3 30M（軽量）",
        model_id="cl-nagoya/ruri-v3-30m",
        prefix="",
        notes="非力なPC向け。",
        approx_size_gb=0.15,
        min_transformers="4.48.0",
    ),
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        label="Qwen3-Embedding 0.6B（多言語）",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        prefix=f"Instruct: {QWEN3_STS_INSTRUCTION}\nQuery:",
        notes="両側に同じ英語の指示を付ける。",
        approx_size_gb=1.2,
        min_transformers="4.51.0",
    ),
    "qwen3-0.6b-noinst": ModelSpec(
        key="qwen3-0.6b-noinst",
        label="Qwen3-Embedding 0.6B（指示なし）",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        prefix="",
        notes="同じモデルを指示文なしで使う。類似度の分布が広い（補正なしでも遊べる）。",
        approx_size_gb=1.2,
        min_transformers="4.51.0",
    ),
    "me5-large": ModelSpec(
        key="me5-large",
        label="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large",
        prefix="query: ",
        notes="類似度が 0.7〜1.0 に偏るので補正推奨。",
        approx_size_gb=2.2,
    ),
    "me5-base": ModelSpec(
        key="me5-base",
        label="multilingual-e5-base（軽量）",
        model_id="intfloat/multilingual-e5-base",
        prefix="query: ",
        notes="類似度が 0.7〜1.0 に偏るので補正推奨。",
        approx_size_gb=1.1,
    ),
}

DEFAULT_MODEL_KEY = "ruri-v3-310m"


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as e:
        raise KeyError(f"未知のモデルキー: {key}（利用可能: {', '.join(MODEL_SPECS)}）") from e
