"""実際のモデルで、いくつかのお題に対する単語のスコアを表にして、モデル間の分布の違いを見る。

    python scripts/compare_models.py                      # 既定の3モデル、既定のお題
    python scripts/compare_models.py --models ruri-v3-310m me5-large --out docs/model_comparison.md
    python scripts/compare_models.py --models ruri-v3-30m me5-base   # 軽量版だけ

出力は Markdown の表（raw / 方式A / 方式B）。ダウンロード済みのモデルだけが対象（オフラインで動く）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantle.embeddings.cache import CachedBackend
from semantle.embeddings.local_st import SentenceTransformerBackend, set_offline_mode
from semantle.embeddings.registry import MODEL_SPECS, ModelSpec, get_spec
from semantle.paths import cache_db_file, reference_words_file
from semantle.reference import load_reference_words
from semantle.service import JudgeService

DEFAULT_KEYS = ["ruri-v3-310m", "qwen3-0.6b", "qwen3-0.6b-noinst", "me5-large"]

# お題 → 近そうな順に並べた候補（意図的に「かなり近い」「まあ近い」「遠い」「反則すれすれ」を混ぜる）
DEFAULT_CASES: dict[str, list[str]] = {
    "拡散モデル": ["生成モデル", "ノイズ", "画像生成", "Stable Diffusion", "ニューラルネットワーク", "統計", "カレー", "拡散"],
    "過学習": ["汎化", "正則化", "学習率", "テストデータ", "暗記", "勉強", "筋肉痛", "バナナ"],
    "査読": ["論文", "リジェクト", "学会", "採択", "締切", "先生", "温泉", "靴下"],
    "温泉": ["露天風呂", "旅館", "湯", "リラックス", "熱海", "合宿", "GPU", "行列"],
    "締切": ["提出", "徹夜", "カレンダー", "焦り", "深夜", "論文", "海", "りんご"],
}


def build_backend(spec: ModelSpec) -> CachedBackend:
    return CachedBackend(SentenceTransformerBackend(spec), cache_db_file())


def run(keys: list[str], cases: dict[str, list[str]], words: list[str]) -> str:
    specs: list[ModelSpec] = [get_spec(k) for k in keys]

    lines: list[str] = ["# モデル間のスコア比較", ""]
    lines.append(f"基準語彙: {len(words)} 語。raw = コサイン類似度、A = 線形補正、B = パーセンタイル。")
    lines.append("")

    services: dict[str, JudgeService] = {}
    timing: list[str] = []
    for spec in specs:
        t0 = time.perf_counter()
        try:
            be = build_backend(spec)
            be.warm_up()
        except Exception as e:  # noqa: BLE001
            lines.append(f"- {spec.label}: 読み込み失敗 ({e})")
            continue
        services[spec.key] = JudgeService(be, words)
        timing.append(f"- {spec.label}: 読み込み {time.perf_counter() - t0:.1f}s")

    for topic, cands in cases.items():
        lines.append(f"## お題「{topic}」")
        lines.append("")
        header = "| 単語 | " + " | ".join(f"{s.label}<br>raw / A / B" for s in specs if s.key in services) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(services))
        tables: dict[str, dict[str, dict[str, float | str]]] = {}
        stats: dict[str, str] = {}
        for key, svc in services.items():
            t0 = time.perf_counter()
            rows = svc.score_table(topic, cands)
            dt = time.perf_counter() - t0
            tables[key] = {str(r["word"]): r for r in rows}
            _, ref = svc.topic_reference_sims(topic)
            stats[key] = f"μ={ref.mean:.3f} min={ref.minimum:.3f} max={ref.maximum:.3f} ({dt:.1f}s)"
        for w in cands:
            cells = []
            for key in services:
                r = tables[key][w]
                cells.append(f"{r['raw']:.3f} / {r['linear']:.3f} / {r['percentile']:.2f}")
            lines.append(f"| {w} | " + " | ".join(cells) + " |")
        lines.append("| *基準語彙との類似度* | " + " | ".join(stats[k] for k in services) + " |")
        lines.append("")

    lines.append("## 読み込み時間")
    lines.extend(timing)
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=DEFAULT_KEYS, help=f"モデルキー（{', '.join(MODEL_SPECS)}）")
    parser.add_argument("--out", type=Path, default=None, help="Markdown の出力先（省略時は標準出力）")
    parser.add_argument("--online", action="store_true", help="Hugging Face Hub への接続を許可")
    args = parser.parse_args()

    set_offline_mode(not args.online)
    words = load_reference_words(reference_words_file())
    text = run(list(args.models), DEFAULT_CASES, words)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"書き出しました: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
