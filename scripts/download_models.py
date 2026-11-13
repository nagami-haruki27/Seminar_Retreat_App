"""ローカルモデルを事前にダウンロードし、基準語彙の埋め込みをキャッシュに作っておく。

合宿前にネットのある場所で実行する:

    python scripts/download_models.py                 # 既定の3モデル（Ruri v3 310M, Qwen3 0.6B, mE5-large）
    python scripts/download_models.py --all           # 登録されている全モデル
    python scripts/download_models.py --models ruri-v3-30m me5-base

終わったら `--check` でオフライン状態でも読めることを確認できる:

    python scripts/download_models.py --check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantle.embeddings.cache import CachedBackend
from semantle.embeddings.local_st import SentenceTransformerBackend, set_offline_mode
from semantle.embeddings.registry import MODEL_SPECS, get_spec
from semantle.paths import cache_db_file, reference_words_file
from semantle.reference import load_reference_words

DEFAULT_KEYS = ["ruri-v3-310m", "qwen3-0.6b", "me5-large"]


def prepare(key: str, words: list[str], check_only: bool) -> bool:
    spec = get_spec(key)
    print(f"\n=== {spec.label}  ({spec.model_id}, 約{spec.approx_size_gb}GB) ===")
    t0 = time.perf_counter()
    try:
        backend = CachedBackend(SentenceTransformerBackend(spec), cache_db_file())
        backend.warm_up()
        print(f"  読み込み OK ({time.perf_counter() - t0:.1f}s)")
        t1 = time.perf_counter()
        vecs = backend.embed(["テスト"] + words)
        print(f"  基準語彙 {len(words)} 語の埋め込み OK (次元={vecs.shape[1]}, {time.perf_counter() - t1:.1f}s)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  失敗: {e}")
        if check_only:
            print("  → ネットのある場所で `python scripts/download_models.py` を実行してください")
        return False


def main() -> int:
    # Windows のコンソールが cp932 でも日本語が化けないようにする
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", help=f"モデルキー（{', '.join(MODEL_SPECS)}）")
    parser.add_argument("--all", action="store_true", help="登録されている全モデル")
    parser.add_argument("--check", action="store_true", help="ダウンロードせず、オフラインで読めるか確認する")
    args = parser.parse_args()

    keys = list(MODEL_SPECS) if args.all else (args.models or DEFAULT_KEYS)
    set_offline_mode(args.check)

    words = load_reference_words(reference_words_file())
    print(f"基準語彙: {len(words)} 語 ({reference_words_file()})")
    print(f"キャッシュ: {cache_db_file()}")

    ok = [prepare(k, words, args.check) for k in keys]
    print("\n結果:", "  ".join(f"{k}:{'OK' if o else 'NG'}" for k, o in zip(keys, ok)))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
