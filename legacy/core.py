"""判定ロジック（モデルに依存しない部分）"""
import numpy as np
import pandas as pd

# 補正の基準にする「お題と無関係そうな」一般的な単語
BASELINE_WORDS = [
    "りんご", "自転車", "雨", "机", "猫", "電車", "山", "靴下", "時計", "海",
    "鉛筆", "パン", "傘", "犬", "窓", "花", "車", "本", "椅子", "雪",
    "牛乳", "帽子", "鍵", "橋", "月", "魚", "財布", "ドア", "石", "紙",
    "音楽", "野球", "病院", "郵便", "砂糖", "階段", "布団", "風", "森", "コップ",
    "歯ブラシ", "新聞", "公園", "ボタン", "卵", "テレビ", "雲", "駅", "皿", "手袋",
    "政治", "祭り", "掃除", "旅行", "料理", "眠り", "冬", "笑顔", "約束", "地図",
]


def adjusted_scores(raw, baseline):
    """無関係な単語の平均類似度を0、完全一致を1に線形変換する"""
    m = float(np.mean(baseline))
    return (np.asarray(raw) - m) / (1.0 - m)


def judge(names, words, raw, scores, theme, target=None):
    df = pd.DataFrame({"チーム": names, "単語": words, "生の類似度": raw, "スコア": scores})
    # お題そのもの・お題を含む単語は反則
    df["反則"] = [
        "お題と同じ" if w == theme else ("お題を含む" if theme in w or w in theme else "")
        for w in df["単語"]
    ]
    key = -df["スコア"] if target is None else (df["スコア"] - target).abs()
    df["_key"] = key
    df["_foul"] = df["反則"] != ""
    df = df.sort_values(["_foul", "_key"]).drop(columns=["_key", "_foul"]).reset_index(drop=True)
    df.insert(0, "順位", [str(i + 1) if f == "" else "-" for i, f in enumerate(df["反則"])])
    return df
