import pytest

from semantle.normalize import FoulKind, check_foul, clean_for_embedding, katakana_to_hiragana, normalize_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  拡散モデル  ", "拡散モデル"),
        ("ＧＰＵ", "GPU"),  # 全角英字 → 半角
        ("ｶﾀｶﾅ", "カタカナ"),  # 半角カナ → 全角カナ
        ("deep　learning", "deep learning"),  # 全角スペース → 半角
        ("a   b", "a b"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_for_embedding(raw: str, expected: str) -> None:
    assert clean_for_embedding(raw) == expected


def test_clean_keeps_case_and_katakana() -> None:
    assert clean_for_embedding("BERT") == "BERT"
    assert clean_for_embedding("ネコ") == "ネコ"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("ネコ", "ねこ"),
        ("ＧＰＵ", "gpu"),
        ("  Transformer ", "transformer"),
        ("ｶﾀｶﾅ", "かたかな"),
        ("ヴァイオリン", "ゔぁいおりん"),
    ],
)
def test_normalize_key_absorbs_variants(a: str, b: str) -> None:
    assert normalize_key(a) == normalize_key(b)


def test_normalize_key_distinguishes_real_differences() -> None:
    assert normalize_key("猫") != normalize_key("ねこ")  # 漢字とかなは区別する（司会判断の領域）
    assert normalize_key("拡散") != normalize_key("拡散モデル")


def test_katakana_to_hiragana_keeps_long_vowel() -> None:
    assert katakana_to_hiragana("コーヒー") == "こーひー"


class TestCheckFoul:
    def test_same(self) -> None:
        assert check_foul("拡散モデル", "拡散モデル") == FoulKind.SAME

    def test_same_with_variants(self) -> None:
        assert check_foul(" 拡散モデル ", "拡散モデル") == FoulKind.SAME
        assert check_foul("ＧＰＵ", "gpu") == FoulKind.SAME
        assert check_foul("ねこ", "ネコ") == FoulKind.SAME

    def test_contains(self) -> None:
        assert check_foul("潜在拡散モデル", "拡散モデル") == FoulKind.CONTAINS
        assert check_foul("拡散モデルの応用", "拡散モデル") == FoulKind.CONTAINS

    def test_contained_is_ok_by_default(self) -> None:
        assert check_foul("拡散", "拡散モデル") is None
        assert check_foul("モデル", "拡散モデル") is None

    def test_contained_when_strict(self) -> None:
        assert check_foul("拡散", "拡散モデル", strict_contained=True) == FoulKind.CONTAINED

    def test_unrelated(self) -> None:
        assert check_foul("カレー", "拡散モデル") is None

    def test_empty_is_not_foul(self) -> None:
        assert check_foul("", "拡散モデル") is None
        assert check_foul("   ", "拡散モデル") is None

    def test_labels(self) -> None:
        assert FoulKind.SAME.label == "お題と同じ"
        assert FoulKind.MANUAL.label == "司会判断"
