import numpy as np
import pytest

from semantle.calibration import CalibrationError, CalibrationMode, calibrate, linear_score, percentile_score


class TestLinear:
    def test_mean_maps_to_zero_and_topic_to_one(self) -> None:
        ref = np.array([0.7, 0.8, 0.9])  # μ = 0.8
        assert linear_score(0.8, ref) == pytest.approx(0.0)
        assert linear_score(1.0, ref) == pytest.approx(1.0)

    def test_midpoint(self) -> None:
        ref = np.array([0.7, 0.8, 0.9])
        assert linear_score(0.9, ref) == pytest.approx(0.5)

    def test_can_be_negative(self) -> None:
        ref = np.array([0.5, 0.5])
        assert linear_score(0.25, ref) == pytest.approx(-0.5)

    def test_e5_like_distribution(self) -> None:
        # E5 のように無関係語でも 0.8 前後になる場合でも 0〜1 に引き直せる
        ref = np.full(100, 0.8)
        assert linear_score(0.9, ref) == pytest.approx(0.5)

    def test_empty_reference_raises(self) -> None:
        with pytest.raises(CalibrationError):
            linear_score(0.5, np.array([]))

    def test_degenerate_reference_raises(self) -> None:
        with pytest.raises(CalibrationError):
            linear_score(0.5, np.array([1.0, 1.0]))


class TestPercentile:
    def test_bounds(self) -> None:
        ref = np.array([0.1, 0.2, 0.3, 0.4])
        assert percentile_score(0.05, ref) == 0.0
        assert percentile_score(0.1, ref) == 0.0
        assert percentile_score(0.4, ref) == 1.0
        assert percentile_score(0.99, ref) == 1.0

    def test_interpolation(self) -> None:
        ref = np.array([0.1, 0.2, 0.3, 0.4])
        assert percentile_score(0.2, ref) == pytest.approx(1 / 3)
        assert percentile_score(0.25, ref) == pytest.approx(0.5)
        assert percentile_score(0.35, ref) == pytest.approx(2.5 / 3)

    def test_unsorted_input(self) -> None:
        ref = np.array([0.4, 0.1, 0.3, 0.2])
        assert percentile_score(0.25, ref) == pytest.approx(0.5)

    def test_monotonic(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.uniform(0, 1, 100)
        xs = np.linspace(-0.2, 1.2, 200)
        ys = [percentile_score(x, ref) for x in xs]
        assert all(b >= a for a, b in zip(ys, ys[1:]))

    def test_duplicates_in_reference(self) -> None:
        ref = np.array([0.2, 0.2, 0.2, 0.8])
        assert percentile_score(0.5, ref) == pytest.approx(2.5 / 3)
        assert percentile_score(0.2, ref) == pytest.approx(0.0)

    def test_single_reference(self) -> None:
        assert percentile_score(0.1, np.array([0.5])) == 0.0
        assert percentile_score(0.9, np.array([0.5])) == 1.0

    def test_empty_reference_raises(self) -> None:
        with pytest.raises(CalibrationError):
            percentile_score(0.5, np.array([]))


def test_calibrate_dispatch() -> None:
    ref = np.array([0.0, 0.5])
    assert calibrate(0.75, ref, CalibrationMode.RAW) == 0.75
    assert calibrate(0.75, ref, CalibrationMode.LINEAR) == pytest.approx((0.75 - 0.25) / 0.75)
    assert calibrate(0.25, ref, CalibrationMode.PERCENTILE) == pytest.approx(0.5)


def test_labels_exist() -> None:
    for mode in CalibrationMode:
        assert mode.label
