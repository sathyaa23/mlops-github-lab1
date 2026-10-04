import pytest

from src.data_utils import (
    mean, median, std, variance,
    min_max_scale, z_score_normalize,
    detect_outliers_iqr, summary, load_column,
)


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_std_sample():
    # population std = 2.0, sample std = sqrt(32/7) = 2.138...
    assert std([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.138, abs=1e-3)


@pytest.mark.parametrize("values,expected", [
    ([0, 5, 10], [0.0, 0.5, 1.0]),
    ([3, 3, 3], [0.0, 0.0, 0.0]),
])
def test_min_max_scale(values, expected):
    assert min_max_scale(values) == pytest.approx(expected)


def test_z_score_mean_is_zero():
    z = z_score_normalize([10, 20, 30, 40])
    assert mean(z) == pytest.approx(0.0, abs=1e-9)


def test_detect_outliers_iqr():
    data = [10, 10, 11, 11, 12, 12, 12, 13, 100]
    assert detect_outliers_iqr(data) == [100]


def test_summary_keys():
    s = summary([1, 2, 3, 4, 5])
    assert set(s) == {"count", "mean", "median", "std", "min", "max", "outliers"}
    assert s["count"] == 5


def test_load_column_from_data_file():
    vals = load_column("data/sample.csv", "value")
    assert len(vals) == 9
    assert 100.0 in vals


def test_variance_raises_on_single_value():
    with pytest.raises(ValueError):
        variance([5])
