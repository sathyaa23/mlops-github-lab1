import os
import tempfile
import unittest

from src.data_utils import (
    mean, median, std, variance,
    min_max_scale, z_score_normalize,
    detect_outliers_iqr, summary, load_column,
)


class TestDataUtils(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_median_even(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_std_sample(self):
        self.assertAlmostEqual(std([2, 4, 4, 4, 5, 5, 7, 9]), 2.138, places=3)

    def test_min_max_scale(self):
        self.assertEqual(min_max_scale([0, 5, 10]), [0.0, 0.5, 1.0])

    def test_min_max_scale_constant(self):
        self.assertEqual(min_max_scale([3, 3, 3]), [0.0, 0.0, 0.0])

    def test_z_score_mean_zero(self):
        z = z_score_normalize([10, 20, 30, 40])
        self.assertAlmostEqual(mean(z), 0.0, places=9)

    def test_detect_outliers(self):
        data = [10, 10, 11, 11, 12, 12, 12, 13, 100]
        self.assertEqual(detect_outliers_iqr(data), [100])

    def test_summary(self):
        s = summary([1, 2, 3, 4, 5])
        self.assertEqual(s["count"], 5)
        self.assertEqual(s["min"], 1)
        self.assertEqual(s["max"], 5)

    def test_variance_raises(self):
        with self.assertRaises(ValueError):
            variance([5])

    def test_load_column(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        ) as f:
            f.write("value\n1\n2\n3\n")
            path = f.name
        try:
            self.assertEqual(load_column(path, "value"), [1.0, 2.0, 3.0])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
