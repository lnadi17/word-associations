"""Validate ranking and directional semantics against the released cleaned-trial indexes."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.explore import _rank
from wordviz.explore_data import MODELS, neighbors, vocabulary


class ExplorerTests(unittest.TestCase):
    def test_direction_and_merged_weight(self):
        self.assertEqual(neighbors("humans", "cat", "forward").counts["dog"], 54)
        self.assertEqual(neighbors("gpt5_mini", "cat", "forward").counts["purr"], 32)
        self.assertEqual(neighbors("gpt5_mini", "cat", "backward").counts["purr"], 100)
        self.assertEqual(neighbors("gpt5_mini", "cat", "merged").counts["purr"], 100)

    def test_membership_uses_all_neighbors(self):
        series = [neighbors(model, "cat", "forward") for model in ("humans", "gpt5_mini")]
        totals = [s.total for s in series]
        left, right = (set(s.counts) for s in series)
        for selection, expected in (("shared", left & right), ("only-0", left - right), ("only-1", right - left)):
            ranked, _ = _rank(series, totals, selection, "share")
            self.assertEqual(set(ranked), expected)
        self.assertEqual(_rank(series, totals, "all", "profile-0")[0][0], "dog")
        self.assertEqual(_rank(series, totals, "all", "profile-1")[0][0], "whisker")

    def test_each_packaged_model_has_searchable_data(self):
        for model in MODELS:
            with self.subTest(model=model):
                self.assertIn("cat", vocabulary(model))
                self.assertTrue(neighbors(model, "cat", "forward").counts)


if __name__ == "__main__":
    unittest.main()
