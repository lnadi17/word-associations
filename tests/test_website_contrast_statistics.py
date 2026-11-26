"""Small-sample inference for the editable social prime contrasts."""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.experiment_data import signed_rank, two_sided_signed_rank
from wordviz.experiments import contrast_results


class ContrastStatisticsTests(unittest.TestCase):
    def test_exact_tails_with_ties_and_zeros(self):
        # The doubled ranks are [3, 3, 6, 9, 9]. Each tail contains the
        # empty subset and two singletons, for six of 32 sign assignments.
        differences = [-1, 1, 3, 5, 5, 0]
        for direction in (1, -1):
            oriented = [direction * value for value in differences]
            effect, p, nonzero, method = two_sided_signed_rank(oriented)
            self.assertEqual(effect, signed_rank(oriented)[0])
            self.assertEqual((p, nonzero, method), (6 / 32, 5, "Exact sign permutation"))
        self.assertEqual(two_sided_signed_rank([1, 1, -2, 0])[1], 1)
        self.assertEqual(two_sided_signed_rank([0, 0]), (None, None, 0, "No nonzero differences"))

    def test_threshold_counts_only_nonzero_pairs(self):
        for n in (5, 20):
            effect, p, nonzero, method = two_sided_signed_rank(list(range(1, n + 1)) + [0, 0])
            self.assertGreater(effect, 0)
            self.assertEqual((p, nonzero, method), (2 / (2 ** n), n, "Exact sign permutation"))
        differences = [-1, 1, 2] * 7 + [0]
        effect, p, nonzero, method = two_sided_signed_rank(differences)
        self.assertEqual(effect, signed_rank(differences)[0])
        self.assertEqual(p, math.erfc(abs(effect) * math.sqrt(len(differences) / 2)))
        self.assertEqual((nonzero, method), (21, "Normal approximation"))

    def test_method_and_p_displayed_for_each_network(self):
        targets = [str(i) for i in range(21)]
        values = {
            model: {word: {"a": int(i < n), "b": 0} for i, word in enumerate(targets)}
            for model, n in [("Humans", 5), ("Gemma", 21)]
        }
        items = contrast_results(["Humans", "Gemma"], targets, "a", "b", values)
        table = next(item for item in items if getattr(item, "className", "") == "table-wrap")
        rows = [[cell.children for cell in row.children] for row in table.children.children[1].children]
        self.assertEqual(rows[0][-1], "Normal approximation")
        self.assertEqual(rows[1][5:], ["0.0625", "Exact sign permutation"])
