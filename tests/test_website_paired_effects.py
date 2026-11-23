"""Directional tests, six-network coverage and the two target mappings."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.experiment_data import MODELS, contrast_signed_rank
from wordviz.experiments import holm_adjust, paired_effect, selection_problem


class PairedStudyEffects(unittest.TestCase):
    def test_exact_direction_with_ties_zeros_and_sign_reversal(self):
        # Mean ranks for the nonzero absolute differences are 1.5, 1.5, 3.
        # Two of the eight permutations have rank sum >= 4.5, plus sum 6.
        positives = contrast_signed_rank([1, -1, 2, 0], directional=True)
        negatives = contrast_signed_rank([-1, 1, -2, 0], directional=True)
        self.assertEqual(positives[1:], (3 / 8, 3, "Exact sign permutation"))
        self.assertEqual(negatives[1], 7 / 8)
        self.assertEqual(positives[0], -negatives[0])
        self.assertEqual(contrast_signed_rank([0] * 25, directional=True)[:3], (None, None, 0))

    def test_holm_preserves_order_and_excludes_undefined_tests(self):
        rows = [{"p": p, "adjusted": None} for p in (.021, .004, None, .020, .85)]
        holm_adjust(rows)
        self.assertEqual([r["adjusted"] for r in rows], [.06, .016, None, .06, .85])

    def test_all_networks_orient_each_set_and_expose_coverage(self):
        pair = {"primes": ["a", "b"], "labels": ["A words", "B words"]}
        groups = [["x", "y", "z"], ["k", "l"]]
        info = {model: {"missing_primes": []} for model in MODELS}
        info["Gemma"]["missing_primes"] = ["b"]
        values = {model: {w: {"a": .8, "b": .2} for w in groups[0] + groups[1]} for model in MODELS}
        del values["Llama3"]["y"]
        del values["Gemma"]["x"]  # An ineligible network cannot reduce the matched sample.
        result = paired_effect(pair, groups, info, values, ["Humans"])
        charts = result[1].children
        self.assertEqual(len(charts), 2)
        for chart in charts:
            self.assertEqual(list(chart.children[1].figure.data[0].x), list(MODELS.values()))
        first = charts[0].children[1].figure.data[0].y
        second = charts[1].children[1].figure.data[0].y
        self.assertGreater(first[0], 0)
        self.assertEqual(second[0], -first[0])
        self.assertIsNone(first[list(MODELS).index("Gemma")])
        coverage = next(child for child in result if child.__class__.__name__ == "Ul")
        self.assertIn("2/3 targets shared across 5", coverage.children[0].children)
        details = next(child for child in result if child.__class__.__name__ == "Details")
        rows = details.children[1].children.children[1].children
        self.assertEqual(len(rows), 12)
        self.assertEqual([c.children for c in rows[0].children][3:9],
                         [2, 2, "+1.000", "0.250", "1.000", "Exact sign permutation"])
        self.assertIn("Missing prime", rows[2].children[-1].children)
        self.assertIn("Remove", selection_problem("religion", "Humans", None, [["a"], ["t"]], pair))
        self.assertIn("one group", selection_problem("religion", "Humans", None, [["w"], ["w"]], pair))


if __name__ == "__main__":
    unittest.main()
