"""Regression checks for the deployed, dependency-light statistics implementation."""

import csv
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.experiment_data import DEFAULT_PAIRS, MODELS, genders, ldt_stats, metadata, signed_rank


def reference(name):
    path = ROOT / "data" / "LDT_analyses" / "Statistical_tests" / name
    with path.open(newline="") as stream:
        return {r[""]: r for r in csv.DictReader(stream)}


class ArchivedExperiments(unittest.TestCase):
    def assert_reference(self, actual, expected):
        self.assertTrue(math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-25),
                        f"{actual!r} differs from archive {expected}")

    def test_ldt_effects_correlations_and_p_values(self):
        ldt = metadata("ldt")
        self.assertEqual(len(ldt["triplets"]), 50)
        wilcoxon = reference("wilcoxonLDT.csv")
        correlation = reference("corrLDT.csv")
        for model in MODELS:
            with self.subTest(network=model):
                effect, wp, rho, sp = ldt_stats(model, ldt)
                self.assert_reference(effect, wilcoxon["effect"][model])
                self.assert_reference(wp, wilcoxon["p"][model])
                self.assert_reference(rho, correlation["r"][model])
                self.assert_reference(sp, correlation["p"][model])

    def test_gender_coverage_and_paper_orientation(self):
        groups = metadata("targets")
        values = genders(groups[0] + groups[1])
        for group, (direction, filename, n_targets) in enumerate([
            (1, "wilxoconGenderF.csv", 23), (-1, "wilxoconGenderM.csv", 25)
        ]):
            expected = reference(filename)
            words = [word for word in groups[group] if all(word in values[m] for m in MODELS)]
            self.assertEqual(len(words), n_targets)
            for model in MODELS:
                with self.subTest(group=group, model=model):
                    differences = [
                        direction * (values[model][w][left] - values[model][w][right])
                        for w in words for left, right in DEFAULT_PAIRS
                    ]
                    effect, p = signed_rank(differences)
                    self.assert_reference(effect, direction * float(expected["effect"][model]))
                    self.assert_reference(p, expected["p"][model])

    def test_tied_ranks_and_zero_differences(self):
        # Two tied values have average rank 1.5. The zero participates in
        # the effect-size denominator but is dropped from the rank test.
        effect, p = signed_rank([1, 1, -2, 0])
        self.assertEqual(effect, 0)
        self.assertEqual(p, 0.5)
        self.assertEqual(signed_rank([0, 0]), (None, None))
        self.assertEqual(signed_rank([]), (None, None))

    def test_reordered_preset_retains_six_network_cohort(self):
        from wordviz.experiments import render_bias

        female, male = metadata("targets")
        pairs = ", ".join(f"{left}/{right}" for left, right in reversed(DEFAULT_PAIRS))
        effect_panel, _ = render_bias("Haiku", "Mistral", female[::-1], male, ["woman", "man"], pairs)
        coverage = next(child for child in effect_panel.children if child.__class__.__name__ == "Ul")
        self.assertIn("23/25", coverage.children[0].children)


if __name__ == "__main__":
    unittest.main()
