"""Regression checks for the deployed, dependency-light statistics implementation."""

import csv
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.models import ARCHIVED_MATRICES
from wordviz.experiment_data import DEFAULT_PAIRS, MODELS, genders, ldt_stats, metadata, signed_rank


def reference(name, cohort=""):
    path = ROOT / "data" / "LDT_analyses" / "Statistical_tests" / cohort / name
    with path.open(newline="") as stream:
        rows = {}
        for record in csv.DictReader(stream):
            name = record.pop("")
            rows[name] = record
        return rows


class ArchivedExperiments(unittest.TestCase):
    def assert_reference(self, actual, expected):
        self.assertTrue(math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-25),
                        f"{actual!r} differs from archive {expected}")

    def test_ldt_effects_correlations_and_p_values(self):
        ldt = metadata("ldt")
        self.assertEqual(len(ldt["triplets"]), 50)
        wilcoxon = reference("wilcoxonLDT.csv")
        correlation = reference("corrLDT.csv")
        for model in wilcoxon["effect"]:
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
            cohort = list(expected["effect"])
            words = [word for word in groups[group] if all(word in values[m] for m in cohort)]
            archived_words = [word for word in groups[group] if all(word in values[m] for m in ARCHIVED_MATRICES)]
            self.assertEqual(len(archived_words), n_targets)
            for model in cohort:
                with self.subTest(group=group, model=model):
                    differences = [
                        direction * (values[model][w][left] - values[model][w][right])
                        for w in words for left, right in DEFAULT_PAIRS
                    ]
                    effect, p = signed_rank(differences)
                    self.assert_reference(effect, direction * float(expected["effect"][model]))
                    self.assert_reference(p, expected["p"][model])

    def test_expanded_network_statistics_match_offline_analysis(self):
        if set(MODELS) == set(ARCHIVED_MATRICES):
            self.skipTest("Only the archived cohort is packaged.")
        wilcoxon = reference("wilcoxonLDT.csv", "all_networks")
        correlation = reference("corrLDT.csv", "all_networks")
        self.assertEqual(set(wilcoxon["effect"]), set(MODELS))
        experiment = metadata("ldt")
        for model in MODELS:
            with self.subTest(model=model):
                effect, p, rho, correlation_p = ldt_stats(model, experiment)
                self.assert_reference(effect, wilcoxon["effect"][model])
                self.assert_reference(p, wilcoxon["p"][model])
                self.assert_reference(rho, correlation["r"][model])
                self.assert_reference(correlation_p, correlation["p"][model])

    def test_missing_ldt_triplet_excludes_its_measured_reaction_times(self):
        trips = [["a", "b", "c", 1., 5.], ["d", "e", "f", 0., 9.], ["g", "h", "i", 2., 4.]]
        values = [[0.8, 0.2], None, [0.7, 0.3]]
        actual = ldt_stats("example", {"triplets": trips, "activation": {"example": values}})
        expected = ldt_stats("example", {"triplets": trips[::2], "activation": {"example": values[::2]}})
        self.assertEqual(actual, expected)
        self.assertEqual(ldt_stats("example", {"triplets": trips, "activation": {"example": [None]*3}}),
                         (None, None, None, None))

    def test_tied_ranks_and_zero_differences(self):
        # Two tied values have average rank 1.5. The zero participates in
        # the effect-size denominator but is dropped from the rank test.
        effect, p = signed_rank([1, 1, -2, 0])
        self.assertEqual(effect, 0)
        self.assertEqual(p, 0.5)
        self.assertEqual(signed_rank([0, 0]), (None, None))
        self.assertEqual(signed_rank([]), (None, None))

    def test_reordered_preset_retains_common_network_cohort(self):
        from wordviz.experiments import render_effect

        female, male = metadata("targets")
        pairs = ", ".join(f"{left}/{right}" for left, right in reversed(DEFAULT_PAIRS))
        effect_panel = render_effect("gender", "Haiku", "Mistral", female[::-1], male, pairs, None, None)
        coverage = next(child for child in effect_panel.children if child.__class__.__name__ == "Ul")
        values = genders(female)
        shared = sum(all(word in values[m] for m in MODELS) for word in female)
        self.assertIn(f"{shared}/25", coverage.children[0].children)

    def test_effect_charts_reserve_the_figure_height_in_the_grid(self):
        from wordviz.experiments import gender_effect

        groups = [["kind"], ["strong"]]
        _, panels, *_ = gender_effect(["Humans"], groups, genders(groups[0] + groups[1]),
                                      [("woman", "man")], canonical=False)
        for panel in panels.children:
            chart = panel.children[1]
            self.assertTrue(chart.responsive)
            self.assertEqual(chart.style["height"], f"{chart.figure.layout.height}px")

    def test_custom_contrast_uses_matched_targets_across_networks(self):
        from wordviz.experiments import contrast_results

        values = {
            "Humans": {"x": {"a": .8, "b": .2}, "y": {"a": .3, "b": .3}, "z": {"a": .1, "b": .9}},
            "Gemma": {"x": {"a": .2, "b": .8}, "y": {"a": .3, "b": .3}, "z": {"a": .7}},
        }
        items = contrast_results(["Humans", "Gemma"], ["x", "y", "z"], "a", "b", values)
        table = next(item for item in items if getattr(item, "className", "") == "table-wrap")
        rows = [[cell.children for cell in row.children] for row in table.children.children[1].children]
        self.assertEqual(rows[0][1:4], [2, 1, "+0.3000"])
        self.assertEqual(rows[1][1:4], [2, 1, "-0.3000"])
        self.assertEqual(float(rows[0][4]), -float(rows[1][4]))
        self.assertEqual(rows[0][5], rows[1][5])

    def test_new_domain_has_no_implicit_reference_prime(self):
        from wordviz.experiments import render_effect
        from wordviz.target_presets import TARGET_PRESETS

        for domain, presets in TARGET_PRESETS.items():
            words = next(iter(presets.values())).words
            result = render_effect(domain, "Humans", None, words, [], None, None, None)
            self.assertIn("Choose two different primes", result.children)

    def test_priming_inclusion_is_independent_of_highlights(self):
        from dash import dcc, html
        from wordviz.experiments import ACCENT, SECONDARY, render_effect

        cohort = ["Humans", "Gpt5_mini", "Gemma"]
        single = render_effect("religion", "Haiku", "Gpt5_mini", ["church", "holy", "cross"], None,
                               None, "christian", "muslim", included=cohort)
        trace = next(c.figure.data[0] for c in single._traverse() if isinstance(c, dcc.Graph))
        self.assertEqual(list(trace.y), [MODELS[m] for m in cohort])
        self.assertEqual(trace.marker.color[1], SECONDARY)
        self.assertNotIn(ACCENT, trace.marker.color)
        rows = next(c.children for c in single._traverse() if isinstance(c, html.Tbody))
        self.assertEqual([r.className for r in rows], ["", "selected-row", ""])

        for highlights in [("Humans", "Humans"), (None, None)]:
            gender = render_effect("gender", *highlights, ["kind"], ["strong"], "woman/man", None, None,
                                   included=["Gemma"])
            for chart in (c for c in gender._traverse() if isinstance(c, dcc.Graph)):
                self.assertEqual(list(chart.figure.data[0].y), [MODELS["Gemma"]])
            self.assertEqual(len(next(c.children for c in gender._traverse() if isinstance(c, html.Tbody))), 2)
        empty = render_effect("gender", "Humans", None, ["kind"], ["strong"], "woman/man", None, None, included=[])
        self.assertEqual(empty.role, "status")
        self.assertIn("at least one network", empty.children)

    def test_published_target_lists_work_in_each_network(self):
        from wordviz.social_data import domain_values
        from wordviz.social_domains import DOMAIN_PRIMES
        from wordviz.target_presets import TARGET_PRESETS

        self.assertEqual(set(TARGET_PRESETS), set(DOMAIN_PRIMES) - {"gender"})
        for domain, presets in TARGET_PRESETS.items():
            words = {word for preset in presets.values() for word in preset.words}
            rows = domain_values(domain, list(words), MODELS)
            for name, preset in presets.items():
                with self.subTest(domain=domain, preset=name):
                    self.assertTrue(preset.study.url.startswith("https://"))
                    self.assertTrue(preset.study.context)
                    self.assertEqual(len(set(preset.words)), len(preset.words))
                    for model in ARCHIVED_MATRICES:
                        self.assertTrue(set(preset.words).issubset(rows[model]), model)

    def test_heatmap_separates_missing_data_and_observed_zero(self):
        from wordviz.experiments import heatmap

        chart = heatmap("Gemma", [["present", "absent"]], ["seen", "missing"],
                        {"Gemma": {"present": {"seen": 0}}}, ["Target words"], ["missing"])
        figure = chart.children.children.figure
        self.assertEqual([list(row) for row in figure.data[0].z], [[0, None], [None, None]])
        self.assertEqual(list(figure.layout.xaxis.ticktext), ["seen", "missing · N/A"])
        self.assertEqual(list(figure.layout.yaxis.ticktext), ["present", "absent · N/A"])


if __name__ == "__main__":
    unittest.main()
