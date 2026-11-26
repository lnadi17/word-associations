"""Regression checks for the published summary values and contrast direction."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from dash import dcc, html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.bias_norms import EMOTIONS, emotion_result, valence_result
from wordviz.models import MATRIX_MODELS
from wordviz.norm_views import ACCENT, CONTEXT, SECONDARY, emotion_view, valence_view
from wordviz.social_domains import DOMAIN_PRIMES


class BiasNormsTests(unittest.TestCase):
    def test_all_networks_plotted_and_highlights_preserve_estimates(self):
        def by_name(view):
            graph = next(child for child in view._traverse() if isinstance(child, dcc.Graph))
            self.assertIs(graph.figure.layout.dragmode, False)
            return {trace.name: trace for trace in graph.figure.data}

        renderers = [lambda a, b: valence_view("religion", a, b),
                     lambda a, b: emotion_view("politics", a, b, "democrat", "republican")]
        for render in renderers:
            original = by_name(render("Humans", "Gpt5_mini"))
            changed = by_name(render("Gemma", "Humans"))
            self.assertEqual(set(original), set(MATRIX_MODELS.values()))
            for name, trace in original.items():
                self.assertEqual((trace.x, trace.y), (changed[name].x, changed[name].y))
            self.assertEqual(original["Humans"].marker.color, ACCENT)
            self.assertEqual(original["GPT-5 mini"].marker.symbol, "diamond")
            self.assertEqual(changed["Humans"].marker.color, SECONDARY)
            self.assertEqual(changed["GPT-5 mini"].marker.color, CONTEXT)

    def test_inclusion_controls_figure_tables_and_empty_state(self):
        only = ["Gpt5_mini"]
        views = [valence_view("religion", "Humans", "Gpt5_mini", only),
                 emotion_view("politics", "Humans", "Gpt5_mini", "democrat", "republican", only)]
        for view in views:
            graph = next(c for c in view._traverse() if isinstance(c, dcc.Graph))
            self.assertEqual([trace.name for trace in graph.figure.data], ["GPT-5 mini"])
            self.assertEqual(graph.figure.data[0].marker.color, SECONDARY)
            bodies = [c for c in view._traverse() if isinstance(c, html.Tbody)]
            self.assertTrue(all(row.className == "selected-row" for tb in bodies for row in tb.children))
        with patch("wordviz.norm_views.valence_result") as compute:
            empty = valence_view("gender", "Humans", "Gemma", [])
            self.assertEqual(empty.role, "status")
            compute.assert_not_called()

    def test_unavailable_networks_keep_coverage_and_never_plot_zero(self):
        def result(domain, model, left, right):
            if model == "Gemma":
                return {"available_primes": [left], "rows": [], "reason": "Prime missing"}
            return {"available_primes": [left, right], "rows": [
                {"emotion": "anger", "effect": 0., "n": 7, "nonzero": 3, "p": 1.}]}
        with patch("wordviz.norm_views.emotion_result", side_effect=result):
            panel = emotion_view("politics", "Gemma", "Gemma", "democrat", "republican", ["Gemma", "Humans"])
        graph = next(c for c in panel._traverse() if isinstance(c, dcc.Graph))
        self.assertEqual([(t.name, t.y, t.marker.color) for t in graph.figure.data], [("Humans", (0.,), CONTEXT)])
        notes = [c.children for c in panel._traverse() if isinstance(c, html.Li) and isinstance(c.children, str)]
        self.assertTrue(any("Gemma" in note and "republican" in note for note in notes))

    def test_all_domain_models_have_valence_and_emotion_coverage(self):
        for domain, primes in DOMAIN_PRIMES.items():
            for model in MATRIX_MODELS:
                with self.subTest(domain=domain, model=model):
                    result = valence_result(domain, model)
                    available = result["available_primes"]
                    self.assertTrue(set(available).issubset(primes))
                    self.assertEqual([row["prime"] for row in result["rows"]], available)
                    self.assertGreater(result["joint"]["n"], 100)
                    self.assertEqual(result["joint"]["df"], len(available) - 1)
                    self.assertLessEqual(0, result["joint"]["partial_r2"])
                    self.assertLessEqual(result["joint"]["partial_r2"], 1)
                    emotion = emotion_result(domain, model, *available[:2])
                    self.assertEqual([r["emotion"] for r in emotion["rows"]], list(EMOTIONS))

    def test_political_reference_and_reversed_contrast(self):
        # Independently checked with SciPy's two-sided, tie-corrected Wilcoxon test.
        left = emotion_result("politics", "Gpt5_mini", "democrat", "republican")["rows"]
        right = emotion_result("politics", "Gpt5_mini", "republican", "democrat")["rows"]
        anger = next(r for r in left if r["emotion"] == "anger")
        self.assertEqual(anger["n"], 926)
        self.assertAlmostEqual(anger["effect"], 0.22143443276346592)
        self.assertAlmostEqual(anger["p"] / 1.602452253268919e-11, 1, places=6)
        for a, b in zip(left, right):
            self.assertEqual(a["p"], b["p"])
            self.assertEqual(a["effect"], -b["effect"])

    def test_identical_or_unknown_primes_return_empty(self):
        for pair in (("democrat", "democrat"), ("democrat", "absent")):
            self.assertFalse(emotion_result("politics", "Humans", *pair)["rows"])

    def test_valence_reference_against_r_lm(self):
        # R lm() and its nested equal-slope model on the matched input matrices.
        result = valence_result("religion", "Gemma")
        row = next(r for r in result["rows"] if r["prime"] == "christian")
        self.assertEqual(row["n"], 10855)
        self.assertAlmostEqual(row["slope"], 0.30699516123381626)
        self.assertAlmostEqual(result["joint"]["statistic"], 1047.0269771358226, places=5)
        self.assertAlmostEqual(result["joint"]["partial_r2"], 0.08801484555710992)
