"""Regression checks for the published summary values and contrast direction."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from dash import dcc, html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.bias_norms import EMOTIONS, emotion_result, valence_result
from wordviz.experiments import format_p
from wordviz.models import MATRIX_MODELS
from wordviz.norm_views import emotion_view, valence_view
from wordviz.social_domains import DOMAIN_PRIMES


class BiasNormsTests(unittest.TestCase):
    def scatter(self, view):
        chart, = [child for child in view._traverse() if isinstance(child, dcc.Graph)]
        self.assertTrue(chart.figure.data)
        self.assertTrue(all(trace.type == "scatter" and trace.mode == "markers" for trace in chart.figure.data))
        for trace in chart.figure.data:
            self.assertEqual(trace.hoverinfo, "none")
            self.assertIsNone(trace.hovertemplate)
        return {trace.name: trace for trace in chart.figure.data}

    def heatmap(self, view):
        chart, = [c for c in view._traverse() if isinstance(c, dcc.Graph)]
        matrix, = chart.figure.data
        self.assertEqual(matrix.type, "heatmap")
        self.assertIs(chart.figure.layout.dragmode, False)
        self.assertEqual(matrix.hoverinfo, "none")
        self.assertIsNone(matrix.hovertemplate)
        return chart.figure, matrix

    def test_heatmaps_align_stored_values_and_only_restyle_highlighted_rows(self):
        def valence(a, b):
            return valence_view("politics", a, b)

        def emotion(a, b):
            return emotion_view("politics", a, b, "democrat", "republican")

        cases = [
            (valence, DOMAIN_PRIMES["politics"], "prime", "slope", lambda m: valence_result("politics", m)),
            (emotion, EMOTIONS, "emotion", "effect", lambda m: emotion_result("politics", m, "democrat", "republican")),
        ]
        for render, columns, category, measure, stored in cases:
            with self.subTest(measure=measure):
                original, matrix = self.heatmap(render("Humans", "Gpt5_mini"))
                changed, restyled = self.heatmap(render("Gemma", "Humans"))
                self.assertEqual(list(matrix.x), list(columns))
                self.assertEqual(list(matrix.y), list(MATRIX_MODELS.values()))
                for model, plotted in zip(MATRIX_MODELS, matrix.z):
                    estimates = {row[category]: row[measure] for row in stored(model)["rows"]}
                    self.assertEqual(list(plotted), [estimates.get(column) for column in columns])
                self.assertGreater(matrix.zmax, 0)
                self.assertEqual(matrix.zmin, -matrix.zmax)
                self.assertGreaterEqual(matrix.zmax, max(abs(v) for row in matrix.z for v in row if v is not None))
                self.assertEqual((matrix.x, matrix.y, matrix.z, matrix.zmin, matrix.zmax),
                                 (restyled.x, restyled.y, restyled.z, restyled.zmin, restyled.zmax))
                changed_labels = [model for model, before, after in zip(
                    MATRIX_MODELS, original.layout.yaxis.ticktext, changed.layout.yaxis.ticktext
                ) if before != after]
                self.assertCountEqual(changed_labels, ["Humans", "Gpt5_mini", "Gemma"])

    def test_valence_scatter_uses_table_estimates_and_stored_intervals(self):
        cohort = ["Humans", "Gemma", "Gpt5_mini"]

        def render(a, b, chart_view):
            return valence_view("religion", a, b, cohort, chart_view=chart_view)

        _, matrix = self.heatmap(render("Humans", "Gemma", "matrix"))
        points = self.scatter(render("Humans", "Gemma", "scatter"))
        swapped = self.scatter(render("Gemma", "Humans", "scatter"))
        self.assertCountEqual(points, matrix.y)
        self.assertCountEqual(points, [MATRIX_MODELS[m] for m in cohort])
        for model in cohort:
            name = MATRIX_MODELS[model]
            with self.subTest(model=name):
                trace = points[name]
                stored = {row["prime"]: row for row in valence_result("religion", model)["rows"]}
                estimates = dict(zip(matrix.x, matrix.z[list(matrix.y).index(name)]))
                categories = [data[0] for data in trace.customdata]
                self.assertCountEqual(categories, [prime for prime, value in estimates.items() if value is not None])
                self.assertEqual(list(trace.y), [estimates[prime] for prime in categories])
                tooltip_row = dict(zip(matrix.x, matrix.customdata[list(matrix.y).index(name)]))
                for category, tooltip in zip(categories, trace.customdata):
                    self.assertEqual(tooltip, tooltip_row[category])
                self.assertEqual(list(trace.error_y.array),
                                 [stored[p]["ci_high"] - stored[p]["slope"] for p in categories])
                self.assertEqual(list(trace.error_y.arrayminus),
                                 [stored[p]["slope"] - stored[p]["ci_low"] for p in categories])
                self.assertEqual((trace.x, trace.y), (swapped[name].x, swapped[name].y))
        self.assertEqual(points[MATRIX_MODELS["Gemma"]].marker.symbol, "diamond")
        self.assertEqual(swapped[MATRIX_MODELS["Humans"]].marker.symbol, "diamond")

    def test_emotion_scatter_reverses_same_estimates_as_table(self):
        cohort = ["Gpt5_mini"]

        def render(left, right, chart_view):
            return emotion_view("politics", "Humans", "Gpt5_mini", left, right, cohort, chart_view=chart_view)

        _, matrix = self.heatmap(render("democrat", "republican", "matrix"))
        forward = self.scatter(render("democrat", "republican", "scatter"))
        backward = self.scatter(render("republican", "democrat", "scatter"))
        name, = matrix.y
        self.assertEqual(list(forward), [name])
        self.assertEqual([item[0] for item in forward[name].customdata], list(matrix.x))
        self.assertEqual(list(forward[name].y), list(matrix.z[0]))
        self.assertEqual(list(forward[name].customdata), list(matrix.customdata[0]))
        self.assertEqual(backward[name].x, forward[name].x)
        self.assertEqual(list(backward[name].y), [-value for value in forward[name].y])
        self.assertIsNone(forward[name].error_y.array)
        anger = list(matrix.x).index("anger")
        for traces in (forward, backward):
            self.assertIn("democrat", traces[name].customdata[anger][3].lower())
            self.assertNotIn("republican", traces[name].customdata[anger][3].lower())
        self.assertIn("+0.22", forward[name].customdata[anger][4])
        self.assertIn("0.22", backward[name].customdata[anger][4])
        self.assertNotIn("+", backward[name].customdata[anger][4])

    def test_inclusion_controls_figure_tables_and_empty_state(self):
        only = ["Gpt5_mini"]
        views = [valence_view("religion", "Humans", "Gpt5_mini", only),
                 emotion_view("politics", "Humans", "Gpt5_mini", "democrat", "republican", only)]
        for view in views:
            _, matrix = self.heatmap(view)
            self.assertEqual(list(matrix.y), [MATRIX_MODELS["Gpt5_mini"]])
            self.assertEqual(len(matrix.z), 1)
            bodies = [c for c in view._traverse() if isinstance(c, html.Tbody)]
            self.assertTrue(all(row.className == "selected-row" for tb in bodies for row in tb.children))
        with patch("wordviz.norm_views.valence_result") as compute:
            empty = valence_view("gender", "Humans", "Gemma", [])
            self.assertEqual(empty.role, "status")
            compute.assert_not_called()

    def test_unavailable_networks_have_empty_cells_distinct_from_zero(self):
        def result(domain, model, left, right):
            if model == "Gemma":
                return {"available_primes": [left], "rows": [], "reason": "Prime missing"}
            return {"available_primes": [left, right], "rows": [
                {"emotion": "anger", "effect": 0., "n": 7, "nonzero": 3, "p": 1.}]}
        with patch("wordviz.norm_views.emotion_result", side_effect=result):
            view = emotion_view("politics", "Gemma", "Gemma", "democrat", "republican", ["Gemma", "Humans"])
            scatter = emotion_view("politics", "Gemma", "Gemma", "democrat", "republican",
                                   ["Gemma", "Humans"], chart_view="scatter")
        points = self.scatter(scatter)
        self.assertEqual(list(points), [MATRIX_MODELS["Humans"]])
        self.assertEqual(list(points[MATRIX_MODELS["Humans"]].y), [0.])
        _, matrix = self.heatmap(view)
        self.assertEqual(list(matrix.y), [MATRIX_MODELS["Gemma"], MATRIX_MODELS["Humans"]])
        self.assertEqual(list(matrix.z[1]), [0. if e == "anger" else None for e in EMOTIONS])
        self.assertEqual(list(matrix.z[0]), [None] * len(EMOTIONS))
        self.assertEqual(list(matrix.customdata[0]), [None] * len(EMOTIONS))
        for category, tooltip in zip(EMOTIONS, matrix.customdata[1]):
            if category == "anger":
                self.assertEqual(tooltip, points[MATRIX_MODELS["Humans"]].customdata[0])
                self.assertEqual(tooltip[4], "Signed-rank effect 0")
                self.assertIn("7", tooltip[5])
                self.assertIn("3 nonzero", tooltip[5])
            else:
                self.assertIsNone(tooltip)
        self.assertGreater(matrix.zmax, 0)
        notes = [c.children for c in view._traverse() if isinstance(c, html.Li) and isinstance(c.children, str)]
        self.assertTrue(any(MATRIX_MODELS["Gemma"] in note and "republican" in note for note in notes))
        with patch("wordviz.norm_views.emotion_result", side_effect=result):
            for mode in ("matrix", "scatter"):
                no_data = emotion_view("politics", "Gemma", None, "democrat", "republican", ["Gemma"], chart_view=mode)
                self.assertFalse(any(isinstance(c, dcc.Graph) for c in no_data._traverse()))
                self.assertTrue(any(getattr(c, "role", None) == "status" for c in no_data._traverse()))

    def test_tooltips_identify_the_network_and_summarize_detailed_statistics(self):
        valence = valence_view("religion", None, None, ["Gemma"])
        emotion = emotion_view("politics", None, None, "democrat", "republican", ["Gpt5_mini"])
        for view, label, expected in [
            (valence, "Valence coefficients and intervals",
             [MATRIX_MODELS["Gemma"], "christian", "10,855", "0.307", "0.282", "0.332", format_p(1.43706483969860e-124)]),
            (emotion, "Emotion word set tests",
             [MATRIX_MODELS["Gpt5_mini"], "anger", 926, 926, "0.221", format_p(1.602452253268919e-11)]),
        ]:
            with self.subTest(label=label):
                _, matrix = self.heatmap(view)
                col = list(matrix.x).index(expected[1])
                category, model, category_label, summary, metric, evidence = matrix.customdata[0][col]
                self.assertEqual((category, model), (expected[1], expected[0]))
                self.assertIn(category, category_label.lower())
                self.assertIn("p < 0.001", evidence)
                self.assertIn(str(expected[2]), evidence)
                if label == "Valence coefficients and intervals":
                    self.assertIn("higher", summary.lower())
                    self.assertIn("pleasant", summary.lower())
                    self.assertEqual(metric, "Slope +0.31 · 95% CI [0.28, 0.33]")
                else:
                    for prime in ("democrat", "republican"):
                        self.assertIn(prime, category_label.lower())
                    self.assertIn("democrat", summary.lower())
                    self.assertEqual(metric, "Signed-rank effect +0.22")
                    self.assertNotIn("nonzero", evidence)
                table = next(c for c in view._traverse() if getattr(c, "aria-label", None) == label)
                row = next(c for c in table._traverse() if isinstance(c, html.Tr) and
                           c.children[0].children == expected[0] and c.children[1].children == expected[1])
                self.assertEqual([cell.children for cell in row.children], expected)

    def test_small_observed_slopes_retain_their_direction_in_tooltips(self):
        models = ["Gpt5_mini", "kona2_12b"]
        _, matrix = self.heatmap(valence_view("politics", None, None, models))
        for model, prime, direction, expected in [
            ("Gpt5_mini", "republican", "lower", "−0.01 < slope < 0"),
            ("kona2_12b", "democrat", "higher", "0 < slope < 0.01"),
        ]:
            with self.subTest(model=model):
                row, col = list(matrix.y).index(MATRIX_MODELS[model]), list(matrix.x).index(prime)
                self.assertTrue(0 < abs(matrix.z[row][col]) < .005)
                _, _, _, summary, metric, _ = matrix.customdata[row][col]
                self.assertIn(direction, summary.lower())
                self.assertEqual(metric.split(" · ", 1)[0], expected)

    def test_valence_color_scale_extends_for_large_slopes(self):
        result = {"available_primes": ["christian"], "joint": None, "rows": [{
            "prime": "christian", "n": 30, "slope": -2.75, "ci_low": -3.4, "ci_high": -2.1, "p": .003,
        }]}
        with patch("wordviz.norm_views.valence_result", return_value=result):
            _, matrix = self.heatmap(valence_view("religion", None, None, ["Humans"]))
        self.assertLessEqual(matrix.zmin, -2.75)
        self.assertEqual(-matrix.zmin, matrix.zmax)

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
