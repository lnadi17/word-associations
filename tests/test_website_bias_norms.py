"""Regression checks for the published summary values and contrast direction."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.bias_norms import EMOTIONS, emotion_result, valence_result
from wordviz.models import MATRIX_MODELS
from wordviz.social_domains import DOMAIN_PRIMES


class BiasNormsTests(unittest.TestCase):
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
