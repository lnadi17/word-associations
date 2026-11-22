"""Compare the on-demand weighted walk with archived values and graph invariants."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.custom_activation import custom_info, custom_values, raw_columns
from wordviz.models import MATRIX_MODELS
from wordviz.social_data import domain_values
from wordviz.social_domains import DOMAIN_PRIMES


class CustomActivationTests(unittest.TestCase):
    def test_arbitrary_domain_matches_published_normalization_on_each_snapshot(self):
        words = ["cat", "doctor", "honest", "school", "garden"]
        primes = DOMAIN_PRIMES["religion"]
        saved = domain_values("religion", words, MATRIX_MODELS)
        for model in MATRIX_MODELS:
            with self.subTest(model=model):
                calculated = custom_values(primes, words, [model])[model]
                self.assertEqual(calculated.keys(), saved[model].keys())
                for word, expected in saved[model].items():
                    self.assertEqual(calculated[word].keys(), expected.keys())
                    np.testing.assert_allclose(list(calculated[word].values()), list(expected.values()),
                                               atol=1e-12, rtol=1e-12)

    def test_novel_primes_retain_mass_and_normalize_in_user_order(self):
        model = "Humans"
        primes = ["cat", "dog", "doctor", "test-word-not-in-graph", "dog"]
        words, available, raw = raw_columns(model, primes)
        self.assertEqual(available, ("cat", "dog", "doctor"))
        np.testing.assert_allclose(raw.sum(axis=0), len(words), rtol=1e-12)
        targets = ["mouse", "patient", "test-word-not-in-graph"]
        actual = custom_values(primes, targets, [model])[model]
        np.testing.assert_allclose(sum(x*x for x in actual["mouse"].values()), 1)
        scaled = raw / np.linalg.norm(raw, axis=0)
        for word in targets[:2]:
            row = scaled[words.index(word)]
            np.testing.assert_allclose(list(actual[word].values()), row / np.linalg.norm(row))
        self.assertNotIn(targets[-1], actual)
        info = custom_info(primes, model)
        self.assertEqual(info["missing_primes"], [primes[3]])
        self.assertEqual(info["primes"], list(available))

    def test_empty_and_oversize_inputs(self):
        words, available, raw = raw_columns("Gemma", ["unknown-prime-for-testing"])
        self.assertEqual(raw.shape, (len(words), 0))
        self.assertEqual(available, ())
        self.assertEqual(custom_values([], ["cat"], ["Gemma"]), {"Gemma": {"cat": {}}})
        self.assertEqual(custom_values(["cat"], [], ["Gemma"]), {"Gemma": {}})
        with self.assertRaises(ValueError):
            raw_columns("Gemma", [str(i) for i in range(13)])
        with self.assertRaises(ValueError):
            raw_columns("unknown-model", ["cat"])
