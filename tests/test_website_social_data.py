"""Verify that the published node lookups preserve each domain's normalization."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz.experiment_data import genders
from wordviz.models import MATRIX_MODELS
from wordviz.social_data import domain_info, domain_suggest, domain_values
from wordviz.social_domains import DOMAIN_PRIMES


class SocialActivationTests(unittest.TestCase):
    def test_domain_norms_and_coverage(self):
        for domain in DOMAIN_PRIMES:
            values = domain_values(domain, ["cat", "outside-test-vocabulary"], MATRIX_MODELS)
            for model, rows in values.items():
                with self.subTest(domain=domain, model=model):
                    info = domain_info(domain, model)
                    self.assertEqual(set(info["primes"]) | set(info["missing_primes"]), set(DOMAIN_PRIMES[domain]))
                    self.assertEqual(set(rows["cat"]), set(info["primes"]))
                    self.assertAlmostEqual(math.sqrt(sum(x*x for x in rows["cat"].values())), 1.)
                    self.assertNotIn("outside-test-vocabulary", rows)

    def test_gender_lookup_uses_original_archive(self):
        self.assertEqual(domain_values("gender", ["kind", "strong"], MATRIX_MODELS),
                         genders(["kind", "strong"]))

    def test_autocomplete_uses_selected_graph_nodes(self):
        self.assertIn("cat", domain_suggest("politics", "cat", ["Humans"]))
        self.assertEqual(domain_suggest("politics", "", ["Humans"]), [])
        self.assertEqual(domain_suggest("politics", "cat", []), [])
