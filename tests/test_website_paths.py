"""Checks on the versioned path stores used in the bias paper presets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz.path_data import MODEL_NAMES, stream


class PathTests(unittest.TestCase):
    def test_shortest_route_counts(self):
        for model, hops, routes in (("humans", 3, 53), ("haiku", 4, 6)):
            with self.subTest(model=model):
                result = stream(model, "feminine", "forceful")
                self.assertEqual((result["hops"], result["routes"]), (hops, routes))
                self.assertEqual(result["layers"][0], ["feminine"])
                self.assertEqual(result["layers"][-1], ["forceful"])

    def test_all_graphs_and_missing_endpoints(self):
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                self.assertEqual(stream(model, "feminine", "feminine")["hops"], 0)
                self.assertEqual(stream(model, "feminine", "absent word fixture")["missing"],
                                 ["absent word fixture"])
