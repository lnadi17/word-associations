"""Regression checks for coverage and weighted activation on new networks."""
import sys
import unittest
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_social_store import spread_graph
from run_fa_statistical_tests import activation_dict


class ActivationPipelineTests(unittest.TestCase):
    def test_weighted_steps_and_missing_prime(self):
        graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        graph.vs["name"] = ["a", "b", "c"]
        graph.es["weight"] = [1, 3]
        levels, steps = spread_graph(graph, ["a", "absent", "c"])
        self.assertEqual(steps, 4)
        self.assertEqual(levels.columns.tolist(), ["a", "c"])
        transition = np.array([[.5, .125, 0], [.5, .5, .5], [0, .375, .5]])
        expected = 3 * np.linalg.matrix_power(transition, 4)[:, [0, 2]]
        np.testing.assert_allclose(levels.to_numpy(), expected)

    def test_statistics_require_present_words_but_retain_genuine_zero(self):
        frame = pd.DataFrame({"p": [0., 1.], "q": [1., .5], "absent": [0., 0.]},
                             index=["target", "second"])
        kept = ("target", "p", "q")
        self.assertEqual(activation_dict(frame, [kept, ("target", "p", "absent"),
                                                 ("target", "unlisted", "q"), ("missing", "p", "q")]),
                         {kept: (0., 1., -1.)})
