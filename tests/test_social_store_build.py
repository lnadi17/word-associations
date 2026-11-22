"""Research-environment check of the sparse update against R spreadr output."""
import sys
import unittest
from pathlib import Path

import igraph as ig
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_social_store import spread_graph


class SparseSpreadTests(unittest.TestCase):
    def test_weighted_graph_agrees_with_spreadr(self):
        graph = ig.Graph.TupleList([("a", "b", 3), ("b", "c", 1), ("c", "d", 4)], weights=True)
        levels, steps = spread_graph(graph, ["a", "c", "absent"])
        self.assertEqual(steps, 6)
        self.assertEqual(list(levels.columns), ["a", "c"])
        # R spreadr defaults, initial activation 4, 2 * unweighted diameter.
        np.testing.assert_allclose(levels.loc[list("abcd"), "a"], [1.36, 1.63, .64, .37], atol=1e-14)
        np.testing.assert_allclose(levels.loc[list("abcd"), "c"], [.384, .622, 1.616, 1.378], atol=1e-14)
