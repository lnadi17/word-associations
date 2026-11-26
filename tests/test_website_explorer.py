"""Validate ranking and directional semantics against the released cleaned-trial indexes."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.explore import _rank, membership_options, retained_option
from wordviz.explore_data import MODELS, neighbors, vocabulary


class ExplorerTests(unittest.TestCase):
    def test_direction_and_merged_weight(self):
        self.assertEqual(neighbors("humans", "cat", "forward").counts["dog"], 54)
        self.assertEqual(neighbors("gpt5_mini", "cat", "forward").counts["purr"], 32)
        self.assertEqual(neighbors("gpt5_mini", "cat", "backward").counts["purr"], 100)
        self.assertEqual(neighbors("gpt5_mini", "cat", "merged").counts["purr"], 100)

    def test_membership_uses_all_neighbors(self):
        series = [neighbors(model, "cat", "forward") for model in ("humans", "gpt5_mini")]
        totals = [s.total for s in series]
        left, right = (set(s.counts) for s in series)
        for selection, expected in (("shared", left & right), ("only-0", left - right), ("only-1", right - left)):
            ranked, _ = _rank(series, totals, selection, "share")
            self.assertEqual(set(ranked), expected)
        self.assertEqual(_rank(series, totals, "all", "profile-0")[0][0], "dog")
        self.assertEqual(_rank(series, totals, "all", "profile-1")[0][0], "whisker")

    def test_each_packaged_model_has_searchable_data(self):
        for model in MODELS:
            with self.subTest(model=model):
                self.assertIn("cat", vocabulary(model))
                self.assertTrue(neighbors(model, "cat", "forward").counts)

    def test_dense_graph_has_a_height_and_preserves_full_node_words(self):
        from wordviz.explore import _graphs

        model = neighbors("humans", "cat", "backward")
        words = sorted(model.counts, key=len, reverse=True)[:30]
        panel = _graphs([model], [model.total], words, "backward").children[0]
        chart = panel.children[1]
        self.assertIn("px", chart.style["height"])
        nodes = chart.figure.data[0]
        self.assertEqual([row[0] for row in nodes.customdata], words)
        self.assertEqual(len(set(zip(nodes.x, nodes.y))), len(words))
        self.assertTrue(any(label.endswith("…") for label in nodes.text))

    def test_filter_follows_its_network_when_indices_change(self):
        previous = membership_options(["cat", "dog"], "humans", "gemma")
        current = membership_options(["dog"], "humans", "gemma")
        self.assertEqual(retained_option(current, previous, "shared"), "shared")
        self.assertEqual(retained_option(current, previous, "only-3"), "only-1")
        self.assertEqual(retained_option(current, previous, "only-0"), "all")

    def test_column_order_preserves_comparison_direction_and_filter_identity(self):
        from wordviz.explore import render

        original, comparison = "humans", "gpt5_mini"
        left, right = [neighbors(m, "cat", "forward") for m in (original, comparison)]
        options = membership_options(["cat"], original, comparison)
        self.assertEqual([(o["label"], o["value"]) for o in options if o["value"].startswith("only-")],
                         [(f"Only {right.label}", "only-1"), (f"Only {left.label}", "only-0")])
        view = render(["cat"], original, comparison, "forward", "compact", "all", "share", 10)
        caption, head, body = view[-1].children.children
        self.assertEqual([cell.children for cell in head.children.children],
                         ["Word", right.label, left.label, "Δ share"])
        self.assertIn(f"{right.label} minus {left.label}", caption.children)
        cells = body.children[0].children
        word = cells[0].children
        for cell, profile in zip(cells[1:3], (right, left)):
            self.assertEqual(cell.children.split(" · ")[0], f"{profile.counts.get(word, 0):,}")
        expected = right.counts.get(word, 0) / right.total - left.counts.get(word, 0) / left.total
        self.assertEqual(cells[3].children, f"{100 * expected:+.1f} pp")


if __name__ == "__main__":
    unittest.main()
