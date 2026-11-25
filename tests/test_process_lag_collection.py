import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lwow.fa_networks import build_filtered_graph
from scripts.process_lag_collection import filtered_graph, load_trials, parse_list


class LAGCollectionTests(unittest.TestCase):
    def test_response_format(self):
        self.assertEqual(parse_list(" Output: black cat, milk, mouse\nExplanation...")[0],
                         ("black cat", "milk", "mouse"))
        self.assertEqual(parse_list("<think>reasoning, not, associations</think>\nred, white, blue")[0],
                         ("red", "white", "blue"))
        for text in ("", "dog cat cat", "house, with an entire essay explaining something, key",
                     "I thought about this:\ncat, dog, house", "<think>here, are, words"):
            self.assertFalse(parse_list(text)[0])

    def test_trial_coverage_and_finish_reasons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            outputs.mkdir()
            path = outputs / "0.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["cue", "variation_index", "response", "finish_reason"])
                for i in range(100):
                    writer.writerow(["sea", i, "water,beach,sun", "length" if i == 99 else "stop"])
            df, report = load_trials(root, expected_cues=1)
            self.assertEqual(len(df), 100)
            self.assertEqual(report["parse_outcomes"], {"accepted": 99, "finish_length": 1})
            self.assertEqual(df.iloc[-1].tolist(), ["sea", "", "", ""])
            with self.assertRaisesRegex(ValueError, "Expected 2"):
                load_trials(root, expected_cues=2)
            with path.open("a") as f:
                f.write('sea,0,"water,beach,sun",stop\n')
            with self.assertRaisesRegex(ValueError, "repeated"):
                load_trials(root, expected_cues=1)

    def test_graph_weights_match_existing_pipeline(self):
        df = pd.DataFrame([("cat", "dog", "animal", ""), ("cat", "animal", "dog", "pet"),
                           ("dog", "cat", "animal", ""), ("pet", "solo", "", "")],
                          columns=["cue", "R1", "R2", "R3"])
        words = {"cat", "dog", "animal", "pet"}
        with patch("scripts.process_lag_collection.get_wn_word_set", return_value=words):
            actual = filtered_graph(df)
        expected = build_filtered_graph(df, wn_word_set=words)
        def weights(g):
            return {frozenset((u, v)): w for u, v, w in g.edges(data="weight")}
        self.assertEqual(weights(actual), weights(expected))
