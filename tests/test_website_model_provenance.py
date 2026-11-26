"""Keep public collection records tied to actual analysis keys and collection sources."""

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))

from wordviz.model_provenance import MODEL_PROVENANCE
from wordviz.model_options import ORIGINS, model_options, pinned_models
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS


class ModelProvenanceTests(unittest.TestCase):
    def test_pins_match_independent_table_and_preserve_selector_values(self):
        table = ROOT / "data/LDT_analyses/Statistical_tests/all_networks/corrLDT.csv"
        with table.open(newline="") as source:
            correlations = next(row for row in csv.DictReader(source) if row[""] == "r")
        ranked = sorted((float(correlations[key]), key.lower()) for key in MATRIX_MODELS
                        if key.lower() in ORIGINS and float(correlations[key]) < 0)
        strongest = {}
        for _, key in ranked:
            strongest.setdefault(ORIGINS[key], key)
        pins = list(strongest.values())[:5]
        self.assertEqual(list(pinned_models()), pins)
        for models in (GRAPH_MODELS, MATRIX_MODELS):
            options = model_options(models)
            values = [opt["value"] for opt in options]
            self.assertCountEqual(values, models.keys())
            self.assertEqual([v.lower() for v in values[:1 + len(pins)]], ["humans", *pins])
            for option in options:
                self.assertIn(models[option["value"]], option["search"])

    def test_served_dataset_labels_and_qualifications(self):
        # Plots and table highlighting identify series by their displayed names.
        self.assertEqual(len(set(GRAPH_MODELS.values())), len(GRAPH_MODELS))
        for key, label in GRAPH_MODELS.items():
            record = MODEL_PROVENANCE[key]
            self.assertEqual(record["label"], label)
            self.assertTrue(record["identifier"] and record["revision"] and record["sources"])
        for size in ("mini", "nano"):
            key = "gpt5_" + size
            config = (ROOT / "configs" / f"lwow_{key}.yaml").read_text()
            self.assertIn(f'model: "{MODEL_PROVENANCE[key]["identifier"]}"', config)

    def test_archived_versions_are_visible_and_searchable_under_the_original_keys(self):
        cases = [
            ("Gpt5_mini", "GPT-5 mini · 2025-08-07", ["gpt-5-mini-2025-08-07"]),
            ("Haiku", "Claude 3.5 Haiku · 2024-10-22", ["claude-3-5-haiku-latest", "claude-3-5-haiku-20241022"]),
            ("Mistral", "Mistral 7B Instruct v0.1 · Q4_K_M", ["mistral-7b-instruct-v0.1.Q4_K_M.gguf"]),
            ("Gemma", "Gemma · checkpoint unrecorded", []),
        ]
        for catalog in (GRAPH_MODELS, MATRIX_MODELS):
            options = model_options(catalog)
            for matrix_key, label, identifiers in cases:
                with self.subTest(matrix_key=matrix_key):
                    key = matrix_key if catalog is MATRIX_MODELS else matrix_key.lower()
                    self.assertEqual(catalog[key], label)
                    for query in [label, *identifiers]:
                        matches = [opt["value"] for opt in options if query.casefold() in opt["search"].casefold()]
                        self.assertEqual(matches, [key], query)

    def test_batched_runs_have_separate_records_and_traceable_files(self):
        run_root = ROOT / "LAG" / "outputs"
        directories = {path.parent.parent.name for path in run_root.glob("*/outputs/0.csv")}
        if not directories:
            self.skipTest("Raw run directories are outside this checkout.")
        represented = []
        config = (ROOT / "LAG/helper/configs.py").read_text()
        for key, record in MODEL_PROVENANCE.items():
            for label, url in record["sources"]:
                if label == "Run files":
                    folder = url.split("/LAG/outputs/")[1].removesuffix("/outputs")
                    represented.append(folder)
                    self.assertEqual(key in GRAPH_MODELS, (ROOT / "website/wordviz/resources/explore" / f"{key}.zip").is_file()
                                     and (ROOT / "website/wordviz/resources/paths" / f"{key}.json.gz").is_file())
                    self.assertIn(f'"{record["identifier"]}"', config)
                    self.assertEqual(folder.rsplit("_", 2)[0], record["identifier"].split("/")[-1])
        self.assertCountEqual(represented, directories)
