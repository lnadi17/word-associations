"""Regressions for custom social-domain statistics and server-only norm inputs."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz import custom_norms, norm_lexicons
from wordviz.bias_norms import emotion_result, valence_result


class CustomNormTests(unittest.TestCase):
    def setUp(self):
        for cache in (custom_norms.custom_emotion, custom_norms.custom_valence,
                      custom_norms._aligned_norms, norm_lexicons.load_lexicons):
            cache.cache_clear()
            self.addCleanup(cache.cache_clear)

    def test_full_graph_normalization_and_model_coverage(self):
        words = tuple("abcdefgh")
        matrix = np.array([[2, 5], [0, 1], [6, 2], [1, 1], [3, 0], [.5, 8], [.3, 1], [1, 5]])
        primes = ("social", "solitary")
        ratings = dict(zip(words, [.7, .8, .1, .4, .2, .8, np.nan, .4]))
        sets = {emotion: set("bcef") for emotion in norm_lexicons.EMOTIONS}
        domain = {"primes": [*primes, "unseen"]}
        def columns(model, requested):
            present = tuple(p for p in requested if p in primes)
            return words, present, matrix[:, [primes.index(p) for p in present]]
        with patch.object(custom_norms, "raw_columns", side_effect=columns), \
             patch.object(custom_norms, "load_lexicons", return_value=(ratings, sets)):
            slopes = valence_result(domain, "Humans")
            self.assertIn("unseen", slopes["reason"])
            self.assertEqual(slopes["rows"][0]["n"], 7)
            self.assertAlmostEqual(slopes["rows"][0]["slope"], -.5929653352724618)
            self.assertAlmostEqual(slopes["rows"][0]["ci_low"], -.9871474906695448)
            self.assertAlmostEqual(slopes["joint"]["p"], .0002451886241950286)
            forward = emotion_result(domain, "Humans", *primes)["rows"][0]
            reverse = emotion_result(domain, "Humans", *reversed(primes))["rows"][0]
            self.assertEqual(forward["n"], 4)
            self.assertAlmostEqual(forward["effect"], -.09205746178983235)
            self.assertAlmostEqual(reverse["effect"], -forward["effect"])
            self.assertEqual(forward["p"], reverse["p"])
            self.assertFalse(emotion_result(domain, "Humans", "social", "unseen")["rows"])
            self.assertFalse(valence_result({"primes": ["social", "unseen"]}, "Humans")["rows"])

    def test_runtime_pack_can_be_prepared_offline_and_verifies_source_hashes(self):
        sample = {"vad": b"word\t0.25\t0.1\t0.1\n", "emotion": b"word\tfear\t1\n"}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources, archives = {}, {}
            for key, data in sample.items():
                name, member = key + ".zip", key + ".txt"
                sources[key] = (name, member, hashlib.sha256(data).hexdigest())
                archives[key] = base / name
                with ZipFile(archives[key], "w") as archive:
                    archive.writestr(member, data)
            dest = base / "server" / "norms.zip"
            with patch.object(norm_lexicons, "SOURCES", sources), patch.object(norm_lexicons, "SERVER_PACK", dest):
                norm_lexicons.prepare_pack(dest, archives)
                ratings, membership = norm_lexicons.load_lexicons()
                self.assertEqual(ratings, {"word": .25})
                self.assertEqual(membership["fear"], {"word"})
                with ZipFile(archives["vad"], "w") as archive:
                    archive.writestr("vad.txt", b"changed version")
                with self.assertRaises(norm_lexicons.NormSourceError):
                    norm_lexicons.prepare_pack(dest, archives)
                norm_lexicons.load_lexicons.cache_clear()
                self.assertEqual(norm_lexicons.load_lexicons()[0], ratings)
