"""The browser's word-list library is validated on every read and write."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.custom_domains import MAX_LISTS, MAX_PRIMES, MAX_TARGETS, get_domain, save_domain, validate_domain
from wordviz.paired_presets import paired_presets, templates


class CustomDomainsTests(unittest.TestCase):
    def save(self):
        return save_domain({}, None, " Life   stages ", "Baby, OLD\nbaby",
                           [" Wellbeing ", "Places"], ["Healthy, tired\nhealthy", "nursing home; school"])

    def test_word_order_case_and_stable_edit_id(self):
        library, key = self.save()
        original = get_domain(library, key)
        self.assertEqual(original["name"], "Life stages")
        self.assertEqual(original["primes"], ["baby", "old"])
        self.assertEqual(original["lists"], {"Wellbeing": ["healthy", "tired"], "Places": ["nursing home", "school"]})
        updated, same_key = save_domain(library, original, "Age", "child, adult", ["Roles"], ["teacher"])
        self.assertEqual(key, same_key)
        self.assertEqual(get_domain(updated, key)["lists"], {"Roles": ["teacher"]})
        with self.assertRaisesRegex(ValueError, "another tab"):
            save_domain(updated, original, "Age", "baby, old", ["Roles"], ["friend"])

    def test_save_primes_alone_then_add_and_remove_last_list(self):
        library, key = save_domain({}, None, "Age", "old, young", [], [])
        for titles, words, expected in [
            (["Traits", " "], ["wise", "\n"], {"Traits": ["wise"]}),
            ([], [], {}),
            ([""], [""], {}),
        ]:
            original = get_domain(library, key)
            library, same_key = save_domain(library, original, "Age", "old, young", titles, words)
            self.assertEqual(key, same_key)
            self.assertEqual(get_domain(library, key)["lists"], expected)
        record = library[key].copy()
        del record["lists"]
        self.assertEqual(validate_domain(record), library[key])

    def test_invalid_lists_names_and_limits(self):
        library, key = self.save()
        for name, primes, titles, words in [
            ("Gender", "old, young", ["Targets"], ["wise"]),
            ("LIFE stages", "old, young", ["Targets"], ["wise"]),
            ("Age", "old, OLD", ["Targets"], ["wise"]),
            ("Age", ",".join(map(str, range(MAX_PRIMES + 1))), ["Targets"], ["wise"]),
            ("Age", "young, old", ["Traits", "TRAITS"], ["wise", "kind"]),
            ("Age", "young, old", ["Targets"], [""]),
            ("Age", "young, old", [" "], ["wise"]),
            ("Age", "young, old", [""] * (MAX_LISTS + 1), [""] * (MAX_LISTS + 1)),
            ("Age", "young, old", ["Targets"], [",".join(map(str, range(MAX_TARGETS + 1)))]),
        ]:
            with self.subTest(name=name, primes=primes, titles=titles):
                with self.assertRaises(ValueError):
                    save_domain(library, None, name, primes, titles, words)
        self.assertEqual(len(library), 1)
        self.assertEqual(library[key]["name"], "Life stages")

    def test_malformed_browser_data_is_ignored(self):
        library, key = self.save()
        record = library[key]
        for value in [None, [], {}, {**record, "id": "gender"}, {**record, "lists": []},
                      {**record, "lists": {"Words": ["x" * 1000]}}, {**record, "primes": "young, old"}]:
            self.assertIsNone(validate_domain(value))
        self.assertIsNone(get_domain({"custom-" + "f" * 32: record}, "custom-" + "f" * 32))
        self.assertIsNone(get_domain(library, []))

    def test_assigning_expected_primes_preserves_order_and_distinguishes_unassigned_lists(self):
        titles, words = ["Experience", "Energy", "Places"], ["wise, careful", "active, quick", "school"]
        library, key = save_domain({}, None, "Age", "old, young", titles, words, ["old", "YOUNG", ""])
        record = get_domain(library, key)
        self.assertEqual(record["list_primes"], {"Experience": "old", "Energy": "young"})
        pair, = paired_presets(record).values()
        self.assertEqual(pair["primes"], ("old", "young"))
        self.assertEqual(pair["groups"], (["wise", "careful"], ["active", "quick"]))
        self.assertIsNone(validate_domain({**record, "list_primes": {"Absent": "young"}}))
        with self.assertRaisesRegex(ValueError, "expected prime"):
            save_domain(library, record, "Age", "old, young", titles, words, ["missing", "young", ""])

    def test_matching_primes_offer_a_study_and_edited_template_keeps_its_source(self):
        key, study = next((k, s) for k, s in templates().items() if s.primes == ("christian", "muslim"))
        primes = ", ".join(study.primes)
        lib, record_id = save_domain({}, None, "Faith", primes, [], [])
        suggested = next(iter(paired_presets(lib[record_id]).values()))
        self.assertEqual(suggested["groups"], study.targets)
        self.assertEqual(suggested["url"], study.url)
        words = [", ".join(study.targets[0][1:]), ", ".join(study.targets[1])]
        lib, same_id = save_domain(lib, lib[record_id], "Faith", primes, list(study.labels),
                                   words, list(study.primes), key)
        self.assertEqual(same_id, record_id)
        edited = next(iter(paired_presets(get_domain(lib, record_id)).values()))
        self.assertIn("Edited target sets", edited["context"])
        self.assertEqual(edited["url"], study.url)

    def test_all_study_sets_have_two_substantial_distinct_sides(self):
        for name, study in templates().items():
            with self.subTest(study=name):
                left, right = map(set, study.targets)
                self.assertFalse(left & right)
                self.assertFalse((left | right) & set(study.primes))
                self.assertEqual(tuple(map(len, study.targets)), (len(left), len(right)))
                self.assertGreaterEqual(min(len(left), len(right)), 20)
                self.assertTrue(study.url and study.citation and study.context)
