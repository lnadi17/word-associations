"""The browser's word-list library is validated on every read and write."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from wordviz.custom_domains import MAX_LISTS, MAX_PRIMES, MAX_TARGETS, get_domain, save_domain, validate_domain


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
