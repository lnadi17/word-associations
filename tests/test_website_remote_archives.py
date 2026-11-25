"""Cold and warm access to the model indexes hosted outside the Vercel function."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz import explore_archives as archives


class VerifiedExploreArchives(unittest.TestCase):
    def test_concurrent_cold_reads_and_warm_reuse(self):
        data = b"archived response index"
        digest = hashlib.sha256(data).hexdigest()
        key = ("dataset", "a" * 40, digest, len(data))
        with tempfile.TemporaryDirectory() as tmp, patch.object(archives, "CACHE", Path(tmp)), \
             patch.object(archives, "urlopen", side_effect=lambda *a, **k: io.BytesIO(data)) as download:
            archives._cached_archive.cache_clear()
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: archives._cached_archive(*key), range(8)))
            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(paths[0].read_bytes(), data)
            download.assert_called_once()
            archives._cached_archive.cache_clear()
            self.assertEqual(archives._cached_archive(*key), paths[0])
            download.assert_called_once()
        archives._cached_archive.cache_clear()

    def test_failed_download_leaves_no_cache_and_can_retry(self):
        data = b"index"
        key = ("dataset", "b" * 40, hashlib.sha256(data).hexdigest(), len(data))
        with tempfile.TemporaryDirectory() as tmp, patch.object(archives, "CACHE", Path(tmp)), \
             patch.object(archives, "urlopen", side_effect=[io.BytesIO(b"other"), io.BytesIO(data)]):
            archives._cached_archive.cache_clear()
            with self.assertRaisesRegex(ValueError, "checksum"):
                archives._cached_archive(*key)
            self.assertFalse(list(Path(tmp).iterdir()))
            self.assertEqual(archives._cached_archive(*key).read_bytes(), data)
        archives._cached_archive.cache_clear()
