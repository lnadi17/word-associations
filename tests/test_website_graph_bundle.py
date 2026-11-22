"""Check the actual lazy-loaded graph asset used by the browser."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

from app import app


class GraphBundleTests(unittest.TestCase):
    def test_graph_route_guards_queued_updates_after_unmount(self):
        with app.test_client() as client:
            path = "/_dash-component-suites/dash/dcc/async-graph.js"
            response = client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"if (!d) return Promise.resolve();", response.data)
            self.assertEqual(client.get(path, headers={"If-None-Match": response.headers["ETag"]}).status_code, 304)
