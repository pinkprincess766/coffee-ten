import hashlib
import json
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


class ResultsManifestTests(unittest.TestCase):
    def test_every_manifest_entry_matches_its_file(self):
        manifests = sorted(RESULTS.glob("*-manifest.json"))
        self.assertTrue(manifests, "no *-manifest.json in bench/results")
        for manifest in manifests:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            for name, entry in entries.items():
                with self.subTest(manifest=manifest.name, file=name):
                    data = (RESULTS / name).read_bytes()
                    self.assertEqual(entry["bytes"], len(data))
                    self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
