import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ct_validate", ROOT / "bench/validate.py")
schema = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schema)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.row = json.loads((ROOT / "crates/coffee-ten-bench/tests/fixtures/report.jsonl").read_text().splitlines()[1])
        self.peak = json.loads((ROOT / "bench/results/2026-10-28-adcf09c-peak.jsonl").read_text())

    def test_current_harness_search_v2_and_peak(self):
        schema.validate(self.row)
        harness = copy.deepcopy(self.row)
        del harness["group"]
        schema.validate(harness)
        schema.validate(self.peak)
        for cache in (None, dict(l1_bytes=65536, l2_bytes=4194304, line_bytes=64)):
            v2 = copy.deepcopy(harness)
            v2["protocol"].update(space="v2", cache=cache, tile_factors=[2, 4, 8, 16])
            schema.validate(v2)

    def test_every_missing_or_extra_field_is_rejected(self):
        for original in [self.row, self.peak]:
            for field in original:
                if field == "group":  # optional on harness records
                    continue
                bad = copy.deepcopy(original)
                del bad[field]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    schema.validate(bad)
            bad = dict(original, unexpected=1)
            with self.assertRaises(ValueError):
                schema.validate(bad)
        for container in [("protocol",), ("rounds", 0), ("reference_compilation",)]:
            target = self.row
            for key in container:
                target = target[key]
            for field in list(target) + ["unexpected"]:
                bad = copy.deepcopy(self.row)
                nested = bad
                for key in container:
                    nested = nested[key]
                if field in nested:
                    del nested[field]
                else:
                    nested[field] = 1
                with self.subTest(container=container, field=field), self.assertRaises(ValueError):
                    schema.validate(bad)

    def test_types_consistency_and_corrupt_rounds(self):
        for field, value in [("n", True), ("verified", 1), ("dirty", 0), ("samples", 0),
                             ("median", None), ("p25", float("nan")), ("median", 17),
                             ("measurement_ns", 0), ("compilation_ns", -1),
                             ("seed", 99), ("cpu", "changed"), ("group", "unknown")]:
            bad = copy.deepcopy(self.row)
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                schema.validate(bad)

        for field, value in [("round_index", True), ("round_index", 1),
                             ("elapsed_ns", 0), ("iterations", -1), ("gflops", float("inf"))]:
            bad = copy.deepcopy(self.row)
            bad["rounds"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                schema.validate(bad)

    def test_rounds_must_meet_the_declared_minimum_duration(self):
        bad = copy.deepcopy(self.row)
        bad["protocol"]["min_sample_ms"] = max(
            sample["elapsed_ns"] for sample in bad["rounds"]
        ) // 1_000_000 + 1
        with self.assertRaisesRegex(ValueError, "minimum"):
            schema.validate(bad)

    def test_strict_json_and_file_line_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps(self.row) + "\n" + json.dumps(self.peak) + "\n")
            self.assertEqual(schema.validate_file(path), 2)
            for text in ["", "\n", '{"n":1,"n":2}\n', '{"n":NaN}\n', "[]\n", "{broken}\n"]:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, str(path)):
                    schema.validate_file(path)


if __name__ == "__main__":
    unittest.main()
