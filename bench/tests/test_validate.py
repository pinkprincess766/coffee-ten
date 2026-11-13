import copy
import importlib.util
import json
import math
import struct
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

    def test_drift_flag_matches_ordered_window_medians(self):
        patterns = [
            ([100] * 9, False),  # Two disjoint windows require ten rounds.
            ([100] * 5 + [105] * 5, False),
            ([100] * 5 + [95] * 5, False),
            ([100] * 5 + [106] * 5, True),
            ([100] * 5 + [94] * 5, True),
            ([1, 100, 100, 100, 100, 100, 100, 100, 100, 1000], False),
        ]
        for rates, expected in patterns:
            row = copy.deepcopy(self.row)
            row["samples"] = row["protocol"]["samples"] = len(rates)
            row["rounds"] = [dict(round_index=i, elapsed_ns=16_000_000,
                                  iterations=rate * 1_000_000, gflops=float(rate))
                             for i, rate in enumerate(rates)]
            row["measurement_ns"] = sum(s["elapsed_ns"] for s in row["rounds"])
            ordered = sorted(rates)
            last = len(rates) - 1
            for field, index in [("min", 0), ("p25", last // 4),
                                 ("median", last // 2), ("p75", 3 * last // 4)]:
                row[field] = float(ordered[index])
            row["drift"] = expected
            with self.subTest(rates=rates, flag=expected):
                schema.validate(row)
            row["drift"] = not expected
            with self.subTest(rates=rates, flag=not expected), self.assertRaisesRegex(ValueError, "drift"):
                schema.validate(row)

    def test_drift_boundary_cases_shared_with_rust(self):
        # stats::drifted reads the same file (drift_cases_shared_with_validate_py), so both
        # sides must flag exactly these cases, including nominal 5% changes that IEEE
        # rounding puts just above or just below the limit.
        fixture = json.loads((ROOT / "crates/coffee-ten-bench/tests/fixtures/drift-cases.json").read_text())
        self.assertEqual(fixture["n"], self.row["n"])
        seconds, nanos = divmod(fixture["elapsed_ns"], 1_000_000_000)
        elapsed_s = float(seconds) + float(nanos) / 1e9
        n = float(fixture["n"])
        flags = []
        for case in fixture["cases"]:
            iterations = case["iterations"]
            rates = [2.0 * ((n * n) * n) * float(it) / elapsed_s / 1e9 for it in iterations]
            row = copy.deepcopy(self.row)
            row["samples"] = row["protocol"]["samples"] = len(rates)
            row["rounds"] = [dict(round_index=i, elapsed_ns=fixture["elapsed_ns"],
                                  iterations=it, gflops=rate)
                             for i, (it, rate) in enumerate(zip(iterations, rates))]
            row["measurement_ns"] = fixture["elapsed_ns"] * len(rates)
            ordered = sorted(rates)
            last = len(rates) - 1
            for field, index in [("min", 0), ("p25", last // 4),
                                 ("median", last // 2), ("p75", 3 * last // 4)]:
                row[field] = ordered[index]
            row["drift"] = case["drift"]
            with self.subTest(case=case["name"]):
                schema.validate(row)
            row["drift"] = not case["drift"]
            with self.subTest(case=case["name"], flipped=True), self.assertRaisesRegex(ValueError, "drift"):
                schema.validate(row)
            flags.append(case["drift"])
        # Both outcomes are covered, and so are the short-run and exact-0.05 cases.
        self.assertIn(True, flags)
        self.assertIn(False, flags)
        self.assertTrue(any(len(c["iterations"]) < 10 for c in fixture["cases"]))

    def test_strict_json_and_file_line_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps(self.row) + "\n" + json.dumps(self.peak) + "\n")
            self.assertEqual(schema.validate_file(path), 2)
            for text in ["", "\n", '{"n":1,"n":2}\n', '{"n":NaN}\n', "[]\n", "{broken}\n"]:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, str(path)):
                    schema.validate_file(path)

    def test_jsonl_uses_lf_boundaries_not_unicode_string_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.jsonl"
            for separator in ["\u0085", "\u2028", "\u2029"]:
                row = copy.deepcopy(self.row)
                row["cpu"] = row["protocol"]["cpu"] = "fixture" + separator + "CPU"
                path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                with self.subTest(separator=ascii(separator)):
                    self.assertEqual(schema.validate_file(path), 1)
            first, second = json.dumps(self.row), json.dumps(self.peak)
            path.write_bytes((first + "\r\n" + second + "\r\n").encode("utf-8"))
            self.assertEqual(schema.validate_file(path), 2)
            path.write_bytes((first + "\r" + second).encode("utf-8"))
            with self.assertRaisesRegex(ValueError, ":1:"):
                schema.validate_file(path)

    def test_self_consistent_rates_must_match_timing(self):
        bad = copy.deepcopy(self.row)
        for sample in bad["rounds"]:
            sample["gflops"] *= 2
        for field in ["min", "p25", "median", "p75"]:
            bad[field] *= 2
        with self.assertRaisesRegex(ValueError, "timing"):
            schema.validate(bad)

    def test_round_rates_match_rust_duration_conversion_bit_for_bit(self):
        # Rust rate bits: 16 * 1e6 / Duration::from_nanos(ns).as_secs_f64() / 1e9.
        for elapsed, bits in [(1_000_000_001, 4580268907422239638),
                              (1_123_456_789, 4579363436459430692),
                              (9_007_199_254_740_993, 4476160764943532032),
                              (5_133_295_621_681_162_413, 4434755519214517698),
                              (2**64 - 1, 4426621169042456576)]:
            value = struct.unpack("=d", struct.pack("=Q", bits))[0]
            row = copy.deepcopy(self.row)
            row["samples"] = row["protocol"]["samples"] = 1
            row["rounds"] = [dict(round_index=0, elapsed_ns=elapsed,
                                  iterations=1_000_000, gflops=value)]
            row["measurement_ns"] = elapsed
            for field in ["min", "p25", "median", "p75"]:
                row[field] = value
            with self.subTest(elapsed=elapsed):
                schema.validate(row)
                corrupted = math.nextafter(value, math.inf)
                row["rounds"][0]["gflops"] = corrupted
                for field in ["min", "p25", "median", "p75"]:
                    row[field] = corrupted
                with self.assertRaisesRegex(ValueError, "timing"):
                    schema.validate(row)


if __name__ == "__main__":
    unittest.main()
