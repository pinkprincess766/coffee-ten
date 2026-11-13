"""bench/validate.py and `coffee-ten-bench report` must agree on which logs are valid.

Both read the same JSONL. The corruptions below are fed through each one; for every case
the two must accept or reject together, except the explicitly listed KNOWN_GAPS (mostly
where validate.py is stricter today). A new disagreement, or a gap that closes without updating
the list, fails the test.

Needs a built `coffee-ten-bench`: $COFFEE_TEN_BENCH, else target/release (bench/run.sh
builds it before CI runs these tests), else target/debug. Skipped if none exists.
"""
import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ct_validate", ROOT / "bench/validate.py")
schema = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schema)
FIXTURE = ROOT / "crates/coffee-ten-bench/tests/fixtures/report.jsonl"
PEAK = ROOT / "bench/results/2026-10-28-adcf09c-peak.jsonl"


def find_binary():
    candidates = [os.environ.get("COFFEE_TEN_BENCH")] + [
        ROOT / "target" / profile / "coffee-ten-bench" for profile in ("release", "debug")]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


BINARY = find_binary()

# validate.py rejects these, report accepts them. Closing a gap: move it out of here.
PYTHON_ONLY_REJECTS = {
    "blank line between records",
    "record: missing commit",
    "record: missing date",
    "record: missing dirty",
    "record: missing order_class",
    "record: extra field",
    "record: bad commit type",
    "record: bad date type",
    "record: bad dirty type",
    "record: bad order_class",
    "record: bad group",
    "round 0: extra field",
    "reference_compilation: extra field",
    "protocol: source_hash not 16 hex digits",
    "protocol: space_hash not 16 hex digits",
    "protocol: unsupported harness_profile",
}
# The reverse: a peak-only log is a valid log, but report has no measurements to rank.
REPORT_ONLY_REJECTS = {"only a peak line"}
KNOWN_GAPS = {name: (False, True) for name in PYTHON_ONLY_REJECTS}
KNOWN_GAPS.update({name: (True, False) for name in REPORT_ONLY_REJECTS})


def mutate(row, path, value=None, delete=False):
    bad = copy.deepcopy(row)
    target = bad
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return json.dumps(bad)


def cases():
    lines = FIXTURE.read_text().splitlines()
    # A real peak line from bench/results; the fixture's own is synthetic but valid too.
    peak, records = PEAK.read_text().strip(), lines[1:]
    row = json.loads(records[0])
    rest = "\n".join(records[1:]) + "\n"
    out = {"valid log": "\n".join([peak] + records) + "\n"}

    def record(name, line):
        out[name] = line + "\n" + rest

    for field in sorted(set(row) - {"group"}):
        record(f"record: missing {field}", mutate(row, [field], delete=True))
    record("record: extra field", mutate(row, ["unexpected"], 1))
    for field in sorted(row["protocol"]):
        record(f"protocol: missing {field}", mutate(row, ["protocol", field], delete=True))
    record("protocol: extra field", mutate(row, ["protocol", "unexpected"], 1))
    for field in sorted(row["rounds"][0]):
        record(f"round 0: missing {field}", mutate(row, ["rounds", 0, field], delete=True))
    record("round 0: extra field", mutate(row, ["rounds", 0, "unexpected"], 1))
    if row["reference_compilation"] is not None:
        record("reference_compilation: extra field",
               mutate(row, ["reference_compilation", "unexpected"], 1))
    for name, path, value in [
        ("bad commit type", ["commit"], 1),
        ("bad date type", ["date"], 1),
        ("bad dirty type", ["dirty"], "yes"),
        ("bad order_class", ["order_class"], "sideways"),
        ("bad group", ["group"], "bogus"),
        ("unverified", ["verified"], False),
        ("other unit", ["unit"], "GB/s"),
        ("kernel differs from protocol", ["kernel"], "relu-matmul-bias"),
        ("cpu differs from protocol", ["cpu"], "other"),
        ("seed differs from protocol", ["seed"], row["seed"] + 1),
        ("quick differs from protocol", ["quick"], not row["quick"]),
        ("samples differs from rounds", ["samples"], row["samples"] + 1),
        ("median differs from rounds", ["median"], row["median"] * 2),
        ("measurement cost differs", ["measurement_ns"], row["measurement_ns"] + 1),
        ("drift flag differs", ["drift"], not row["drift"]),
        ("compilation_wall_ns set", ["compilation_wall_ns"], 1),
        ("negative compilation_ns", ["compilation_ns"], -1),
        ("string cache_hit", ["cache_hit"], "true"),
    ]:
        record(f"record: {name}", mutate(row, path, value))
    for name, field, value in [
        ("unsupported version", "version", 2),
        ("unsupported space", "space", "v9"),
        ("zero n", "n", 0),
        ("zero samples", "samples", 0),
        ("unsupported kernel", "kernel", "conv"),
        ("string quick", "quick", "no"),
        ("rounds below min_sample_ms", "min_sample_ms", 10_000),
        ("source_hash not 16 hex digits", "source_hash", "0123"),
        ("space_hash not 16 hex digits", "space_hash", "ABCDEF0123456789"),
        ("unsupported harness_profile", "harness_profile", "fast"),
    ]:
        record(f"protocol: {name}", mutate(row, ["protocol", field], value))
    for name, field, value in [
        ("non-sequential round_index", "round_index", 5),
        ("zero elapsed_ns", "elapsed_ns", 0),
        ("rate does not match timing", "gflops", row["rounds"][0]["gflops"] * 2),
        ("null rate", "gflops", None),
    ]:
        record(f"round 0: {name}", mutate(row, ["rounds", 0, field], value))
    for name, text in [
        ("empty file", ""),
        ("only a peak line", peak + "\n"),
        ("not an object", "[]\n" + rest),
        ("truncated JSON", records[0][:-1] + "\n" + rest),
        ("NaN token", records[0].replace('"median":', '"median":NaN,"x":', 1) + "\n" + rest),
        ("duplicate key", records[0][:-1] + ',"n":' + str(row["n"]) + "}\n" + rest),
        ("offline curve record", '{"kind":"replay_curve"}\n' + rest),
        ("blank line between records", records[0] + "\n\n" + rest),
        ("malformed peak line", '{"kind":"peak","fixture":true}\n' + records[0] + "\n" + rest),
        ("peak line with an extra field", peak[:-1] + ',"extra":1}\n' + "\n".join(records) + "\n"),
        ("peak line with unordered statistics",
         peak.replace('"median":', '"median":-', 1) + "\n" + "\n".join(records) + "\n"),
    ]:
        out[name] = text
    return out


@unittest.skipIf(BINARY is None, "coffee-ten-bench is not built; run bench/run.sh or cargo build")
class ValidatorParityTests(unittest.TestCase):
    def verdicts(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            path.write_text(text, encoding="utf-8")
            try:
                schema.validate_file(path)
                python = True
            except ValueError:
                python = False
            result = subprocess.run([str(BINARY), "report", str(path)],
                                    capture_output=True, text=True, timeout=60)
            # Exit 1 is a data error; anything else (usage error, crash) is a test failure.
            self.assertIn(result.returncode, (0, 1), result.stderr)
            return python, result.returncode == 0

    def test_both_validators_accept_and_reject_the_same_logs(self):
        all_cases = cases()
        self.assertTrue(set(KNOWN_GAPS) <= set(all_cases), sorted(set(KNOWN_GAPS) - set(all_cases)))
        gaps = set()
        for name, text in all_cases.items():
            python, rust = self.verdicts(text)
            with self.subTest(case=name):
                if name == "valid log":
                    self.assertEqual((python, rust), (True, True))
                elif name in KNOWN_GAPS:
                    self.assertEqual((python, rust), KNOWN_GAPS[name], "gap closed: update KNOWN_GAPS")
                    gaps.add(name)
                else:
                    self.assertEqual(python, rust, f"validate.py {python}, report {rust}")
                    self.assertFalse(python, "corruption accepted by both")
        self.assertEqual(gaps, set(KNOWN_GAPS))


if __name__ == "__main__":
    unittest.main()
