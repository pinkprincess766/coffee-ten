"""Failure artifacts survive native panic wrappers and interruption exceptions."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fuzz"))
from ct_fuzz_cases import generate
from ct_fuzz_minimize import save_failure


class SyntheticNativePanic(BaseException):
    """Same exception ancestry as PyO3 PanicException; does not execute native code."""


class MinimizeControlErrors(unittest.TestCase):
    def test_oracle_base_exceptions_retain_original_and_error_report(self):
        case = next(generate())
        for exception_type in (SyntheticNativePanic, KeyboardInterrupt, SystemExit):
            with self.subTest(exception_type=exception_type), tempfile.TemporaryDirectory() as directory:
                error = exception_type("synthetic failure")
                def fail(_):
                    raise error
                with self.assertRaises(exception_type) as raised:
                    save_failure(case, fail, directory, backend="artifact-test")
                self.assertIs(raised.exception, error)
                directory = Path(directory)
                self.assertEqual((directory / "original.tsv").read_text(), case.row() + "\n")
                self.assertFalse((directory / "minimal.tsv").exists())
                report = json.loads((directory / "report.json").read_text())
                self.assertEqual(report["status"], "error")
                self.assertIs(report["verified"], False)
                self.assertEqual(report["error"], "synthetic failure")


if __name__ == "__main__":
    unittest.main()
