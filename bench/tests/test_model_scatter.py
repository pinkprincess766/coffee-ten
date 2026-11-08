import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ct_model_scatter", BENCH / "model_scatter.py")
scatter = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(BENCH), *sys.path]):
    SPEC.loader.exec_module(scatter)


class ModelScatterTests(unittest.TestCase):
    def test_failed_comparison_cleans_temporary_input(self):
        with tempfile.TemporaryDirectory() as directory:
            peak, run = Path(directory) / "peak.jsonl", Path(directory) / "run.jsonl"
            scratch = Path(directory) / "scratch"
            scratch.mkdir()
            peak.write_text('{"kind":"peak"}\n')
            for text, error in [('{"group":"random"}\n', subprocess.CalledProcessError),
                                ('{broken}\n', ValueError)]:
                run.write_text(text)
                with self.subTest(text=text), patch.object(scatter.tempfile, "tempdir", str(scratch)), patch.object(
                    scatter.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "bench")
                ), self.assertRaises(error):
                    scatter.compare(["bench"], peak, run)
                self.assertEqual(list(scratch.iterdir()), [], "failed comparison leaked temporary data")

    def test_compare_accepts_signed_spearman_from_successful_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            peak, run = Path(directory) / "peak.jsonl", Path(directory) / "run.jsonl"
            peak.write_text('{"kind":"peak"}\n')
            run.write_text('{"group":"random"}\n')
            for rho in ("-1.000", "-0.000", "0.853", "1.000"):
                output = (
                    "analytic model v1.8 (PREDICTION) against the measurements\n"
                    "machine: fixture\n"
                    "  256    30.000    20.000   0.67    ldst  schedule\n"
                    f"Spearman rank correlation {rho}; median |ln(predicted/measured)| "
                    "0.405 (a factor 1.50)\n"
                )
                with self.subTest(rho=rho), patch.object(
                    scatter.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=output)
                ) as execute:
                    points, parsed_rho, error, version, machine = scatter.compare(["bench"], peak, run)
                    self.assertEqual((points, parsed_rho, error, version, machine),
                                     ([(30.0, 20.0)], rho, "1.50", "v1.8", "machine: fixture"))
                    arguments = execute.call_args.args[0]
                    self.assertEqual(arguments[:4], ["bench", "model", "compare", "--from"])
                    self.assertFalse(Path(arguments[4]).exists())


if __name__ == "__main__":
    unittest.main()
