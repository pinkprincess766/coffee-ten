import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BENCH = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ct_hypothesis_ratio", BENCH / "hypothesis_ratio.py")
ratio = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(BENCH), *sys.path]):
    SPEC.loader.exec_module(ratio)


class ShortTableTests(unittest.TestCase):
    def test_table_prints_only_reached_curve_checkpoints(self):
        for count in [1, 3, 16, 63, 64]:
            records = [dict(group="random", schedule=f"r{i}", median=float(i))
                       for i in range(1, count + 1)]
            records += [dict(group="model", schedule="m", median=100.0)]
            records += [dict(group="final", schedule=s, median=m, p25=m-0.5, p75=m+0.5)
                        for s, m in [(f"r{count}", 7.0), ("m", 8.0)]]
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "short.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    ratio.table(path)
                text = output.getvalue()
                self.assertIn("ratio 1.143", text)
                self.assertIn(f"{count:.2f} from {count} on", text)
                for checkpoint in ratio.CURVE_M:
                    self.assertEqual(f"after {checkpoint}" in text, checkpoint <= count)


class SharedWinnerTests(unittest.TestCase):
    def test_same_winner_without_final_run_has_exact_identity_ratio(self):
        records = [dict(group=group, schedule="shared", median=median, p25=median-1,
                        p75=median+1, seed=1, commit="fixture")
                   for group, median in [("random", 10.0), ("model", 12.0)]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            model, random = ratio.ratio(path)
            self.assertIs(model, random)
            self.assertEqual(model["median"], 10.0)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ratio.table(path)
            self.assertIn("no final run", output.getvalue())
            self.assertIn("ratio 1.000", output.getvalue())
            pyplot, matplotlib, figure, axes = (MagicMock() for _ in range(4))
            matplotlib.pyplot = pyplot
            pyplot.subplots.return_value = figure, axes
            with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}), patch.object(
                sys, "argv", ["ratio", "--input", str(path), "--out", str(Path(directory) / "figure.png")]
            ), contextlib.redirect_stdout(io.StringIO()):
                ratio.main()
            self.assertEqual(axes.errorbar.call_args.kwargs["yerr"], [[0.0], [0.0]])

    def test_missing_final_run_with_different_winners_is_rejected(self):
        records = [dict(group=group, schedule=group, median=1.0)
                   for group in ["random", "model"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaises(SystemExit):
                ratio.ratio(path)


if __name__ == "__main__":
    unittest.main()
