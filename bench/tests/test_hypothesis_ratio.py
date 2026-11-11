import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
