"""plot.py and roofline.py keep "drift not evaluated" apart from "no drift".

With fewer than stats::MIN_ROUNDS_FOR_DRIFT (10) rounds the harness never checks drift and
stores `false`. Since #194 the report prints "not evaluated" there; the plots now mark such
points with a separate × marker and legend entry instead of drawing them as drift-free.
"""
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
RESULTS = BENCH / "results"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"ct_drift_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCH), *sys.path]):
        spec.loader.exec_module(module)
    return module


PLOT = load_script("plot")


def record(n=128, samples=30, drift=False, rounds=None, schedule=""):
    value = dict(kernel="matmul", n=n, unit="GFLOP/s", verified=True, schedule=schedule,
                 median=2.0, p25=1.5, p75=2.5, samples=samples, drift=drift)
    if rounds is not None:
        value["rounds"] = [{"round_index": i} for i in range(rounds)]
    return value


def render(name, path):
    script = load_script(name)
    matplotlib, pyplot, figure, axes = (MagicMock() for _ in range(4))
    matplotlib.pyplot = pyplot
    pyplot.subplots.return_value = figure, axes
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}))
        stack.enter_context(patch.object(sys, "argv", [name, "--input", str(path), "--out", str(Path(directory) / "out.png")]))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        script.main()
    crosses = [c for c in axes.plot.call_args_list if len(c.args) > 2 and c.args[2] == "x"]
    labels = [c.kwargs.get("label") for c in crosses]
    return axes.set_title.call_args.args[0], crosses, labels


def write_log(directory, records, peak=True):
    path = Path(directory) / "log.jsonl"
    lines = [json.dumps(r) for r in records]
    if peak:
        lines.insert(0, json.dumps({"kind": "peak", "fma_gflops": {"median": 100.0},
                                    "triad_gbs": {"median": 50.0}}))
    path.write_text("\n".join(lines) + "\n")
    return path


class DriftStateTests(unittest.TestCase):
    def test_three_states_from_rounds_or_samples(self):
        for value, expected in [
            (record(samples=30, drift=False), False),
            (record(samples=30, drift=True), True),
            (record(samples=10, drift=False), False),
            (record(samples=9, drift=False), None),
            (record(samples=5, drift=True), None),  # never stored, but still not a checked flag
            (record(samples=30, rounds=5), None),  # raw rounds win over the declared count
            (record(samples=5, rounds=10, drift=True), True),
            ({k: v for k, v in record().items() if k != "samples"}, None),
        ]:
            with self.subTest(value={k: value.get(k) for k in ("samples", "drift")},
                              rounds=len(value.get("rounds", []))):
                self.assertIs(PLOT.drift_state(value), expected)

    def test_saved_logs_load_with_the_expected_states(self):
        # S3 (default input): 30 rounds, every flag was checked. N20: 5 rounds, none was.
        s3 = PLOT.load(PLOT.DEFAULT)
        states = {p[4] for points in s3.values() for p in points}
        self.assertEqual(states, {True, False})
        n20 = PLOT.load(RESULTS / "2026-11-07-n20-v1-n256.jsonl")
        self.assertEqual({p[4] for points in n20.values() for p in points}, {None})
        for path in sorted(RESULTS.glob("*.jsonl")):
            with self.subTest(log=path.name):
                for points in PLOT.load(path).values():
                    for point in points:
                        self.assertIn(point[4], (True, False, None))


class DriftMarkerTests(unittest.TestCase):
    def test_short_runs_get_a_cross_and_a_legend_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_log(directory, [record(n=64, samples=30), record(n=128, samples=5),
                                         record(n=256, samples=30, drift=True)])
            for name in ("plot", "roofline"):
                with self.subTest(script=name):
                    title, crosses, labels = render(name, path)
                    self.assertIn(PLOT.NOT_EVALUATED_NOTE, title)
                    # One cross on the n = 128 point, one empty legend handle.
                    self.assertIn("drift not evaluated (< 10 rounds)", labels)
                    points = [c for c in crosses if c.args[0]]
                    self.assertEqual(len(points), 1)
                    self.assertEqual(list(points[0].args[1]), [2.0])

    def test_fully_checked_logs_render_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_log(directory, [record(n=64), record(n=128, drift=True)])
            for name in ("plot", "roofline"):
                with self.subTest(script=name):
                    title, crosses, _ = render(name, path)
                    self.assertNotIn("not evaluated", title)
                    self.assertEqual(crosses, [])


if __name__ == "__main__":
    unittest.main()
