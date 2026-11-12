import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BENCH = Path(__file__).resolve().parents[1]


def render(name, **inputs):
    spec = importlib.util.spec_from_file_location(f"ct_ranges_{name}", BENCH / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCH), *sys.path]):
        spec.loader.exec_module(script)
    matplotlib, pyplot, figure, axes = (MagicMock() for _ in range(4))
    matplotlib.pyplot = pyplot
    pyplot.subplots.return_value = figure, axes
    axes.plot.return_value = [MagicMock()]
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}))
        stack.enter_context(patch.object(sys, "argv", [name, "--input", "fixture.jsonl", "--out", str(Path(directory) / "out.png")]))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for key, value in inputs.items():
            stack.enter_context(patch.object(script, key, return_value=value))
        script.main()
    return axes


class PlotRangeTests(unittest.TestCase):
    def test_hypothesis_error_bars_and_ratio_are_visible(self):
        model = dict(median=4.0, p25=0.1, p75=40.0, seed=42, commit="fixture", n=64)
        random = dict(median=1.0, p25=0.5, p75=1.5)
        axes = render("hypothesis_ratio", ratio=(model, random))
        low, high = axes.set_ylim.call_args.args
        self.assertLessEqual(low, model["p25"] / random["p75"])
        self.assertGreaterEqual(high, model["p75"] / random["p25"])

    def test_roofline_includes_measured_points_outside_the_default_window(self):
        axes = render("roofline", ceilings=(100.0, 10.0),
                      load={"naive": [(64, 0.1, 0.09, 0.11, False), (128, 300.0, 290.0, 310.0, False)]})
        low, high = axes.set_ylim.call_args.args
        self.assertLessEqual(low, 0.1)
        self.assertGreaterEqual(high, 300.0)

    def test_scatter_includes_all_points_and_extends_the_reference_diagonal(self):
        points = [(0.1, 500.0), (400.0, 0.01)]
        axes = render("model_scatter", compare=(points, "0.5", "2.0", "v1.8", "machine"))
        for method in [axes.set_xlim, axes.set_ylim]:
            low, high = method.call_args.args
            self.assertLessEqual(low, 0.01)
            self.assertGreaterEqual(high, 500.0)
        limits = list(axes.set_xlim.call_args.args)
        self.assertEqual(axes.plot.return_value[0].set_data.call_args.args, (limits, limits))


class PlotMetadataTests(unittest.TestCase):
    def test_hypothesis_labels_use_the_recorded_size_and_seed(self):
        model = dict(median=4.0, p25=3.0, p75=5.0, seed=42, commit="fixture", n=64)
        random = dict(median=3.0, p25=2.5, p75=3.5)
        axes = render("hypothesis_ratio", ratio=(model, random))
        labels = " ".join(axes.set_xticklabels.call_args.args[0])
        self.assertIn("N=64", labels)
        self.assertIn("seed=42", labels)
        self.assertNotIn("256", axes.set_title.call_args.args[0])
        self.assertNotIn("1, 2, 3, 5", axes.set_xlabel.call_args.args[0])

    def test_scatter_does_not_claim_an_unread_size_or_sample_count(self):
        axes = render("model_scatter", compare=([(1.0, 2.0)], "0.5", "2.0", "v1.8", "machine"))
        self.assertNotIn("N =", axes.set_title.call_args.args[0])
        self.assertNotIn("samples", axes.set_xlabel.call_args.args[0])
        self.assertIn("median", axes.set_xlabel.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
