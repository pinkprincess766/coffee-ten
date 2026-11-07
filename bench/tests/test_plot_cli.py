import importlib.util
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

BENCH = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"ct_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCH), *sys.path]):
        spec.loader.exec_module(module)
    return module


class PlotCliTests(unittest.TestCase):
    def test_default_and_explicit_inputs_reach_save_with_their_filename(self):
        for name in ("plot", "roofline"):
            script = load_script(name)
            for explicit in (False, True):
                with self.subTest(script=name, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "custom-session.jsonl" if explicit else script.DEFAULT
                    output = Path(directory) / "figures" / "chart.png"
                    figure, axes = Mock(), Mock()
                    pyplot = types.ModuleType("matplotlib.pyplot")
                    pyplot.subplots = Mock(return_value=(figure, axes))
                    matplotlib = types.ModuleType("matplotlib")
                    matplotlib.use = Mock()
                    matplotlib.pyplot = pyplot
                    arguments = [str(BENCH / f"{name}.py"), "--out", str(output)]
                    if explicit:
                        arguments.extend(["--input", str(source)])
                    with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}), \
                            patch.object(sys, "argv", arguments), \
                            patch.object(script, "load", return_value={"naive": [(64, 8.0, 7.0, 9.0, True)]}) as load, \
                            patch.object(script, "ceilings", return_value=(100.0, 50.0), create=True), \
                            redirect_stdout(io.StringIO()) as stdout:
                        script.main()
                    self.assertEqual(load.call_args.args, (str(source) if explicit else source,))
                    self.assertIn(source.name, axes.set_title.call_args.args[0])
                    axes.plot.assert_called()
                    figure.savefig.assert_called_once_with(str(output), dpi=120, bbox_inches="tight")
                    self.assertEqual(stdout.getvalue(), f"{output}\n")


if __name__ == "__main__":
    unittest.main()
