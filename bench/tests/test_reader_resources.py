import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"ct_resources_{name}", BENCH / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCH), *sys.path]):
        spec.loader.exec_module(value)
    return value


class ReaderResourceTests(unittest.TestCase):
    def test_sources_close_on_success_and_malformed_json(self):
        records = [dict(kernel="matmul", n=64, unit="GFLOP/s", verified=True,
                        schedule="", median=2, p25=1, p75=3),
                   dict(kind="peak", fma_gflops=dict(median=10), triad_gbs=dict(median=5))]
        records += [dict(group=group, schedule=schedule, median=median, p25=1, p75=4)
                    for group, schedule, median in [("random", "r", 2), ("model", "m", 3),
                                                     ("final", "r", 2), ("final", "m", 3)]]
        text = "".join(json.dumps(r) + "\n" for r in records)
        hypothesis = module("hypothesis_ratio")
        readers = [module("plot").load, module("roofline").ceilings,
                   hypothesis.ratio, hypothesis.table]
        for reader in readers:
            for broken in [False, True]:
                sources = []

                def opened(*args, **kwargs):
                    source = io.StringIO("{broken}\n" if broken else text)
                    sources.append(source)
                    return source

                with self.subTest(reader=reader.__name__, broken=broken), patch("builtins.open", opened), contextlib.redirect_stdout(io.StringIO()):
                    if broken:
                        with self.assertRaises(ValueError):
                            reader("fixture.jsonl")
                    else:
                        reader("fixture.jsonl")
                    self.assertTrue(sources)
                    self.assertTrue(all(source.closed for source in sources))


if __name__ == "__main__":
    unittest.main()
