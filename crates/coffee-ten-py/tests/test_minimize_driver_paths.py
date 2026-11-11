"""Explicit Rust driver paths are relative to the caller, not the repository."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fuzz"))
from ct_fuzz_cases import generate
from ct_fuzz_minimize import rust_mismatch


@unittest.skipUnless(os.name == "posix", "the harmless test driver is a shell script")
class MinimizeDriverPaths(unittest.TestCase):
    def test_relative_driver_is_resolved_before_switching_to_repository(self):
        case = next(generate())
        caller = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "nested").mkdir()
            for relative in (Path("driver.sh"), Path("nested/driver.sh")):
                executable = directory / relative
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o755)
            try:
                os.chdir(directory)
                for relative in (Path("driver.sh"), Path("nested/driver.sh")):
                    with self.subTest(relative=relative):
                        self.assertFalse(rust_mismatch(case, relative))
            finally:
                os.chdir(caller)


if __name__ == "__main__":
    unittest.main()
