import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RunArgumentTests(unittest.TestCase):
    def run_script(self, sizes, pause):
        with tempfile.TemporaryDirectory() as directory:
            commands = {
                "cargo": "exit 97", "clang": "exit 97", "sleep": "exit 97",
                "git": 'case "$1" in status) exit 0;; rev-parse) echo fixture;; *) exit 97;; esac',
                "pmset": 'echo "Now drawing from \'AC Power\'"; echo "lowpowermode 0"',
                "uname": "echo arm64",
            }
            for name, body in commands.items():
                path = Path(directory) / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            return subprocess.run(
                ["/bin/bash", str(ROOT / "bench/run.sh"), "--dry-run", "--yes",
                 "--allow-dirty", "--sizes", sizes, "--pause", pause],
                env=dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"]),
                text=True, capture_output=True, timeout=10,
            )

    def test_invalid_arguments_fail_before_printing_any_build_or_measurement(self):
        for sizes, pause in [(s, "0") for s in ["", "0", "1025", "bad", "64 bad", "*", "18446744073709551616"]] + [
            ("64", p) for p in ["-1", "oops", "0.5"]
        ]:
            with self.subTest(sizes=sizes, pause=pause):
                result = self.run_script(sizes, pause)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("+ cargo", result.stdout)

    def test_decimal_pause_and_size_boundaries_remain_usable(self):
        for pause in ["0", "000", "08", "+8"]:
            with self.subTest(pause=pause):
                result = self.run_script("1 1024 00064 +128", pause)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("+ sleep" in result.stdout, pause in ("08", "+8"))
                self.assertIn("--n 1024", result.stdout)


if __name__ == "__main__":
    unittest.main()
