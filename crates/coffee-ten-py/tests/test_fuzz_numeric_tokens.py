"""Python replay rows use the same unsigned-token grammar as the Rust driver."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "fuzz"))
from ct_fuzz_cases import Case, generate


class FuzzNumericTokens(unittest.TestCase):
    def test_python_only_integer_spellings_are_not_replayable(self):
        row = next(generate()).row()
        fields = row.split("\t")
        mutations = [(1, fields[1][:4] + "_" + fields[1][4:]),
                     (1, "0x" + fields[1]), (0, "٠"), (0, "-0")]
        mutations += [(i, fields[i] + " ") for i in (0, 1, 3, 4, 5, 6)]
        for field, token in mutations:
            bad = fields.copy()
            bad[field] = token
            with self.subTest(field=field, token=token), self.assertRaises(ValueError):
                Case.parse("\t".join(bad))

    def test_ascii_tokens_supported_by_rust_remain_valid(self):
        case = next(generate())
        fields = case.row().split("\t")
        for i in (0, 1, 4, 5, 6):
            fields[i] = "+" + fields[i].upper()
        fields[3] = ",".join("+" + dim for dim in fields[3].split(","))
        self.assertEqual(Case.parse("\t".join(fields)).row(), case.row())


if __name__ == "__main__":
    unittest.main()
