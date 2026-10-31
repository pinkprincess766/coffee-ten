#!/usr/bin/env python3
"""Fig. 3: analytic model prediction vs measurement for hypothesis A runs (no new measurements).

    python3 bench/model_scatter.py [--input run.jsonl ...] [--peak P3.jsonl] [--bin coffee-ten-bench] [-o docs/model-vs-measured.png]

Default input: runs 2 and 3 with the P3 peak line (`bench/results/2026-10-21-561ede3-peak.jsonl`,
the machine both runs were taken on); `--input` may be given several times. Predictions are not
recomputed here: for each run this script writes a temporary file with the peak line followed
by the run's `"group":"random"` records, and runs the existing Rust tool
`coffee-ten-bench model compare --from <file>`, so the model is the current one in the tree (the
committed figure was made with v1.3). The model version, points, Spearman and median error are
read from its report, so the caption numbers are the tool's own.
Dev-only tool; needs matplotlib (not a project dependency) and cargo (or --bin).
"""
import argparse, json, pathlib, re, subprocess, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot import ROOT  # noqa: E402

RES = ROOT / "bench/results"
PEAK = RES / "2026-10-21-561ede3-peak.jsonl"
RUNS = [RES / "2026-10-21-hypothesis-a-run2.jsonl", RES / "2026-10-22-hypothesis-a-run3.jsonl"]
ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S+)\s+")
VERSION = re.compile(r"analytic model (v[0-9.]+)")
SUMMARY = re.compile(r"Spearman rank correlation ([0-9.]+); median \|ln\(predicted/measured\)\| "
                     r"([0-9.]+) \(a factor ([0-9.]+)\)")


def compare(binary, peak, run):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.writelines(l for l in open(peak) if json.loads(l).get("kind") == "peak")
        f.writelines(l for l in open(run) if json.loads(l).get("group") == "random")
    cmd = (binary + ["model", "compare", "--from", f.name])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    pathlib.Path(f.name).unlink()
    pts = [(float(m[2]), float(m[3])) for m in map(ROW.match, out.splitlines()) if m]
    s = SUMMARY.search(out)
    v = VERSION.search(out)
    if not s or not v or not pts:
        sys.exit(f"unexpected `model compare` output:\n{out}")
    return pts, s[1], s[3], v[1], out.splitlines()[1]


def label(path):
    """`run N` for a hypothesis A log named `...-runN.jsonl`, else the file name."""
    m = re.search(r"run(\d+)\.jsonl$", str(path))
    return f"run {m[1]}" if m else pathlib.Path(path).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", action="append", help="hypothesis A log (repeatable; default: runs 2 and 3)")
    ap.add_argument("--peak", default=PEAK)
    ap.add_argument("--bin", help="prebuilt coffee-ten-bench (default: cargo run)")
    ap.add_argument("-o", "--out", default=ROOT / "docs/model-vs-measured.png")
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("bench/model_scatter.py needs matplotlib: pip install matplotlib")
    binary = [a.bin] if a.bin else ["cargo", "run", "-q", "-p", "coffee-ten-bench", "--"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    lo, hi = 0.5, 200.0
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, label="y = x")
    for i, path in enumerate(a.input or RUNS):
        name = label(path)
        pts, rho, err, version, machine = compare(binary, a.peak, path)
        meas, pred = zip(*pts)
        ax.plot(meas, pred, "o", ms=3, alpha=0.45, mec="none", color=f"C{i}",
                label=f"{name}: n = {len(pts)}, Spearman {rho}, median error ×{err}")
        print(f"{name}: {len(pts)} points, Spearman {rho}, median error x{err}")
    print(machine)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("measured, GFLOP/s (median of 5 samples)")
    ax.set_ylabel(f"predicted by model {version}, GFLOP/s")
    ax.set_title("matmul N = 256, hypothesis A random candidates\n"
                 f"machine: bench/peak line of {pathlib.Path(a.peak).name}", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
