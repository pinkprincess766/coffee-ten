#!/usr/bin/env python3
"""Plot matmul GFLOP/s vs N from committed bench/results/*.jsonl (no new measurements).

    python3 bench/plot.py [results.jsonl] [-o docs/bench-gflops.png]

Default input: S3 (2026-10-18-4928ffa), the only session with all four schedules.
Dev-only tool; needs matplotlib (not a project dependency).
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "bench/results/2026-10-18-4928ffa.jsonl"
ORDER = ["naive", "tiled", "tiled + vec", "4×16 register tile"]


def label(schedule):
    if not schedule:
        return "naive"
    if "ii, 4)" in schedule and "ji, 16)" in schedule:
        return "4×16 register tile"
    return "tiled + vec" if "vectorize" in schedule else "tiled"


def load(path):
    series = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kernel") != "matmul" or not 64 <= r["n"] <= 1024:
            continue
        series.setdefault(label(r["schedule"]), []).append((r["n"], r["median"], r["p25"], r["p75"]))
    return {k: sorted(v) for k, v in series.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=DEFAULT)
    ap.add_argument("-o", "--out", default=ROOT / "docs/bench-gflops.png")
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("bench/plot.py needs matplotlib: pip install matplotlib")
    series = load(a.results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ORDER:
        if name not in series:
            continue
        n, med, lo, hi = zip(*series[name])
        ax.errorbar(n, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]],
                    marker="o", capsize=3, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024], ["64", "128", "256", "512", "1024"])
    ax.set_xlabel("N")
    ax.set_ylabel("GFLOP/s (median, p25–p75)")
    ax.set_title(f"matmul A @ B, single thread — {pathlib.Path(a.results).name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(a.out)


if __name__ == "__main__":
    main()
