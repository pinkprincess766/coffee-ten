#!/usr/bin/env python3
"""Plot matmul GFLOP/s vs N from committed bench/results/*.jsonl (no new measurements).

    python3 bench/plot.py [--input results.jsonl] [-o docs/bench-gflops.png]

Default input: S3 (2026-10-18-4928ffa), the only session with all four schedules.
Dev-only tool; needs matplotlib (not a project dependency).
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "bench/results/2026-10-18-4928ffa.jsonl"
ORDER = ["naive", "tiled", "tiled + vec", "4×16 register tile"]
# stats::MIN_ROUNDS_FOR_DRIFT: with fewer rounds the harness never checks drift, so a
# stored `false` means "not evaluated", not "no drift".
MIN_ROUNDS_FOR_DRIFT = 10
NOT_EVALUATED_NOTE = "× = drift not evaluated (< 10 rounds)"


def label(schedule):
    if not schedule:
        return "naive"
    if "ii, 4)" in schedule and "ji, 16)" in schedule:
        return "4×16 register tile"
    return "tiled + vec" if "vectorize" in schedule else "tiled"


def drift_state(record):
    """True (drift), False (checked, no drift) or None (too few rounds: not evaluated)."""
    rounds = record.get("rounds")
    count = len(rounds) if isinstance(rounds, list) else record.get("samples")
    if not isinstance(count, int) or count < MIN_ROUNDS_FOR_DRIFT:
        return None
    return record.get("drift") is True


def load(path):
    """Points per schedule label: (n, median, p25, p75, drift_state(record)), sorted by n."""
    series = {}
    with open(path) as source:
        for line in source:
            r = json.loads(line)
            if r.get("kernel") != "matmul" or not 64 <= r["n"] <= 1024:
                continue
            if r.get("unit") != "GFLOP/s" or r.get("verified") is not True:
                continue
            series.setdefault(label(r["schedule"]), []).append(
                (r["n"], r["median"], r["p25"], r["p75"], drift_state(r)))
    return {k: sorted(v) for k, v in series.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=DEFAULT, help="results JSONL (default: S3)")
    ap.add_argument("-o", "--out", default=ROOT / "docs/bench-gflops.png")
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("bench/plot.py needs matplotlib: pip install matplotlib")
    series = load(a.input)
    unevaluated = any(p[4] is None for points in series.values() for p in points)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(ORDER):
        if name not in series:
            continue
        n, med, lo, hi, drift = zip(*series[name])
        ax.errorbar(n, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]],
                    marker="o", capsize=3, label=name, color=f"C{i}")
        dn = [x for x, d in zip(n, drift) if d]
        dm = [y for y, d in zip(med, drift) if d]
        if dn:
            ax.plot(dn, dm, "o", mfc="white", mec=f"C{i}", mew=1.5, zorder=3, label="_nolegend_")
        un = [x for x, d in zip(n, drift) if d is None]
        um = [y for y, d in zip(med, drift) if d is None]
        if un:
            ax.plot(un, um, "x", color="black", ms=5, zorder=4, label="_nolegend_")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024], ["64", "128", "256", "512", "1024"])
    ax.set_xlabel("N")
    ax.set_ylabel("GFLOP/s (median, p25–p75)")
    ax.set_title(f"matmul A @ B, single thread — {pathlib.Path(a.input).name}\n"
                 "hollow marker = drift (first/last 5 rounds differ by > 5%)"
                 + (f"; {NOT_EVALUATED_NOTE}" if unevaluated else ""), fontsize=10)
    ax.grid(alpha=0.3)
    if unevaluated:
        ax.plot([], [], "x", color="black", label="drift not evaluated (< 10 rounds)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
