#!/usr/bin/env python3
"""Roofline plot of the S3 matmul points from committed bench/results/*.jsonl (no new measurements).

    python3 bench/roofline.py [results.jsonl] [-o docs/bench-roofline.png]

Ceilings come from the `"kind":"peak"` line of the same file (`bench/peak`): FMA peak
`fma_gflops` and memory bandwidth `triad_gbs`, both medians; not datasheet numbers.
Roofline after Williams, Waterman, Patterson, CACM 2009 (README [L2]).

AI (FLOP/byte) per point = 2·N³ / (16·N²) = N / 8: compulsory DRAM traffic only
(A and B read once, C read and written once, float32). That is the minimum traffic, so it
is an UPPER bound on the real AI of every variant (lower bound on traffic): the tiled and
naive nests may move more. See BENCHMARKS.md, "Roofline".
Dev-only tool; needs matplotlib (not a project dependency).
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot import DEFAULT, ORDER, ROOT, load  # noqa: E402


def ai(n):
    return 2 * n**3 / (16 * n**2)


def ceilings(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "peak":
            return r["fma_gflops"]["median"], r["triad_gbs"]["median"]
    sys.exit(f"{path}: no \"kind\":\"peak\" line")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=DEFAULT)
    ap.add_argument("-o", "--out", default=ROOT / "docs/bench-roofline.png")
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("bench/roofline.py needs matplotlib: pip install matplotlib")
    peak, bw = ceilings(a.results)
    series = load(a.results)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = sorted([2**e for e in range(-3, 8)] + [peak / bw])
    ax.plot(xs, [min(peak, bw * x) for x in xs], "k-", lw=1.5, label="_nolegend_")
    ax.text(0.15, bw * 0.15 * 1.25, f"triad {bw:.1f} GB/s", rotation=38, fontsize=8)
    ax.text(16, peak * 1.06, f"FMA peak {peak:.1f} GFLOP/s", fontsize=8)
    ax.axvline(peak / bw, color="grey", ls=":", lw=1)
    ax.text(peak / bw * 1.05, 1.2, f"ridge {peak / bw:.2f}", fontsize=8, color="grey")
    for i, name in enumerate(ORDER):
        if name not in series:
            continue
        n, med, _, _, drift = zip(*series[name])
        x = [ai(v) for v in n]
        ax.plot(x, med, "o-", color=f"C{i}", label=name)
        for xv, yv, d in zip(x, med, drift):
            if d:
                ax.plot([xv], [yv], "o", mfc="white", mec=f"C{i}", mew=1.5, zorder=3)
    for v in [64, 128, 256, 512, 1024]:
        ax.annotate(f"N={v}\nAI≤{ai(v):g}", (ai(v), 1.0), ha="center", fontsize=7, color="grey")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.8, peak * 1.6)
    ax.set_xlabel("arithmetic intensity, FLOP/byte (compulsory DRAM traffic: upper bound)")
    ax.set_ylabel("GFLOP/s (median)")
    ax.set_title(f"matmul roofline, single thread — {pathlib.Path(a.results).name}\n"
                 "ceilings: bench/peak medians of the same session; hollow = drift", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
