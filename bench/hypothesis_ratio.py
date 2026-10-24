#!/usr/bin/env python3
"""Fig. 4: hypothesis A ratio per run, from committed bench/results (no new measurements).

    python3 bench/hypothesis_ratio.py [run1.jsonl run2.jsonl run3.jsonl] [-o docs/hypothesis-a-ratio.png]

Ratio as in BENCHMARKS.md (spec §7.1): median GFLOP/s of the best of the model's top 10
over the median of the best of 1000 random candidates, both from the run's interleaved
decision run (the two `"group":"final"` records, 30 rounds). `search` logs them best random
first, best model second (`log_records(.., "final", &[&random, &model])` in
crates/coffee-ten-bench/src/search.rs); the same ratio as its report. Criterion: ratio ≥ 0.9.
Dev-only tool; needs matplotlib (not a project dependency).
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot import ROOT  # noqa: E402

RES = ROOT / "bench/results"
DEFAULT = [RES / "2026-10-19-hypothesis-a.jsonl", RES / "2026-10-21-hypothesis-a-run2.jsonl",
           RES / "2026-10-22-hypothesis-a-run3.jsonl"]


def ratio(path):
    recs = [json.loads(line) for line in open(path)]
    final = [r for r in recs if r.get("group") == "final"]
    if len(final) != 2:
        sys.exit(f"{path}: expected two final records (best random, best model)")
    rnd, m = final
    return m, rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", default=DEFAULT)
    ap.add_argument("-o", "--out", default=ROOT / "docs/hypothesis-a-ratio.png")
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("bench/hypothesis_ratio.py needs matplotlib: pip install matplotlib")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, path in enumerate(a.results, 1):
        m, r = ratio(path)
        q = m["median"] / r["median"]
        lo, hi = m["p25"] / r["p75"], m["p75"] / r["p25"]
        print(f"run {i} (seed {m['seed']}, {m['commit']}): {m['median']:.2f} / {r['median']:.2f}"
              f" = {q:.3f}")
        ax.errorbar([i], [q], yerr=[[q - lo], [hi - q]], fmt="o", capsize=4,
                    color="C2" if q >= 0.9 else "C3")
        ax.annotate(f"{q:.3f}", (i, q), xytext=(8, -3), textcoords="offset points", fontsize=9)
    ax.axhline(1.0, color="k", lw=1, label="1.0: model best = random best")
    ax.axhline(0.9, color="grey", ls="--", lw=1, label="0.9: criterion (spec §7.1)")
    ax.set_xticks(range(1, len(a.results) + 1))
    ax.set_xticklabels([f"run {i}" for i in range(1, len(a.results) + 1)])
    ax.set_xlim(0.5, len(a.results) + 0.5)
    ax.set_xlabel("hypothesis A run (seed = run number), N = 256")
    ax.set_ylim(0.5, 1.5)
    ax.set_ylabel("ratio, model-best / random-best GFLOP/s (dimensionless)")
    ax.set_title("hypothesis A, matmul N = 256: decision-run medians\n"
                 "bars: p25/p75 combined (model p25 / random p75 .. model p75 / random p25)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
