# bench

`bench/run.sh` takes the numbers for BENCHMARKS.md in one command. It refuses a dirty
tree, checks the power conditions, asks you to confirm the rest, builds
`coffee-ten-bench` and `bench/peak`, runs `peak` once, then runs `matmul` and
`relu-matmul-bias` for each N. Each (kernel, N) is one invocation with three variants:
naive, tiled, tiled + vectorized. Output: `bench/results/<UTC date>-<commit>.jsonl`,
one `peak` line and one line per variant.

    bench/run.sh                                 # full run, asks to confirm conditions
    bench/run.sh --yes                           # skip the prompt
    bench/run.sh --dry-run                       # print the commands only
    bench/run.sh --quick --sizes "64" --pause 0  # smoke run -> bench/results/quick-*

- `--sizes "64 128"`: N values (default `64 128 256 512 1024`)
- `--pause S`: seconds between (kernel, N) runs (default 60, `0` skips)
- `--allow-dirty`: run on uncommitted changes; not reproducible, do not publish
- `--quick`: 3 samples, not a measurement; the file is git-ignored

Method: variants interleaved within each run; 30 samples per variant, each at least
10 ms; median, p25 and p75 (IQR) reported; a drift flag when the rate moves more than
5% between the first and last 5 rounds; pauses between runs.

Conditions: AC power, Low Power Mode off, heavy apps closed, no other jobs during the
run. `run.sh` checks the first two and warns; you confirm the rest. Check them before
copying numbers into BENCHMARKS.md.

FMA contraction is allowed in benchmark builds (`-ffp-contract=fast`, as for peak);
the bench header prints it.
