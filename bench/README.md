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

## Result files

A results file is JSONL: one JSON object per line, appended and never rewritten. Three kinds
of line exist:

- a harness record: one per variant, from `coffee-ten-bench --log` (what `bench/run.sh`
  runs);
- a search record: a harness record with one more field, `group`, last, from
  `coffee-ten-bench search --log`;
- a peak line: from `bench/peak`, with `"kind":"peak"` put first; `bench/run.sh` writes one
  as the first line of its file, and a peak-only run is a file of its own. Search logs have
  none (see Peak line below). The other lines have no `kind`.

### Record fields

The rows are in the order the keys are written. Every record has every field; the order and
the key set are checked by a test in `crates/coffee-ten-bench/src/search.rs`, which reads
this table, so change the table and the writer together. Writers: `harness` is `record_json`
in `crates/coffee-ten-bench/src/run.rs`, `meta` is `Metadata` in
`crates/coffee-ten-bench/src/meta.rs` (taken once per process, so every line of a run has the
same values), `search` is `group_json` in `crates/coffee-ten-bench/src/search.rs`. Only
search lines have `group`; the files `run.sh` writes have none.

| Field | Type | Unit | Meaning | Writer |
| --- | --- | --- | --- | --- |
| `kernel` | string | - | `matmul` or `relu-matmul-bias` | harness |
| `n` | integer | - | matrix size, all dimensions N | harness |
| `graph_hash` | string | - | 16 hex digits: FNV-1a 64 of the lowered kernel's text before any schedule; equal hash, same kernel | harness |
| `schedule` | string | - | canonical schedule text; `""` for the naive kernel (not the variant text as typed) | harness |
| `order_class` | string | - | `preserving` (bit for bit equal to naive) or `reordering` (equal within the error bound) | harness |
| `verified` | bool | - | always `true`: a variant that fails the check against naive aborts the run before it is logged | harness |
| `quick` | bool | - | the run was `--quick` (3 samples): not a measurement | harness |
| `samples` | integer | - | number of samples (rounds) behind the statistics | harness |
| `median` | number or null | GFLOP/s | sorted rates r, element r[(n-1)/2], no interpolation | harness |
| `p25` | number or null | GFLOP/s | r[(n-1)/4] | harness |
| `p75` | number or null | GFLOP/s | r[3(n-1)/4] | harness |
| `min` | number or null | GFLOP/s | r[0], the slowest sample | harness |
| `drift` | bool | - | median of the first 5 rounds and of the last 5 differ by more than 5% of the first; always `false` below 10 samples | harness |
| `unit` | string | - | always `GFLOP/s`, the unit of `median`, `p25`, `p75`, `min` | harness |
| `seed` | integer | - | `--seed`: seeds the input matrices and the round order (in `search` also the candidate sample) | meta |
| `commit` | string | - | `git rev-parse --short HEAD`; `unknown` if git fails | meta |
| `dirty` | bool or null | - | tracked files differ from `commit`; `null` if git is unavailable | meta |
| `clang` | string | - | first line of `clang --version`; `unknown` if it fails | meta |
| `flags` | string | - | the C compiler flags of the benchmark build, space-separated | meta |
| `machine` | string | - | `sysctl -n hw.model`; `unknown` off macOS | meta |
| `cpu` | string | - | `sysctl -n machdep.cpu.brand_string`; `unknown` off macOS | meta |
| `os` | string | - | `sw_vers -productVersion`; `unknown` off macOS | meta |
| `date` | string | - | UTC start of the process, `YYYY-MM-DDTHH:MM:SSZ`; `unknown` if `date` fails | meta |
| `group` | string | - | `search` lines only, always last; which part of the search measured the line (below) | search |

A rate that is NaN or infinite is written as `null`; all other numbers are the shortest text
that reads back to the same value. The variant text as typed on the command line is not
logged; `schedule` is its canonical form. Files from before `search` have the same 23 fields
without `group`.

`group` values:

- `naive`: the reference kernel, measured first; `schedule` is `""`; not counted in `--budget`
- `random`: the random sample, in the order measured
- `model`: the model's top K (`--model-top`), in rank order
- `local`: the steps of the hill climb (`--local`), in the order measured
- `final`: the interleaved decision run, two lines: best random, then best model

### Peak line

The line `bench/peak` prints last, with `"kind":"peak"` put in front by `run.sh`. For a
standalone run, the files `bench/results/*-peak.jsonl` hold only this line. It has no
metadata fields. Every row in the table with type `stats` is an object `{"median", "p25",
"p75", "min", "n"}` over `n` samples, in the unit of its field. Older files have fewer
rows; each new kernel adds rows at the end, so the order below is the order of the newest
file. These rows are not checked by a test.

| Field | Type | Unit | Meaning | Writer |
| --- | --- | --- | --- | --- |
| `kind` | string | - | always `peak` | run.sh |
| `fma_gflops` | stats | GFLOP/s | vector FMA throughput, registers only | peak.c |
| `triad_gbs` | stats | GB/s | STREAM triad, memory | peak.c |
| `fma_chains` | integer | - | independent FMA chains in `fma_gflops` | peak.c |
| `triad_n` | integer | floats | length of each triad array | peak.c |
| `triad_l2_gbs` | stats | GB/s | STREAM triad, working set in L2 | peak.c |
| `triad_l2_n` | integer | floats | length of each L2 triad array | peak.c |
| `fma_mem_acc_gflops` | stats | GFLOP/s | vector FMA with accumulator loaded and stored | peak.c |
| `mem_acc_n` | integer | floats | length of the arrays of the `mem_acc` kernels | peak.c |
| `fma_mem_acc_scalar_gflops` | stats | GFLOP/s | the same, scalar | peak.c |
| `fma_chain_gflops` | stats | GFLOP/s | one dependent scalar FMA chain | peak.c |
| `fma_chain256_gflops` | stats | GFLOP/s | 256 chains of length 256, as the naive matmul reduction | peak.c |
| `fma_store_gflops` | stats | GFLOP/s | vector FMA, every result stored, nothing loaded | peak.c |
| `fma_mem_acc_scalar_unrolled_gflops` | stats | GFLOP/s | scalar load/FMA/store, 16 floats per iteration | peak.c |
| `fma_col_reload_gflops` | stats | GFLOP/s | scalar matmul loop nest that reloads a column of C | peak.c |
| `fma_col_unrolled_gflops` | stats | GFLOP/s | scalar matmul tile of 16 C values in registers, A read down a column | peak.c |

The kernels are described in `bench/peak/README.md`.
