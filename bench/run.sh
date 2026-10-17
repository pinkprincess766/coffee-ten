#!/usr/bin/env bash
# One benchmark run for BENCHMARKS.md: peak once, then matmul and relu-matmul-bias.
# Usage: bench/run.sh [--quick] [--sizes "64 128"] [--pause SECONDS]
#                     [--dry-run] [--yes] [--allow-dirty]
set -euo pipefail

cd "$(dirname "$0")/.."

sizes="64 128 256 512 1024"
pause_s=60
quick=0
dry_run=0
yes=0
allow_dirty=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) quick=1 ;;
    --dry-run) dry_run=1 ;;
    --yes) yes=1 ;;
    --allow-dirty) allow_dirty=1 ;;
    --sizes) sizes="${2:?--sizes needs a value}"; shift ;;
    --pause) pause_s="${2:?--pause needs a value}"; shift ;;
    *) echo "usage: bench/run.sh [--quick] [--sizes \"64 128\"] [--pause S] [--dry-run] [--yes] [--allow-dirty]" >&2; exit 2 ;;
  esac
  shift
done

dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$dirty" ]]; then
  if [[ $allow_dirty -eq 0 ]]; then
    echo "error: uncommitted changes; a number must come from a commit (or pass --allow-dirty):" >&2
    echo "$dirty" >&2
    exit 1
  fi
  echo "warning: --allow-dirty: this run is not reproducible from the commit" >&2
fi

commit="$(git rev-parse --short HEAD)"
day="$(date -u +%Y-%m-%d)"
if [[ $quick -eq 1 ]]; then
  results="bench/results/quick-$day-$commit.jsonl"
else
  results="bench/results/$day-$commit.jsonl"
fi
if [[ $dry_run -eq 0 && -s "$results" ]]; then
  echo "error: $results already has runs; move it aside, one file holds one peak and one series" >&2
  exit 1
fi

warn() { echo "WARNING: $*" >&2; }
if ! grep -q "'AC Power'" <<< "$(pmset -g batt)"; then
  warn "not on AC power (pmset -g batt)"
fi
if grep -Eq 'lowpowermode[[:space:]]+1' <<< "$(pmset -g)"; then
  warn "Low Power Mode is on (pmset -g)"
fi
echo "commit $commit  results $results"
echo "Conditions: close heavy apps; do not use the machine during the run."
echo "Also: no builds, downloads or other jobs while it runs."
if [[ $dry_run -eq 0 && $yes -eq 0 ]]; then
  reply=""
  read -r -p "Conditions checked and met? [y/N] " reply || true
  [[ "$reply" == [yY] ]] || { echo "aborted" >&2; exit 1; }
fi

# Prints the command, then runs it unless --dry-run.
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ $dry_run -eq 0 ]]; then "$@"; fi
}

target_dir="${CARGO_TARGET_DIR:-target}"
peak_bin="$target_dir/peak"
bench_bin="$target_dir/release/coffee-ten-bench"
# Extra flag only on x86, as bench/peak/README.md says; the array is empty on arm64.
march=()
if [[ "$(uname -m)" == x86_64 ]]; then march=(-march=native); fi
mode=()
if [[ $quick -eq 1 ]]; then mode=(--quick); fi

run cargo build --release --locked -p coffee-ten-bench
run clang -std=c99 -O2 -ffp-contract=fast ${march[@]+"${march[@]}"} -o "$peak_bin" bench/peak/peak.c

# The peak line is the JSON object that peak prints last; its first "{" is replaced.
if [[ $dry_run -eq 1 ]]; then
  run "$peak_bin" ${mode[@]+"${mode[@]}"}
  echo "+ append the peak JSON line as {\"kind\":\"peak\",...} to $results"
else
  peak_out="$("$peak_bin" ${mode[@]+"${mode[@]}"})"
  echo "$peak_out"
  peak_json="$(grep '^{"fma_gflops"' <<< "$peak_out")" || { echo "error: no JSON line from peak" >&2; exit 1; }
  printf '{"kind":"peak",%s\n' "${peak_json:1}" >> "$results"
fi

tiled="split(1, i, io, ii, 32); split(1, j, jo, ji, 64); reorder(1, io, jo, k, ii, ji)"
tiled_vec="$tiled; vectorize(1, ji)"
# 4x16 register tile of C kept in vector registers across k; nest 1 is the matmul update
# nest in both kernels.
regtile="split(1, i, io, ii, 4); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)"
first=1
for kernel in matmul relu-matmul-bias; do
  for n in $sizes; do
    if [[ $first -eq 0 && $pause_s -gt 0 ]]; then
      run sleep "$pause_s"
    fi
    first=0
    run "$bench_bin" --kernel "$kernel" --n "$n" \
      --variant "" --variant "$tiled" --variant "$tiled_vec" --variant "$regtile" \
      --log "$results" ${mode[@]+"${mode[@]}"}
  done
done

if [[ $dry_run -eq 1 ]]; then
  echo "dry run: nothing was run; results would be $results"
else
  echo "results: $results"
fi
echo "Numbers go into BENCHMARKS.md only after you checked the conditions above."
