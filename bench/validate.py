#!/usr/bin/env python3
"""Strict current C16/N1/N2 JSONL validation for CI; Python standard library only."""
import json
import math
import re
import sys
from pathlib import Path

RECORD_FIELDS = set("kernel n graph_hash schedule order_class verified quick samples median p25 p75 min drift rounds protocol measurement_ns compilation_ns cache_hit reference_compilation compilation_wall_ns unit seed commit dirty clang flags machine cpu os date".split())
PROTOCOL_FIELDS = set("version n kernel graph_hash space space_hash source_hash cpu machine os arch target_os clang flags rustc harness_profile rustflags samples min_sample_ms input_seed quick warmup_ms candidate_warmup_ms".split())
PEAK_STATS = set("fma_gflops triad_gbs triad_l2_gbs fma_mem_acc_gflops fma_mem_acc_scalar_gflops fma_chain_gflops fma_chain256_gflops fma_store_gflops fma_mem_acc_scalar_unrolled_gflops fma_col_reload_gflops fma_col_unrolled_gflops".split())
PEAK_INTS = set("fma_chains triad_n triad_l2_n mem_acc_n".split())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, expected, label):
    require(type(value) is dict, f"{label}: expected object")
    missing, extra = expected - value.keys(), value.keys() - expected
    require(not missing and not extra, f"{label}: missing={sorted(missing)}, extra={sorted(extra)}")


def uint(value, label, bits=64, positive=False):
    require(type(value) is int and (1 if positive else 0) <= value < 2**bits,
            f"{label}: expected {'positive' if positive else 'nonnegative'} u{bits}")


def string(value, label):
    require(type(value) is str, f"{label}: expected string")


def boolean(value, label):
    require(type(value) is bool, f"{label}: expected boolean")


def rate(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{label}: expected finite positive rate")


def fingerprint(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{16}", value) is not None,
            f"{label}: expected 16 lowercase hex digits")


def stats(value):
    fields(value, {"median", "p25", "p75", "min", "n"}, "peak stats")
    uint(value["n"], "peak n", positive=True)
    for name in ("median", "p25", "p75", "min"):
        rate(value[name], name)
    require(value["min"] <= value["p25"] <= value["median"] <= value["p75"], "unordered peak statistics")


def protocol(key):
    require(type(key) is dict, "protocol: expected object")
    space = key.get("space")
    require(space in ("v1", "v2"), "unsupported space")
    fields(key, PROTOCOL_FIELDS | ({"cache", "tile_factors"} if space == "v2" else set()), "protocol")
    require(type(key["version"]) is int and key["version"] == 1, "unsupported protocol version")
    for name in ("n", "samples"):
        uint(key[name], name, positive=True)
    for name in ("min_sample_ms", "input_seed", "warmup_ms", "candidate_warmup_ms"):
        uint(key[name], name)
    boolean(key["quick"], "protocol quick")
    for name in PROTOCOL_FIELDS - {"version", "n", "samples", "min_sample_ms", "input_seed", "quick", "warmup_ms", "candidate_warmup_ms"}:
        string(key[name], f"protocol {name}")
    for name in ("graph_hash", "space_hash", "source_hash"):
        fingerprint(key[name], name)
    require(key["kernel"] in ("matmul", "relu-matmul-bias"), "unsupported kernel")
    require(key["harness_profile"] in ("debug", "release"), "unsupported harness profile")
    if space == "v2":
        factors = key["tile_factors"]
        require(type(factors) is list and bool(factors), "tile_factors: expected nonempty array")
        for factor in factors:
            uint(factor, "tile factor", positive=True)
        require(factors == sorted(set(factors)), "tile_factors: must be sorted and unique")
        cache = key["cache"]
        if cache is not None:
            fields(cache, {"l1_bytes", "l2_bytes", "line_bytes"}, "cache")
            for name, value in cache.items():
                uint(value, name, positive=True)
            line = cache["line_bytes"]
            require(line >= 4 and line & (line - 1) == 0 and cache["l2_bytes"] >= cache["l1_bytes"] >= line, "invalid cache sizes")


def compilation(value):
    fields(value, {"elapsed_ns", "cache_hit"}, "reference compilation")
    uint(value["elapsed_ns"], "reference elapsed_ns", bits=128)
    boolean(value["cache_hit"], "reference cache_hit")


def validate(value):
    require(type(value) is dict, "expected record object")
    if value.get("kind") == "peak":
        fields(value, {"kind"} | PEAK_STATS | PEAK_INTS, "peak")
        for name in PEAK_STATS:
            stats(value[name])
        for name in PEAK_INTS:
            uint(value[name], name, positive=True)
        return
    fields(value, RECORD_FIELDS | ({"group"} if "group" in value else set()), "record")
    key = value["protocol"]
    protocol(key)
    for name in ("schedule", "unit", "commit", "date", "order_class"):
        string(value[name], name)
    for name in ("kernel", "graph_hash", "cpu", "machine", "os", "clang", "flags"):
        string(value[name], name)
        require(value[name] == key[name], f"inconsistent {name}")
    require(value["unit"] == "GFLOP/s", "unsupported unit")
    require(value["order_class"] in ("preserving", "reordering"), "invalid order_class")
    for name in ("verified", "quick", "drift", "cache_hit"):
        boolean(value[name], name)
    require(value["verified"], "unverified measurement")
    require(value["dirty"] is None or type(value["dirty"]) is bool, "dirty: expected bool or null")
    for name in ("n", "samples"):
        uint(value[name], name, positive=True)
        require(value[name] == key[name], f"inconsistent {name}")
    uint(value["seed"], "seed")
    require(value["seed"] == key["input_seed"] and value["quick"] == key["quick"], "inconsistent seed/quick")
    for name in ("measurement_ns", "compilation_ns"):
        uint(value[name], name, bits=128)
    require(value["compilation_wall_ns"] is None, "compilation_wall_ns: reserved null")
    if value["reference_compilation"] is not None:
        compilation(value["reference_compilation"])
    if "group" in value:
        require(value["group"] in ("naive", "random", "model", "local", "final"), "invalid group")
        if value["group"] == "naive":
            require(value["schedule"] == "", "naive schedule must be empty")
    raw = value["rounds"]
    require(type(raw) is list and len(raw) == value["samples"], "wrong round count")
    for index, sample in enumerate(raw):
        fields(sample, {"round_index", "elapsed_ns", "iterations", "gflops"}, "round")
        uint(sample["round_index"], "round_index")
        require(sample["round_index"] == index, "wrong round order")
        for name in ("elapsed_ns", "iterations"):
            uint(sample[name], name, positive=True)
        require(sample["elapsed_ns"] >= key["min_sample_ms"] * 1_000_000,
                "raw round is below the protocol minimum duration")
        rate(sample["gflops"], "gflops")
        seconds, nanos = divmod(sample["elapsed_ns"], 1_000_000_000)
        # Mirror Rust Duration::as_secs_f64, including its two separate conversions.
        elapsed_s = float(seconds) + float(nanos) / 1e9
        n = float(value["n"])
        expected = 2.0 * ((n * n) * n) * float(sample["iterations"]) / elapsed_s / 1e9
        require(float(sample["gflops"]) == expected, "raw rate does not match timing")
    rates = sorted(sample["gflops"] for sample in raw)
    last = len(rates) - 1
    for name, index in [("min", 0), ("p25", last // 4), ("median", last // 2), ("p75", 3 * last // 4)]:
        rate(value[name], name)
        require(value[name] == rates[index], f"{name} differs from raw rounds")
    require(value["measurement_ns"] == sum(sample["elapsed_ns"] for sample in raw), "measurement cost differs from rounds")
    # Keep the live harness rule: disjoint five-round windows, strictly over 5%.
    drift = False
    if len(raw) >= 10:
        first = sorted(sample["gflops"] for sample in raw[:5])[2]
        last = sorted(sample["gflops"] for sample in raw[-5:])[2]
        drift = abs(last - first) / first > 0.05
    require(value["drift"] == drift, "drift differs from raw rounds")


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, f"duplicate key {name}")
        result[name] = value
    return result


def validate_file(path):
    count = 0
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            require(bool(line.strip()), "empty JSONL line")
            value = json.loads(line, object_pairs_hook=unique_object,
                               parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-JSON number {token}")))
            validate(value)
            count += 1
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{path}:{index}: {exc}") from exc
    require(count > 0, f"{path}: empty log")
    return count


def main(paths):
    require(bool(paths), "usage: python3 bench/validate.py <current-log.jsonl> [...]")
    for path in paths:
        print(f"{path}: {validate_file(path)} valid current-schema records")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
