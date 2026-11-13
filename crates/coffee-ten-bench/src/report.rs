//! Read-only Markdown analysis of saved N1/N2 measurement records.
use crate::{
    BenchError,
    json::{self, Value},
    run::{RoundSample, fnv1a_64},
    stats::{
        BootstrapConfig, MIN_ROUNDS_FOR_DRIFT, bootstrap_independent_ratio, bootstrap_median,
        bootstrap_paired_ratio,
    },
};
use std::fmt::Write as _;

pub const USAGE: &str = "usage: coffee-ten-bench report <run.jsonl> [--top <K>] [--seed <SEED>] [--resamples <B>]\n       coffee-ten-bench report --compare <A.jsonl> <B.jsonl> [--paired] [--seed <SEED>] [--resamples <B>]";

#[derive(Debug)]
pub(crate) struct Row {
    pub protocol: Value,
    pub schedule: String,
    pub group: String,
    pub rounds: Vec<RoundSample>,
    pub median: f64,
    pub measurement_ns: u128,
    pub compilation_ns: u128,
    pub drift: bool,
}

/// Errors that carry the usage text are about the command line (`BenchError::Usage`, exit 2);
/// everything else is about the input data (`BenchError::Setup`, exit 1).
fn error(message: impl std::fmt::Display) -> BenchError {
    let message = format!("report: {message}");
    if message.ends_with(USAGE) {
        BenchError::Usage(message)
    } else {
        BenchError::Setup(message)
    }
}

/// String fields of the persisted comparison key (`ProtocolKey::to_json`).
const TEXT_FIELDS: [&str; 15] = [
    "kernel",
    "graph_hash",
    "space",
    "space_hash",
    "source_hash",
    "cpu",
    "machine",
    "os",
    "arch",
    "target_os",
    "clang",
    "flags",
    "rustc",
    "harness_profile",
    "rustflags",
];
/// Unsigned integer fields of the persisted comparison key.
const UINT_FIELDS: [&str; 7] = [
    "version",
    "n",
    "samples",
    "min_sample_ms",
    "input_seed",
    "warmup_ms",
    "candidate_warmup_ms",
];
/// Boolean fields of the persisted comparison key.
const BOOL_FIELDS: [&str; 1] = ["quick"];
/// Fields written only for non-V1 schedule spaces.
const SPACE_FIELDS: [&str; 2] = ["cache", "tile_factors"];

/// Validate the complete persisted comparison key, including fields not printed below.
/// Equality between two incomplete objects is not evidence of matching provenance, and an
/// unknown field is provenance nobody checks, so the key must be exactly what the writer emits.
fn validate_protocol(protocol: &Value) -> Result<(), String> {
    let Value::Object(fields) = protocol else {
        return Err("protocol must be an object".into());
    };
    let v1 = protocol
        .get("space")
        .and_then(Value::text)
        .map_err(|e| format!("protocol space: {e}"))?
        == "v1";
    for key in fields.keys() {
        let known = TEXT_FIELDS.contains(&key.as_str())
            || UINT_FIELDS.contains(&key.as_str())
            || BOOL_FIELDS.contains(&key.as_str())
            || (!v1 && SPACE_FIELDS.contains(&key.as_str()));
        if !known {
            return Err(format!("unknown protocol field {key}"));
        }
    }
    for field in TEXT_FIELDS {
        protocol
            .get(field)
            .and_then(Value::text)
            .map_err(|e| format!("protocol {field}: {e}"))?;
    }
    for field in UINT_FIELDS {
        protocol
            .get(field)
            .and_then(Value::uint::<u64>)
            .map_err(|e| format!("protocol {field}: {e}"))?;
    }
    for field in BOOL_FIELDS {
        protocol
            .get(field)
            .and_then(Value::boolean)
            .map_err(|e| format!("protocol {field}: {e}"))?;
    }
    match protocol.get("space")?.text()? {
        "v1" => {}
        "v2" => {
            let factors = protocol.get("tile_factors")?.array()?;
            if factors.is_empty() {
                return Err("empty tile factors".into());
            }
            let mut previous = 0;
            for factor in factors {
                let factor = factor.uint::<usize>()?;
                if factor <= previous {
                    return Err("tile factors must be positive, sorted and unique".into());
                }
                previous = factor;
            }
            let cache = protocol.get("cache")?;
            if let Value::Object(cache_fields) = cache
                && let Some(key) = cache_fields
                    .keys()
                    .find(|k| !["l1_bytes", "l2_bytes", "line_bytes"].contains(&k.as_str()))
            {
                return Err(format!("unknown protocol cache field {key}"));
            }
            if cache != &Value::Null
                && crate::space::CacheInfo::new(
                    cache.get("l1_bytes")?.uint()?,
                    cache.get("l2_bytes")?.uint()?,
                    cache.get("line_bytes")?.uint()?,
                )
                .is_none()
            {
                return Err("invalid cache sizes".into());
            }
        }
        _ => return Err("unsupported schedule space".into()),
    }
    Ok(())
}

/// Record metadata that the report does not use but `bench/validate.py` requires, checked
/// with the same rules so the two validators accept the same logs.
const ORDER_CLASSES: [&str; 2] = ["preserving", "reordering"];
const GROUPS: [&str; 5] = ["naive", "random", "model", "local", "final"];

fn check_metadata(row: &Value) -> Result<(), String> {
    for name in ["commit", "date"] {
        row.get(name)?.text().map_err(|e| format!("{name}: {e}"))?;
    }
    let dirty = row.get("dirty")?;
    if dirty != &Value::Null && dirty.boolean().is_err() {
        return Err("dirty: expected bool or null".into());
    }
    let order_class = row
        .get("order_class")?
        .text()
        .map_err(|e| format!("order_class: {e}"))?;
    if !ORDER_CLASSES.contains(&order_class) {
        return Err(format!("invalid order_class {order_class:?}"));
    }
    if let Ok(group) = row.get("group") {
        let group = group.text().map_err(|e| format!("group: {e}"))?;
        if !GROUPS.contains(&group) {
            return Err(format!("invalid group {group:?}"));
        }
        if group == "naive" && !row.get("schedule")?.text()?.is_empty() {
            return Err("naive schedule must be empty".into());
        }
    }
    Ok(())
}

fn parse_row(row: &Value) -> Result<Row, String> {
    let protocol = row.get("protocol").map_err(
        |_| "N1/N2 log required: raw rounds and protocol are unavailable in legacy logs",
    )?;
    check_known_fields(row)?;
    validate_protocol(protocol)?;
    check_metadata(row)?;
    if protocol.get("version")?.uint::<u32>()? != crate::protocol::VERSION {
        return Err("unsupported protocol version".into());
    }
    let n = protocol.get("n")?.uint::<usize>()?;
    if n == 0 || n != row.get("n")?.uint::<usize>()? {
        return Err("inconsistent matrix size".into());
    }
    for field in [
        "kernel",
        "graph_hash",
        "cpu",
        "machine",
        "os",
        "clang",
        "flags",
    ] {
        if row.get(field)?.text()? != protocol.get(field)?.text()? {
            return Err(format!("inconsistent {field}"));
        }
    }
    if row.get("unit")?.text()? != "GFLOP/s"
        || !row.get("verified")?.boolean()?
        || row.get("quick")?.boolean()? != protocol.get("quick")?.boolean()?
        || row.get("seed")?.uint::<u64>()? != protocol.get("input_seed")?.uint::<u64>()?
    {
        return Err("inconsistent measurement metadata".into());
    }
    let count = row.get("samples")?.uint::<usize>()?;
    let raw = row.get("rounds")?.array()?;
    let declared = protocol.get("samples")?.uint::<usize>()?;
    if count == 0 || count != raw.len() || count != declared {
        return Err(format!(
            "wrong raw round count: {} rounds, samples {count}, protocol samples {declared}",
            raw.len()
        ));
    }
    let checked = crate::rounds::validate(raw, n, protocol.get("min_sample_ms")?.uint::<u64>()?)?;
    crate::rounds::check_aggregates(row, &checked)?;
    let drift = row.get("drift")?.boolean()?;
    if row.get("compilation_wall_ns")? != &Value::Null {
        return Err("unsupported compilation wall accounting".into());
    }
    row.get("cache_hit")?.boolean()?;
    let mut compilation_ns = row.get("compilation_ns")?.uint::<u128>()?;
    let reference = row.get("reference_compilation")?;
    if reference != &Value::Null {
        reference.get("cache_hit")?.boolean()?;
        compilation_ns = compilation_ns
            .checked_add(reference.get("elapsed_ns")?.uint()?)
            .ok_or("compilation cost overflow")?;
    }
    let group = match row.get("group") {
        Ok(value) => value.text()?,
        Err(_) => "harness",
    };
    Ok(Row {
        protocol: protocol.clone(),
        schedule: row.get("schedule")?.text()?.into(),
        group: group.into(),
        rounds: checked.rounds,
        median: checked.stats.median,
        measurement_ns: checked.measurement_ns,
        compilation_ns,
        drift,
    })
}

/// Every field a current record may carry (`bench/validate.py` `RECORD_FIELDS`, plus the
/// optional `group`). Like the protocol key, the record, its raw rounds and its reference
/// compilation are closed: a field nobody checks is rejected rather than ignored.
const RECORD_FIELDS: [&str; 31] = [
    "kernel",
    "n",
    "graph_hash",
    "schedule",
    "order_class",
    "verified",
    "quick",
    "samples",
    "median",
    "p25",
    "p75",
    "min",
    "drift",
    "rounds",
    "protocol",
    "measurement_ns",
    "compilation_ns",
    "cache_hit",
    "reference_compilation",
    "compilation_wall_ns",
    "unit",
    "seed",
    "commit",
    "dirty",
    "clang",
    "flags",
    "machine",
    "cpu",
    "os",
    "date",
    "group",
];
const ROUND_FIELDS: [&str; 4] = ["round_index", "elapsed_ns", "iterations", "gflops"];
const REFERENCE_FIELDS: [&str; 2] = ["elapsed_ns", "cache_hit"];

fn unknown_field<'a>(value: &'a Value, known: &[&str]) -> Option<&'a str> {
    match value {
        Value::Object(fields) => fields
            .keys()
            .map(String::as_str)
            .find(|key| !known.contains(key)),
        _ => None,
    }
}

fn check_known_fields(row: &Value) -> Result<(), String> {
    if let Some(key) = unknown_field(row, &RECORD_FIELDS) {
        return Err(format!("unknown record field {key}"));
    }
    if let Ok(reference) = row.get("reference_compilation")
        && let Some(key) = unknown_field(reference, &REFERENCE_FIELDS)
    {
        return Err(format!("unknown reference_compilation field {key}"));
    }
    if let Ok(rounds) = row.get("rounds").and_then(Value::array) {
        for (index, round) in rounds.iter().enumerate() {
            if let Some(key) = unknown_field(round, &ROUND_FIELDS) {
                return Err(format!("round {index}: unknown field {key}"));
            }
        }
    }
    Ok(())
}

pub(crate) fn read(text: &str) -> Result<Vec<Row>, BenchError> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        // As in `bench/validate.py`: every line is a record, so a blank one is an error
        // (an interrupted or hand-edited append), not something to step over.
        if line.trim().is_empty() {
            return Err(error(format!("line {}: empty JSONL line", index + 1)));
        }
        let parse = || -> Result<Option<Row>, String> {
            let value = json::parse(line)?;
            if !matches!(value, Value::Object(_)) {
                return Err("expected one JSON object per line".into());
            }
            if let Ok(kind) = value.get("kind") {
                if kind.text().map_err(|e| format!("kind: {e}"))? == "peak" {
                    check_peak(&value)?;
                    return Ok(None);
                }
                return Err("expected live measurements, not offline curve records".into());
            }
            parse_row(&value).map(Some)
        };
        if let Some(row) = parse().map_err(|e| error(format!("line {}: {e}", index + 1)))? {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return Err(error("no measurement records"));
    }
    Ok(rows)
}

/// `bench/peak` statistics a peak line carries (`bench/validate.py` `PEAK_STATS`).
const PEAK_STATS: [&str; 11] = [
    "fma_gflops",
    "triad_gbs",
    "triad_l2_gbs",
    "fma_mem_acc_gflops",
    "fma_mem_acc_scalar_gflops",
    "fma_chain_gflops",
    "fma_chain256_gflops",
    "fma_store_gflops",
    "fma_mem_acc_scalar_unrolled_gflops",
    "fma_col_reload_gflops",
    "fma_col_unrolled_gflops",
];
/// Positive problem sizes of the peak benchmarks (`PEAK_INTS`).
const PEAK_INTS: [&str; 4] = ["fma_chains", "triad_n", "triad_l2_n", "mem_acc_n"];
const PEAK_STAT_FIELDS: [&str; 5] = ["median", "p25", "p75", "min", "n"];

/// The report ranks no peak line, but it checks one as `bench/validate.py` does instead of
/// skipping it unread: exactly the current fields, positive sizes and ordered statistics.
fn check_peak(peak: &Value) -> Result<(), String> {
    let Value::Object(fields) = peak else {
        return Err("peak: expected object".into());
    };
    let known = |key: &str| key == "kind" || PEAK_STATS.contains(&key) || PEAK_INTS.contains(&key);
    if let Some(key) = fields.keys().find(|key| !known(key)) {
        return Err(format!("peak: unknown field {key}"));
    }
    for name in PEAK_INTS {
        let value = peak
            .get(name)
            .map_err(|e| format!("peak: {e}"))?
            .uint::<u64>()
            .map_err(|e| format!("peak {name}: {e}"))?;
        if value == 0 {
            return Err(format!("peak {name}: expected positive integer"));
        }
    }
    for name in PEAK_STATS {
        let stats = peak.get(name).map_err(|e| format!("peak: {e}"))?;
        let Value::Object(fields) = stats else {
            return Err(format!("peak {name}: expected object"));
        };
        if let Some(key) = fields
            .keys()
            .find(|key| !PEAK_STAT_FIELDS.contains(&key.as_str()))
        {
            return Err(format!("peak {name}: unknown field {key}"));
        }
        let count = stats
            .get("n")
            .map_err(|e| format!("peak {name}: {e}"))?
            .uint::<u64>()
            .map_err(|e| format!("peak {name} n: {e}"))?;
        if count == 0 {
            return Err(format!("peak {name} n: expected positive integer"));
        }
        let mut rates = [0.0; 4];
        for (rate, field) in rates.iter_mut().zip(["min", "p25", "median", "p75"]) {
            *rate = stats
                .get(field)
                .map_err(|e| format!("peak {name}: {e}"))?
                .number()
                .map_err(|e| format!("peak {name} {field}: {e}"))?;
            if *rate <= 0.0 {
                return Err(format!(
                    "peak {name} {field}: expected finite positive rate"
                ));
            }
        }
        if !rates.is_sorted() {
            return Err(format!("peak {name}: unordered statistics"));
        }
    }
    Ok(())
}

/// The Drift column. With fewer than `MIN_ROUNDS_FOR_DRIFT` rounds the check never ran, so a
/// stored `false` would read as "no drift" when nothing was tested.
fn drift_cell(row: &Row) -> &'static str {
    if row.rounds.len() < MIN_ROUNDS_FOR_DRIFT {
        "not evaluated"
    } else if row.drift {
        "true"
    } else {
        "false"
    }
}

/// Text from the log as inert Markdown: no HTML, no table breaks, no code spans, no
/// emphasis, no links. Every special character becomes a numeric entity that renders as itself.
fn cell(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace('`', "&#96;")
        .replace('\\', "&#92;")
        .replace('*', "&#42;")
        .replace('_', "&#95;")
        .replace('[', "&#91;")
        .replace(']', "&#93;")
        .replace('~', "&#126;")
        .replace(['\n', '\r'], " ")
}

/// Each protocol key gets its own table; rounds from different rows are never pooled.
/// Costs include all input rows, including those outside top-k and reference compilation.
pub fn markdown(text: &str, top: usize, bootstrap: BootstrapConfig) -> Result<String, BenchError> {
    if top == 0 {
        return Err(error("--top must be positive"));
    }
    let rows = read(text)?;
    let mut groups: Vec<(&Value, Vec<&Row>)> = Vec::new();
    for row in &rows {
        if let Some((_, entries)) = groups.iter_mut().find(|(key, _)| *key == &row.protocol) {
            entries.push(row);
        } else {
            groups.push((&row.protocol, vec![row]));
        }
    }
    let mut out = format!(
        "# Run report\n\nInput fingerprint: `{:016x}`.\n\n95% percentile bootstrap: {} resamples, seed {}. Intervals describe supplied rounds, not process restarts or winner-selection uncertainty.\n",
        fnv1a_64(text.as_bytes()),
        bootstrap.resamples,
        bootstrap.seed
    );
    for (index, (key, mut entries)) in groups.into_iter().enumerate() {
        let mut measured = 0u128;
        let mut compiled = 0u128;
        for row in &entries {
            measured = measured
                .checked_add(row.measurement_ns)
                .ok_or_else(|| error("measurement cost overflow"))?;
            compiled = compiled
                .checked_add(row.compilation_ns)
                .ok_or_else(|| error("compilation cost overflow"))?;
        }
        let kernel = cell(key.get("kernel").and_then(Value::text).map_err(error)?);
        let n = key.get("n").and_then(Value::uint::<usize>).map_err(error)?;
        writeln!(out, "\n## Protocol {}: {kernel}, n={n}\n", index + 1).unwrap();
        for field in [
            "space",
            "space_hash",
            "source_hash",
            "cpu",
            "clang",
            "harness_profile",
        ] {
            writeln!(
                out,
                "- {field}: {}",
                cell(key.get(field).and_then(Value::text).map_err(error)?)
            )
            .unwrap();
        }
        if key.get("quick").and_then(Value::boolean).map_err(error)? {
            out.push_str("\n**Quick smoke data; not a performance measurement.**\n");
        }
        writeln!(out, "\nRecords: {}. Measurement: {measured} ns. Compilation API work: {compiled} ns (includes reference; summed work, not wall time).\n", entries.len()).unwrap();
        // Costs above cover every record; say how many of them the table below ranks.
        writeln!(
            out,
            "Showing {} of {} records (--top {top}).\n",
            top.min(entries.len()),
            entries.len()
        )
        .unwrap();
        out.push_str("| Rank | Group | Schedule | Median GFLOP/s | 95% CI | Rounds | Drift |\n| --- | --- | --- | ---: | --- | ---: | --- |\n");
        entries.sort_by(|a, b| b.median.total_cmp(&a.median));
        for (rank, row) in entries.into_iter().take(top).enumerate() {
            let ci = bootstrap_median(&row.rounds, bootstrap).map_err(error)?;
            writeln!(
                out,
                "| {} | {} | {} | {:.6} | [{:.6}, {:.6}] | {} | {} |",
                rank + 1,
                cell(&row.group),
                if row.schedule.is_empty() {
                    "(naive)".into()
                } else {
                    cell(&row.schedule)
                },
                row.median,
                ci.lower,
                ci.upper,
                row.rounds.len(),
                drift_cell(row)
            )
            .unwrap();
        }
    }
    Ok(out)
}

/// A non-negative integer flag value; the error names the flag and the rejected text, as the
/// main harness does (`cli.rs`), instead of a bare `invalid digit found in string`.
fn flag_value<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, BenchError> {
    value.parse().map_err(|_| {
        error(format!(
            "{flag} needs a non-negative integer, got {value:?}\n{USAGE}"
        ))
    })
}

pub fn command(args: impl IntoIterator<Item = String>) -> Result<String, BenchError> {
    let mut args = args.into_iter().peekable();
    let comparing = args.peek().map(String::as_str) == Some("--compare");
    if comparing {
        args.next();
    }
    let path = args
        .next()
        .filter(|s| !s.starts_with('-'))
        .ok_or_else(|| error(USAGE))?;
    let other = if comparing {
        Some(
            args.next()
                .filter(|s| !s.starts_with('-'))
                .ok_or_else(|| error(USAGE))?,
        )
    } else {
        None
    };
    let mut top = 10;
    let mut bootstrap = BootstrapConfig::default();
    let mut paired = false;
    while let Some(flag) = args.next() {
        if flag == "--paired" && comparing {
            paired = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| error(format!("{flag} needs a value\n{USAGE}")))?;
        match flag.as_str() {
            "--top" if !comparing => top = flag_value(&flag, &value)?,
            "--seed" => bootstrap.seed = flag_value(&flag, &value)?,
            "--resamples" => bootstrap.resamples = flag_value(&flag, &value)?,
            _ => return Err(error(format!("unknown argument {flag}\n{USAGE}"))),
        }
    }
    let text = std::fs::read_to_string(&path).map_err(|e| error(format!("{path}: {e}")))?;
    if let Some(other) = other {
        let other_text =
            std::fs::read_to_string(&other).map_err(|e| error(format!("{other}: {e}")))?;
        compare(&text, &other_text, bootstrap, paired)
    } else {
        markdown(&text, top, bootstrap)
    }
}

/// Each differing protocol field with both values as JSON, sorted by field name, so a mismatch
/// says what to fix (a different build, machine, flag or sampling) without diffing logs by hand.
fn key_diff(a: &Value, a_name: &str, b: &Value, b_name: &str) -> String {
    let show = |key: &Value, field: &str| {
        key.get(field)
            .map_or_else(|_| "missing".to_string(), Value::to_json)
    };
    a.differing_keys(b)
        .unwrap_or_default()
        .into_iter()
        .map(|field| {
            format!(
                "{field} ({a_name}: {}, {b_name}: {})",
                show(a, field),
                show(b, field)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn best<'a>(rows: &'a [Row], file: &str) -> Result<&'a Row, BenchError> {
    let mut best = &rows[0];
    for row in &rows[1..] {
        if row.protocol != rows[0].protocol {
            return Err(error(format!(
                "comparison requires exactly one protocol per file; {file} mixes keys: {}",
                key_diff(&rows[0].protocol, "first", &row.protocol, "later")
            )));
        }
        if row.median > best.median {
            best = row;
        }
    }
    Ok(best)
}

/// Compare the first best observation of each file, never silently choosing a shared subset.
/// `paired` is an explicit caller assertion of paired acquisition, not inferred from IDs.
pub fn compare(
    a: &str,
    b: &str,
    config: BootstrapConfig,
    paired: bool,
) -> Result<String, BenchError> {
    let a_rows = read(a)?;
    let b_rows = read(b)?;
    let a_best = best(&a_rows, "A")?;
    let b_best = best(&b_rows, "B")?;
    if a_best.protocol != b_best.protocol {
        return Err(error(format!(
            "protocol key mismatch between A and B: {}",
            key_diff(&a_best.protocol, "A", &b_best.protocol, "B")
        )));
    }
    let bootstrap = if paired {
        bootstrap_paired_ratio
    } else {
        bootstrap_independent_ratio
    };
    let ci = bootstrap(&a_best.rounds, &b_best.rounds, config).map_err(error)?;
    let mut out = format!(
        "# Run comparison\n\nA fingerprint: `{:016x}`. B fingerprint: `{:016x}`.\n\n",
        fnv1a_64(a.as_bytes()),
        fnv1a_64(b.as_bytes())
    );
    if a_best
        .protocol
        .get("quick")
        .and_then(Value::boolean)
        .map_err(error)?
    {
        out.push_str("**Quick smoke data; not a performance measurement.**\n\n");
    }
    writeln!(out, "Ratio: median(best A) / median(best B). 95% percentile bootstrap, {} resamples, seed {}, {} resampling.\n", config.resamples, config.seed, if paired { "paired (asserted by caller)" } else { "independent" }).unwrap();
    out.push_str("| Run | Best schedule | Median GFLOP/s | Rounds | Drift |\n| --- | --- | ---: | ---: | --- |\n");
    for (name, row) in [("A", a_best), ("B", b_best)] {
        writeln!(
            out,
            "| {name} | {} | {:.6} | {} | {} |",
            if row.schedule.is_empty() {
                "(naive)".into()
            } else {
                cell(&row.schedule)
            },
            row.median,
            row.rounds.len(),
            drift_cell(row)
        )
        .unwrap();
    }
    writeln!(out, "\nMedian ratio A/B: {:.6}; 95% CI [{:.6}, {:.6}].\n\nIntervals are conditional on the selected winners and supplied rounds; they do not adjust for winner selection or independent process restarts.", ci.estimate, ci.lower, ci.upper).unwrap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{protocol::ProtocolKey, space::CacheInfo};

    fn key(space: &str, cache: Option<CacheInfo>, tile_factors: Option<Vec<usize>>) -> Value {
        let key = ProtocolKey {
            version: crate::protocol::VERSION,
            kernel: "matmul".into(),
            n: 64,
            graph_hash: "0123456789abcdef".into(),
            space: space.into(),
            space_hash: "0011223344556677".into(),
            cache,
            tile_factors,
            source_hash: "ffeeddccbbaa9988".into(),
            cpu: "cpu".into(),
            machine: "machine".into(),
            os: "os".into(),
            arch: "aarch64".into(),
            target_os: "macos".into(),
            clang: "clang".into(),
            flags: "-O3".into(),
            rustc: "rustc".into(),
            harness_profile: "release".into(),
            rustflags: String::new(),
            samples: 30,
            min_sample_ms: 10,
            input_seed: 1,
            quick: false,
            warmup_ms: 300,
            candidate_warmup_ms: 50,
        };
        json::parse(&key.to_json()).unwrap()
    }

    fn keys() -> Vec<Value> {
        vec![
            key("v1", None, None),
            key("v2", None, Some(vec![2, 4, 8])),
            key(
                "v2",
                Some(CacheInfo::new(32 * 1024, 1024 * 1024, 64).unwrap()),
                Some(vec![4, 16]),
            ),
        ]
    }

    #[test]
    fn writer_output_is_a_complete_key() {
        for key in keys() {
            validate_protocol(&key).unwrap();
        }
    }

    /// Guards against a new `ProtocolKey` field being written but never required here, and
    /// against a required field silently becoming optional.
    #[test]
    fn removing_or_nulling_any_written_field_is_rejected() {
        for key in keys() {
            let Value::Object(fields) = &key else {
                unreachable!()
            };
            for name in fields.keys() {
                let mut removed = fields.clone();
                removed.remove(name);
                let err = validate_protocol(&Value::Object(removed)).unwrap_err();
                assert!(err.contains(name.as_str()), "{name}: {err}");
                if name == "cache" {
                    continue; // null is the written value for unknown hardware
                }
                let mut nulled = fields.clone();
                nulled.insert(name.clone(), Value::Null);
                assert!(validate_protocol(&Value::Object(nulled)).is_err(), "{name}");
            }
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        for key in keys() {
            let Value::Object(mut fields) = key else {
                unreachable!()
            };
            fields.insert("extra".into(), Value::String("x".into()));
            let err = validate_protocol(&Value::Object(fields)).unwrap_err();
            assert!(err.contains("unknown protocol field extra"), "{err}");
        }
        // Space-specific fields are not accepted on V1, where the writer never emits them.
        let Value::Object(mut v1) = key("v1", None, None) else {
            unreachable!()
        };
        v1.insert("cache".into(), Value::Null);
        assert!(validate_protocol(&Value::Object(v1)).is_err());
        let Value::Object(mut v2) = key(
            "v2",
            Some(CacheInfo::new(32 * 1024, 1024 * 1024, 64).unwrap()),
            Some(vec![4]),
        ) else {
            unreachable!()
        };
        let Some(Value::Object(cache)) = v2.get_mut("cache") else {
            unreachable!()
        };
        cache.insert("l3_bytes".into(), Value::Number("1".into()));
        assert!(validate_protocol(&Value::Object(v2)).is_err());
    }
}
