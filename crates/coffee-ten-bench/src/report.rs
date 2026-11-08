//! Read-only Markdown analysis of saved N1/N2 measurement records.
use crate::{
    BenchError,
    json::{self, Value},
    run::{RoundSample, fnv1a_64},
    stats::{
        BootstrapConfig, bootstrap_independent_ratio, bootstrap_median, bootstrap_paired_ratio,
        drifted, summarize,
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

fn error(message: impl std::fmt::Display) -> BenchError {
    BenchError::Setup(format!("report: {message}"))
}

fn parse_row(row: &Value) -> Result<Row, String> {
    let protocol = row.get("protocol").map_err(
        |_| "N1/N2 log required: raw rounds and protocol are unavailable in legacy logs",
    )?;
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
    if count == 0 || count != raw.len() || count != protocol.get("samples")?.uint::<usize>()? {
        return Err("wrong raw round count".into());
    }
    let mut rounds = Vec::with_capacity(count);
    let mut measurement_ns = 0u128;
    let minimum_ns = u128::from(protocol.get("min_sample_ms")?.uint::<u64>()?) * 1_000_000;
    for (index, value) in raw.iter().enumerate() {
        let round = RoundSample {
            round_index: value.get("round_index")?.uint()?,
            elapsed_ns: value.get("elapsed_ns")?.uint()?,
            iterations: value.get("iterations")?.uint()?,
            gflops: value.get("gflops")?.number()?,
        };
        if round.round_index != index || round.elapsed_ns == 0 || round.iterations == 0 {
            return Err("invalid raw round timing or order".into());
        }
        if u128::from(round.elapsed_ns) < minimum_ns {
            return Err("raw round is below the protocol minimum duration".into());
        }
        let expected = 2.0 * (n as f64).powi(3) * round.iterations as f64
            / std::time::Duration::from_nanos(round.elapsed_ns).as_secs_f64()
            / 1e9;
        if round.gflops <= 0.0 || round.gflops.to_bits() != expected.to_bits() {
            return Err("raw rate does not match timing".into());
        }
        measurement_ns = measurement_ns
            .checked_add(u128::from(round.elapsed_ns))
            .ok_or("measurement cost overflow")?;
        rounds.push(round);
    }
    let rates: Vec<_> = rounds.iter().map(|r| r.gflops).collect();
    let stats = summarize(&rates).ok_or("empty rounds")?;
    for (field, expected) in [
        ("median", stats.median),
        ("p25", stats.p25),
        ("p75", stats.p75),
        ("min", stats.min),
    ] {
        if row.get(field)?.number()?.to_bits() != expected.to_bits() {
            return Err(format!("{field} differs from raw rounds"));
        }
    }
    let drift = row.get("drift")?.boolean()?;
    if drift != drifted(&rates) || measurement_ns != row.get("measurement_ns")?.uint::<u128>()? {
        return Err("inconsistent measurement cost or drift".into());
    }
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
        rounds,
        median: stats.median,
        measurement_ns,
        compilation_ns,
        drift,
    })
}

pub(crate) fn read(text: &str) -> Result<Vec<Row>, BenchError> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parse = || -> Result<Option<Row>, String> {
            let value = json::parse(line)?;
            if let Ok(kind) = value.get("kind") {
                if kind.text()? == "peak" {
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

fn cell(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace('`', "&#96;")
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
                row.drift
            )
            .unwrap();
        }
    }
    Ok(out)
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
            "--top" if !comparing => top = value.parse().map_err(error)?,
            "--seed" => bootstrap.seed = value.parse().map_err(error)?,
            "--resamples" => bootstrap.resamples = value.parse().map_err(error)?,
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

fn best(rows: &[Row]) -> Result<&Row, BenchError> {
    let mut best = &rows[0];
    for row in &rows[1..] {
        if row.protocol != best.protocol {
            return Err(error("comparison requires exactly one protocol per file"));
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
    let a_best = best(&a_rows)?;
    let b_best = best(&b_rows)?;
    if a_best.protocol != b_best.protocol {
        return Err(error("protocol key mismatch between A and B"));
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
            row.drift
        )
        .unwrap();
    }
    writeln!(out, "\nMedian ratio A/B: {:.6}; 95% CI [{:.6}, {:.6}].\n\nIntervals are conditional on the selected winners and supplied rounds; they do not adjust for winner selection or independent process restarts.", ci.estimate, ci.lower, ci.upper).unwrap();
    Ok(out)
}
