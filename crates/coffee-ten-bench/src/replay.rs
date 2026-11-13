//! Offline random-search baseline over a complete, single-protocol measurement table.
use crate::{
    BenchError,
    cli::KernelKind,
    json::{self, Value},
    log::append_lines,
    meta::{Metadata, UNKNOWN},
    protocol::ProtocolKey,
    run::{RunInfo, build_kernel, fnv1a_64, json_string},
    search::{SearchConfig, best_index_so_far, sample},
    space::matmul_space_for,
};
use coffee_ten_loop::OrderClass;
use std::{collections::HashMap, fmt::Write as _};

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub schedule: String,
    pub median: f64,
    pub measurement_ns: u128,
    pub compilation_ns: u128,
    pub cache_hit: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Replay {
    pub protocol: ProtocolKey,
    pub table_hash: String,
    pub policy_seed: u64,
    pub space_size: usize,
    pub naive: Entry,
    pub reference_compilation_ns: u128,
    pub records: Vec<Entry>,
}
fn err(message: impl Into<String>) -> BenchError {
    BenchError::Setup(format!("replay: {}", message.into()))
}

fn read_entry(row: &Value, key: &ProtocolKey, class: OrderClass) -> Result<Entry, String> {
    let expected = json::parse(&key.to_json())?;
    let protocol = row.get("protocol")?;
    if protocol != &expected {
        return Err(key_mismatch(protocol, &expected));
    }
    for (field, value) in [
        ("kernel", key.kernel.as_str()),
        ("graph_hash", &key.graph_hash),
        ("cpu", &key.cpu),
        ("machine", &key.machine),
        ("os", &key.os),
        ("clang", &key.clang),
        ("flags", &key.flags),
        ("unit", "GFLOP/s"),
        (
            "order_class",
            match class {
                OrderClass::Preserving => "preserving",
                OrderClass::Reordering => "reordering",
            },
        ),
    ] {
        if row.get(field)?.text()? != value {
            return Err(format!("inconsistent {field}"));
        }
    }
    for (field, consistent) in [
        ("n", row.get("n")?.uint::<usize>()? == key.n),
        (
            "samples",
            row.get("samples")?.uint::<usize>()? == key.samples,
        ),
        ("seed", row.get("seed")?.uint::<u64>()? == key.input_seed),
        ("quick", row.get("quick")?.boolean()? == key.quick),
        ("verified", row.get("verified")?.boolean()?),
    ] {
        if !consistent {
            return Err(format!("inconsistent measurement metadata: {field}"));
        }
    }
    if row.get("compilation_wall_ns")? != &Value::Null {
        return Err("unsupported compilation wall accounting".into());
    }
    let rounds = row.get("rounds")?.array()?;
    if rounds.len() != key.samples || rounds.is_empty() {
        return Err(format!(
            "wrong raw round count: {} rounds, protocol samples {}",
            rounds.len(),
            key.samples
        ));
    }
    let checked = crate::rounds::validate(rounds, key.n, key.min_sample_ms)?;
    crate::rounds::check_aggregates(row, &checked)?;
    Ok(Entry {
        schedule: row.get("schedule")?.text()?.into(),
        median: checked.stats.median,
        measurement_ns: checked.measurement_ns,
        compilation_ns: row.get("compilation_ns")?.uint()?,
        cache_hit: row.get("cache_hit")?.boolean()?,
    })
}

/// Names the key fields that differ, so a wrong `--samples`, `--n`, `--space` or a table from
/// another machine is diagnosed without diffing JSON by hand. Values are not printed: some are
/// long (`flags`, `cpu`), and the field names are enough to find the flag or file to fix.
fn key_mismatch(found: &Value, expected: &Value) -> String {
    let (Value::Object(found), Value::Object(expected)) = (found, expected) else {
        return "protocol key mismatch: protocol must be an object".into();
    };
    let mut fields: Vec<&str> = expected
        .keys()
        .chain(found.keys())
        .map(String::as_str)
        .filter(|k| found.get(*k) != expected.get(*k))
        .collect();
    fields.sort_unstable();
    fields.dedup();
    format!("protocol key mismatch: {}", fields.join(", "))
}

/// Validates the entire table before selecting any candidates. Does not compile or run C.
pub fn from_table(
    config: &SearchConfig,
    metadata: &Metadata,
    text: &str,
) -> Result<Replay, BenchError> {
    if config.local || config.model_top.is_some() || config.from.is_some() {
        return Err(err("only random search is supported"));
    }
    if config.samples == 0 || config.budget == 0 || !(1..=crate::cli::MAX_N).contains(&config.n) {
        return Err(err("invalid search dimensions or budget"));
    }
    for identity in [
        &metadata.cpu_brand,
        &metadata.hw_model,
        &metadata.os_version,
        &metadata.clang,
        &metadata.flags,
    ] {
        if identity.is_empty() || identity == UNKNOWN {
            return Err(err("unknown CPU/platform/compiler identity"));
        }
    }
    let space = matmul_space_for(config.n, config.n, config.n, config.space);
    let classes: HashMap<&str, OrderClass> = space
        .iter()
        .map(|c| (c.schedule.as_str(), c.class))
        .collect();
    let graph = build_kernel(KernelKind::Matmul, config.n)?;
    let info = RunInfo {
        kernel: "matmul",
        n: config.n,
        graph_hash: fnv1a_64(graph.to_string().as_bytes()),
        min_sample_ms: config.min_sample_ms,
        quick: config.quick,
    };
    let mut key = None;
    let mut naive = None;
    let mut reference_compilation_ns = None;
    let mut entries = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parse_row = || -> Result<(), String> {
            let row = json::parse(line)?;
            if let Ok(kind) = row.get("kind") {
                if kind.text()? == "peak" {
                    return Ok(());
                }
                return Err("not a live measurement record".into());
            }
            let protocol = row.get("protocol")?;
            if key.is_none() {
                // Input data stay fixed; --seed controls only selection during replay.
                let mut input_metadata = metadata.clone();
                input_metadata.seed = protocol.get("input_seed")?.uint()?;
                key = Some(ProtocolKey::for_space(
                    &info,
                    config.samples,
                    &input_metadata,
                    config.space,
                ));
            }
            let schedule = row.get("schedule")?.text()?;
            let group = row.get("group")?.text()?;
            let class = if group == "naive" && schedule.is_empty() {
                OrderClass::Preserving
            } else if group == "random" {
                *classes.get(schedule).ok_or("unknown candidate schedule")?
            } else {
                return Err("table must contain only naive and random groups".into());
            };
            let entry = read_entry(&row, key.as_ref().unwrap(), class)?;
            let reference = row.get("reference_compilation")?;
            if group == "naive" {
                if naive.replace(entry).is_some() {
                    return Err("duplicate naive reference".into());
                }
                reference.get("cache_hit")?.boolean()?;
                reference_compilation_ns = Some(reference.get("elapsed_ns")?.uint::<u128>()?);
            } else {
                if reference != &Value::Null {
                    return Err("reference compilation charged more than once".into());
                }
                if entries.insert(schedule.to_string(), entry).is_some() {
                    return Err("duplicate schedule".into());
                }
            }
            Ok(())
        };
        let mut parse_row = parse_row;
        parse_row().map_err(|message| err(format!("line {}: {message}", index + 1)))?;
    }
    let naive = naive.ok_or_else(|| err("missing naive reference"))?;
    if entries.len() != space.len() {
        return Err(err(format!(
            "incomplete table: {} of {} candidates",
            entries.len(),
            space.len()
        )));
    }
    let space_size = space.len();
    let records = sample(space, config.budget, config.seed)
        .into_iter()
        .map(|c| {
            entries
                .remove(&c.schedule)
                .expect("validated complete table")
        })
        .collect();
    Ok(Replay {
        protocol: key.unwrap(),
        table_hash: format!("{:016x}", fnv1a_64(text.as_bytes())),
        policy_seed: config.seed,
        space_size,
        naive,
        reference_compilation_ns: reference_compilation_ns.unwrap(),
        records,
    })
}

fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    if a == b {
        return true;
    }
    if let (Ok(a), Ok(b)) = (a.canonicalize(), b.canonicalize())
        && a == b
    {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(a), Ok(b)) = (a.metadata(), b.metadata()) {
            return a.dev() == b.dev() && a.ino() == b.ino();
        }
    }
    false
}

/// Reads the table and optionally appends explicitly offline curve records.
pub fn run(config: &SearchConfig, metadata: &Metadata) -> Result<Replay, BenchError> {
    let path = config
        .replay
        .as_ref()
        .ok_or_else(|| err("--replay table is required"))?;
    if config.log.as_ref().is_some_and(|log| same_file(log, path)) {
        return Err(err("replay output must differ from the input table"));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let result = from_table(config, metadata, &text)?;
    if let Some(log) = &config.log {
        append_lines(log, &result.json_lines()?)
            .map_err(|e| err(format!("cannot append to {}: {e}", log.display())))?;
    }
    Ok(result)
}
impl Replay {
    /// Separate schema: these are stored costs, not new live measurements.
    pub fn json_lines(&self) -> Result<Vec<String>, BenchError> {
        let best = best_index_so_far(&self.records.iter().map(|r| r.median).collect::<Vec<_>>());
        let mut measurement = self.naive.measurement_ns;
        let mut compilation = self
            .naive
            .compilation_ns
            .checked_add(self.reference_compilation_ns)
            .ok_or_else(|| err("compilation cost overflow"))?;
        self.records.iter().zip(best).enumerate().map(|(index, (row, best))| {
            measurement = measurement.checked_add(row.measurement_ns).ok_or_else(|| err("measurement cost overflow"))?;
            compilation = compilation.checked_add(row.compilation_ns).ok_or_else(|| err("compilation cost overflow"))?;
            Ok(format!("{{\"kind\":\"replay_curve\",\"offline\":true,\"protocol\":{},\"table_hash\":{},\"policy_seed\":{},\"step\":{},\"schedule\":{},\"gflops\":{},\"best_schedule\":{},\"best_gflops\":{},\"charged_measurement_ns\":{},\"charged_compilation_ns\":{},\"cache_hit\":{},\"cumulative_measurement_ns\":{},\"cumulative_compilation_ns\":{}}}", self.protocol.to_json(), json_string(&self.table_hash), self.policy_seed, index + 1, json_string(&row.schedule), row.median, json_string(&self.records[best].schedule), self.records[best].median, row.measurement_ns, row.compilation_ns, row.cache_hit, measurement, compilation))
        }).collect()
    }
    pub fn report(&self) -> String {
        let mut text = format!(
            "offline replay: {} candidates of {}, policy seed {}, fixed input seed {}\ntable {} (stored rates and costs; no new measurements)\n",
            self.records.len(),
            self.space_size,
            self.policy_seed,
            self.protocol.input_seed,
            self.table_hash
        );
        if self.protocol.quick {
            text.push_str("quick: true (smoke data, not a measurement)\n");
        }
        text.push_str("step  best GFLOP/s  schedule\n");
        let best = best_index_so_far(&self.records.iter().map(|r| r.median).collect::<Vec<_>>());
        for step in crate::search::curve_points(self.records.len()) {
            let row = &self.records[best[step - 1]];
            let _ = writeln!(text, "{step}  {:.3}  {}", row.median, row.schedule);
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::matmul_space;
    use crate::{
        run::REFERENCES_BUILT,
        search::{parse_search_args, search},
    };
    use std::sync::OnceLock;

    fn config() -> SearchConfig {
        parse_search_args(
            ["--n", "2", "--budget", "999", "--seed", "17", "--quick"].map(str::to_string),
        )
        .unwrap()
    }
    fn metadata() -> Metadata {
        Metadata {
            hw_model: "toy-hardware".into(),
            cpu_brand: "toy-CPU".into(),
            os_version: "toy-OS".into(),
            clang: "toy-clang".into(),
            flags: coffee_ten_codegen_c::BENCH_FLAGS.join(" "),
            git_rev: "fixture".into(),
            git_dirty: Some(false),
            date_utc: "fixture".into(),
            seed: 17,
        }
    }
    /// Actual quick measurements of the complete tiny space, captured once per test process.
    fn table() -> &'static str {
        static TABLE: OnceLock<String> = OnceLock::new();
        TABLE.get_or_init(|| {
            let path = std::env::temp_dir().join(format!(
                "coffee-ten-replay-toy-{}.jsonl",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let mut config = config();
            config.log = Some(path.clone());
            let measured = search(&config, &metadata()).unwrap();
            assert_eq!(measured.records.len(), matmul_space(2, 2, 2).len());
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            text
        })
    }
    fn encode(v: &Value) -> String {
        match v {
            Value::Null => "null".into(),
            Value::Bool(b) => b.to_string(),
            Value::Number(s) => s.clone(),
            Value::String(s) => json_string(s),
            Value::Array(a) => format!("[{}]", a.iter().map(encode).collect::<Vec<_>>().join(",")),
            Value::Object(o) => format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, v)| format!("{}:{}", json_string(k), encode(v)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
    fn change_first(
        mut change: impl FnMut(&mut std::collections::BTreeMap<String, Value>),
    ) -> String {
        let mut lines: Vec<_> = table().lines().map(str::to_string).collect();
        let Value::Object(mut row) = json::parse(&lines[0]).unwrap() else {
            panic!()
        };
        change(&mut row);
        lines[0] = encode(&Value::Object(row));
        lines.join("\n")
    }
    #[test]
    fn full_toy_live_order_and_replay_agree_without_building_references() {
        let text = table();
        let before = REFERENCES_BUILT.with(std::cell::Cell::get);
        let replay = from_table(&config(), &metadata(), text).unwrap();
        let live_schedules: Vec<_> = text
            .lines()
            .skip(1)
            .map(|line| {
                json::parse(line)
                    .unwrap()
                    .get("schedule")
                    .unwrap()
                    .text()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            replay
                .records
                .iter()
                .map(|r| r.schedule.clone())
                .collect::<Vec<_>>(),
            live_schedules
        );
        assert_eq!(replay.protocol.input_seed, 17);
        for seed in [0, 1, 17, 999, u64::MAX] {
            let mut config = config();
            config.seed = seed;
            config.budget = 3;
            let run = from_table(&config, &metadata(), text).unwrap();
            let expected = sample(matmul_space(2, 2, 2), 3, seed);
            assert_eq!(
                run.records.iter().map(|r| &r.schedule).collect::<Vec<_>>(),
                expected.iter().map(|r| &r.schedule).collect::<Vec<_>>()
            );
            let again = from_table(&config, &metadata(), text).unwrap();
            assert_eq!(run.json_lines().unwrap(), again.json_lines().unwrap());
            let lines = run.json_lines().unwrap();
            let first = json::parse(&lines[0]).unwrap();
            assert_eq!(first.get("kind").unwrap().text().unwrap(), "replay_curve");
            assert!(first.get("offline").unwrap().boolean().unwrap());
            assert_eq!(
                first
                    .get("cumulative_measurement_ns")
                    .unwrap()
                    .uint::<u128>()
                    .unwrap(),
                run.naive.measurement_ns + run.records[0].measurement_ns
            );
            assert_eq!(
                first
                    .get("cumulative_compilation_ns")
                    .unwrap()
                    .uint::<u128>()
                    .unwrap(),
                run.naive.compilation_ns
                    + run.reference_compilation_ns
                    + run.records[0].compilation_ns
            );
        }
        assert_eq!(REFERENCES_BUILT.with(std::cell::Cell::get), before);
    }
    #[test]
    fn expanded_v2_table_roundtrips_and_rejects_v1_or_different_cache_keys() {
        use crate::space::{CacheInfo, SpaceConfig, matmul_space_for};
        let path =
            std::env::temp_dir().join(format!("coffee-ten-replay-v2-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut config = config();
        config.n = 4;
        config.space = SpaceConfig::V2(Some(CacheInfo::new(64, 512, 8).unwrap()));
        config.log = Some(path.clone());
        let live = search(&config, &metadata()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let before = REFERENCES_BUILT.with(std::cell::Cell::get);
        let replay = from_table(&config, &metadata(), &text).unwrap();
        assert_eq!(
            replay.records.len(),
            matmul_space_for(4, 4, 4, config.space).len()
        );
        assert_eq!(
            replay
                .records
                .iter()
                .map(|r| &r.schedule)
                .collect::<Vec<_>>(),
            live.records.iter().map(|r| &r.schedule).collect::<Vec<_>>()
        );
        assert_eq!(replay.protocol.space, "v2");
        assert!(
            replay
                .protocol
                .to_json()
                .contains("\"tile_factors\":[2,4,8,16,32,64,128]")
        );
        config.space = SpaceConfig::V1;
        assert!(from_table(&config, &metadata(), &text).is_err());
        config.space = SpaceConfig::V2(None);
        assert!(from_table(&config, &metadata(), &text).is_err());
        // Both caches produce the same candidate set at n=4, but remain different keys.
        config.space = SpaceConfig::V2(Some(CacheInfo::new(128, 512, 8).unwrap()));
        assert!(from_table(&config, &metadata(), &text).is_err());
        assert_eq!(REFERENCES_BUILT.with(std::cell::Cell::get), before);
    }

    #[test]
    fn multi_seed_csv_curves_reuse_the_validated_random_baseline() {
        let text = table();
        let before = REFERENCES_BUILT.with(std::cell::Cell::get);
        let mut runs = Vec::new();
        for seed in [1, 42, 999] {
            let mut config = config();
            config.seed = seed;
            config.budget = 3;
            runs.push(from_table(&config, &metadata(), text).unwrap());
        }
        let csv = crate::curves::csv(&runs).unwrap();
        assert_eq!(csv.lines().count(), 1 + 3 * 4);
        assert_eq!(csv, crate::curves::csv(&runs).unwrap());
        for run in &runs {
            let mut cost = run.naive.measurement_ns;
            let mut best = run.naive.median;
            for (step, row) in run.records.iter().enumerate() {
                cost += row.measurement_ns;
                best = best.max(row.median);
                assert!(csv.lines().any(|line| line.starts_with(&format!(
                    "{},{},{cost},",
                    run.policy_seed,
                    step + 1
                ))));
            }
            let last = csv
                .lines()
                .find(|line| line.starts_with(&format!("{},3,", run.policy_seed)))
                .unwrap();
            assert_eq!(
                last.split(',')
                    .nth(4)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    .to_bits(),
                best.to_bits()
            );
        }
        assert_eq!(REFERENCES_BUILT.with(std::cell::Cell::get), before);
    }

    #[test]
    fn every_protocol_field_is_enforced_even_with_reordered_json_keys() {
        let unchanged = change_first(|_| {});
        from_table(&config(), &metadata(), &unchanged).unwrap();
        let first = json::parse(table().lines().next().unwrap()).unwrap();
        let Value::Object(protocol) = first.get("protocol").unwrap() else {
            panic!()
        };
        for field in protocol.keys() {
            let bad = change_first(|row| {
                let Value::Object(protocol) = row.get_mut("protocol").unwrap() else {
                    panic!()
                };
                protocol.insert(field.clone(), Value::String("incompatible".into()));
            });
            assert!(
                from_table(&config(), &metadata(), &bad).is_err(),
                "accepted changed {field}"
            );
        }
    }
    #[test]
    fn rejects_self_consistent_rounds_below_the_protocol_minimum() {
        let bad = change_first(|row| {
            let samples = row.get("samples").unwrap().uint::<usize>().unwrap();
            let rate = (16.0 / std::time::Duration::from_nanos(1).as_secs_f64() / 1e9).to_string();
            let rounds = (0..samples)
                .map(|i| {
                    Value::Object(
                        [
                            ("round_index".into(), Value::Number(i.to_string())),
                            ("elapsed_ns".into(), Value::Number("1".into())),
                            ("iterations".into(), Value::Number("1".into())),
                            ("gflops".into(), Value::Number(rate.clone())),
                        ]
                        .into(),
                    )
                })
                .collect();
            row.insert("rounds".into(), Value::Array(rounds));
            row.insert("measurement_ns".into(), Value::Number(samples.to_string()));
            for field in ["median", "p25", "p75", "min"] {
                row.insert(field.into(), Value::Number(rate.clone()));
            }
            row.insert("drift".into(), Value::Bool(false));
        });
        assert!(from_table(&config(), &metadata(), &bad).is_err());
    }

    #[test]
    fn rejects_incomplete_duplicate_mixed_and_corrupt_tables() {
        let text = table();
        let lines: Vec<_> = text.lines().collect();
        for bad in [
            lines[1..].join("\n"),
            lines[..lines.len() - 1].join("\n"),
            format!("{text}\n{}", lines[1]),
            format!("{text}\n{}", lines[0]),
            "".into(),
            "{bad}".into(),
        ] {
            assert!(from_table(&config(), &metadata(), &bad).is_err());
        }
        for field in [
            "rounds",
            "protocol",
            "reference_compilation",
            "measurement_ns",
            "cache_hit",
        ] {
            let bad = change_first(|row| {
                row.remove(field);
            });
            assert!(from_table(&config(), &metadata(), &bad).is_err(), "{field}");
        }
        for (field, value) in [
            ("median", Value::Number("0".into())),
            ("measurement_ns", Value::Number("0".into())),
            ("verified", Value::Bool(false)),
            ("schedule", Value::String("unknown".into())),
            ("group", Value::String("final".into())),
            ("seed", Value::Number("18".into())),
        ] {
            let bad = change_first(|row| {
                row.insert(field.into(), value.clone());
            });
            assert!(from_table(&config(), &metadata(), &bad).is_err(), "{field}");
        }
        for field in ["iterations", "elapsed_ns", "gflops", "round_index"] {
            let bad = change_first(|row| {
                let Value::Array(rounds) = row.get_mut("rounds").unwrap() else {
                    panic!()
                };
                let Value::Object(round) = &mut rounds[0] else {
                    panic!()
                };
                round.insert(
                    field.into(),
                    Value::Number(if field == "round_index" { "1" } else { "0" }.into()),
                );
            });
            assert!(from_table(&config(), &metadata(), &bad).is_err(), "{field}");
        }
        let mut unknown = metadata();
        unknown.cpu_brand = UNKNOWN.into();
        assert!(from_table(&config(), &unknown, text).is_err());
        for flag in ["--samples", "--n", "--min-sample-ms"] {
            let mut config = config();
            match flag {
                "--samples" => config.samples += 1,
                "--n" => config.n += 1,
                _ => config.min_sample_ms += 1,
            }
            assert!(from_table(&config, &metadata(), text).is_err());
        }
    }
    #[test]
    fn file_replay_writes_curves_and_refuses_input_aliases() {
        let dir =
            std::env::temp_dir().join(format!("coffee-ten-replay-files-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("table.jsonl");
        let output = dir.join("curves.jsonl");
        std::fs::write(&input, table()).unwrap();
        let _ = std::fs::remove_file(&output);
        let mut config = config();
        config.replay = Some(input.clone());
        config.log = Some(output.clone());
        let replay = run(&config, &metadata()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            replay.json_lines().unwrap().join("\n") + "\n"
        );
        std::fs::remove_file(&output).unwrap();
        std::fs::hard_link(&input, &output).unwrap();
        assert!(run(&config, &metadata()).is_err());
        assert_eq!(std::fs::read_to_string(&input).unwrap(), table());
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn cli_checks_modes_and_invalid_input_never_produces_a_log() {
        let args = ["--n", "2", "--budget", "3", "--replay", "table.jsonl"];
        let parsed = parse_search_args(args.map(str::to_string)).unwrap();
        assert_eq!(parsed.replay, Some("table.jsonl".into()));
        for mode in [
            vec!["--local", "--from", "peak.jsonl"],
            vec!["--model-top", "1", "--from", "peak.jsonl"],
        ] {
            assert!(parse_search_args(args.into_iter().chain(mode).map(str::to_string)).is_err());
        }
        let output = std::env::temp_dir().join(format!(
            "coffee-ten-replay-output-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&output);
        let mut config = config();
        config.replay = Some("/nonexistent/coffee-ten-replay.jsonl".into());
        config.log = Some(output.clone());
        assert!(run(&config, &metadata()).is_err());
        assert!(!output.exists());
        config.replay = Some(output);
        assert!(run(&config, &metadata()).is_err());
        assert!(search(&config, &metadata()).is_err());
    }

    #[test]
    fn two_file_replays_with_one_seed_are_byte_identical() {
        let directory = std::env::temp_dir().join(format!(
            "coffee-ten-n17-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        let input = directory.join("complete.jsonl");
        let first = directory.join("first.jsonl");
        let second = directory.join("second.jsonl");
        let changed_seed = directory.join("changed-seed.jsonl");
        std::fs::write(&input, table()).unwrap();
        let before = REFERENCES_BUILT.with(std::cell::Cell::get);
        let mut config = config();
        config.budget = 3;
        config.seed = 17;
        config.replay = Some(input.clone());
        config.log = Some(first.clone());
        run(&config, &metadata()).unwrap();
        config.log = Some(second.clone());
        run(&config, &metadata()).unwrap();
        let first_bytes = std::fs::read(&first).unwrap();
        assert_eq!(first_bytes, std::fs::read(&second).unwrap());
        let rows: Vec<_> = std::str::from_utf8(&first_bytes)
            .unwrap()
            .lines()
            .map(|line| json::parse(line).unwrap())
            .collect();
        assert_eq!(rows.len(), config.budget);
        for (step, row) in rows.iter().enumerate() {
            assert_eq!(row.get("policy_seed").unwrap().uint::<u64>().unwrap(), 17);
            assert_eq!(row.get("step").unwrap().uint::<usize>().unwrap(), step + 1);
            assert_eq!(
                row.get("protocol")
                    .unwrap()
                    .get("input_seed")
                    .unwrap()
                    .uint::<u64>()
                    .unwrap(),
                17
            );
        }
        config.seed = 42;
        config.log = Some(changed_seed.clone());
        run(&config, &metadata()).unwrap();
        let changed = std::fs::read_to_string(&changed_seed).unwrap();
        let schedules = |text: &str| {
            text.lines()
                .map(|line| {
                    json::parse(line)
                        .unwrap()
                        .get("schedule")
                        .unwrap()
                        .text()
                        .unwrap()
                        .to_string()
                })
                .collect::<Vec<_>>()
        };
        assert_ne!(
            schedules(std::str::from_utf8(&first_bytes).unwrap()),
            schedules(&changed)
        );
        assert_eq!(std::fs::read_to_string(&input).unwrap(), table());
        assert_eq!(REFERENCES_BUILT.with(std::cell::Cell::get), before);
    }

    fn error_of(text: &str) -> String {
        from_table(&config(), &metadata(), text)
            .unwrap_err()
            .to_string()
    }
    fn change_round(field: &str, value: Value) -> String {
        change_first(|row| {
            let Value::Array(rounds) = row.get_mut("rounds").unwrap() else {
                panic!()
            };
            let Value::Object(round) = &mut rounds[0] else {
                panic!()
            };
            round.insert(field.into(), value.clone());
        })
    }
    #[test]
    fn errors_name_the_round_the_field_and_the_counts() {
        let samples = config().samples;
        for (bad, expected) in [
            (
                change_round("elapsed_ns", Value::Number("0".into())),
                "round 0 needs positive elapsed_ns and iterations".to_string(),
            ),
            (
                change_round("iterations", Value::Number("0".into())),
                "round 0 needs positive elapsed_ns and iterations".into(),
            ),
            (
                change_round("round_index", Value::Number("1".into())),
                "invalid raw round order: round 0 has round_index 1".into(),
            ),
            (
                change_round("gflops", Value::Null),
                "round 0 gflops: expected finite number".into(),
            ),
            (
                change_round("gflops", Value::Number("1.5".into())),
                "raw rate does not match timing: round 0 stores 1.5".into(),
            ),
            (
                change_first(|row| {
                    let Value::Array(rounds) = row.get_mut("rounds").unwrap() else {
                        panic!()
                    };
                    let Value::Object(round) = &mut rounds[0] else {
                        panic!()
                    };
                    round.remove("iterations");
                }),
                "round 0 iterations: missing iterations".into(),
            ),
            (
                change_first(|row| {
                    let Value::Array(rounds) = row.get_mut("rounds").unwrap() else {
                        panic!()
                    };
                    rounds.pop();
                }),
                format!(
                    "wrong raw round count: {} rounds, protocol samples {samples}",
                    samples - 1
                ),
            ),
            (
                change_first(|row| {
                    row.insert("seed".into(), Value::Number("18".into()));
                }),
                "inconsistent measurement metadata: seed".into(),
            ),
            (
                change_first(|row| {
                    row.insert("verified".into(), Value::Bool(false));
                }),
                "inconsistent measurement metadata: verified".into(),
            ),
            (
                change_first(|row| {
                    row.insert("measurement_ns".into(), Value::Number("1".into()));
                }),
                "inconsistent measurement cost: measurement_ns 1, raw rounds sum to".into(),
            ),
            (
                change_first(|row| {
                    let drift = row.get("drift").unwrap().boolean().unwrap();
                    row.insert("drift".into(), Value::Bool(!drift));
                }),
                "inconsistent drift: stored flag differs from raw rounds".into(),
            ),
        ] {
            let err = error_of(&bad);
            assert!(err.starts_with("replay: line 1: "), "{err}");
            assert!(err.contains(&expected), "{expected}: {err}");
        }
    }
    #[test]
    fn a_protocol_mismatch_names_the_differing_fields() {
        let text = table();
        // A replay run with other flags than the table's acquisition.
        let mut more_samples = config();
        more_samples.samples += 1;
        let err = from_table(&more_samples, &metadata(), text)
            .unwrap_err()
            .to_string();
        assert!(err.ends_with("protocol key mismatch: samples"), "{err}");
        let mut slower = config();
        slower.min_sample_ms += 1;
        let err = from_table(&slower, &metadata(), text)
            .unwrap_err()
            .to_string();
        assert!(
            err.ends_with("protocol key mismatch: min_sample_ms"),
            "{err}"
        );
        // Several fields, in sorted order; extra or missing fields are named too.
        let mut other = metadata();
        other.cpu_brand = "other-CPU".into();
        other.os_version = "other-OS".into();
        let err = from_table(&config(), &other, text).unwrap_err().to_string();
        assert!(err.ends_with("protocol key mismatch: cpu, os"), "{err}");
        let bad = change_first(|row| {
            let Value::Object(protocol) = row.get_mut("protocol").unwrap() else {
                panic!()
            };
            protocol.remove("rustc");
            protocol.insert("extra".into(), Value::Null);
        });
        assert!(
            error_of(&bad).ends_with("protocol key mismatch: extra, rustc"),
            "{}",
            error_of(&bad)
        );
    }
}
