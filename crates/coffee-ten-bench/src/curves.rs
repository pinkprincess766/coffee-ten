//! Offline random-baseline curves, with stored costs kept distinct from live timings.
use crate::{
    BenchError, meta,
    replay::{self, Replay},
    search::parse_search_args,
};
use std::{collections::HashSet, fmt::Write as _};

fn error(msg: impl std::fmt::Display) -> BenchError {
    BenchError::Setup(format!("curve: {msg}"))
}

fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
}

/// Every candidate prefix, including the fixed naive-reference cost at step zero.
/// The incumbent includes naive: an unsuccessful candidate cannot lower the curve.
pub fn csv(replays: &[Replay]) -> Result<String, BenchError> {
    let first = replays
        .first()
        .ok_or_else(|| error("at least one seed is required"))?;
    let mut seeds = HashSet::new();
    let mut out = String::from(
        "policy_seed,step,cumulative_measurement_ns,cumulative_compilation_ns,best_gflops,best_schedule,table_hash,protocol\n",
    );
    for replay in replays {
        if replay.protocol != first.protocol || replay.table_hash != first.table_hash {
            return Err(error("all curves must use one table and protocol"));
        }
        if !seeds.insert(replay.policy_seed) {
            return Err(error("duplicate policy seed"));
        }
        let mut incumbent = &replay.naive;
        let mut measured = replay.naive.measurement_ns;
        let mut compiled = replay
            .naive
            .compilation_ns
            .checked_add(replay.reference_compilation_ns)
            .ok_or_else(|| error("compilation cost overflow"))?;
        for step in 0..=replay.records.len() {
            if step > 0 {
                let row = &replay.records[step - 1];
                measured = measured
                    .checked_add(row.measurement_ns)
                    .ok_or_else(|| error("measurement cost overflow"))?;
                compiled = compiled
                    .checked_add(row.compilation_ns)
                    .ok_or_else(|| error("compilation cost overflow"))?;
                if row.median > incumbent.median {
                    incumbent = row;
                }
            }
            writeln!(
                out,
                "{},{step},{measured},{compiled},{},{},{},{}",
                replay.policy_seed,
                incumbent.median,
                quoted(&incumbent.schedule),
                replay.table_hash,
                quoted(&replay.protocol.to_json())
            )
            .unwrap();
        }
    }
    Ok(out)
}

/// Uses the same complete-table validation and seeded sample as N3. No timing or log writes.
pub fn command(args: impl IntoIterator<Item = String>) -> Result<String, BenchError> {
    let mut args = args.into_iter();
    let mut search_args = Vec::new();
    let mut seeds = vec![1];
    while let Some(arg) = args.next() {
        if arg == "--seeds" {
            let text = args
                .next()
                .ok_or_else(|| error("--seeds needs comma-separated integers"))?;
            seeds = text
                .split(',')
                .map(|s| s.parse::<u64>().map_err(error))
                .collect::<Result<_, _>>()?;
        } else {
            if matches!(
                arg.as_str(),
                "--log" | "--seed" | "--local" | "--model-top" | "--from"
            ) {
                return Err(error(
                    "use --seeds and --replay; live modes and log writes are unavailable",
                ));
            }
            search_args.push(arg);
        }
    }
    let mut config = parse_search_args(search_args)?;
    let path = config
        .replay
        .as_ref()
        .ok_or_else(|| error("--replay <complete-table.jsonl> is required"))?;
    let table = std::fs::read_to_string(path).map_err(error)?;
    let metadata = meta::collect(config.seed);
    let replays = seeds
        .into_iter()
        .map(|seed| {
            config.seed = seed;
            replay::from_table(&config, &metadata, &table)
        })
        .collect::<Result<Vec<_>, _>>()?;
    csv(&replays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{protocol::ProtocolKey, replay::Entry, run::RunInfo};

    fn fixture(seed: u64) -> Replay {
        let metadata = meta::Metadata {
            hw_model: "toy".into(),
            cpu_brand: "toy".into(),
            os_version: "toy".into(),
            clang: "toy".into(),
            flags: "toy".into(),
            git_rev: "toy".into(),
            git_dirty: Some(false),
            date_utc: "toy".into(),
            seed: 17,
        };
        let mut protocol = ProtocolKey::for_run(
            &RunInfo {
                kernel: "matmul",
                n: 2,
                graph_hash: 1,
                min_sample_ms: 1,
                quick: true,
            },
            3,
            &metadata,
        );
        protocol.source_hash = "fixture".into();
        let entry = |schedule: &str, median, measurement_ns, compilation_ns| Entry {
            schedule: schedule.into(),
            median,
            measurement_ns,
            compilation_ns,
            cache_hit: false,
        };
        Replay {
            protocol,
            table_hash: "table-fixture".into(),
            policy_seed: seed,
            space_size: 3,
            naive: entry("", 2.0, 10, 100),
            reference_compilation_ns: 50,
            records: vec![
                entry("vectorize(1, j)", 4.0, 20, 200),
                entry("slower", 1.0, 30, 300),
                entry("tie", 4.0, 40, 400),
            ],
        }
    }

    #[test]
    fn every_prefix_has_separate_costs_stable_ties_and_csv_quoting() {
        let a = fixture(1);
        let mut b = fixture(42);
        b.records.reverse();
        let out = csv(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(out, csv(&[a, b]).unwrap());
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(lines.len(), 9);
        for (line, prefix) in lines[1..5].iter().zip([
            "1,0,10,150,2,\"\",table-fixture,",
            "1,1,30,350,4,\"vectorize(1, j)\",table-fixture,",
            "1,2,60,650,4,\"vectorize(1, j)\",table-fixture,",
            "1,3,100,1050,4,\"vectorize(1, j)\",table-fixture,",
        ]) {
            assert!(line.starts_with(prefix), "{line}");
        }
        assert!(lines[6].starts_with("42,1,50,550,4,\"tie\""));
        assert!(out.contains("\"\"input_seed\"\":17"));
        assert_eq!(quoted("a,\"b\"\n"), "\"a,\"\"b\"\"\n\"");
    }

    #[test]
    fn rejects_mixed_tables_protocols_duplicate_seeds_and_overflow() {
        let a = fixture(1);
        assert!(csv(&[]).is_err());
        assert!(csv(&[a.clone(), a.clone()]).is_err());
        let mut b = fixture(2);
        b.table_hash = "other".into();
        assert!(csv(&[a.clone(), b]).is_err());
        let mut b = fixture(2);
        b.protocol.n = 3;
        assert!(csv(&[a.clone(), b]).is_err());
        let mut a = a;
        a.records[0].measurement_ns = u128::MAX;
        assert!(csv(&[a]).is_err());
        for args in [
            vec![],
            vec!["--seeds", "1,,2"],
            vec!["--log", "out"],
            vec!["--seed", "2"],
        ] {
            assert!(command(args.into_iter().map(str::to_string)).is_err());
        }
    }
}
