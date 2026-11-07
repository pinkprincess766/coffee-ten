use coffee_ten_bench::{
    protocol::{space_hash, space_hash_for},
    run::fnv1a_64,
    search::{parse_search_args, sample},
    space::{CacheInfo, SpaceConfig, enumerate_matmul_for, matmul_space, parse_space_config_args},
};

fn selected_hash(n: usize, seed: u64) -> String {
    let mut bytes = Vec::new();
    for candidate in sample(matmul_space(n, n, n), 8, seed) {
        bytes.extend_from_slice(&(candidate.schedule.len() as u64).to_le_bytes());
        bytes.extend_from_slice(candidate.schedule.as_bytes());
    }
    format!("{:016x}", fnv1a_64(&bytes))
}

#[test]
fn documented_schedule_space_anchors_match_computed_hashes_and_counts() {
    let document = include_str!("../../../docs/schedule-space.md");
    let mut anchors = 0;
    for line in document.lines().filter(|line| line.starts_with("| v1 |")) {
        let fields: Vec<_> = line.split('|').map(str::trim).collect();
        assert_eq!(fields.len(), 6);
        let n = fields[2].parse::<usize>().unwrap();
        let count = fields[3].parse::<usize>().unwrap();
        assert_eq!(matmul_space(n, n, n).len(), count, "documented n={n}");
        assert_eq!(space_hash(n), fields[4], "documented n={n}");
        anchors += 1;
    }
    assert_eq!(
        anchors, 2,
        "documentation must retain both baseline anchors"
    );
}

#[test]
fn default_space_and_seeded_sample_are_frozen_before_n5() {
    // Captured from main ee9393b before implementing N5. Changes need a separate,
    // explained baseline change, never a fixture update to accommodate V2.
    for (n, count, hash, selections) in [
        (
            16,
            133,
            "f659b70a6a6219db",
            [
                "64e43d1e8bff08f2",
                "64e43d1e8bff08f2",
                "2d1d128bca2b6cd1",
                "9af6d82e741c8381",
            ],
        ),
        (
            256,
            7334,
            "252d60d1d76478c7",
            [
                "58958b11a355a7fb",
                "58958b11a355a7fb",
                "08333ae74092fabb",
                "10aefd88e3121160",
            ],
        ),
    ] {
        assert_eq!(matmul_space(n, n, n).len(), count);
        assert_eq!(space_hash(n), hash);
        assert_eq!(space_hash_for(n, SpaceConfig::V1), hash);
        for (seed, expected) in [0, 1, 17, u64::MAX].into_iter().zip(selections) {
            assert_eq!(selected_hash(n, seed), expected);
        }
    }
}

#[test]
fn v2_only_appends_unique_legal_preserving_candidates_and_fallback_keeps_v1() {
    use coffee_ten_loop::{OrderClass, Schedule};
    use std::collections::HashSet;
    let v2 = SpaceConfig::V2(Some(CacheInfo::new(64, 512, 8).unwrap()));
    let v1 = matmul_space(4, 4, 4);
    let expanded = enumerate_matmul_for(4, 4, 4, v2);
    assert!(expanded.candidates.len() > v1.len());
    assert_eq!(&expanded.candidates[..v1.len()], v1);
    assert_eq!(expanded.dropped, 0);
    let mut unique = HashSet::new();
    for c in &expanded.candidates {
        Schedule::parse(&c.schedule).unwrap();
        assert_eq!(c.class, OrderClass::Preserving);
        assert!(unique.insert(&c.schedule));
    }
    let fallback = enumerate_matmul_for(4, 4, 4, SpaceConfig::V2(None));
    assert_eq!(fallback.candidates, v1);
    assert_ne!(space_hash_for(4, SpaceConfig::V2(None)), space_hash(4));
    assert_ne!(
        space_hash_for(4, v2),
        space_hash_for(4, SpaceConfig::V2(None))
    );
}

#[test]
fn cli_defaults_to_v1_and_rejects_unsupported_modes_before_measurements() {
    let args = ["--n", "4", "--budget", "1"];
    let config = parse_search_args(args.map(str::to_string)).unwrap();
    assert_eq!(config.space, SpaceConfig::V1);
    let config = parse_search_args(
        args.into_iter()
            .chain(["--space", "v2"])
            .map(str::to_string),
    )
    .unwrap();
    assert!(matches!(config.space, SpaceConfig::V2(_)));
    let space = parse_space_config_args(["--n", "4"].map(str::to_string)).unwrap();
    assert_eq!((space.n, space.space), (4, SpaceConfig::V1));
    assert!(matches!(
        parse_space_config_args(["--n", "4", "--space", "v2"].map(str::to_string))
            .unwrap()
            .space,
        SpaceConfig::V2(_)
    ));
    assert!(parse_space_config_args(["--n", "4", "--space", "v3"].map(str::to_string)).is_err());
    assert!(
        parse_search_args(
            args.into_iter()
                .chain(["--space", "v2", "--local", "--from", "peak.jsonl"])
                .map(str::to_string)
        )
        .is_err()
    );
    assert!(
        parse_search_args(
            args.into_iter()
                .chain(["--space", "v2", "--model-top", "1", "--from", "peak.jsonl"])
                .map(str::to_string)
        )
        .is_err()
    );
}

#[test]
fn cache_derived_splits_on_each_axis_match_the_f32_reference_bit_for_bit() {
    use coffee_ten_codegen_c::compile;
    use coffee_ten_ir::{Graph, Shape};
    use coffee_ten_loop::{Schedule, apply, lower};
    let n = 4;
    let mut graph = Graph::new();
    let a = graph.input("A", Shape::new(vec![n, n])).unwrap();
    let b = graph.input("B", Shape::new(vec![n, n])).unwrap();
    let out = graph.matmul(a, b).unwrap();
    let kernel = lower(&graph, out, "n5_test").unwrap();
    let candidates = enumerate_matmul_for(
        n,
        n,
        n,
        SpaceConfig::V2(Some(CacheInfo::new(64, 512, 8).unwrap())),
    )
    .candidates;
    let dir = std::env::temp_dir().join(format!("coffee-ten-n5-bitwise-{}", std::process::id()));
    let a: Vec<f32> = (0..n * n).map(|i| (i as f32 - 7.0) / 13.0).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (9.0 - i as f32) / 17.0).collect();
    let mut expected = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                expected[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    for (axis, outer, inner) in [("i", "io", "ii"), ("j", "jo", "ji"), ("k", "ko", "ki")] {
        let fragment = format!("split(1, {axis}, {outer}, {inner}, 2)");
        let candidate = candidates
            .iter()
            .find(|c| c.schedule.contains(&fragment))
            .unwrap();
        let scheduled = apply(&kernel, &Schedule::parse(&candidate.schedule).unwrap()).unwrap();
        let compiled = compile(&scheduled, &dir).unwrap(); // TEST_FLAGS: FMA contraction off
        let mut a = a.clone();
        let mut b = b.clone();
        let mut output = vec![f32::NAN; n * n];
        compiled.run(&mut [&mut a, &mut b, &mut output]).unwrap();
        for (got, want) in output.iter().zip(&expected) {
            assert_eq!(got.to_bits(), want.to_bits(), "{}", candidate.schedule);
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
