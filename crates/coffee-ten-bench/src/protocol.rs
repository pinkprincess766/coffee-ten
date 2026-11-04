//! A reproducibility key and compilation accounting, without fitting or new dependencies.
use crate::{
    meta::Metadata,
    run::{Record, RunInfo, fnv1a_64, json_string},
    space::matmul_space,
};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

pub const VERSION: u32 = 1;
pub const SOURCE_HASH: &str = env!("COFFEE_TEN_SOURCE_HASH");
pub const RUSTC: &str = env!("COFFEE_TEN_RUSTC");

/// Hash ordered, length-delimited canonical schedules and their arithmetic classes.
/// A reproducibility fingerprint, not a security hash. Ordering affects seeded sampling.
pub fn space_hash(n: usize) -> String {
    static CACHE: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry(n)
        .or_insert_with(|| {
            let mut bytes = b"coffee-ten-matmul-space-v1\0".to_vec();
            for c in matmul_space(n, n, n) {
                bytes.extend_from_slice(&(c.schedule.len() as u64).to_le_bytes());
                bytes.extend_from_slice(c.schedule.as_bytes());
                bytes.push(match c.class {
                    coffee_ten_loop::OrderClass::Preserving => 0,
                    coffee_ten_loop::OrderClass::Reordering => 1,
                });
            }
            format!("{:016x}", fnv1a_64(&bytes))
        })
        .clone()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolKey {
    pub version: u32,
    pub kernel: String,
    pub n: usize,
    pub graph_hash: String,
    pub space: String,
    pub space_hash: String,
    pub source_hash: String,
    pub cpu: String,
    pub machine: String,
    pub os: String,
    pub arch: String,
    pub target_os: String,
    pub clang: String,
    pub flags: String,
    pub rustc: String,
    pub harness_profile: String,
    pub rustflags: String,
    pub samples: usize,
    pub min_sample_ms: u64,
    pub input_seed: u64,
    pub quick: bool,
    pub warmup_ms: u64,
    pub candidate_warmup_ms: u64,
}
impl ProtocolKey {
    pub fn for_record(r: &Record, m: &Metadata) -> Self {
        Self::for_run(
            &RunInfo {
                kernel: r.kernel,
                n: r.n,
                graph_hash: r.graph_hash,
                min_sample_ms: r.min_sample_ms,
                quick: r.quick,
            },
            r.samples,
            m,
        )
    }
    pub fn for_run(r: &RunInfo, samples: usize, m: &Metadata) -> Self {
        Self {
            version: VERSION,
            kernel: r.kernel.into(),
            n: r.n,
            graph_hash: format!("{:016x}", r.graph_hash),
            space: "v1".into(),
            space_hash: space_hash(r.n),
            source_hash: SOURCE_HASH.into(),
            cpu: m.cpu_brand.clone(),
            machine: m.hw_model.clone(),
            os: m.os_version.clone(),
            arch: std::env::consts::ARCH.into(),
            target_os: std::env::consts::OS.into(),
            clang: m.clang.clone(),
            flags: m.flags.clone(),
            rustc: RUSTC.into(),
            harness_profile: env!("COFFEE_TEN_PROFILE").into(),
            rustflags: env!("COFFEE_TEN_RUSTFLAGS").into(),
            samples,
            min_sample_ms: r.min_sample_ms,
            input_seed: m.seed,
            quick: r.quick,
            warmup_ms: if r.quick {
                crate::run::QUICK_WARMUP_MS
            } else {
                crate::run::WARMUP_MS
            },
            candidate_warmup_ms: if r.quick {
                crate::run::QUICK_WARMUP_MS
            } else {
                crate::run::CANDIDATE_WARMUP_MS
            },
        }
    }
    pub fn to_json(&self) -> String {
        let mut fields = vec![
            format!("\"version\":{}", self.version),
            format!("\"n\":{}", self.n),
        ];
        for (name, value) in [
            ("kernel", &self.kernel),
            ("graph_hash", &self.graph_hash),
            ("space", &self.space),
            ("space_hash", &self.space_hash),
            ("source_hash", &self.source_hash),
            ("cpu", &self.cpu),
            ("machine", &self.machine),
            ("os", &self.os),
            ("arch", &self.arch),
            ("target_os", &self.target_os),
            ("clang", &self.clang),
            ("flags", &self.flags),
            ("rustc", &self.rustc),
            ("harness_profile", &self.harness_profile),
            ("rustflags", &self.rustflags),
        ] {
            fields.push(format!("{}:{}", json_string(name), json_string(value)));
        }
        fields.extend([
            format!("\"samples\":{}", self.samples),
            format!("\"min_sample_ms\":{}", self.min_sample_ms),
            format!("\"input_seed\":{}", self.input_seed),
            format!("\"quick\":{}", self.quick),
            format!("\"warmup_ms\":{}", self.warmup_ms),
            format!("\"candidate_warmup_ms\":{}", self.candidate_warmup_ms),
        ]);
        format!("{{{}}}", fields.join(","))
    }
}

/// One successful compiler API call, including emission, cache lookup and library loading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Compilation {
    pub elapsed_ns: u128,
    pub cache_hit: bool,
}
impl Compilation {
    pub fn to_json(self) -> String {
        format!(
            "{{\"elapsed_ns\":{},\"cache_hit\":{}}}",
            self.elapsed_ns, self.cache_hit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_and_space_fingerprints_are_present_and_deterministic() {
        assert_eq!(SOURCE_HASH.len(), 16);
        assert!(SOURCE_HASH.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(space_hash(16), space_hash(16));
        assert_ne!(space_hash(16), space_hash(32));
    }
}
