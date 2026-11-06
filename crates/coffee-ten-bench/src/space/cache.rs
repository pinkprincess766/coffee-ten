//! Opt-in capacity-derived candidates; no measurements or model-based filtering.
use super::TILE_FACTORS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheInfo {
    l1_bytes: u64,
    l2_bytes: u64,
    line_bytes: u64,
}

impl CacheInfo {
    /// Ignore missing/nonsensical hardware data instead of guessing a cache layout.
    pub fn new(l1_bytes: u64, l2_bytes: u64, line_bytes: u64) -> Option<Self> {
        (line_bytes >= 4
            && line_bytes.is_power_of_two()
            && l1_bytes >= line_bytes
            && l2_bytes >= l1_bytes)
            .then_some(Self {
                l1_bytes,
                l2_bytes,
                line_bytes,
            })
    }

    pub fn to_json(self) -> String {
        format!(
            "{{\"l1_bytes\":{},\"l2_bytes\":{},\"line_bytes\":{}}}",
            self.l1_bytes, self.l2_bytes, self.line_bytes
        )
    }

    #[cfg(any(target_os = "macos", test))]
    fn detect_with(mut read: impl FnMut(&str) -> Option<u64>) -> Option<Self> {
        Self::new(
            read("hw.l1dcachesize")?,
            read("hw.l2cachesize")?,
            read("hw.cachelinesize")?,
        )
    }

    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::detect_with(|name| {
                crate::meta::command_stdout("sysctl", &["-n", name])
                    .parse()
                    .ok()
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    fn factors(self) -> Vec<usize> {
        let line_floats = self.line_bytes / 4;
        let mut factors = vec![line_floats];
        for capacity in [self.l1_bytes, self.l2_bytes] {
            // Reserve half the cache; three square f32 tiles: 3 * 4 * t^2 <= C/2.
            let root = (capacity / 24).isqrt();
            if root >= line_floats {
                // Power-of-two divisors are usable for the default power-of-two sizes.
                factors.push(1u64 << (63 - root.leading_zeros()));
            }
        }
        factors
            .into_iter()
            .filter_map(|x| usize::try_from(x).ok())
            .filter(|&x| x >= 2)
            .collect()
    }
}

/// Resolved once per invocation. V1 never queries hardware; V2(None) is a visible fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SpaceConfig {
    #[default]
    V1,
    V2(Option<CacheInfo>),
}

impl SpaceConfig {
    pub fn from_arg(value: &str) -> Result<Self, crate::BenchError> {
        match value {
            "v1" => Ok(Self::V1),
            "v2" => Ok(Self::V2(CacheInfo::detect())),
            _ => Err(crate::BenchError::Usage(format!(
                "--space must be v1 or v2, got {value:?}"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2(_) => "v2",
        }
    }

    pub fn cache(self) -> Option<CacheInfo> {
        match self {
            Self::V1 => None,
            Self::V2(cache) => cache,
        }
    }

    pub fn tile_factors(self) -> Vec<usize> {
        let mut factors = TILE_FACTORS.to_vec();
        if let Some(cache) = self.cache() {
            factors.extend(cache.factors());
        }
        factors.sort_unstable();
        factors.dedup();
        factors
    }

    pub fn description(self) -> String {
        let cache = self
            .cache()
            .map_or("null (fallback: legacy factors)".into(), CacheInfo::to_json);
        format!(
            "space {}  tile factors {:?}\ncache {}\n",
            self.name(),
            self.tile_factors(),
            cache
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_cache_formula_and_legacy_union() {
        let cache = CacheInfo::new(32 * 1024, 2 * 1024 * 1024, 64).unwrap();
        // sqrt(32768/24) -> 36 -> 32; sqrt(2097152/24) -> 295 -> 256.
        assert_eq!(cache.factors(), [16, 32, 256]);
        assert_eq!(
            SpaceConfig::V2(Some(cache)).tile_factors(),
            [8, 16, 32, 64, 128, 256]
        );
        assert_eq!(
            CacheInfo::new(128 * 1024, 16 * 1024 * 1024, 128)
                .unwrap()
                .factors(),
            [32, 64, 512]
        );
    }

    #[test]
    fn unavailable_or_invalid_hardware_keeps_old_factors() {
        assert!(CacheInfo::detect_with(|_| None).is_none());
        assert!(
            CacheInfo::detect_with(|name| if name == "hw.l2cachesize" {
                None
            } else {
                Some(64)
            })
            .is_none()
        );
        for (l1, l2, line) in [
            (0, 1024, 64),
            (1024, 512, 64),
            (1024, 2048, 0),
            (1024, 2048, 3),
            (1024, 2048, 48),
        ] {
            assert!(CacheInfo::new(l1, l2, line).is_none());
        }
        assert_eq!(SpaceConfig::V2(None).tile_factors(), TILE_FACTORS);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(CacheInfo::detect(), None);
    }

    #[test]
    fn extreme_values_do_not_overflow_capacity_arithmetic() {
        let cache = CacheInfo::new(u64::MAX, u64::MAX, 64).unwrap();
        assert!(!cache.factors().is_empty());
        assert!(CacheInfo::new(8, 16, 4).unwrap().factors().is_empty());
    }
}
