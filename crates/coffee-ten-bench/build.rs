//! Fingerprint actual build inputs, independently of the Git commit or dirty status.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let entry = entry.expect("source entry");
        let path = entry.path();
        if entry.file_type().expect("file type").is_dir() {
            if !matches!(
                entry.file_name().to_str(),
                Some("target" | ".git" | ".claude")
            ) {
                files(&path, out);
            }
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("rs" | "toml")
        ) {
            out.push(path);
        }
    }
}
fn hash(bytes: &[u8], state: &mut u64) {
    for &b in bytes {
        *state = (*state ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let mut paths = Vec::new();
    files(&root.join("crates"), &mut paths);
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "bench/peak/peak.c",
    ] {
        paths.push(root.join(name));
    }
    paths.sort_by_key(|p| p.strip_prefix(&root).unwrap().to_path_buf());
    let mut state = 0xcbf29ce484222325;
    hash(b"coffee-ten-source-v1\0", &mut state);
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    for path in paths {
        let relative = path
            .strip_prefix(&root)
            .unwrap()
            .to_str()
            .expect("UTF-8 source path")
            .replace('\\', "/");
        let bytes = fs::read(&path).expect("read build input");
        hash(&(relative.len() as u64).to_le_bytes(), &mut state);
        hash(relative.as_bytes(), &mut state);
        hash(&(bytes.len() as u64).to_le_bytes(), &mut state);
        hash(&bytes, &mut state);
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rustc-env=COFFEE_TEN_SOURCE_HASH={state:016x}");
    let version = std::process::Command::new(env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .expect("rustc version");
    assert!(version.status.success());
    println!(
        "cargo:rustc-env=COFFEE_TEN_RUSTC={}",
        String::from_utf8(version.stdout).unwrap().trim()
    );
}
