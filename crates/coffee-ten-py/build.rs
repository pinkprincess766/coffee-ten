//! The module is built as a Python extension (see .cargo/config.toml): libpython is not
//! linked, the interpreter that imports the module provides its symbols. Linux allows
//! undefined symbols in a shared library; the macOS linker needs to be told, or a plain
//! `cargo build` of this cdylib fails. maturin passes the same flags for wheels.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
