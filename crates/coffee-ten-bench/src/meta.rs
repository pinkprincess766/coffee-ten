//! Header metadata of a run: machine, toolchain, code version, time.

use coffee_ten_codegen_c::BENCH_FLAGS;
use std::process::Command;

/// What a command that failed or printed nothing is reported as.
pub const UNKNOWN: &str = "unknown";

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub hw_model: String,
    pub cpu_brand: String,
    pub os_version: String,
    pub clang: String,
    pub flags: String,
    pub git_rev: String,
    pub date_utc: String,
    pub seed: u64,
}

/// Runs the system commands; any that fails gives `UNKNOWN`, so Linux does not crash.
pub fn collect(seed: u64) -> Metadata {
    Metadata {
        hw_model: command_stdout("sysctl", &["-n", "hw.model"]),
        cpu_brand: command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]),
        os_version: command_stdout("sw_vers", &["-productVersion"]),
        clang: first_line(&command_stdout("clang", &["--version"])),
        flags: BENCH_FLAGS.join(" "),
        git_rev: command_stdout("git", &["rev-parse", "--short", "HEAD"]),
        date_utc: command_stdout("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        seed,
    }
}

/// Trimmed stdout of `program args`, or `UNKNOWN` if it cannot start, fails or prints nothing.
pub fn command_stdout(program: &str, args: &[&str]) -> String {
    let Ok(output) = Command::new(program).args(args).output() else {
        return UNKNOWN.to_string();
    };
    if !output.status.success() {
        return UNKNOWN.to_string();
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        UNKNOWN.to_string()
    } else {
        text
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or(UNKNOWN).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failing_commands_give_unknown() {
        assert_eq!(command_stdout("coffee-ten-no-such-program", &[]), UNKNOWN);
        assert_eq!(command_stdout("sh", &["-c", "echo text; exit 3"]), UNKNOWN);
        assert_eq!(command_stdout("sh", &["-c", "exit 0"]), UNKNOWN);
    }

    #[test]
    fn stdout_is_trimmed() {
        assert_eq!(command_stdout("sh", &["-c", "echo '  x y  '"]), "x y");
    }

    #[test]
    fn clang_version_is_one_line() {
        assert_eq!(
            first_line("Apple clang 17\nTarget: arm64\n"),
            "Apple clang 17"
        );
        assert_eq!(first_line(""), UNKNOWN);
    }
}
