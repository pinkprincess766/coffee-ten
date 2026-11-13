//! Exit status of the offline subcommands: 0 success, 2 wrong command line, 1 rejected data.
use std::process::Command;

const LOG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/report.jsonl");

fn code(args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .args(args)
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8(out.stderr).unwrap())
}

#[test]
fn report_separates_command_line_errors_from_data_errors() {
    assert_eq!(code(&["report", LOG, "--resamples", "10"]).0, Some(0));
    for args in [
        &["report"][..],
        &["report", "--compare", LOG],
        &["report", LOG, "--top"],
        &["report", LOG, "--bogus", "1"],
        &["report", "--compare", LOG, LOG, "--top", "1"],
    ] {
        let (status, stderr) = code(args);
        assert_eq!(status, Some(2), "{args:?}: {stderr}");
        assert!(
            stderr.contains("usage: coffee-ten-bench report"),
            "{args:?}"
        );
    }
    let dir = std::env::temp_dir().join(format!("coffee-ten-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("bad.jsonl");
    std::fs::write(&bad, "{\"kind\":\"replay_curve\"}\n").unwrap();
    let bad = bad.to_str().unwrap();
    for args in [
        &["report", "/nonexistent/run.jsonl"][..],
        &["report", bad],
        &["report", "--compare", LOG, bad],
        &["report", LOG, "--resamples", "0"],
    ] {
        let (status, stderr) = code(args);
        assert_eq!(status, Some(1), "{args:?}: {stderr}");
        assert!(stderr.starts_with("error: report: "), "{args:?}");
        assert!(!stderr.contains("usage:"), "{args:?}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn curve_separates_command_line_errors_from_data_errors() {
    // Rejected by curve itself or by the shared search argument parser: usage errors.
    for args in [
        &["curve", "--bogus", "1"][..],
        &["curve", "--seeds", "1,,2"],
        &["curve", "--log", "out"],
        &["curve", "--n", "2", "--budget", "1"],
    ] {
        let (status, stderr) = code(args);
        assert_eq!(status, Some(2), "{args:?}: {stderr}");
        assert!(stderr.contains("usage: coffee-ten-bench curve"), "{args:?}");
    }
    // A well-formed command whose table cannot be read: a data error.
    let (status, stderr) = code(&[
        "curve",
        "--n",
        "2",
        "--budget",
        "1",
        "--replay",
        "/nonexistent/table.jsonl",
    ]);
    assert_eq!(status, Some(1), "{stderr}");
}
