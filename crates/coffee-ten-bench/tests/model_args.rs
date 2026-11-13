//! `coffee-ten-bench model` argument errors for `--peak` and `--from`: nothing on stdout,
//! exit 2 for usage errors and 1 for unusable files, and a message naming what is accepted.
use std::process::Command;

use coffee_ten_bench::model::USAGE;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .arg("model")
        .args(args)
        .output()
        .unwrap()
}

#[track_caller]
fn assert_error(args: &[&str], code: i32, stderr: &str) {
    let out = run(args);
    assert_eq!(out.status.code(), Some(code), "{args:?}");
    assert!(out.stdout.is_empty(), "{args:?}");
    assert_eq!(String::from_utf8(out.stderr).unwrap(), stderr, "{args:?}");
}

#[test]
fn peak_must_be_a_finite_number_greater_than_zero() {
    for value in [
        "-1", "-0", "0", "NaN", "nan", "inf", "-inf", "1e400", "", "fast",
    ] {
        assert_error(
            &["--n", "8", "--peak", value],
            2,
            &format!(
                "error: --peak needs a finite number greater than 0, got {value:?}\n{USAGE}\n"
            ),
        );
    }
    assert_error(
        &["--n", "8", "--peak"],
        2,
        &format!("error: --peak needs a value\n{USAGE}\n"),
    );
    // The same rule holds for the other speeds.
    assert_error(
        &["--n", "8", "--mem-gbs", "NaN"],
        2,
        &format!("error: --mem-gbs needs a finite number greater than 0, got \"NaN\"\n{USAGE}\n"),
    );
    // A tiny positive peak is accepted: the model takes it as given.
    assert!(run(&["--n", "8", "--peak", "1e-9"]).status.success());
}

#[test]
fn from_must_name_a_non_empty_file() {
    let dir = std::env::temp_dir().join(format!("coffee-ten-model-args-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let empty = dir.join("empty.jsonl");
    let blank = dir.join("blank.jsonl");
    std::fs::write(&empty, "").unwrap();
    std::fs::write(&blank, "\n \n\t\n").unwrap();
    let missing = dir.join("missing.jsonl");
    let (empty, blank, missing) = (
        empty.to_str().unwrap(),
        blank.to_str().unwrap(),
        missing.to_str().unwrap(),
    );
    let usage_error = |flag: &str| format!("error: {flag} needs a file path, got \"\"\n{USAGE}\n");
    let empty_file = |flag: &str, path: &str| {
        format!("error: {flag} {path}: the file is empty, so it has no peak line and no records\n")
    };

    assert_error(&["--n", "8", "--from", ""], 2, &usage_error("--from"));
    assert_error(
        &["--n", "8", "--from"],
        2,
        &format!("error: --from needs a value\n{USAGE}\n"),
    );
    assert_error(
        &["--n", "8", "--from", empty],
        1,
        &empty_file("--from", empty),
    );
    assert_error(
        &["--n", "8", "--from", blank],
        1,
        &empty_file("--from", blank),
    );
    let out = run(&["--n", "8", "--from", missing]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8(out.stderr)
            .unwrap()
            .starts_with(&format!("error: cannot read {missing}: "))
    );

    assert_error(&["compare", "--from", ""], 2, &usage_error("--from"));
    assert_error(
        &["compare", "--from", empty],
        1,
        &empty_file("--from", empty),
    );
    assert_error(
        &["compare", "--from", blank],
        1,
        &empty_file("--from", blank),
    );
    let run_log = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-18-4928ffa.jsonl"
    );
    assert_error(
        &["compare", "--from", run_log, "--peak", ""],
        2,
        &usage_error("--peak"),
    );
    assert_error(
        &["compare", "--from", run_log, "--peak", empty],
        1,
        &empty_file("--peak", empty),
    );
    let _ = std::fs::remove_dir_all(&dir);
}
