//! `coffee-ten-bench space` argument errors: exit code 2, nothing on stdout, and a message
//! that names the accepted values followed by the space usage.
use std::process::Command;

use coffee_ten_bench::space::SPACE_USAGE;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .arg("space")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn bad_arguments_name_the_accepted_values_and_print_the_usage() {
    for (args, message) in [
        (&["--n", "0"][..], "--n must be in 1..=1024, got 0"),
        (&["--n", "1025"], "--n must be in 1..=1024, got 1025"),
        (
            &["--n", "-1"],
            "--n needs a non-negative integer, got \"-1\"",
        ),
        (&["--n", "x"], "--n needs a non-negative integer, got \"x\""),
        (&[], "--n is required"),
        (&["--space", "v2"], "--n is required"),
        (&["--n"], "--n needs a value"),
        (
            &["--n", "4", "--space", "v3"],
            "--space must be v1 or v2, got \"v3\"",
        ),
        (
            &["--n", "4", "--space", "V2"],
            "--space must be v1 or v2, got \"V2\"",
        ),
        (
            &["--n", "4", "--space", ""],
            "--space must be v1 or v2, got \"\"",
        ),
        (&["--n", "4", "--space"], "--space needs a value"),
        (
            &["--n", "4", "--spaec", "v2"],
            "unknown argument --spaec, expected --n or --space",
        ),
    ] {
        let out = run(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
        assert_eq!(
            String::from_utf8(out.stderr).unwrap(),
            format!("error: {message}\n{SPACE_USAGE}\n"),
            "{args:?}"
        );
    }
}

#[test]
fn the_default_space_is_v1_and_lists_the_same_candidates() {
    let default = run(&["--n", "4"]);
    let v1 = run(&["--space", "v1", "--n", "4"]);
    assert!(default.status.success() && v1.status.success());
    assert!(!default.stdout.is_empty());
    assert_eq!(default.stdout, v1.stdout);
}
