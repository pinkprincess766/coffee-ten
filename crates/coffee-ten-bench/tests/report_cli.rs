//! Command-line errors of `coffee-ten-bench report`.
use coffee_ten_bench::report;

const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/report.jsonl");

fn error(args: &[&str]) -> String {
    report::command(args.iter().map(|s| s.to_string()))
        .unwrap_err()
        .to_string()
}

#[test]
fn numeric_flag_errors_name_the_flag_and_the_value() {
    for flag in ["--top", "--seed", "--resamples"] {
        for value in ["abc", "-1", "1.5", "", "99999999999999999999999"] {
            let err = error(&[PATH, flag, value]);
            let expected = format!("report: {flag} needs a non-negative integer, got {value:?}\n");
            assert!(err.starts_with(&expected), "{flag} {value:?}: {err}");
            assert!(err.ends_with(report::USAGE), "{err}");
        }
    }
    for flag in ["--seed", "--resamples"] {
        let err = error(&["--compare", PATH, PATH, flag, "x"]);
        assert!(
            err.starts_with(&format!(
                "report: {flag} needs a non-negative integer, got \"x\""
            )),
            "{err}"
        );
    }
}

#[test]
fn valid_values_and_other_errors_are_unchanged() {
    assert!(
        report::command(
            [PATH, "--top", "1", "--seed", "0", "--resamples", "3"].map(str::to_string)
        )
        .is_ok()
    );
    // Zero parses; the semantic checks still reject it with their own messages.
    assert_eq!(
        error(&[PATH, "--top", "0"]),
        "report: --top must be positive"
    );
    assert_eq!(
        error(&[PATH, "--resamples", "0"]),
        "report: bootstrap resamples must be positive"
    );
    assert!(error(&[PATH, "--top"]).starts_with("report: --top needs a value\n"));
    assert!(
        error(&["--compare", PATH, PATH, "--top", "1"])
            .starts_with("report: unknown argument --top\n")
    );
}
