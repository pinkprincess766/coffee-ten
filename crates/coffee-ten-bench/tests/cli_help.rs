//! `--help` / `-h` print the right usage to stdout and exit 0 without doing any work.
use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn every_subcommand_answers_help_with_its_usage() {
    for (args, usage) in [
        (&[][..], "usage: coffee-ten-bench --kernel"),
        (
            &["report"][..],
            "usage: coffee-ten-bench report <run.jsonl>",
        ),
        (
            &["report", "--compare"][..],
            "usage: coffee-ten-bench report <run.jsonl>",
        ),
        (&["search"][..], "usage: coffee-ten-bench search --n <N>"),
        (
            &["space"][..],
            "usage: coffee-ten-bench space --n <N> [--space v1|v2]",
        ),
        (
            &["space", "reduce"][..],
            "usage: coffee-ten-bench space --n <N>",
        ),
        (&["model"][..], "usage: coffee-ten-bench model --n <N>"),
        (
            &["model", "compare"][..],
            "usage: coffee-ten-bench model --n <N>",
        ),
    ] {
        for help in ["--help", "-h"] {
            let mut full = args.to_vec();
            full.push(help);
            let out = run(&full);
            assert!(out.status.success(), "{full:?}: {out:?}");
            let stdout = String::from_utf8(out.stdout).unwrap();
            assert!(stdout.starts_with(usage), "{full:?}: {stdout}");
            assert!(out.stderr.is_empty(), "{full:?}");
        }
    }
    let top = String::from_utf8(run(&["--help"]).stdout).unwrap();
    assert!(top.contains("subcommands: report, search, space, model, curve"));
    let space = String::from_utf8(run(&["space", "--help"]).stdout).unwrap();
    assert!(space.contains("coffee-ten-bench space reduce --shape"));
}

#[test]
fn help_wins_over_other_arguments_and_runs_nothing() {
    // A complete harness command line would measure; with --help it only prints usage.
    let out = run(&["--kernel", "matmul", "--n", "4", "--variant", "", "--help"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.starts_with("usage: coffee-ten-bench --kernel"),
        "{stdout}"
    );
    assert!(!stdout.contains("GFLOP/s"));
}

#[test]
fn errors_still_go_to_stderr_with_exit_code_2() {
    for args in [
        &["--n"][..],
        &["space", "--n", "0"],
        &["search", "--budget"],
    ] {
        let out = run(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
        assert!(
            String::from_utf8(out.stderr)
                .unwrap()
                .contains("usage: coffee-ten-bench")
        );
    }
    // The space usage on errors now lists --space, like --help.
    let err = String::from_utf8(run(&["space", "--n", "0"]).stderr).unwrap();
    assert!(err.contains("[--space v1|v2]"), "{err}");
}
