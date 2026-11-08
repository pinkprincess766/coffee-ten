use std::{fs, process::Command};

#[test]
fn explicit_peak_source_cannot_fall_back_to_the_runs_peak() {
    let directory =
        std::env::temp_dir().join(format!("coffee-ten-model-peak-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let peak = directory.join("selected-peak.jsonl");
    let run = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/results/2026-10-18-4928ffa.jsonl"
    );
    for text in ["", "{\"kernel\":\"matmul\"}\n"] {
        fs::write(&peak, text).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
            .args(["model", "compare", "--from", run, "--peak"])
            .arg(&peak)
            .output()
            .unwrap();
        assert!(!output.status.success(), "accepted peak source {text:?}");
        assert!(output.stdout.is_empty());
        let message = String::from_utf8(output.stderr).unwrap();
        assert!(message.contains("no peak line"), "{message}");
    }
    fs::write(
        &peak,
        r#"{"kind":"peak","fma_gflops":{"median":101.5},"triad_gbs":{"median":90.25},"triad_l2_gbs":{"median":110}}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_coffee-ten-bench"))
        .args(["model", "compare", "--from", run, "--peak"])
        .arg(&peak)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(
        report.contains("machine: peak 101.5 GFLOP/s, mem 90.25 GB/s"),
        "{report}"
    );
    assert!(report.contains(&format!("peak line from {}", peak.display())));
    fs::remove_dir_all(directory).unwrap();
}
