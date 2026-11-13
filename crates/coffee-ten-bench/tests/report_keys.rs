//! Protocol keys across rows and files: what counts as the same experiment for `report`
//! tables and `report --compare`.
use coffee_ten_bench::{report, stats::BootstrapConfig};

const LOG: &str = include_str!("fixtures/report.jsonl");
const B: &str = include_str!("fixtures/report-b.jsonl");
const CONFIG: BootstrapConfig = BootstrapConfig {
    resamples: 20,
    seed: 5,
};

/// The fixture's protocol object text (V1, no nested objects).
fn protocol() -> &'static str {
    let line = LOG.lines().nth(1).unwrap();
    let start = line.find("\"protocol\":{").unwrap() + "\"protocol\":".len();
    let end = start + line[start..].find('}').unwrap() + 1;
    &line[start..end]
}

fn fields() -> Vec<(&'static str, &'static str)> {
    let body = protocol();
    body[1..body.len() - 1]
        .split(',')
        .map(|field| field.split_once(':').unwrap())
        .collect()
}

/// A different value of the same JSON type.
fn changed(value: &str) -> String {
    match value {
        "true" => "false".into(),
        "false" => "true".into(),
        _ if value.starts_with('"') => format!("\"{}x\"", &value[1..value.len() - 1]),
        _ => (value.parse::<u64>().unwrap() + 1).to_string(),
    }
}

#[test]
fn any_changed_key_field_prevents_a_comparison() {
    let fields = fields();
    assert_eq!(fields.len(), 23);
    let mut mismatches = 0;
    for (name, value) in fields {
        let token = format!("{name}:{value}");
        let other = B.replace(&token, &format!("{name}:{}", changed(value)));
        assert_ne!(other, B, "{name}");
        let err = report::compare(LOG, &other, CONFIG, false)
            .unwrap_err()
            .to_string();
        // Fields cross-checked against the row (n, kernel, seed, ...) or validated on their own
        // (version, space) already make B unreadable; every other change is a key mismatch.
        if report::markdown(&other, 1, CONFIG).is_ok() {
            assert!(err.contains("protocol key mismatch"), "{name}: {err}");
            mismatches += 1;
        }
        // Order of arguments does not matter.
        assert!(
            report::compare(&other, LOG, CONFIG, false).is_err(),
            "{name}"
        );
    }
    // At least the provenance fields that are never echoed in the row reach the key check.
    assert!(mismatches >= 10, "{mismatches}");
}

#[test]
fn json_key_order_inside_the_protocol_does_not_matter() {
    let mut fields = fields();
    fields.reverse();
    let reordered = format!(
        "{{{}}}",
        fields
            .iter()
            .map(|(name, value)| format!("{name}:{value}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let other = B.replace(protocol(), &reordered);
    assert_ne!(other, B);
    let expected = report::compare(B, LOG, CONFIG, false).unwrap();
    let got = report::compare(&other, LOG, CONFIG, false).unwrap();
    // Only the input-byte fingerprint of A differs.
    let tail = |s: &str| s[s.find("B fingerprint").unwrap()..].to_string();
    assert_eq!(tail(&got), tail(&expected));
    // In one file, both spellings are one protocol and share one table.
    let mixed = report::markdown(&format!("{LOG}{other}"), 10, CONFIG).unwrap();
    assert_eq!(mixed.matches("## Protocol ").count(), 1);
    assert!(mixed.contains("Records: 6."));
}

#[test]
fn a_file_with_two_protocols_cannot_be_either_side_of_a_comparison() {
    let other = LOG.replace(
        "\"rustflags\":\"\"",
        "\"rustflags\":\"-C target-cpu=native\"",
    );
    assert_ne!(other, LOG);
    let mixed = format!("{LOG}{other}");
    for (a, b) in [(mixed.as_str(), LOG), (LOG, mixed.as_str())] {
        let err = report::compare(a, b, CONFIG, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("comparison requires exactly one protocol per file"),
            "{err}"
        );
    }
    // The report keeps them apart even though rustflags is not printed in the header.
    let out = report::markdown(&mixed, 10, CONFIG).unwrap();
    assert_eq!(out.matches("## Protocol ").count(), 2);
    assert_eq!(out.matches("Records: 3.").count(), 2);
}

/// Replace the last occurrence, which in the fixture is the row's copy (after `protocol`).
fn replace_last(text: &str, from: &str, to: &str) -> String {
    let index = text.rfind(from).unwrap();
    format!("{}{to}{}", &text[..index], &text[index + from.len()..])
}

#[test]
fn row_fields_must_agree_with_their_protocol() {
    let line = format!("{}\n", LOG.lines().nth(1).unwrap());
    assert!(report::markdown(&line, 1, CONFIG).is_ok());
    for (from, to, expected) in [
        (
            "\"cpu\":\"fixture CPU\"",
            "\"cpu\":\"other CPU\"",
            "inconsistent cpu",
        ),
        (
            "\"os\":\"fixture OS\"",
            "\"os\":\"other OS\"",
            "inconsistent os",
        ),
        (
            "\"flags\":\"-O3\"",
            "\"flags\":\"-O2\"",
            "inconsistent flags",
        ),
        (
            "\"clang\":\"fixture clang\"",
            "\"clang\":\"other clang\"",
            "inconsistent clang",
        ),
        (
            "\"machine\":\"fixture Mac\"",
            "\"machine\":\"other Mac\"",
            "inconsistent machine",
        ),
        (
            "\"seed\":17",
            "\"seed\":18",
            "inconsistent measurement metadata",
        ),
    ] {
        let bad = replace_last(&line, from, to);
        assert_ne!(bad, line);
        let err = report::markdown(&bad, 1, CONFIG).unwrap_err().to_string();
        assert!(err.contains(expected), "{from}: {err}");
    }
    // The row's own kernel, graph hash, size and quick flag come first in the line.
    for (from, to, expected) in [
        (
            "{\"kernel\":\"matmul\"",
            "{\"kernel\":\"conv\"",
            "inconsistent kernel",
        ),
        (
            "\"graph_hash\":\"0123456789abcdef\",\"schedule\"",
            "\"graph_hash\":\"fedcba9876543210\",\"schedule\"",
            "inconsistent graph_hash",
        ),
        (
            "{\"kernel\":\"matmul\",\"n\":2",
            "{\"kernel\":\"matmul\",\"n\":3",
            "inconsistent matrix size",
        ),
        (
            "\"verified\":true,\"quick\":true",
            "\"verified\":true,\"quick\":false",
            "inconsistent measurement metadata",
        ),
    ] {
        let bad = line.replacen(from, to, 1);
        assert_ne!(bad, line, "{from}");
        let err = report::markdown(&bad, 1, CONFIG).unwrap_err().to_string();
        assert!(err.contains(expected), "{from}: {err}");
    }
}
