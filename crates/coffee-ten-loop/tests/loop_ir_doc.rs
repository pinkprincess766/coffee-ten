//! Every example in `docs/loop-ir.md` is checked here, so the doc cannot drift from the code.
//!
//! The doc is read at compile time and cut into fenced code blocks. A fence starts at column 0
//! with three backticks and an info string: a kind, then optional words. Every block must have
//! one of these kinds; any other block fails the test, so no example goes unchecked.
//!
//! - `loop-ir [name=N]`: kernel text. It must parse with `parse_kernel`, pass `validate`, and
//!   print back (`Display`) to the same text. `name=N` lets a schedule block refer to it.
//! - `loop-ir bad`: kernel text that must fail. The next block is an `error` block with the
//!   exact `Display` of the error from `parse_kernel` (or, if it parses, from `validate`).
//! - `schedule`: schedule text. It must parse with `Schedule::parse`, and its `Display` must
//!   parse back to the same schedule.
//! - `schedule on=N`: also applied to the kernel named `N` (an earlier block). The next block
//!   is either a `loop-ir` block, which must equal the printed result, or an `error` block,
//!   which must equal the `Display` of the `apply` error.
//! - `schedule bad`: schedule text that must fail to parse. The next block is an `error` block.
//!   Schedule text is the block without its last newline, so a byte offset in an error counts
//!   from the first character of the block.
//! - `error`: the expected one-line error text. It must follow a block that asks for it.
//! - `grammar`: reference text; not checked.

use coffee_ten_ir::{Graph, Shape};
use coffee_ten_loop::{Kernel, Schedule, apply, lower, parse_kernel};
use std::collections::HashMap;

const DOC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/loop-ir.md"
));

struct Block {
    /// 1-based line of the opening fence.
    line: usize,
    kind: String,
    words: Vec<String>,
    text: String,
}

impl Block {
    /// The value of a `key=value` word.
    fn value(&self, key: &str) -> Option<&str> {
        self.words
            .iter()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }

    fn is_bad(&self) -> bool {
        self.words.iter().any(|w| w == "bad")
    }

    /// `docs/loop-ir.md:LINE`, for failure messages.
    fn at(&self) -> String {
        format!("docs/loop-ir.md:{}", self.line)
    }
}

/// The fenced blocks of `doc`, in order. Fences must start at column 0 and be backticks.
fn blocks(doc: &str) -> Vec<Block> {
    let mut found = Vec::new();
    let mut open: Option<Block> = None;
    for (n, line) in doc.lines().enumerate() {
        let is_fence_like =
            line.trim_start().starts_with("```") || line.trim_start().starts_with("~~~");
        if let Some(block) = open.as_mut() {
            if line == "```" {
                found.extend(open.take());
            } else {
                block.text.push_str(line);
                block.text.push('\n');
            }
        } else if let Some(info) = line.strip_prefix("```") {
            let mut words = info.split_whitespace();
            open = Some(Block {
                line: n + 1,
                kind: words.next().unwrap_or("").to_string(),
                words: words.map(String::from).collect(),
                text: String::new(),
            });
        } else {
            assert!(
                !is_fence_like,
                "docs/loop-ir.md:{}: fence not at column 0 or not ```",
                n + 1
            );
        }
    }
    assert!(open.is_none(), "unclosed code block");
    found
}

/// The text of the `error` block after block `i`, which is marked as used.
fn expected_error<'a>(all: &'a [Block], i: usize, used: &mut [bool]) -> &'a str {
    let next = all
        .get(i + 1)
        .filter(|b| b.kind == "error")
        .unwrap_or_else(|| panic!("{}: an error block must follow", all[i].at()));
    used[i + 1] = true;
    next.text.trim_end()
}

/// The error of kernel text: from the parser, or if it parses, from `validate`.
fn kernel_error(text: &str) -> Option<String> {
    match parse_kernel(text) {
        Err(e) => Some(e.to_string()),
        Ok(kernel) => kernel.validate().err().map(|e| e.to_string()),
    }
}

#[test]
fn the_extractor_reads_kinds_words_and_text() {
    let doc = "text\n```schedule on=m bad\nunroll(0, i)\n```\nmore\n```\nx\n```\n";
    let found = blocks(doc);
    assert_eq!(found.len(), 2);
    assert_eq!((found[0].line, found[0].kind.as_str()), (2, "schedule"));
    assert_eq!(found[0].value("on"), Some("m"));
    assert!(found[0].is_bad());
    assert_eq!(found[0].text, "unroll(0, i)\n");
    assert_eq!(found[1].kind, "");
}

#[test]
fn every_example_in_the_doc_holds() {
    let all = blocks(DOC);
    let mut kernels: HashMap<&str, Kernel> = HashMap::new();
    let mut used = vec![false; all.len()];
    // How many blocks of each case were checked, so that a broken extractor cannot pass.
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (i, b) in all.iter().enumerate() {
        match (b.kind.as_str(), b.is_bad()) {
            ("grammar" | "error", _) => {}
            ("loop-ir", true) => {
                let want = expected_error(&all, i, &mut used);
                assert_eq!(kernel_error(&b.text).as_deref(), Some(want), "{}", b.at());
                *seen.entry("bad kernel").or_default() += 1;
            }
            ("loop-ir", false) => {
                let kernel = parse_kernel(&b.text)
                    .unwrap_or_else(|e| panic!("{}: does not parse: {e}", b.at()));
                kernel
                    .validate()
                    .unwrap_or_else(|e| panic!("{}: does not validate: {e}", b.at()));
                assert_eq!(kernel.to_string(), b.text, "{}: prints differently", b.at());
                if let Some(name) = b.value("name") {
                    let clash = kernels.insert(name, kernel);
                    assert!(clash.is_none(), "{}: name {name} is used twice", b.at());
                }
                *seen.entry("kernel").or_default() += 1;
            }
            ("schedule", true) => {
                let want = expected_error(&all, i, &mut used);
                let got = Schedule::parse(b.text.trim_end())
                    .expect_err(&b.at())
                    .to_string();
                assert_eq!(got, want, "{}", b.at());
                *seen.entry("bad schedule").or_default() += 1;
            }
            ("schedule", false) => {
                let schedule = Schedule::parse(b.text.trim_end())
                    .unwrap_or_else(|e| panic!("{}: does not parse: {e}", b.at()));
                let again = Schedule::parse(&schedule.to_string());
                assert_eq!(again.as_ref(), Ok(&schedule), "{}", b.at());
                *seen.entry("schedule").or_default() += 1;
                let Some(name) = b.value("on") else { continue };
                let kernel = kernels
                    .get(name)
                    .unwrap_or_else(|| panic!("{}: no earlier kernel named {name}", b.at()));
                let next = all
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("{}: a result block must follow", b.at()));
                match (apply(kernel, &schedule), next.kind.as_str()) {
                    (Ok(result), "loop-ir") => {
                        assert_eq!(result.to_string(), next.text, "{}: result differs", b.at());
                        *seen.entry("applied").or_default() += 1;
                    }
                    (Err(e), "error") => {
                        let want = expected_error(&all, i, &mut used);
                        assert_eq!(e.to_string(), want, "{}", b.at());
                        *seen.entry("refused").or_default() += 1;
                    }
                    (got, kind) => panic!("{}: apply gave {got:?}, doc has a {kind} block", b.at()),
                }
            }
            (kind, _) => panic!("{}: block kind {kind:?} is not checked", b.at()),
        }
    }
    for (i, b) in all.iter().enumerate() {
        assert!(
            b.kind != "error" || used[i],
            "{}: error block with no source",
            b.at()
        );
    }
    for case in [
        "kernel",
        "bad kernel",
        "schedule",
        "bad schedule",
        "applied",
        "refused",
    ] {
        assert!(seen.contains_key(case), "the doc has no {case} example");
    }
}

/// The naive matmul in the doc is what `lower` prints for an 8x4 times 4x32 product.
#[test]
fn the_naive_matmul_block_is_what_lower_prints() {
    let mut g = Graph::new();
    let a = g
        .input("A", Shape::new(vec![8, 4]))
        .expect("test graph builds");
    let b = g
        .input("B", Shape::new(vec![4, 32]))
        .expect("test graph builds");
    let product = g.matmul(a, b).expect("test graph builds");
    let lowered = lower(&g, product, "matmul").expect("test graph lowers");
    let all = blocks(DOC);
    let doc_matmul = all
        .iter()
        .find(|b| b.value("name") == Some("matmul"))
        .expect("the doc names a matmul block");
    assert_eq!(lowered.to_string(), doc_matmul.text);
}
