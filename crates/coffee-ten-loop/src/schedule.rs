//! Schedules: named loop-IR transformations and their one-line text form.

use std::fmt;

/// One loop-IR transformation of nest `nest`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Split loop `loop_name` of nest `nest` into `outer` (extent ceil(n/factor)) and `inner` (extent factor).
    Split {
        nest: usize,
        loop_name: String,
        outer: String,
        inner: String,
        factor: usize,
    },
    /// New loop order of nest `nest`, outermost first.
    Reorder {
        nest: usize,
        order: Vec<String>,
    },
    Vectorize {
        nest: usize,
        loop_name: String,
    },
    /// Ask the backend to unroll loop `loop_name` of nest `nest` fully.
    Unroll {
        nest: usize,
        loop_name: String,
    },
}

/// Steps in the order they are applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    pub steps: Vec<Step>,
}

/// A syntax error in schedule text. `pos` is the byte offset in the text; `line` and `column`
/// are 1-based, with `column` counted in characters. `msg` names what was expected and what
/// was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleParseError {
    pub pos: usize,
    pub line: usize,
    pub column: usize,
    pub msg: String,
}

impl Schedule {
    /// Parses the text form, e.g. `split(1, k, ko, ki, 4); vectorize(1, j)`.
    /// Checks syntax and names only; loops and legality are not checked here.
    pub fn parse(text: &str) -> Result<Schedule, ScheduleParseError> {
        let mut p = Parser { text, pos: 0 };
        let mut steps = Vec::new();
        p.skip_ws();
        if p.peek().is_none() {
            return Ok(Schedule::default());
        }
        loop {
            steps.push(p.step()?);
            p.skip_ws();
            if p.peek() != Some(b';') {
                break;
            }
            p.pos += 1;
        }
        if p.peek().is_some() {
            return Err(p.expected("';' or end of text"));
        }
        Ok(Schedule { steps })
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, step) in self.steps.iter().enumerate() {
            if i > 0 {
                f.write_str("; ")?;
            }
            write_step(f, step)?;
        }
        Ok(())
    }
}

fn write_step(f: &mut fmt::Formatter<'_>, step: &Step) -> fmt::Result {
    match step {
        Step::Split {
            nest,
            loop_name,
            outer,
            inner,
            factor,
        } => write!(f, "split({nest}, {loop_name}, {outer}, {inner}, {factor})"),
        Step::Reorder { nest, order } => write!(f, "reorder({nest}, {})", order.join(", ")),
        Step::Vectorize { nest, loop_name } => write!(f, "vectorize({nest}, {loop_name})"),
        Step::Unroll { nest, loop_name } => write!(f, "unroll({nest}, {loop_name})"),
    }
}

impl fmt::Display for ScheduleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "schedule text line {}, column {}: {}",
            self.line, self.column, self.msg
        )
    }
}

impl std::error::Error for ScheduleParseError {}

/// The step names, as listed in errors.
const STEPS: &str = "split, reorder, vectorize or unroll";

/// Cursor over the text. `pos` only moves over ASCII bytes, so it is always a char boundary.
struct Parser<'a> {
    text: &'a str,
    pos: usize,
}

impl Parser<'_> {
    fn error_at(&self, pos: usize, msg: String) -> ScheduleParseError {
        let before = &self.text[..pos];
        let line_start = before.rfind('\n').map_or(0, |i| i + 1);
        ScheduleParseError {
            pos,
            line: before.matches('\n').count() + 1,
            column: before[line_start..].chars().count() + 1,
            msg,
        }
    }

    /// `expected <what>, found <the token at pos>`.
    fn expected(&self, what: &str) -> ScheduleParseError {
        self.error_at(self.pos, format!("expected {what}, found {}", self.found()))
    }

    /// The token at `pos`: a quoted word (letters, digits and `_`) or character, or
    /// `end of text`.
    fn found(&self) -> String {
        let rest = &self.text[self.pos..];
        let word_len = rest
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        match rest.chars().next() {
            None => "end of text".to_string(),
            Some(_) if word_len > 0 => format!("'{}'", &rest[..word_len]),
            Some(c) => format!("'{c}'"),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Consumes the byte `c` after optional whitespace.
    fn expect(&mut self, c: u8) -> Result<(), ScheduleParseError> {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.expected(&format!("'{}'", char::from(c))))
        }
    }

    /// An identifier; `what` names it in the error.
    fn ident(&mut self, what: &str) -> Result<String, ScheduleParseError> {
        self.skip_ws();
        if !self
            .peek()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        {
            return Err(self.expected(what));
        }
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.pos += 1;
        }
        Ok(self.text[start..self.pos].to_string())
    }

    /// An identifier that is not one of `seen`.
    fn loop_name(&mut self, seen: &[&str]) -> Result<String, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        let name = self.ident("loop name")?;
        if seen.contains(&name.as_str()) {
            return Err(self.error_at(start, format!("repeated loop name '{name}'")));
        }
        Ok(name)
    }

    fn number(&mut self) -> Result<usize, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(self.expected("number"));
        }
        self.text[start..self.pos]
            .parse()
            .map_err(|_| self.error_at(start, format!("number too large (at most {})", usize::MAX)))
    }

    fn factor(&mut self) -> Result<usize, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        let factor = self.number()?;
        if factor == 0 {
            return Err(self.error_at(start, "factor must be at least 1".to_string()));
        }
        Ok(factor)
    }

    fn step(&mut self) -> Result<Step, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        match self.ident(&format!("step ({STEPS})"))?.as_str() {
            "split" => self.split(),
            "reorder" => self.reorder(),
            "vectorize" => self.vectorize(),
            "unroll" => self.unroll(),
            name => Err(self.error_at(start, format!("unknown step '{name}', expected {STEPS}"))),
        }
    }

    fn split(&mut self) -> Result<Step, ScheduleParseError> {
        self.expect(b'(')?;
        let nest = self.number()?;
        self.expect(b',')?;
        let loop_name = self.loop_name(&[])?;
        self.expect(b',')?;
        let outer = self.loop_name(&[loop_name.as_str()])?;
        self.expect(b',')?;
        let inner = self.loop_name(&[loop_name.as_str(), outer.as_str()])?;
        self.expect(b',')?;
        let factor = self.factor()?;
        self.expect(b')')?;
        Ok(Step::Split {
            nest,
            loop_name,
            outer,
            inner,
            factor,
        })
    }

    fn reorder(&mut self) -> Result<Step, ScheduleParseError> {
        self.expect(b'(')?;
        let nest = self.number()?;
        let mut order: Vec<String> = Vec::new();
        loop {
            self.expect(b',')?;
            let seen: Vec<&str> = order.iter().map(String::as_str).collect();
            let name = self.loop_name(&seen)?;
            order.push(name);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {}
                Some(b')') => break,
                _ => return Err(self.expected("',' or ')'")),
            }
        }
        self.expect(b')')?;
        Ok(Step::Reorder { nest, order })
    }

    fn vectorize(&mut self) -> Result<Step, ScheduleParseError> {
        self.expect(b'(')?;
        let nest = self.number()?;
        self.expect(b',')?;
        let loop_name = self.loop_name(&[])?;
        self.expect(b')')?;
        Ok(Step::Vectorize { nest, loop_name })
    }

    fn unroll(&mut self) -> Result<Step, ScheduleParseError> {
        self.expect(b'(')?;
        let nest = self.number()?;
        self.expect(b',')?;
        let loop_name = self.loop_name(&[])?;
        self.expect(b')')?;
        Ok(Step::Unroll { nest, loop_name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "split(1, k, ko, ki, 4); reorder(1, i, ko, j, ki); vectorize(1, j)";

    fn example() -> Schedule {
        Schedule {
            steps: vec![
                Step::Split {
                    nest: 1,
                    loop_name: "k".to_string(),
                    outer: "ko".to_string(),
                    inner: "ki".to_string(),
                    factor: 4,
                },
                Step::Reorder {
                    nest: 1,
                    order: ["i", "ko", "j", "ki"].map(String::from).to_vec(),
                },
                Step::Vectorize {
                    nest: 1,
                    loop_name: "j".to_string(),
                },
            ],
        }
    }

    fn parse_error(text: &str) -> ScheduleParseError {
        Schedule::parse(text).expect_err(text)
    }

    /// Asserts the error of `text`: its byte offset, its column on line 1 and its `Display`.
    #[track_caller]
    fn assert_error(text: &str, pos: usize, column: usize, msg: &str) {
        let e = parse_error(text);
        assert_eq!(
            (e.pos, e.line, e.column, e.msg.as_str()),
            (pos, 1, column, msg),
            "{text}"
        );
        assert_eq!(
            e.to_string(),
            format!("schedule text line 1, column {column}: {msg}")
        );
    }

    #[test]
    fn example_parses_and_prints_back() {
        let s = Schedule::parse(EXAMPLE).expect("example is valid");
        assert_eq!(s, example());
        assert_eq!(s.to_string(), EXAMPLE);
    }

    #[test]
    fn unroll_parses_and_prints_back() {
        let s = Schedule::parse("unroll( 1,ii )").expect("valid");
        assert_eq!(
            s.steps,
            [Step::Unroll {
                nest: 1,
                loop_name: "ii".to_string()
            }]
        );
        assert_eq!(s.to_string(), "unroll(1, ii)");
        assert_eq!(Schedule::parse(&s.to_string()), Ok(s));
    }

    #[test]
    fn empty_and_whitespace_only_are_empty_schedules() {
        for text in ["", "  \n"] {
            let s = Schedule::parse(text).expect("empty text is valid");
            assert_eq!(s, Schedule::default());
            assert_eq!(s.to_string(), "");
        }
    }

    #[test]
    fn spacing_is_free_and_prints_canonically() {
        let s = Schedule::parse(" split ( 1 ,k,ko , ki,4 ) ;vectorize(1,j) ").expect("valid");
        assert_eq!(s.to_string(), "split(1, k, ko, ki, 4); vectorize(1, j)");
        let canonical = Schedule::parse("split(1, k, ko, ki, 4); vectorize(1, j)").expect("valid");
        assert_eq!(s, canonical);
        let tabs_and_newlines = Schedule::parse("\tsplit(1,k,ko,ki,4)\n;\nvectorize(1,j)\n");
        assert_eq!(tabs_and_newlines, Ok(canonical));
    }

    #[test]
    fn factor_one_is_accepted() {
        let s = Schedule::parse("split(0, a, b, c, 1)").expect("factor 1 is valid");
        assert_eq!(s.to_string(), "split(0, a, b, c, 1)");
    }

    #[test]
    fn nest_number_is_limited_to_usize() {
        let max = usize::MAX.to_string();
        let s = Schedule::parse(&format!("vectorize({max}, j)")).expect("usize::MAX is valid");
        assert_eq!(s.to_string(), format!("vectorize({max}, j)"));
        let too_big = (usize::MAX as u128 + 1).to_string();
        assert_error(
            &format!("vectorize({too_big}, j)"),
            10,
            11,
            &format!("number too large (at most {max})"),
        );
    }

    #[test]
    fn zero_factor_is_rejected() {
        // `split(1, k, ko, ki, ` is 20 bytes, so the factor `0` is at byte 20, column 21.
        assert_error(
            "split(1, k, ko, ki, 0)",
            20,
            21,
            "factor must be at least 1",
        );
    }

    #[test]
    fn unknown_step_is_rejected() {
        // `splt` starts at byte 0.
        assert_error(
            "splt(1, k)",
            0,
            1,
            "unknown step 'splt', expected split, reorder, vectorize or unroll",
        );
        // Step names are case-sensitive.
        assert_error(
            "Vectorize(1, j)",
            0,
            1,
            "unknown step 'Vectorize', expected split, reorder, vectorize or unroll",
        );
    }

    #[test]
    fn missing_close_paren_is_rejected_at_end() {
        // `vectorize(1, j` is 14 bytes; the error is at end of input.
        assert_error("vectorize(1, j", 14, 15, "expected ')', found end of text");
    }

    #[test]
    fn trailing_semicolon_is_rejected_at_end() {
        // `vectorize(1, j);` is 16 bytes; the step expected after `;` is at end of input.
        assert_error(
            "vectorize(1, j);",
            16,
            17,
            "expected step (split, reorder, vectorize or unroll), found end of text",
        );
    }

    #[test]
    fn reorder_without_loops_is_rejected() {
        // `reorder(1` is 9 bytes; the `)` at byte 9 is where a `,` is expected.
        assert_error("reorder(1)", 9, 10, "expected ',', found ')'");
    }

    #[test]
    fn repeated_reorder_loop_is_rejected() {
        // `reorder(1, i, ` is 14 bytes; the second `i` is at byte 14.
        assert_error("reorder(1, i, i)", 14, 15, "repeated loop name 'i'");
    }

    #[test]
    fn split_names_must_differ() {
        // `split(1, k, ` is 12 bytes; `outer` = `k` repeats `loop_name` at byte 12.
        assert_error("split(1, k, k, ki, 4)", 12, 13, "repeated loop name 'k'");
    }

    #[test]
    fn identifier_cannot_start_with_digit() {
        // `vectorize(1, ` is 13 bytes; `1j` starts at byte 13.
        assert_error("vectorize(1, 1j)", 13, 14, "expected loop name, found '1j'");
    }

    #[test]
    fn number_overflow_is_rejected() {
        // `vectorize(` is 10 bytes; the nest number starts at byte 10.
        assert_error(
            "vectorize(99999999999999999999, j)",
            10,
            11,
            &format!("number too large (at most {})", usize::MAX),
        );
    }

    #[test]
    fn typical_typos_name_the_expected_token() {
        let cases = [
            // A missing comma between arguments.
            ("split(1 k, ko, ki, 4)", 8, "expected ',', found 'k'"),
            // `:` or `,` instead of `;` between steps.
            (
                "vectorize(1, j): unroll(1, i)",
                15,
                "expected ';' or end of text, found ':'",
            ),
            (
                "vectorize(1, j), unroll(1, i)",
                15,
                "expected ';' or end of text, found ','",
            ),
            // Two steps without a separator.
            (
                "vectorize(1, j) unroll(1, i)",
                16,
                "expected ';' or end of text, found 'unroll'",
            ),
            // A doubled `)`.
            (
                "vectorize(1, j))",
                15,
                "expected ';' or end of text, found ')'",
            ),
            // A missing `(`.
            ("vectorize 1, j)", 10, "expected '(', found '1'"),
            // A missing `)` before the next step.
            (
                "vectorize(1, j; unroll(1, i)",
                14,
                "expected ')', found ';'",
            ),
            // An empty argument.
            ("split(1, , ko, ki, 4)", 9, "expected loop name, found ','"),
            // A loop name where the nest number goes.
            ("vectorize(j)", 10, "expected number, found 'j'"),
            // A loop name where the factor goes.
            ("split(1, k, ko, ki, f)", 20, "expected number, found 'f'"),
            // A missing factor.
            ("split(1, k, ko, ki)", 18, "expected ',', found ')'"),
            // Too many arguments.
            ("vectorize(1, j, i)", 14, "expected ')', found ','"),
            // A reorder list cut off by the next step.
            (
                "reorder(1, i, j; vectorize(1, j)",
                15,
                "expected ',' or ')', found ';'",
            ),
            ("reorder(1, i j)", 13, "expected ',' or ')', found 'j'"),
            // Something that is not a word or punctuation.
            ("vectorize(1, -j)", 13, "expected loop name, found '-'"),
            (
                "+",
                0,
                "expected step (split, reorder, vectorize or unroll), found '+'",
            ),
        ];
        for (text, pos, msg) in cases {
            assert_error(text, pos, pos + 1, msg);
        }
    }

    #[test]
    fn line_and_column_count_lines_and_characters() {
        // The error is at the end of line 3: `\t` is one column, then 11 more.
        let e = parse_error("split(1, k, ko, ki, 4);\nvectorize(1, j);\n\tunroll(1, x");
        assert_eq!((e.pos, e.line, e.column), (53, 3, 13));
        assert_eq!(
            e.to_string(),
            "schedule text line 3, column 13: expected ')', found end of text"
        );
        // A non-ASCII character is one column however many bytes it has.
        let e = parse_error("vectorize(1, é)");
        assert_eq!((e.pos, e.line, e.column), (13, 1, 14));
        assert_eq!(e.msg, "expected loop name, found 'é'");
        let e = parse_error("vectorize(1, é é)");
        assert_eq!((e.pos, e.line, e.column), (13, 1, 14));
        // `\r\n` line ends: the `\r` is whitespace before the newline, not a column.
        let e = parse_error("vectorize(1, j);\r\nunrol(1, i)");
        assert_eq!((e.pos, e.line, e.column), (18, 2, 1));
        assert_eq!(
            e.msg,
            "unknown step 'unrol', expected split, reorder, vectorize or unroll"
        );
    }
}
