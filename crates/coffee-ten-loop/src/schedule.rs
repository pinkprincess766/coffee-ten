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

/// `pos` is the byte offset in the input text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleParseError {
    pub pos: usize,
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
            return Err(error_at(p.pos, "expected ';' or end of text"));
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
        write!(f, "schedule text at byte {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ScheduleParseError {}

fn error_at(pos: usize, msg: &str) -> ScheduleParseError {
    ScheduleParseError {
        pos,
        msg: msg.to_string(),
    }
}

/// Cursor over the text. `pos` only moves over ASCII bytes, so it is always a char boundary.
struct Parser<'a> {
    text: &'a str,
    pos: usize,
}

impl Parser<'_> {
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
            Err(error_at(self.pos, &format!("expected '{}'", char::from(c))))
        }
    }

    fn ident(&mut self) -> Result<String, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        if !self
            .peek()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        {
            return Err(error_at(start, "expected identifier"));
        }
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
        let name = self.ident()?;
        if seen.contains(&name.as_str()) {
            return Err(error_at(start, "repeated loop name"));
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
            return Err(error_at(start, "expected number"));
        }
        self.text[start..self.pos]
            .parse()
            .map_err(|_| error_at(start, "number too large"))
    }

    fn factor(&mut self) -> Result<usize, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        let factor = self.number()?;
        if factor == 0 {
            return Err(error_at(start, "factor must be at least 1"));
        }
        Ok(factor)
    }

    fn step(&mut self) -> Result<Step, ScheduleParseError> {
        self.skip_ws();
        let start = self.pos;
        match self.ident()?.as_str() {
            "split" => self.split(),
            "reorder" => self.reorder(),
            "vectorize" => self.vectorize(),
            "unroll" => self.unroll(),
            _ => Err(error_at(start, "unknown step")),
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
            if self.peek() != Some(b',') {
                break;
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
        assert_eq!(
            Schedule::parse(&format!("vectorize({too_big}, j)")).expect_err("one past usize::MAX"),
            ScheduleParseError {
                pos: 10,
                msg: "number too large".to_string(),
            }
        );
    }

    #[test]
    fn zero_factor_is_rejected() {
        // `split(1, k, ko, ki, ` is 20 bytes, so the factor `0` is at byte 20.
        assert_eq!(
            parse_error("split(1, k, ko, ki, 0)"),
            ScheduleParseError {
                pos: 20,
                msg: "factor must be at least 1".to_string(),
            }
        );
    }

    #[test]
    fn unknown_step_is_rejected() {
        // `splt` starts at byte 0.
        assert_eq!(
            parse_error("splt(1, k)"),
            ScheduleParseError {
                pos: 0,
                msg: "unknown step".to_string(),
            }
        );
    }

    #[test]
    fn missing_close_paren_is_rejected_at_end() {
        // `vectorize(1, j` is 14 bytes; the error is at end of input.
        assert_eq!(
            parse_error("vectorize(1, j"),
            ScheduleParseError {
                pos: 14,
                msg: "expected ')'".to_string(),
            }
        );
    }

    #[test]
    fn trailing_semicolon_is_rejected_at_end() {
        // `vectorize(1, j);` is 16 bytes; the step expected after `;` is at end of input.
        assert_eq!(
            parse_error("vectorize(1, j);"),
            ScheduleParseError {
                pos: 16,
                msg: "expected identifier".to_string(),
            }
        );
    }

    #[test]
    fn reorder_without_loops_is_rejected() {
        // `reorder(1` is 9 bytes; the `)` at byte 9 is where a `,` is expected.
        assert_eq!(
            parse_error("reorder(1)"),
            ScheduleParseError {
                pos: 9,
                msg: "expected ','".to_string(),
            }
        );
    }

    #[test]
    fn repeated_reorder_loop_is_rejected() {
        // `reorder(1, i, ` is 14 bytes; the second `i` is at byte 14.
        assert_eq!(
            parse_error("reorder(1, i, i)"),
            ScheduleParseError {
                pos: 14,
                msg: "repeated loop name".to_string(),
            }
        );
    }

    #[test]
    fn split_names_must_differ() {
        // `split(1, k, ` is 12 bytes; `outer` = `k` repeats `loop_name` at byte 12.
        assert_eq!(
            parse_error("split(1, k, k, ki, 4)"),
            ScheduleParseError {
                pos: 12,
                msg: "repeated loop name".to_string(),
            }
        );
    }

    #[test]
    fn identifier_cannot_start_with_digit() {
        // `vectorize(1, ` is 13 bytes; `1j` starts at byte 13.
        assert_eq!(
            parse_error("vectorize(1, 1j)"),
            ScheduleParseError {
                pos: 13,
                msg: "expected identifier".to_string(),
            }
        );
    }

    #[test]
    fn number_overflow_is_rejected() {
        // `vectorize(` is 10 bytes; the nest number starts at byte 10.
        assert_eq!(
            parse_error("vectorize(99999999999999999999, j)"),
            ScheduleParseError {
                pos: 10,
                msg: "number too large".to_string(),
            }
        );
    }
}
