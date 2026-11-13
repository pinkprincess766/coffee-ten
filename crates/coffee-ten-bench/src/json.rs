//! Small strict JSON reader for replay records. Integer tokens retain all 128 bits.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Value {
    pub fn get(&self, key: &str) -> Result<&Self, String> {
        match self {
            Self::Object(fields) => fields.get(key).ok_or_else(|| format!("missing {key}")),
            _ => Err("expected object".into()),
        }
    }
    pub fn text(&self) -> Result<&str, String> {
        if let Self::String(s) = self {
            Ok(s)
        } else {
            Err("expected string".into())
        }
    }
    pub fn uint<T: std::str::FromStr>(&self) -> Result<T, String> {
        if let Self::Number(s) = self
            && s.bytes().all(|b| b.is_ascii_digit())
        {
            return s.parse().map_err(|_| "integer out of range".into());
        }
        Err("expected unsigned integer".into())
    }
    pub fn number(&self) -> Result<f64, String> {
        if let Self::Number(s) = self {
            let n: f64 = s.parse().map_err(|_| "invalid number")?;
            if n.is_finite() {
                return Ok(n);
            }
        }
        Err("expected finite number".into())
    }
    pub fn boolean(&self) -> Result<bool, String> {
        if let Self::Bool(b) = self {
            Ok(*b)
        } else {
            Err("expected boolean".into())
        }
    }
    pub fn array(&self) -> Result<&[Self], String> {
        if let Self::Array(a) = self {
            Ok(a)
        } else {
            Err("expected array".into())
        }
    }
}

pub(crate) fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { text, pos: 0 };
    let result = p.value(0)?;
    p.ws();
    if p.pos != text.len() {
        return Err("trailing JSON data".into());
    }
    Ok(result)
}
struct Parser<'a> {
    text: &'a str,
    pos: usize,
}
impl Parser<'_> {
    fn ws(&mut self) {
        while self
            .peek()
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }
    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.take(byte) {
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", byte as char, self.pos))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 32 {
            return Err("JSON nesting limit".into());
        }
        self.ws();
        Ok(match self.peek() {
            Some(b'"') => Value::String(self.string()?),
            Some(b'{') => {
                self.pos += 1;
                self.ws();
                let mut fields = BTreeMap::new();
                if !self.take(b'}') {
                    loop {
                        self.ws();
                        let key = self.string()?;
                        self.ws();
                        self.expect(b':')?;
                        let value = self.value(depth + 1)?;
                        if fields.insert(key, value).is_some() {
                            return Err("duplicate JSON key".into());
                        }
                        self.ws();
                        if self.take(b'}') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Value::Object(fields)
            }
            Some(b'[') => {
                self.pos += 1;
                self.ws();
                let mut items = Vec::new();
                if !self.take(b']') {
                    loop {
                        items.push(self.value(depth + 1)?);
                        self.ws();
                        if self.take(b']') {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
                Value::Array(items)
            }
            Some(b'-' | b'0'..=b'9') => Value::Number(self.number()?),
            _ => {
                let mut found = None;
                for (word, value) in [
                    ("null", Value::Null),
                    ("true", Value::Bool(true)),
                    ("false", Value::Bool(false)),
                ] {
                    if self.text[self.pos..].starts_with(word) {
                        self.pos += word.len();
                        found = Some(value);
                        break;
                    }
                }
                found.ok_or_else(|| format!("invalid JSON value at byte {}", self.pos))?
            }
        })
    }
    fn digits(&mut self) -> Result<(), String> {
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            Err("expected digit".into())
        } else {
            Ok(())
        }
    }
    fn number(&mut self) -> Result<String, String> {
        let start = self.pos;
        self.take(b'-');
        if !self.take(b'0') {
            self.digits()?;
        }
        if self.take(b'.') {
            self.digits()?;
        }
        if self.take(b'e') || self.take(b'E') {
            if !self.take(b'+') {
                self.take(b'-');
            }
            self.digits()?;
        }
        Ok(self.text[start..self.pos].into())
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let bytes = self
            .text
            .as_bytes()
            .get(self.pos..self.pos + 4)
            .ok_or("short unicode escape")?;
        let mut value = 0;
        for b in bytes {
            value = value * 16 + (*b as char).to_digit(16).ok_or("bad unicode escape")?;
        }
        self.pos += 4;
        Ok(value)
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.text[self.pos..]
                .chars()
                .next()
                .ok_or("unterminated string")?;
            self.pos += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let escape = self.peek().ok_or("short escape")?;
                    self.pos += 1;
                    out.push(match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xd800..=0xdbff).contains(&code) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let low = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err("bad surrogate pair".into());
                                }
                                code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                            }
                            char::from_u32(code).ok_or("invalid unicode scalar")?
                        }
                        _ => return Err("invalid string escape".into()),
                    });
                }
                c if c < ' ' => return Err("unescaped control character".into()),
                _ => out.push(c),
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_integer_bits_and_decodes_strings() {
        let v =
            parse(r#"{"n":340282366920938463463374607431768211455,"s":"Привет\n\uD83D\uDC08"}"#)
                .unwrap();
        assert_eq!(v.get("n").unwrap().uint::<u128>().unwrap(), u128::MAX);
        assert_eq!(v.get("s").unwrap().text().unwrap(), "Привет\n🐈");
    }
    #[test]
    fn malformed_json_is_rejected() {
        for s in [
            "01",
            "1.",
            "1e",
            "+1",
            "[1,]",
            "{\"x\":1,}",
            "{\"x\":1,\"x\":2}",
            "true false",
            r#""\uD800""#,
            r#""\q""#,
            "\"\n\"",
            "NaN",
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
        assert!(parse(&"[".repeat(34)).is_err());
        assert!(parse("1e999").unwrap().number().is_err());
    }
    #[test]
    fn nesting_limit_is_exactly_33_levels() {
        // Depth 0 is the outer value; depths 0..=32 are allowed.
        let ok = format!("{}{}", "[".repeat(33), "]".repeat(33));
        assert!(parse(&ok).is_ok());
        let deep = format!("{}{}", "[".repeat(34), "]".repeat(34));
        assert_eq!(parse(&deep).unwrap_err(), "JSON nesting limit");
    }
    #[test]
    fn unicode_escapes_must_be_complete_scalar_values() {
        for s in [
            r#""\uDC00""#,       // lone low surrogate
            r#""\uD800\u0041""#, // high surrogate followed by a non-surrogate
            r#""\uD800x""#,      // high surrogate without a second escape
            r#""\u12""#,         // short escape
            r#""\u12G4""#,       // non-hex digit
            r#""\u00é0""#,       // non-ASCII inside the escape
            "\"abc",             // unterminated
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
        assert_eq!(parse(r#""\u00e9\/""#).unwrap(), Value::String("é/".into()));
    }
    #[test]
    fn unsigned_integers_are_digits_only_and_range_checked() {
        let n = |s: &str| parse(s).unwrap();
        assert_eq!(n("18446744073709551615").uint::<u64>(), Ok(u64::MAX));
        assert_eq!(
            n("18446744073709551616").uint::<u64>(),
            Err("integer out of range".into())
        );
        for s in ["-1", "-0", "1.0", "1e3", "0.5"] {
            assert!(n(s).uint::<u64>().is_err(), "{s}");
            assert!(n(s).number().is_ok(), "{s}");
        }
        assert_eq!(n("0").uint::<u64>(), Ok(0));
        assert!(n("\"1\"").uint::<u64>().is_err());
        assert!(n("true").number().is_err());
    }
    #[test]
    fn whitespace_empty_input_and_keys() {
        assert_eq!(
            parse(" \t\r\n{ } \n").unwrap(),
            Value::Object(BTreeMap::new())
        );
        for s in ["", "   ", "{1:2}", "{\"a\" 1}", "{\"a\":}", "nul", "tru"] {
            assert!(parse(s).is_err(), "{s:?}");
        }
        // Keys are compared after decoding, so an escaped duplicate is still a duplicate.
        assert_eq!(
            parse(r#"{"a":1,"\u0061":2}"#).unwrap_err(),
            "duplicate JSON key"
        );
    }
}
