//! The JSONL log: lines are appended to the file, never truncated.

use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Appends each line, followed by a newline, to `path`. Creates the file if it is missing.
/// If existing data have no final newline, complete that line before appending more.
/// Nonempty appends require read access to inspect that boundary as well as append access.
pub fn append_lines(path: &Path, lines: &[String]) -> io::Result<()> {
    let mut text = String::new();
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(!text.is_empty())
        .open(path)?;
    if !text.is_empty() && file.metadata()?.len() != 0 {
        file.seek(SeekFrom::End(-1))?;
        let mut last = [0];
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            text.insert(0, '\n');
        }
    }
    file.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_log(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "coffee-ten-bench-log-{name}-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        path
    }

    #[test]
    fn creates_a_missing_file_with_one_line_per_record() {
        let path = temp_log("creates");
        append_lines(&path, &["first".to_string(), "second".to_string()]).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(text, "first\nsecond\n");
    }

    #[test]
    fn appending_completes_an_unterminated_json_record() {
        let path = temp_log("unterminated");
        fs::write(&path, "{\"old\":1}").unwrap();
        append_lines(&path, &["{\"new\":2}".to_string()]).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(text, "{\"old\":1}\n{\"new\":2}\n");
    }

    #[test]
    fn appending_no_lines_does_not_change_existing_bytes() {
        for original in ["", "unterminated", "terminated\n"] {
            let path = temp_log("no-lines");
            fs::write(&path, original).unwrap();
            append_lines(&path, &[]).unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), original);
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn appending_keeps_the_existing_lines() {
        let path = temp_log("keeps");
        fs::write(&path, "old\n").unwrap();
        append_lines(&path, &["a".to_string(), "b".to_string()]).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(text, "old\na\nb\n");
    }
}
