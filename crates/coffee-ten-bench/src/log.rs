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

    #[test]
    fn a_directory_path_is_an_error_and_the_directory_is_unchanged() {
        let dir = temp_log("dir");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("inside.jsonl"), "kept\n").unwrap();
        for lines in [vec![], vec!["line".to_string()]] {
            append_lines(&dir, &lines).unwrap_err();
        }
        let entries: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, ["inside.jsonl"]);
        assert_eq!(
            fs::read_to_string(dir.join("inside.jsonl")).unwrap(),
            "kept\n"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_parent_directory_is_an_error_and_creates_nothing() {
        let dir = temp_log("no-parent");
        let _ = fs::remove_dir_all(&dir);
        let err = append_lines(&dir.join("log.jsonl"), &["line".to_string()]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(!dir.exists());
    }

    /// Permission bits do not bind a privileged user (root in a container): `None` there, so the
    /// tests below skip rather than assert something the OS does not enforce.
    #[cfg(unix)]
    fn with_mode(name: &str, original: &str, mode: u32) -> Option<PathBuf> {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_log(name);
        fs::write(&path, original).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        let readable = fs::File::open(&path).is_ok();
        let writable = OpenOptions::new().append(true).open(&path).is_ok();
        if readable == (mode & 0o400 != 0) && writable == (mode & 0o200 != 0) {
            Some(path)
        } else {
            eprintln!("skipped: permission bits are not enforced for this user");
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
            let _ = fs::remove_file(path);
            None
        }
    }

    #[cfg(unix)]
    fn cleanup(path: PathBuf) -> String {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();
        text
    }

    #[cfg(unix)]
    #[test]
    fn write_only_file_rejects_a_nonempty_append_without_changing_it() {
        for original in ["", "unterminated", "terminated\n"] {
            let Some(path) = with_mode("write-only", original, 0o200) else {
                return;
            };
            let err = append_lines(&path, &["new".to_string()]).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
            assert_eq!(cleanup(path), original);
        }
    }

    #[cfg(unix)]
    #[test]
    fn write_only_file_accepts_an_empty_append_without_reading_it() {
        let Some(path) = with_mode("write-only-empty", "unterminated", 0o200) else {
            return;
        };
        append_lines(&path, &[]).unwrap();
        assert_eq!(cleanup(path), "unterminated");
    }

    #[cfg(unix)]
    #[test]
    fn read_only_file_rejects_every_append_without_changing_it() {
        for lines in [vec![], vec!["new".to_string()]] {
            let Some(path) = with_mode("read-only", "old\n", 0o400) else {
                return;
            };
            let err = append_lines(&path, &lines).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
            assert_eq!(cleanup(path), "old\n");
        }
    }
}
