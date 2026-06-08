// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by providers that consume SSE (Server-Sent Events) streams.

use crate::error::ProviderError;

/// Locate the end (exclusive) of the next SSE event in `buffer`.
///
/// SSE events are terminated by `\n\n` (or `\r\n\r\n`). Returns `None` if no
/// boundary has been received yet, in which case the caller should keep
/// reading bytes into the buffer and try again.
pub(crate) fn find_event_end(buffer: &[u8]) -> Option<usize> {
    let lf = buffer.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = buffer
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Drain one SSE event from `buffer` and return its text representation.
///
/// Strips trailing `\r`/`\n` and validates UTF-8. Returns `None` when no
/// complete event is available (caller should read more bytes and retry).
pub(crate) fn drain_event(buffer: &mut Vec<u8>) -> Option<Result<String, ProviderError>> {
    let end = find_event_end(buffer)?;
    let event_bytes: Vec<u8> = buffer.drain(..end).collect();
    let trim_len = event_bytes
        .iter()
        .rev()
        .take_while(|&&b| b == b'\r' || b == b'\n')
        .count();
    match std::str::from_utf8(&event_bytes[..event_bytes.len() - trim_len]) {
        Ok(t) => Some(Ok(t.to_owned())),
        Err(e) => Some(Err(ProviderError::Decode(serde::de::Error::custom(
            format!("non-utf8 SSE event: {e}"),
        )))),
    }
}

/// Extract the `data:` payload strings from an SSE event text.
///
/// Returns an iterator of trimmed data strings (empty lines skipped).
/// Multiple `data:` lines in a single event are yielded separately.
pub(crate) fn data_lines(event_text: &str) -> impl Iterator<Item = &str> {
    event_text.lines().filter_map(|line| {
        line.strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
            .map(|d| d.trim_start())
            .filter(|d| !d.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_when_no_boundary() {
        assert_eq!(find_event_end(b""), None);
        assert_eq!(find_event_end(b"data: partial"), None);
    }

    #[test]
    fn finds_lf_lf_boundary() {
        assert_eq!(find_event_end(b"data: x\n\nrest"), Some(9));
    }

    #[test]
    fn finds_crlf_crlf_boundary() {
        assert_eq!(find_event_end(b"data: x\r\n\r\nrest"), Some(11));
    }

    #[test]
    fn picks_earliest_when_both_present() {
        let buf = b"data: x\n\ndata: y\r\n\r\n";
        assert_eq!(find_event_end(buf), Some(9));
    }

    #[test]
    fn drain_event_returns_none_when_incomplete() {
        let mut buf = b"data: partial".to_vec();
        assert!(drain_event(&mut buf).is_none());
    }

    #[test]
    fn drain_event_extracts_text() {
        let mut buf = b"data: hello\n\n".to_vec();
        let result = drain_event(&mut buf).unwrap().unwrap();
        assert_eq!(result, "data: hello");
        assert!(buf.is_empty());
    }

    #[test]
    fn data_lines_extracts_payload() {
        let event = "data: {\"foo\":1}\n\n";
        let lines: Vec<_> = data_lines(event).collect();
        assert_eq!(lines, vec![r#"{"foo":1}"#]);
    }

    #[test]
    fn data_lines_skips_empty() {
        let event = "data: \ndata: hello\n";
        let lines: Vec<_> = data_lines(event).collect();
        assert_eq!(lines, vec!["hello"]);
    }

    #[test]
    fn data_lines_handles_no_prefix() {
        let event = ": comment\nevent: ping\n";
        let lines: Vec<_> = data_lines(event).collect();
        assert!(lines.is_empty());
    }
}
