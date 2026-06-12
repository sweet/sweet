// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;

use sweet_core::{MemoryItem, Message, Session, SessionError, SessionId};

/// SQLite-backed session.
///
/// Maintains an in-memory cache of the *live* transcript alongside the
/// database for fast reads. By default opens an in-memory transient store
/// (`:memory:`); pass a file path to `open()` for persistence.
///
/// Each database file stores a single session. Opening an existing file loads
/// the most recent session's live items.
///
/// # Archived rows
///
/// [`replace_range`](Session::replace_range) (compaction) does not delete the
/// rows it replaces — it marks them `archived`. Archived rows are invisible
/// to the [`Session`] trait (`items`, `messages`, `token_count`,
/// `total_tokens`) but are retained on disk and readable via
/// [`full_messages`](Self::full_messages) / [`full_items`](Self::full_items),
/// so a UI can show the complete history of a heavily compacted session
/// without a second store. [`clear`](Session::clear) is a full teardown and
/// deletes archived rows too.
///
/// Ordering uses a `position REAL` column instead of rowid: replacements are
/// inserted at fractional positions bisected into the gap after the span they
/// replace, which keeps both the live order and the full-transcript order
/// stable across reopen without rewriting rows. Each compaction bisects a
/// given gap at most once per replacement item, so position precision is
/// nowhere near f64 limits in practice.
pub struct SqliteSession {
    id: SessionId,
    conn: Mutex<Connection>,
    cache: Vec<MemoryItem>,
    /// Live rows' positions, parallel to `cache`.
    positions: Vec<f64>,
    /// Highest position ever used in this session (including archived rows),
    /// so appends never collide with an archived tail.
    max_position: f64,
}

impl SqliteSession {
    /// Create a new transient in-memory session.
    pub fn new() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Open (or create) a session stored at `path`.
    ///
    /// If the database already contains a session, the most recent one is loaded
    /// into the cache. Otherwise a new session row is created.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, rusqlite::Error> {
        Self::open_with_id(path, SessionId::new())
    }

    /// Open (or create) a session stored at `path`, using `id` when a new
    /// session row must be inserted.
    ///
    /// If the database already contains a session row, that row's id is used
    /// regardless of the supplied `id`.
    pub fn open_with_id(
        path: impl AsRef<std::path::Path>,
        id: SessionId,
    ) -> Result<Self, rusqlite::Error> {
        let mut conn = Connection::open(path)?;
        Self::init_schema(&conn)?;

        let tx = conn.transaction()?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;

        let id = match existing {
            Some(id_str) => id_str.parse::<SessionId>().map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?,
            None => {
                tx.execute(
                    "INSERT INTO sessions (id, created_at) VALUES (?1, datetime('now'))",
                    params![id.to_string()],
                )?;
                id
            }
        };
        tx.commit()?;

        let mut cache = Vec::new();
        let mut positions = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT kind, data, position FROM items
                 WHERE session_id = ?1 AND archived = 0 ORDER BY position",
            )?;
            let rows = stmt.query_map(params![id.to_string()], |row| {
                let kind: String = row.get(0)?;
                let data: String = row.get(1)?;
                let position: f64 = row.get(2)?;
                Ok((kind, data, position))
            })?;

            for row in rows {
                let (kind, data, position) = row?;
                if kind == "message" {
                    let msg: Message = serde_json::from_str(&data).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            0,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;
                    cache.push(MemoryItem::Message(msg));
                    positions.push(position);
                }
            }
        }

        let max_position: f64 = conn.query_row(
            "SELECT COALESCE(MAX(position), 0) FROM items WHERE session_id = ?1",
            params![id.to_string()],
            |row| row.get(0),
        )?;

        Ok(Self {
            id,
            conn: Mutex::new(conn),
            cache,
            positions,
            max_position,
        })
    }

    fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items (
                rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                kind       TEXT NOT NULL,
                data       TEXT NOT NULL,
                tokens     INTEGER,
                metadata   TEXT,
                archived   INTEGER NOT NULL DEFAULT 0,
                position   REAL NOT NULL
            )",
            [],
        )?;
        Ok(())
    }

    /// The full transcript: every item, archived and live, in transcript
    /// order. Archived originals appear in place; compaction summaries (live,
    /// `compacted: true`) appear immediately after the span they replaced.
    pub fn full_messages(&self) -> sweet_core::error::Result<Vec<Message>> {
        Ok(self
            .full_items()?
            .into_iter()
            .map(|(item, _)| match item {
                MemoryItem::Message(msg) => msg,
            })
            .collect())
    }

    /// Like [`full_messages`](Self::full_messages), but yields each item with
    /// its archived flag, for UIs that want to style replaced history.
    pub fn full_items(&self) -> sweet_core::error::Result<Vec<(MemoryItem, bool)>> {
        let conn = self
            .conn
            .lock()
            .expect("sqlite connection mutex not poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT kind, data, archived FROM items
                 WHERE session_id = ?1 ORDER BY position",
            )
            .map_err(SessionError::storage)?;
        let rows = stmt
            .query_map(params![self.id.to_string()], |row| {
                let kind: String = row.get(0)?;
                let data: String = row.get(1)?;
                let archived: bool = row.get(2)?;
                Ok((kind, data, archived))
            })
            .map_err(SessionError::storage)?;

        let mut items = Vec::new();
        for row in rows {
            let (kind, data, archived) = row.map_err(SessionError::storage)?;
            if kind == "message" {
                let msg: Message = serde_json::from_str(&data).map_err(SessionError::storage)?;
                items.push((MemoryItem::Message(msg), archived));
            }
        }
        Ok(items)
    }
}

impl Session for SqliteSession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn push(&mut self, item: MemoryItem) -> sweet_core::error::Result<()> {
        let position = self.max_position + 1.0;
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            match &item {
                MemoryItem::Message(msg) => {
                    let data = serde_json::to_string(msg).map_err(SessionError::storage)?;
                    conn.execute(
                        "INSERT INTO items (session_id, kind, data, tokens, position)
                         VALUES (?1, 'message', ?2, ?3, ?4)",
                        params![
                            self.id.to_string(),
                            data,
                            msg.token_count.map(|t| t as i64),
                            position
                        ],
                    )
                    .map_err(SessionError::storage)?;
                }
            }
        }
        self.cache.push(item);
        self.positions.push(position);
        self.max_position = position;
        Ok(())
    }

    fn items(&self) -> &[MemoryItem] {
        &self.cache
    }

    fn messages(&self) -> Vec<Message> {
        self.cache
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.clone(),
            })
            .collect()
    }

    fn clear(&mut self) -> sweet_core::error::Result<()> {
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            // Full teardown: archived history goes too.
            conn.execute(
                "DELETE FROM items WHERE session_id = ?1",
                params![self.id.to_string()],
            )
            .map_err(SessionError::storage)?;
        }
        self.cache.clear();
        self.positions.clear();
        self.max_position = 0.0;
        Ok(())
    }

    fn token_count(&self) -> usize {
        self.cache
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.text_content().chars().count() / 4,
            })
            .sum()
    }

    fn total_tokens(&self) -> usize {
        {
            let conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            let total: i64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(tokens), 0) FROM items
                     WHERE session_id = ?1 AND archived = 0",
                    params![self.id.to_string()],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            total as usize
        }
    }

    fn context_size(&self) -> usize {
        sweet_core::last_context_size(&self.cache).unwrap_or_else(|| self.token_count())
    }

    fn replace_range(
        &mut self,
        range: std::ops::Range<usize>,
        replacement: Vec<MemoryItem>,
    ) -> sweet_core::error::Result<()> {
        // Archive the replaced rows in place and insert the replacements at
        // fractional positions bisected into the gap between the end of the
        // replaced span and the next live row. The originals keep their
        // positions, so the full transcript stays ordered with the summary
        // directly after the span it replaced.
        let lower = if range.end > range.start {
            self.positions[range.end - 1]
        } else if range.start > 0 {
            self.positions[range.start - 1]
        } else {
            0.0
        };
        let upper = self.positions.get(range.end).copied();

        let new_positions: Vec<f64> = match upper {
            Some(upper) => {
                let step = (upper - lower) / (replacement.len() as f64 + 1.0);
                (1..=replacement.len())
                    .map(|k| lower + step * k as f64)
                    .collect()
            }
            // No live successor: extend the tail in whole steps.
            None => (1..=replacement.len()).map(|k| lower + k as f64).collect(),
        };

        {
            let mut conn = self
                .conn
                .lock()
                .expect("sqlite connection mutex not poisoned");
            let tx = conn.transaction().map_err(SessionError::storage)?;

            for position in &self.positions[range.clone()] {
                // Exact float equality is sound here: positions are written
                // as SQLite REAL (8-byte IEEE), so the f64s in `positions`
                // round-trip bit-for-bit through the database.
                tx.execute(
                    "UPDATE items SET archived = 1
                     WHERE session_id = ?1 AND archived = 0 AND position = ?2",
                    params![self.id.to_string(), position],
                )
                .map_err(SessionError::storage)?;
            }
            for (item, position) in replacement.iter().zip(&new_positions) {
                insert_item(&tx, &self.id, item, *position)?;
            }

            tx.commit().map_err(SessionError::storage)?;
        }

        self.cache.splice(range.clone(), replacement);
        self.positions.splice(range, new_positions.iter().copied());
        self.max_position = self
            .max_position
            .max(new_positions.last().copied().unwrap_or(0.0));
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn insert_item(
    tx: &rusqlite::Transaction<'_>,
    session_id: &SessionId,
    item: &MemoryItem,
    position: f64,
) -> sweet_core::error::Result<()> {
    match item {
        MemoryItem::Message(msg) => {
            let data = serde_json::to_string(msg).map_err(SessionError::storage)?;
            tx.execute(
                "INSERT INTO items (session_id, kind, data, tokens, position)
                 VALUES (?1, 'message', ?2, ?3, ?4)",
                params![
                    session_id.to_string(),
                    data,
                    msg.token_count.map(|t| t as i64),
                    position
                ],
            )
            .map_err(SessionError::storage)?;
        }
    }
    Ok(())
}
