// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::error::Error as StdError;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::message::Message;

/// Structured error type for session storage backends.
///
/// `Storage` wraps any `std::error::Error + Send + Sync` so concrete backends
/// (SQLite, JSON files, etc.) can plug their own error types in without
/// forcing `sweet-core` to depend on any specific storage crate.
#[derive(thiserror::Error, Debug)]
pub enum SessionError {
    #[error("session storage error: {0}")]
    Storage(#[source] Box<dyn StdError + Send + Sync>),
}

impl SessionError {
    pub fn storage<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Storage(Box::new(err))
    }
}

/// A time-sortable identifier for a session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct SessionId(uuid::Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::str::FromStr for SessionId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(uuid::Uuid::parse_str(s)?))
    }
}

/// A single item stored in a session.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum MemoryItem {
    Message(Message),
}

/// The most recent provider-reported context size (`prompt_tokens`) recorded
/// on a message in `items`, scanning newest-first. `None` when no provider
/// response is on record yet.
pub fn last_context_size(items: &[MemoryItem]) -> Option<usize> {
    items.iter().rev().find_map(|item| match item {
        MemoryItem::Message(msg) => msg.context_tokens,
    })
}

/// Abstraction over conversational storage.
///
/// Implementations own their storage directly (Vec, SQLite connection, etc.).
/// Session does not own compaction logic; that lives on `Agent` (in the `sweet-agent` crate).
pub trait Session: Send + Sync {
    /// Unique identifier for this session.
    fn id(&self) -> &SessionId;

    /// Append an item to the session.
    fn push(&mut self, item: MemoryItem) -> Result<()>;

    /// All items in the session, in order.
    fn items(&self) -> &[MemoryItem];

    /// Filtered view of only [`MemoryItem::Message`] variants, in order.
    fn messages(&self) -> Vec<Message>;

    /// Remove all items from the session.
    fn clear(&mut self) -> Result<()>;

    /// Rough token estimate (characters / 4).
    fn token_count(&self) -> usize;

    /// Total tokens in the session (actual or estimated).
    fn total_tokens(&self) -> usize;

    /// Current context size based on the most recent `prompt_tokens` reported
    /// by the provider. Falls back to `token_count()` if no API response has
    /// been received yet.
    fn context_size(&self) -> usize;

    /// Replace items in `range` with `replacement`.
    fn replace_range(
        &mut self,
        range: std::ops::Range<usize>,
        replacement: Vec<MemoryItem>,
    ) -> Result<()>;

    /// Downcast support for `dyn Session`. Implementations return `self`.
    fn as_any(&self) -> &dyn Any;
}

/// Simple in-memory session backed by a `Vec<MemoryItem>`.
pub struct InMemorySession {
    id: SessionId,
    items: Vec<MemoryItem>,
}

impl InMemorySession {
    pub fn new() -> Self {
        Self {
            id: SessionId::new(),
            items: Vec::new(),
        }
    }
}

impl Default for InMemorySession {
    fn default() -> Self {
        Self::new()
    }
}

impl Session for InMemorySession {
    fn id(&self) -> &SessionId {
        &self.id
    }

    fn push(&mut self, item: MemoryItem) -> Result<()> {
        self.items.push(item);
        Ok(())
    }

    fn items(&self) -> &[MemoryItem] {
        &self.items
    }

    fn messages(&self) -> Vec<Message> {
        self.items
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.clone(),
            })
            .collect()
    }

    fn clear(&mut self) -> Result<()> {
        self.items.clear();
        Ok(())
    }

    fn token_count(&self) -> usize {
        self.items
            .iter()
            .map(|item| match item {
                MemoryItem::Message(msg) => msg.text_content().chars().count() / 4,
            })
            .sum()
    }

    fn total_tokens(&self) -> usize {
        self.token_count()
    }

    fn context_size(&self) -> usize {
        last_context_size(&self.items).unwrap_or_else(|| self.token_count())
    }

    fn replace_range(
        &mut self,
        range: std::ops::Range<usize>,
        replacement: Vec<MemoryItem>,
    ) -> Result<()> {
        self.items.splice(range, replacement);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Read-only handle to a session's message history, shared with consumers
/// that need point-in-time snapshots (e.g. subagent handlers).
///
/// A [`SharedSession`] wrapper mirrors every `push`, `clear`, and
/// `replace_range` into the handle's snapshot. Consumers clone the handle and
/// call [`snapshot_messages`](Self::snapshot_messages) when they need the
/// current transcript. Writes from the consumer do not flow back through the
/// handle.
///
/// Construct only via [`SharedSession::new`] — a standalone handle is never
/// written to and would always snapshot empty.
#[derive(Clone)]
pub struct SharedSessionHandle {
    snapshot: Arc<Mutex<Vec<Message>>>,
}

impl SharedSessionHandle {
    /// Copy the current messages. Cheap relative to the agent loop — one clone
    /// per consumer invocation.
    pub fn snapshot_messages(&self) -> Vec<Message> {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Append a single message. The O(1) hot path used on every push.
    fn append(&self, message: Message) {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(message);
    }

    /// Replace the entire snapshot. Used for clear and replace_range.
    fn replace_all(&self, messages: Vec<Message>) {
        *self.snapshot.lock().unwrap_or_else(|e| e.into_inner()) = messages;
    }
}

/// Session wrapper that mirrors every write into a [`SharedSessionHandle`].
///
/// Used by orchestrators so their workers can snapshot the transcript at
/// invocation time. The wrapper delegates reads to `inner` directly (no
/// locking on the read path); each mutating method refreshes the shared
/// handle after the inner mutation succeeds.
pub struct SharedSession {
    inner: Box<dyn Session>,
    handle: SharedSessionHandle,
}

impl SharedSession {
    /// Wrap `inner` and return the wrapper together with a handle that
    /// mirrors its messages. The handle is initialised from `inner`'s current
    /// state, so resumed sessions are visible to consumers on first call.
    pub fn new(inner: Box<dyn Session>) -> (Self, SharedSessionHandle) {
        let handle = SharedSessionHandle {
            snapshot: Arc::new(Mutex::new(Vec::new())),
        };
        handle.replace_all(inner.messages());
        (
            Self {
                inner,
                handle: handle.clone(),
            },
            handle,
        )
    }
}

impl Session for SharedSession {
    fn id(&self) -> &SessionId {
        self.inner.id()
    }

    fn push(&mut self, item: MemoryItem) -> Result<()> {
        // Extract the message before the move into `inner.push`. Mirroring the
        // append directly keeps the hot path O(1); a full
        // `inner.messages()` snapshot per push would be O(n²) across a
        // long session.
        let mirror = match &item {
            MemoryItem::Message(msg) => msg.clone(),
        };
        self.inner.push(item)?;
        self.handle.append(mirror);
        Ok(())
    }

    fn items(&self) -> &[MemoryItem] {
        self.inner.items()
    }

    fn messages(&self) -> Vec<Message> {
        self.inner.messages()
    }

    fn clear(&mut self) -> Result<()> {
        self.inner.clear()?;
        self.handle.replace_all(Vec::new());
        Ok(())
    }

    fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    fn context_size(&self) -> usize {
        self.inner.context_size()
    }

    fn replace_range(&mut self, range: Range<usize>, replacement: Vec<MemoryItem>) -> Result<()> {
        // Rare path (compaction). A full rebuild is simpler than tracking the
        // overlap between `range` and the mirror.
        self.inner.replace_range(range, replacement)?;
        self.handle.replace_all(self.inner.messages());
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};

    #[test]
    fn in_memory_session_push_and_messages() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("hello")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant("hi")))
            .unwrap();

        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.messages()[0].role, Role::User);
        assert_eq!(session.messages()[1].role, Role::Assistant);
    }

    #[test]
    fn in_memory_session_clear() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("hello")))
            .unwrap();
        session.clear().unwrap();
        assert!(session.messages().is_empty());
        assert!(session.items().is_empty());
    }

    #[test]
    fn in_memory_session_token_count() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("abcd".repeat(4))))
            .unwrap();
        // 16 chars / 4 = 4 tokens
        assert_eq!(session.token_count(), 4);
    }

    #[test]
    fn in_memory_session_replace_range() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("a")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::user("b")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::user("c")))
            .unwrap();

        let mut compacted = Message::user("summary");
        compacted.compacted = true;
        session
            .replace_range(0..2, vec![MemoryItem::Message(compacted)])
            .unwrap();

        assert_eq!(session.items().len(), 2);
        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.messages()[0].role, Role::User);
        assert_eq!(session.messages()[0].text_content(), "summary");
        assert!(session.messages()[0].compacted);
        assert_eq!(session.messages()[1].role, Role::User);
        assert_eq!(session.messages()[1].text_content(), "c");
    }

    #[test]
    fn session_id_is_v7() {
        let id = SessionId::new();
        let uuid = id.to_string();
        // UUID v7 starts with version nibble = 7 in the 13th hex char position
        // e.g. xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
        let parts: Vec<&str> = uuid.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert!(parts[2].starts_with('7'));
    }

    #[test]
    fn shared_session_mirrors_pushes_into_handle() {
        let (mut shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        assert!(handle.snapshot_messages().is_empty());

        shared
            .push(MemoryItem::Message(Message::user("first")))
            .unwrap();
        shared
            .push(MemoryItem::Message(Message::assistant("ok")))
            .unwrap();

        let snap = handle.snapshot_messages();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].text_content(), "first");
        assert_eq!(snap[1].text_content(), "ok");
    }

    #[test]
    fn shared_session_clear_propagates() {
        let (mut shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        shared
            .push(MemoryItem::Message(Message::user("hi")))
            .unwrap();
        assert_eq!(handle.snapshot_messages().len(), 1);

        shared.clear().unwrap();
        assert!(handle.snapshot_messages().is_empty());
    }

    #[test]
    fn shared_session_replace_range_propagates() {
        let (mut shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        shared
            .push(MemoryItem::Message(Message::user("a")))
            .unwrap();
        shared
            .push(MemoryItem::Message(Message::user("b")))
            .unwrap();
        shared
            .push(MemoryItem::Message(Message::user("c")))
            .unwrap();

        shared
            .replace_range(0..2, vec![MemoryItem::Message(Message::user("summary"))])
            .unwrap();

        let snap = handle.snapshot_messages();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].text_content(), "summary");
        assert_eq!(snap[1].text_content(), "c");
    }

    #[test]
    fn shared_session_handle_initialises_from_existing_state() {
        let mut existing = InMemorySession::new();
        existing
            .push(MemoryItem::Message(Message::user("prior")))
            .unwrap();
        let (_shared, handle) = SharedSession::new(Box::new(existing));
        let snap = handle.snapshot_messages();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].text_content(), "prior");
    }
}
