// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! SQLite-backed [`Memory`] with hybrid keyword + semantic recall, using
//! `sqlite-vec` for vector similarity search instead of brute-force cosine.

use std::sync::{Arc, Mutex, OnceLock};

use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use sweet_core::{
    rrf_merge, unix_now, Embedder, Memory, MemoryError, MemoryHit, MemoryId, MemoryQuery,
    MemoryRecord, MemoryScope,
};

use async_trait::async_trait;

use crate::sqlite_shared::{
    collect_records, filter_record, fts_match_expr, open_conn, row_to_record, scope_filter,
    vec_to_blob, CANDIDATE_LIMIT, MEMORIES_TABLE_DDL, RECORD_COLUMNS, RECORD_COLUMNS_QUALIFIED,
};

/// sqlite-vec returns results well beyond what scope/tag filters will accept,
/// so we over-fetch and trim in Rust.
const VEC_CANDIDATE_OVERFETCH: usize = 200;

/// Registers `sqlite3_vec_init` as a process-wide auto-extension exactly once.
fn ensure_vec_extension_loaded() {
    static VEC_EXT: OnceLock<()> = OnceLock::new();
    VEC_EXT.get_or_init(|| unsafe {
        // SAFETY: `sqlite3_auto_extension` is thread-safe and idempotent per
        // SQLite docs. `sqlite_vec::sqlite3_vec_init` is the canonical init
        // function exported by the `sqlite-vec` C library; its signature
        // matches the `sqlite3_auto_extension` callback contract (three
        // `*mut` parameters returning `c_int`). The transmute from a typed
        // function pointer through `*const ()` to
        // `Option<unsafe extern "C" fn(...)>` is the pattern documented in
        // the sqlite-vec README.
        type AutoExtensionEntry = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int;
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            AutoExtensionEntry,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    });
}

/// Persistent [`Memory`] store backed by SQLite + `sqlite-vec`.
///
/// Like [`SqliteMemory`](super::SqliteMemory), keyword recall uses an
/// external-content FTS5 index kept in sync by triggers. Semantic recall uses
/// a `vec0` virtual table powered by `sqlite-vec` for KNN search instead of
/// brute-force cosine similarity. The two rankings are fused via Reciprocal
/// Rank Fusion.
///
/// The vector dimensionality is fixed at store creation time and persisted in
/// a `_meta` table. Reopening with a different dimensionality is an error —
/// you would need to create a new database or re-embed everything.
///
/// Opens in WAL mode with a busy timeout so multiple processes can share the
/// file.
pub struct SqliteVecMemory {
    conn: Mutex<Connection>,
    embedder: Option<Arc<dyn Embedder>>,
    /// Dimensionality of the vec0 table; vectors of any other size are
    /// rejected by sqlite-vec, so they degrade to keyword-only instead.
    dims: usize,
}

impl std::fmt::Debug for SqliteVecMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVecMemory")
            .field("embedder", &self.embedder.as_ref().map(|e| e.id()))
            .finish()
    }
}

impl SqliteVecMemory {
    /// Open (or create) the store at `path` with the given vector
    /// dimensionality. Pass `":memory:"` for a transient store.
    ///
    /// The dimensionality must match across reopens — it is validated against
    /// the `_meta` table on subsequent opens. The attached embedder must
    /// produce vectors of exactly this size; other sizes degrade to
    /// keyword-only recall (see [`with_embedder`](Self::with_embedder)).
    pub fn open(
        path: impl AsRef<std::path::Path>,
        vector_dimensions: usize,
    ) -> Result<Self, MemoryError> {
        ensure_vec_extension_loaded();
        let conn = open_conn(path)?;
        Self::init_schema(&conn, vector_dimensions)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder: None,
            dims: vector_dimensions,
        })
    }

    /// Attach an embedder; subsequent saves are embedded and searches add a
    /// semantic ranking. Embedding failure during save — including vectors
    /// whose size doesn't match the store's dimensionality — degrades that
    /// record to keyword-only recall rather than failing the save.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    fn init_schema(conn: &Connection, vector_dimensions: usize) -> Result<(), MemoryError> {
        // Verify sqlite-vec is loaded.
        let _version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .map_err(|e| {
                MemoryError::storage(rusqlite::Error::InvalidParameterName(format!(
                    "sqlite-vec extension not loaded (is the sqlite-vec feature enabled?): {e}"
                )))
            })?;

        // Create _meta table first (idempotent), then check for dimension
        // mismatch on reopen.
        conn.execute_batch("CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT);")
            .map_err(MemoryError::storage)?;

        let dims_str = vector_dimensions.to_string();
        let existing_dims: Option<String> = conn
            .query_row(
                "SELECT value FROM _meta WHERE key = 'vector_dimensions'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(MemoryError::storage)?;

        match existing_dims {
            Some(d) if d != dims_str => {
                return Err(MemoryError::storage(rusqlite::Error::InvalidParameterName(
                    format!(
                    "vector dimensionality mismatch: database has {d}, but {dims_str} was requested"
                ),
                )));
            }
            None => {
                conn.execute(
                    "INSERT INTO _meta (key, value) VALUES ('vector_dimensions', ?1)",
                    params![dims_str],
                )
                .map_err(MemoryError::storage)?;
            }
            _ => {}
        }

        // Shared memories table + FTS5, then vec0 virtual table.
        conn.execute_batch(MEMORIES_TABLE_DDL)
            .map_err(MemoryError::storage)?;

        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                embedding float[{vector_dimensions}]
            );"
        ))
        .map_err(MemoryError::storage)
    }

    /// Embed `text` if an embedder is attached; `None` (with a warning) when
    /// embedding fails or the vector doesn't match the store's
    /// dimensionality — memory durability beats vector coverage.
    async fn try_embed(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        match embedder.embed(&[text.to_string()]).await {
            Ok(mut vectors) => self.check_dims(vectors.pop()),
            Err(err) => {
                tracing::warn!("embedding failed, saving keyword-only memory: {err}");
                None
            }
        }
    }

    /// `None` (with a warning) for a vector the vec0 table would reject.
    fn check_dims(&self, vector: Option<Vec<f32>>) -> Option<Vec<f32>> {
        match vector {
            Some(v) if v.len() == self.dims => Some(v),
            Some(v) => {
                tracing::warn!(
                    "embedder produced {} dimensions but the store expects {}; \
                     degrading to keyword-only",
                    v.len(),
                    self.dims
                );
                None
            }
            None => None,
        }
    }

    /// Keyword candidates, best (lowest bm25) first.
    fn fts_candidates(
        &self,
        text: &str,
        query: &MemoryQuery,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        let match_expr = fts_match_expr(text);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }
        let (scope_clause, scope_params) = scope_filter(&query.scopes);
        let sql = format!(
            "SELECT {RECORD_COLUMNS_QUALIFIED} FROM memories_fts
             JOIN memories ON memories.rowid = memories_fts.rowid
             WHERE memories_fts MATCH ?1{scope_clause}
             ORDER BY bm25(memories_fts) LIMIT {CANDIDATE_LIMIT}"
        );
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
        let params_iter = std::iter::once(match_expr).chain(scope_params);
        let rows = stmt
            .query_map(params_from_iter(params_iter), row_to_record)
            .map_err(MemoryError::storage)?;
        collect_records(rows, query)
    }

    /// Semantic candidates via sqlite-vec KNN search, closest first.
    ///
    /// The vec0 virtual table returns results ordered by L2 distance. For
    /// normalized vectors (most embedding models output unit vectors), L2
    /// distance and cosine similarity produce identical rankings.
    ///
    /// sqlite-vec requires the LIMIT to be directly on the vec0 scan —
    /// additional WHERE filters break the KNN plan. So we do the KNN query
    /// first with just the MATCH + k, then filter by embedder model,
    /// scope, and tags in Rust.
    fn vector_candidates(
        &self,
        query_vector: &[f32],
        embedder_id: &str,
        query: &MemoryQuery,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        let query_blob = vec_to_blob(query_vector);
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());

        // KNN query — using `k = N` for broad SQLite compatibility.
        let sql = format!(
            "SELECT {RECORD_COLUMNS_QUALIFIED}, memories.embedding_model
             FROM memories_vec v
             JOIN memories ON memories.rowid = v.rowid
             WHERE v.embedding MATCH ?1
               AND k = {VEC_CANDIDATE_OVERFETCH}
             ORDER BY v.distance"
        );

        let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
        let rows = stmt
            .query_map(params![query_blob], |row| {
                let record = row_to_record(row)?;
                let model: Option<String> = row.get(8)?;
                Ok((record, model))
            })
            .map_err(MemoryError::storage)?;

        let mut candidates = Vec::new();
        for row in rows {
            let (record, model) = row.map_err(MemoryError::storage)?;
            // Filter by embedder model and tags in Rust.
            if model.as_deref() != Some(embedder_id) {
                continue;
            }
            if let Some(record) = filter_record(record, query) {
                candidates.push(record);
                if candidates.len() >= CANDIDATE_LIMIT {
                    break;
                }
            }
        }
        Ok(candidates)
    }

    /// Look up the rowid for a memory by its id.
    fn get_rowid(conn: &Connection, id: &MemoryId) -> Result<i64, MemoryError> {
        conn.query_row(
            "SELECT rowid FROM memories WHERE id = ?1",
            params![id.to_string()],
            |row| row.get(0),
        )
        .optional()
        .map_err(MemoryError::storage)?
        .ok_or_else(|| MemoryError::NotFound(id.to_string()))
    }
}

#[async_trait]
impl Memory for SqliteVecMemory {
    async fn save(
        &self,
        scope: MemoryScope,
        content: &str,
        tags: &[String],
        source_session: Option<&str>,
    ) -> Result<MemoryRecord, MemoryError> {
        // Embed before taking the lock; the guard can't be held across await.
        let embedding = self.try_embed(content).await;
        let now = unix_now();
        let record = MemoryRecord {
            id: MemoryId::new(),
            scope,
            content: content.to_string(),
            tags: tags.to_vec(),
            source_session: source_session.map(str::to_string),
            created_at: now,
            updated_at: now,
        };
        let tags_json = serde_json::to_string(&record.tags).map_err(MemoryError::storage)?;
        // One transaction: a memories row without its vec0 twin would be
        // silently invisible to semantic search (and rowid reuse after a
        // partial delete could attach an orphaned vector to the wrong row).
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(MemoryError::storage)?;
        tx.execute(
            "INSERT INTO memories
             (id, scope_kind, scope_key, content, tags, source_session,
              created_at, updated_at, embedding, embedding_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                record.id.to_string(),
                record.scope.kind(),
                record.scope.key(),
                record.content,
                tags_json,
                record.source_session,
                record.created_at,
                record.updated_at,
                embedding.as_deref().map(vec_to_blob),
                embedding
                    .is_some()
                    .then(|| self.embedder.as_ref().map(|e| e.id().to_string()))
                    .flatten(),
            ],
        )
        .map_err(MemoryError::storage)?;

        // Also insert into the vec0 virtual table if we have an embedding.
        if let Some(ref vec) = embedding {
            let rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO memories_vec(rowid, embedding) VALUES (?1, ?2)",
                params![rowid, vec_to_blob(vec)],
            )
            .map_err(MemoryError::storage)?;
        }
        tx.commit().map_err(MemoryError::storage)?;

        Ok(record)
    }

    async fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, MemoryError> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        conn.query_row(
            &format!("SELECT {RECORD_COLUMNS} FROM memories WHERE id = ?1"),
            params![id.to_string()],
            row_to_record,
        )
        .optional()
        .map_err(MemoryError::storage)
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<MemoryHit>, MemoryError> {
        let text = query.text.as_deref().filter(|t| !t.trim().is_empty());

        let Some(text) = text else {
            // List mode: newest first within the filters.
            let (scope_clause, scope_params) = scope_filter(&query.scopes);
            let sql = format!(
                "SELECT {RECORD_COLUMNS} FROM memories WHERE 1=1{scope_clause}
                 ORDER BY updated_at DESC, id DESC"
            );
            let records = {
                let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
                let mut stmt = conn.prepare(&sql).map_err(MemoryError::storage)?;
                let rows = stmt
                    .query_map(params_from_iter(scope_params), row_to_record)
                    .map_err(MemoryError::storage)?;
                collect_records(rows, query)?
            };
            return Ok(records
                .into_iter()
                .take(query.limit)
                .map(|record| MemoryHit { record, score: 0.0 })
                .collect());
        };

        // Embed the query before any lock is taken. A query vector the vec0
        // table would reject degrades the search to keyword-only.
        let query_embedding = match &self.embedder {
            Some(embedder) => {
                let vector = embedder
                    .embed(&[text.to_string()])
                    .await
                    .map_err(|e| MemoryError::Embedding(e.into()))?
                    .pop();
                self.check_dims(vector)
                    .map(|v| (v, embedder.id().to_string()))
            }
            None => None,
        };

        let keyword = self.fts_candidates(text, query)?;
        let vector = match &query_embedding {
            Some((qv, embedder_id)) => self.vector_candidates(qv, embedder_id, query)?,
            None => Vec::new(),
        };

        let mut by_id: Vec<MemoryRecord> = Vec::new();
        for record in keyword.iter().chain(vector.iter()) {
            if !by_id.iter().any(|r| r.id == record.id) {
                by_id.push(record.clone());
            }
        }
        let rankings = [
            keyword.into_iter().map(|r| r.id).collect::<Vec<_>>(),
            vector.into_iter().map(|r| r.id).collect::<Vec<_>>(),
        ];
        let fused = rrf_merge(&rankings);

        Ok(fused
            .into_iter()
            .take(query.limit)
            .filter_map(|(id, score)| {
                by_id.iter().find(|r| r.id == id).map(|record| MemoryHit {
                    record: record.clone(),
                    score,
                })
            })
            .collect())
    }

    async fn update(
        &self,
        id: &MemoryId,
        content: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<MemoryRecord, MemoryError> {
        let mut record = self
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        // Re-embed only when the content changes.
        let new_embedding = match content {
            Some(text) => Some(self.try_embed(text).await),
            None => None,
        };

        if let Some(text) = content {
            record.content = text.to_string();
        }
        if let Some(tags) = tags {
            record.tags = tags.to_vec();
        }
        record.updated_at = unix_now();
        let tags_json = serde_json::to_string(&record.tags).map_err(MemoryError::storage)?;

        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(MemoryError::storage)?;
        let rowid = Self::get_rowid(&tx, id)?;

        let updated = match new_embedding {
            Some(embedding) => {
                let n = tx
                    .execute(
                        "UPDATE memories SET content = ?2, tags = ?3, updated_at = ?4,
                         embedding = ?5, embedding_model = ?6 WHERE id = ?1",
                        params![
                            id.to_string(),
                            record.content,
                            tags_json,
                            record.updated_at,
                            embedding.as_deref().map(vec_to_blob),
                            embedding
                                .is_some()
                                .then(|| self.embedder.as_ref().map(|e| e.id().to_string()))
                                .flatten(),
                        ],
                    )
                    .map_err(MemoryError::storage)?;

                // Update the vec0 table: delete old entry, insert new if we
                // have an embedding.
                tx.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![rowid])
                    .map_err(MemoryError::storage)?;

                if let Some(ref vec) = embedding {
                    tx.execute(
                        "INSERT INTO memories_vec(rowid, embedding) VALUES (?1, ?2)",
                        params![rowid, vec_to_blob(vec)],
                    )
                    .map_err(MemoryError::storage)?;
                }
                n
            }
            None => tx
                .execute(
                    "UPDATE memories SET content = ?2, tags = ?3, updated_at = ?4 WHERE id = ?1",
                    params![id.to_string(), record.content, tags_json, record.updated_at],
                )
                .map_err(MemoryError::storage)?,
        };
        if updated == 0 {
            return Err(MemoryError::NotFound(id.to_string()));
        }
        tx.commit().map_err(MemoryError::storage)?;
        Ok(record)
    }

    async fn delete(&self, id: &MemoryId) -> Result<bool, MemoryError> {
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.transaction().map_err(MemoryError::storage)?;

        // Get rowid before deleting from memories.
        let rowid_result: Option<i64> = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(MemoryError::storage)?;

        let Some(rowid) = rowid_result else {
            return Ok(false);
        };

        // Delete from vec0 and memories together (FTS5 trigger handles the
        // FTS cleanup): a vec0 row outliving its memories row would attach
        // to whatever record later reuses the rowid.
        tx.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![rowid])
            .map_err(MemoryError::storage)?;
        let deleted = tx
            .execute(
                "DELETE FROM memories WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(MemoryError::storage)?;
        tx.commit().map_err(MemoryError::storage)?;
        Ok(deleted > 0)
    }
}
