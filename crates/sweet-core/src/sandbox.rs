// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Sandbox traits and types for tool execution isolation.
//!
//! Defines the abstraction boundary between tool logic and the underlying
//! execution environment. Tools depend on [`CommandRunner`] and [`Filesystem`]
//! traits; concrete implementations live in separate crates (`sweet-sandbox`
//! for local OS sandboxing).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Command execution
// ---------------------------------------------------------------------------

/// Output from a command execution.
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Executes a shell command, optionally within a sandbox.
#[async_trait]
pub trait CommandRunner: Send + Sync {
    async fn run(
        &self,
        command: &str,
        cwd: Option<&Path>,
        env: Option<&HashMap<String, String>>,
    ) -> Result<CommandOutput, SandboxError>;
}

// ---------------------------------------------------------------------------
// Filesystem operations
// ---------------------------------------------------------------------------

/// Metadata for a file or directory.
pub struct FileMetadata {
    pub size: u64,
    pub is_dir: bool,
    pub is_symlink: bool,
    pub modified: Option<SystemTime>,
    pub created: Option<SystemTime>,
    /// Unix permissions (e.g. `0o644`). `None` on non-Unix or remote backends.
    #[cfg(unix)]
    pub unix_permissions: Option<u32>,
}

/// A directory entry returned by [`Filesystem::list_dir`].
pub struct DirEntry {
    pub name: String,
    pub path: PathBuf,
    pub metadata: FileMetadata,
}

/// A single match from [`Filesystem::search`].
pub struct SearchMatch {
    pub path: PathBuf,
    pub line_number: usize,
    pub line: String,
}

/// Filesystem operations that tools use. Abstracts over the local filesystem,
/// a restricted (sandboxed) filesystem, and any other backend.
#[async_trait]
pub trait Filesystem: Send + Sync {
    async fn read(&self, path: &Path) -> Result<Vec<u8>, SandboxError>;
    async fn read_to_string(&self, path: &Path) -> Result<String, SandboxError>;
    async fn write(&self, path: &Path, content: &[u8]) -> Result<(), SandboxError>;
    async fn metadata(&self, path: &Path) -> Result<FileMetadata, SandboxError>;
    async fn list_dir(&self, path: &Path) -> Result<Vec<DirEntry>, SandboxError>;
    async fn create_dir_all(&self, path: &Path) -> Result<(), SandboxError>;
    async fn remove_file(&self, path: &Path) -> Result<(), SandboxError>;
    async fn remove_dir_all(&self, path: &Path) -> Result<(), SandboxError>;
    async fn rename(&self, src: &Path, dst: &Path) -> Result<(), SandboxError>;
    async fn exists(&self, path: &Path) -> bool;

    /// Recursive directory walk with glob-pattern filtering.
    /// Returns matching **file** paths (no directories). Respects
    /// `.gitignore` when `DirectFs` overrides the default impl.
    async fn walk(&self, pattern: &str, base: &Path) -> Result<Vec<PathBuf>, SandboxError> {
        let glob = globset::Glob::new(pattern)
            .map_err(|e| SandboxError::Backend(format!("invalid glob: {e}")))?
            .compile_matcher();
        let mut results = Vec::new();
        self.walk_recursive(base, &glob, &mut results).await?;
        Ok(results)
    }

    /// Recursive directory walk returning both files and directories.
    /// Respects `.gitignore` when `DirectFs` overrides the default impl.
    /// Default: naive recursive `list_dir` without `.gitignore` support.
    async fn walk_entries(&self, base: &Path) -> Result<Vec<DirEntry>, SandboxError> {
        let mut results = Vec::new();
        self.walk_entries_recursive(base, &mut results).await?;
        Ok(results)
    }

    /// Content search across files under `base`.
    /// Returns up to `limit` matches. `regex` controls whether `pattern` is
    /// treated as a regex. Default impl: `walk` + `read_to_string` + matching.
    /// `DirectFs` overrides with `ignore::WalkBuilder` for speed and
    /// `.gitignore` support.
    async fn search(
        &self,
        pattern: &str,
        base: &Path,
        regex: bool,
        limit: usize,
    ) -> Result<Vec<SearchMatch>, SandboxError> {
        let re = if regex {
            regex::Regex::new(pattern)
                .map_err(|e| SandboxError::Backend(format!("invalid regex: {e}")))?
        } else {
            regex::Regex::new(&regex::escape(pattern))
                .map_err(|e| SandboxError::Backend(format!("invalid regex: {e}")))?
        };

        let files = self.walk("**", base).await?;
        let mut results = Vec::new();

        for file_path in files {
            if results.len() >= limit {
                break;
            }
            let content = match self.read_to_string(&file_path).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            for (line_no, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(SearchMatch {
                        path: file_path.clone(),
                        line_number: line_no + 1,
                        line: line.to_string(),
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Recursive helper for the default `walk` implementation.
    async fn walk_recursive(
        &self,
        dir: &Path,
        matcher: &globset::GlobMatcher,
        results: &mut Vec<PathBuf>,
    ) -> Result<(), SandboxError> {
        let entries = self.list_dir(dir).await?;
        for entry in entries {
            if entry.metadata.is_dir {
                Box::pin(self.walk_recursive(&entry.path, matcher, results)).await?;
            } else if matcher.is_match(&entry.path)
                || matcher.is_match(entry.path.file_name().unwrap_or_default())
            {
                results.push(entry.path);
            }
        }
        Ok(())
    }

    /// Recursive helper for the default `walk_entries` implementation.
    async fn walk_entries_recursive(
        &self,
        dir: &Path,
        results: &mut Vec<DirEntry>,
    ) -> Result<(), SandboxError> {
        let entries = self.list_dir(dir).await?;
        for entry in entries {
            if entry.metadata.is_dir {
                let path = entry.path.clone();
                results.push(entry);
                Box::pin(self.walk_entries_recursive(&path, results)).await?;
            } else {
                results.push(entry);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sandbox bundle
// ---------------------------------------------------------------------------

/// Bundles a [`CommandRunner`] and [`Filesystem`] together. Prevents mixing
/// a local filesystem with a remote runner. Constructed once at startup,
/// shared via `Arc<dyn Sandbox>`.
///
/// Policy is fixed at construction time. Neither platform sandbox
/// (macOS Seatbelt, Linux bwrap) can filter network traffic by domain or IP
/// at the kernel layer, so the only honest model is a one-shot decision —
/// to change it mid-session, the caller restarts.
pub trait Sandbox: Send + Sync {
    fn runner(&self) -> Arc<dyn CommandRunner>;
    fn fs(&self) -> Arc<dyn Filesystem>;
}

// ---------------------------------------------------------------------------
// Sandbox policy
// ---------------------------------------------------------------------------

/// Sandbox policy for command execution.
///
/// Set once at startup and fixed for the session — neither macOS Seatbelt
/// nor Linux bwrap can filter network traffic by domain or IP at the kernel
/// layer, so the only honest model is a one-shot decision. To change policy
/// mid-session, the caller restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SandboxPolicy {
    /// No OS-level sandboxing; commands run directly (default).
    #[default]
    Off,
    /// OS-level sandbox enabled, network allowed.
    Sandbox,
    /// OS-level sandbox enabled, outbound network blocked.
    Restricted,
}
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("path denied: {path} ({reason})")]
    PathDenied { path: PathBuf, reason: String },
    #[error("{0}")]
    Backend(String),
}

// ---------------------------------------------------------------------------
// Local default implementations
// ---------------------------------------------------------------------------

/// Unsandboxed command runner using `tokio::process::Command`.
pub struct DirectRunner;

#[async_trait]
impl CommandRunner for DirectRunner {
    async fn run(
        &self,
        command: &str,
        cwd: Option<&Path>,
        env: Option<&HashMap<String, String>>,
    ) -> Result<CommandOutput, SandboxError> {
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c").arg(command);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if let Some(vars) = env {
            for (k, v) in vars {
                cmd.env(k, v);
            }
        }
        // If the future driving us is dropped (turn cancelled), SIGKILL the
        // child shell rather than orphaning it to launchd/init.
        cmd.kill_on_drop(true);
        let output = cmd.output().await?;
        let exit_code = output.status.code().unwrap_or(-1);
        Ok(CommandOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code,
        })
    }
}

/// Unsandboxed filesystem using `tokio::fs` directly.
pub struct DirectFs;

#[async_trait]
impl Filesystem for DirectFs {
    async fn read(&self, path: &Path) -> Result<Vec<u8>, SandboxError> {
        Ok(tokio::fs::read(path).await?)
    }

    async fn read_to_string(&self, path: &Path) -> Result<String, SandboxError> {
        Ok(tokio::fs::read_to_string(path).await?)
    }

    async fn write(&self, path: &Path, content: &[u8]) -> Result<(), SandboxError> {
        Ok(tokio::fs::write(path, content).await?)
    }

    async fn metadata(&self, path: &Path) -> Result<FileMetadata, SandboxError> {
        let meta = tokio::fs::metadata(path).await?;
        #[cfg(unix)]
        let unix_permissions = {
            use std::os::unix::fs::PermissionsExt;
            Some(meta.permissions().mode())
        };
        Ok(FileMetadata {
            size: meta.len(),
            is_dir: meta.is_dir(),
            is_symlink: false, // tokio::fs::metadata follows symlinks
            modified: meta.modified().ok(),
            created: meta.created().ok(),
            #[cfg(unix)]
            unix_permissions,
        })
    }

    async fn list_dir(&self, path: &Path) -> Result<Vec<DirEntry>, SandboxError> {
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(path).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let meta = entry.metadata().await?;
            #[cfg(unix)]
            let unix_permissions = {
                use std::os::unix::fs::PermissionsExt;
                Some(meta.permissions().mode())
            };
            entries.push(DirEntry {
                path: entry.path(),
                name,
                metadata: FileMetadata {
                    size: meta.len(),
                    is_dir: meta.is_dir(),
                    is_symlink: false,
                    modified: meta.modified().ok(),
                    created: meta.created().ok(),
                    #[cfg(unix)]
                    unix_permissions,
                },
            });
        }
        Ok(entries)
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), SandboxError> {
        Ok(tokio::fs::create_dir_all(path).await?)
    }

    async fn remove_file(&self, path: &Path) -> Result<(), SandboxError> {
        Ok(tokio::fs::remove_file(path).await?)
    }

    async fn remove_dir_all(&self, path: &Path) -> Result<(), SandboxError> {
        Ok(tokio::fs::remove_dir_all(path).await?)
    }

    async fn rename(&self, src: &Path, dst: &Path) -> Result<(), SandboxError> {
        Ok(tokio::fs::rename(src, dst).await?)
    }

    async fn exists(&self, path: &Path) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }

    async fn walk(&self, pattern: &str, base: &Path) -> Result<Vec<PathBuf>, SandboxError> {
        let glob = globset::Glob::new(pattern)
            .map_err(|e| SandboxError::Backend(format!("invalid glob: {e}")))?
            .compile_matcher();
        let mut results = Vec::new();

        let walker = ignore::WalkBuilder::new(base)
            .standard_filters(true)
            .build();
        for entry in walker {
            let entry = entry.map_err(|e| SandboxError::Backend(format!("walk error: {e}")))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let relative = path.strip_prefix(base).unwrap_or(path);
            if !glob.is_match(relative) && !glob.is_match(path.file_name().unwrap_or_default()) {
                continue;
            }
            results.push(path.to_path_buf());
        }

        Ok(results)
    }

    async fn walk_entries(&self, base: &Path) -> Result<Vec<DirEntry>, SandboxError> {
        let mut results = Vec::new();
        let walker = ignore::WalkBuilder::new(base)
            .standard_filters(true)
            .build();
        for entry in walker {
            let entry = entry.map_err(|e| SandboxError::Backend(format!("walk error: {e}")))?;
            let path = entry.path();
            // Skip the root entry itself
            if path == base {
                continue;
            }
            let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let meta = tokio::fs::metadata(path).await?;
            #[cfg(unix)]
            let unix_permissions = {
                use std::os::unix::fs::PermissionsExt;
                Some(meta.permissions().mode())
            };
            results.push(DirEntry {
                name,
                path: path.to_path_buf(),
                metadata: FileMetadata {
                    size: meta.len(),
                    is_dir,
                    is_symlink: false,
                    modified: meta.modified().ok(),
                    created: meta.created().ok(),
                    #[cfg(unix)]
                    unix_permissions,
                },
            });
        }
        Ok(results)
    }

    async fn search(
        &self,
        pattern: &str,
        base: &Path,
        regex: bool,
        limit: usize,
    ) -> Result<Vec<SearchMatch>, SandboxError> {
        let re = if regex {
            regex::Regex::new(pattern)
                .map_err(|e| SandboxError::Backend(format!("invalid regex: {e}")))?
        } else {
            regex::Regex::new(&regex::escape(pattern))
                .map_err(|e| SandboxError::Backend(format!("invalid regex: {e}")))?
        };

        let meta = tokio::fs::metadata(base).await?;

        let mut results = Vec::new();

        if meta.is_file() {
            let content = tokio::fs::read_to_string(base).await?;
            for (line_no, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(SearchMatch {
                        path: base.to_path_buf(),
                        line_number: line_no + 1,
                        line: line.to_string(),
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        } else {
            let walker = ignore::WalkBuilder::new(base)
                .standard_filters(true)
                .build();
            for entry in walker {
                if results.len() >= limit {
                    break;
                }
                let entry = entry.map_err(|e| SandboxError::Backend(format!("walk error: {e}")))?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let content = match tokio::fs::read_to_string(path).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                for (line_no, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        results.push(SearchMatch {
                            path: path.to_path_buf(),
                            line_number: line_no + 1,
                            line: line.to_string(),
                        });
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Unsandboxed local sandbox: bundles `DirectRunner` + `DirectFs`.
pub struct DirectSandbox {
    runner: Arc<DirectRunner>,
    fs: Arc<DirectFs>,
}

impl DirectSandbox {
    pub fn new() -> Self {
        Self {
            runner: Arc::new(DirectRunner),
            fs: Arc::new(DirectFs),
        }
    }
}

impl Default for DirectSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for DirectSandbox {
    fn runner(&self) -> Arc<dyn CommandRunner> {
        Arc::clone(&self.runner) as Arc<dyn CommandRunner>
    }

    fn fs(&self) -> Arc<dyn Filesystem> {
        Arc::clone(&self.fs) as Arc<dyn Filesystem>
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_runner_executes_echo() {
        let runner = DirectRunner;
        let output = runner.run("echo hello", None, None).await.unwrap();
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn local_runner_returns_exit_code() {
        let runner = DirectRunner;
        let output = runner.run("exit 42", None, None).await.unwrap();
        assert_eq!(output.exit_code, 42);
    }

    #[tokio::test]
    async fn local_runner_captures_stderr() {
        let runner = DirectRunner;
        let output = runner.run("echo error >&2", None, None).await.unwrap();
        assert!(output.stderr.contains("error"));
    }

    #[tokio::test]
    async fn local_fs_read_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let fs = DirectFs;

        fs.write(&file_path, b"hello world").await.unwrap();
        let content = fs.read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn local_fs_exists() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("exists.txt");
        let fs = DirectFs;

        assert!(!fs.exists(&file_path).await);
        fs.write(&file_path, b"content").await.unwrap();
        assert!(fs.exists(&file_path).await);
    }

    #[tokio::test]
    async fn local_fs_list_dir() {
        let dir = tempfile::tempdir().unwrap();
        let fs = DirectFs;
        fs.write(&dir.path().join("a.txt"), b"a").await.unwrap();
        fs.write(&dir.path().join("b.txt"), b"b").await.unwrap();

        let entries = fs.list_dir(dir.path()).await.unwrap();
        assert_eq!(entries.len(), 2);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a.txt"));
        assert!(names.contains(&"b.txt"));
    }

    #[tokio::test]
    async fn local_fs_create_dir_all() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        let fs = DirectFs;
        fs.create_dir_all(&nested).await.unwrap();
        assert!(nested.is_dir());
    }

    #[tokio::test]
    async fn local_fs_rename() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("old.txt");
        let dst = dir.path().join("new.txt");
        let fs = DirectFs;
        fs.write(&src, b"data").await.unwrap();
        fs.rename(&src, &dst).await.unwrap();
        assert!(!src.exists());
        assert_eq!(fs.read_to_string(&dst).await.unwrap(), "data");
    }

    #[tokio::test]
    async fn local_fs_remove_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("temp.txt");
        let fs = DirectFs;
        fs.write(&path, b"x").await.unwrap();
        fs.remove_file(&path).await.unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn local_fs_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.txt");
        let fs = DirectFs;
        fs.write(&path, b"12345").await.unwrap();

        let meta = fs.metadata(&path).await.unwrap();
        assert_eq!(meta.size, 5);
        assert!(!meta.is_dir);
        assert!(meta.modified.is_some());
    }

    #[tokio::test]
    async fn local_sandbox_bundles_runner_and_fs() {
        let sandbox = DirectSandbox::new();
        let runner = sandbox.runner();
        let fs = sandbox.fs();

        let output = runner.run("echo bundled", None, None).await.unwrap();
        assert!(output.stdout.contains("bundled"));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("via_sandbox.txt");
        fs.write(&path, b"sandboxed").await.unwrap();
        assert_eq!(fs.read_to_string(&path).await.unwrap(), "sandboxed");
    }

    #[tokio::test]
    async fn local_fs_walk_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        let fs = DirectFs;
        fs.write(&dir.path().join("a.rs"), b"").await.unwrap();
        fs.write(&dir.path().join("b.txt"), b"").await.unwrap();
        fs.create_dir_all(&dir.path().join("sub")).await.unwrap();
        fs.write(&dir.path().join("sub/c.rs"), b"").await.unwrap();

        let results = fs.walk("*.rs", dir.path()).await.unwrap();
        assert_eq!(results.len(), 2);
        let names: Vec<String> = results
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.rs".to_string()));
        assert!(names.contains(&"c.rs".to_string()));
    }

    #[tokio::test]
    async fn local_fs_walk_matches_directory_scoped_glob() {
        let dir = tempfile::tempdir().unwrap();
        let fs = DirectFs;
        fs.write(&dir.path().join("a.rs"), b"").await.unwrap();
        fs.create_dir_all(&dir.path().join("sub")).await.unwrap();
        fs.write(&dir.path().join("sub/c.rs"), b"").await.unwrap();

        // A glob with a directory component matches against the path relative
        // to `base`, so it selects only the nested file, not the top-level one.
        let results = fs.walk("sub/*.rs", dir.path()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].ends_with("sub/c.rs"));
    }

    #[tokio::test]
    async fn local_fs_search_finds_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let fs = DirectFs;
        fs.write(
            &dir.path().join("hello.txt"),
            b"line one\nmatch here\nline three\n",
        )
        .await
        .unwrap();

        let matches = fs.search("match", dir.path(), false, 10).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line_number, 2);
        assert!(matches[0].line.contains("match here"));
    }
}
