// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! A filesystem wrapper that restricts both read and write operations to
//! allowed directory roots.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::sandbox::{
    DirEntry, DirectFs, FileMetadata, Filesystem, SandboxError, SearchMatch,
};

use crate::tool_paths;

/// System paths that contain no user secrets and are needed for tooling
/// (compilers, linkers, interpreters, DNS, SSL certs, etc.).
/// Only these paths (plus the project root) are readable.
const SYSTEM_READ_PATHS: &[&str] = &[
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
    "/tmp",
    "/opt",
    "/nix",
    "/var",
    #[cfg(target_os = "macos")]
    "/System",
    #[cfg(target_os = "macos")]
    "/Library",
    #[cfg(target_os = "macos")]
    "/private",
    #[cfg(target_os = "macos")]
    "/Applications",
    #[cfg(target_os = "macos")]
    "/Developer",
];

/// A filesystem that restricts both read and write operations to a set of
/// allowed directory roots.
///
/// **Read roots**: system paths, tool paths (from `$PATH`), every write root,
/// and any extra read roots. No access to the user's home directory beyond
/// those.
///
/// **Write roots**: the roots passed to [`new`](Self::new) - the project root
/// plus any extra write roots (e.g. `$CARGO_HOME` for cargo's caches). Writes
/// outside them are denied; every write root is readable too.
///
/// All roots are canonicalized at construction time. Every read and write call
/// is validated against the respective root set.
pub struct RestrictedFs {
    inner: Arc<dyn Filesystem>,
    canonical_read_roots: Vec<PathBuf>,
    canonical_write_roots: Vec<PathBuf>,
}

impl RestrictedFs {
    /// Create a new restricted filesystem.
    ///
    /// `inner`: the underlying filesystem (typically `DirectFs`).
    /// `write_roots`: directories where writes/removes/renames are allowed.
    /// `extra_read_roots`: additional directories where reads are allowed,
    ///   beyond system paths and write roots.
    pub fn new(
        inner: Arc<dyn Filesystem>,
        write_roots: Vec<PathBuf>,
        extra_read_roots: Vec<PathBuf>,
    ) -> Self {
        let canonical_write_roots: Vec<PathBuf> = write_roots
            .iter()
            .map(|r| dunce::canonicalize(r).unwrap_or_else(|_| r.clone()))
            .collect();

        // Read roots = system paths + write roots + extra read roots
        let mut read_roots: Vec<PathBuf> = SYSTEM_READ_PATHS
            .iter()
            .filter_map(|p| {
                if Path::new(p).exists() {
                    dunce::canonicalize(p).ok()
                } else {
                    None
                }
            })
            .collect();
        read_roots.extend(canonical_write_roots.iter().cloned());
        read_roots.extend(
            extra_read_roots
                .iter()
                .filter_map(|r| dunce::canonicalize(r).ok()),
        );

        // Safe individual config files (e.g. ~/.gitconfig). These are exact
        // file paths - Path::starts_with is component-aware, so
        // /home/user/.gitconfig_backup does NOT match /home/user/.gitconfig.
        read_roots.extend(
            tool_paths::resolve_safe_config_files()
                .into_iter()
                .filter_map(|r| dunce::canonicalize(r).ok()),
        );

        Self {
            inner,
            canonical_read_roots: read_roots,
            canonical_write_roots,
        }
    }

    /// Convenience constructor for the common case: restrict writes to a
    /// single project root, using `DirectFs` as the underlying filesystem.
    /// Reads are allowed from system paths + tool paths + the project root.
    pub fn with_local_fs(project_root: PathBuf) -> Self {
        Self::with_local_fs_and_reads(project_root, Vec::new(), Vec::new())
    }

    /// Like [`with_local_fs`](Self::with_local_fs) but also allows reads from
    /// `extra_read_roots` (e.g. a directory holding session state outside the
    /// project root, which the agent must be able to read back even though the
    /// rest of the home directory is hidden). Writes stay limited to the
    /// project root - to also grant *write* access beyond it (e.g. `$CARGO_HOME`)
    /// use [`with_local_fs_and_reads_and_writes`](Self::with_local_fs_and_reads_and_writes).
    ///
    /// `extra_secret_dirs` lists home-relative directories (e.g. `".myapp"`)
    /// to keep out of the resolved tool roots, on top of the built-in
    /// credential directories - see `tool_paths::resolve_tool_roots`.
    pub fn with_local_fs_and_reads(
        project_root: PathBuf,
        extra_read_roots: Vec<PathBuf>,
        extra_secret_dirs: Vec<String>,
    ) -> Self {
        Self::with_local_fs_and_reads_and_writes(
            project_root,
            extra_read_roots,
            Vec::new(),
            extra_secret_dirs,
        )
    }

    /// Like [`with_local_fs_and_reads`](Self::with_local_fs_and_reads) but also
    /// grants write access to `extra_write_roots` on top of the project root.
    /// The cargo registry/cache under `$CARGO_HOME` is the motivating case:
    /// `cargo build` writes there, yet it lives outside the project root.
    /// Write roots are folded into the read set by [`new`](Self::new), so a
    /// write root is always also readable.
    ///
    /// Prefer this over the primitive [`new`](Self::new) when you want the same
    /// readable tool paths (cargo, rustup, `$PATH`) the other convenience
    /// constructors mount; `new` does not resolve those.
    pub fn with_local_fs_and_reads_and_writes(
        project_root: PathBuf,
        extra_read_roots: Vec<PathBuf>,
        extra_write_roots: Vec<PathBuf>,
        extra_secret_dirs: Vec<String>,
    ) -> Self {
        let mut read_roots = tool_paths::resolve_tool_roots(&extra_secret_dirs);
        read_roots.extend(extra_read_roots);
        let mut write_roots = vec![project_root];
        write_roots.extend(extra_write_roots);
        Self::new(Arc::new(DirectFs), write_roots, read_roots)
    }

    /// Check whether `path` is readable (under a read root).
    fn check_read_allowed(&self, path: &Path) -> Result<(), SandboxError> {
        let canonical = self.canonicalize(path)?;
        for root in &self.canonical_read_roots {
            if canonical.starts_with(root) {
                return Ok(());
            }
        }
        Err(SandboxError::PathDenied {
            path: path.to_path_buf(),
            reason: format!(
                "path resolves outside readable roots: {}",
                canonical.display()
            ),
        })
    }

    /// Check whether `path` is writable (under a write root).
    fn check_write_allowed(&self, path: &Path) -> Result<(), SandboxError> {
        let canonical = self.canonicalize(path)?;
        for root in &self.canonical_write_roots {
            if canonical.starts_with(root) {
                return Ok(());
            }
        }
        Err(SandboxError::PathDenied {
            path: path.to_path_buf(),
            reason: format!(
                "path resolves outside writable roots: {}",
                canonical.display()
            ),
        })
    }

    /// Check that a creation target (where intermediate components may
    /// not exist yet) resolves to a path within write roots.
    fn check_write_allowed_for_creation(&self, path: &Path) -> Result<(), SandboxError> {
        if let Ok(canonical) = self.canonicalize(path) {
            return self.check_against_write_roots(path, &canonical);
        }
        let mut suffix = PathBuf::new();
        let mut current = path;
        loop {
            if current.exists() || current.parent().is_none() {
                break;
            }
            if let Some(name) = current.file_name() {
                suffix = PathBuf::from(name).join(&suffix);
            }
            current = match current.parent() {
                Some(p) => p,
                None => break,
            };
        }
        let canonical_ancestor =
            dunce::canonicalize(current).map_err(|_| SandboxError::PathDenied {
                path: path.to_path_buf(),
                reason: "no existing ancestor could be canonicalized".to_string(),
            })?;
        let full_resolved = canonical_ancestor.join(&suffix);
        self.check_against_write_roots(path, &full_resolved)
    }

    fn canonicalize(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        if let Ok(canonical) = dunce::canonicalize(path) {
            return Ok(canonical);
        }
        if let Some(parent) = path.parent() {
            if let Some(name) = path.file_name() {
                if let Ok(canonical_parent) = dunce::canonicalize(parent) {
                    return Ok(canonical_parent.join(name));
                }
            }
        }
        Err(SandboxError::PathDenied {
            path: path.to_path_buf(),
            reason: "path could not be canonicalized (intermediate component may not exist)"
                .to_string(),
        })
    }

    fn check_against_write_roots(
        &self,
        original: &Path,
        canonical: &Path,
    ) -> Result<(), SandboxError> {
        for root in &self.canonical_write_roots {
            if canonical.starts_with(root) {
                return Ok(());
            }
        }
        Err(SandboxError::PathDenied {
            path: original.to_path_buf(),
            reason: format!(
                "path resolves outside writable roots: {}",
                canonical.display()
            ),
        })
    }
}

#[async_trait]
impl Filesystem for RestrictedFs {
    async fn read(&self, path: &Path) -> Result<Vec<u8>, SandboxError> {
        self.check_read_allowed(path)?;
        self.inner.read(path).await
    }

    async fn read_to_string(&self, path: &Path) -> Result<String, SandboxError> {
        self.check_read_allowed(path)?;
        self.inner.read_to_string(path).await
    }

    async fn write(&self, path: &Path, content: &[u8]) -> Result<(), SandboxError> {
        self.check_write_allowed(path)?;
        self.inner.write(path, content).await
    }

    async fn metadata(&self, path: &Path) -> Result<FileMetadata, SandboxError> {
        self.check_read_allowed(path)?;
        self.inner.metadata(path).await
    }

    async fn list_dir(&self, path: &Path) -> Result<Vec<DirEntry>, SandboxError> {
        self.check_read_allowed(path)?;
        self.inner.list_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), SandboxError> {
        self.check_write_allowed_for_creation(path)?;
        self.inner.create_dir_all(path).await
    }

    async fn remove_file(&self, path: &Path) -> Result<(), SandboxError> {
        self.check_write_allowed(path)?;
        self.inner.remove_file(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> Result<(), SandboxError> {
        self.check_write_allowed(path)?;
        self.inner.remove_dir_all(path).await
    }

    async fn rename(&self, src: &Path, dst: &Path) -> Result<(), SandboxError> {
        self.check_read_allowed(src)?;
        self.check_write_allowed(dst)?;
        self.inner.rename(src, dst).await
    }

    async fn exists(&self, path: &Path) -> bool {
        if self.check_read_allowed(path).is_err() {
            return false;
        }
        self.inner.exists(path).await
    }

    async fn walk(&self, pattern: &str, base: &Path) -> Result<Vec<PathBuf>, SandboxError> {
        self.check_read_allowed(base)?;
        self.inner.walk(pattern, base).await
    }

    async fn walk_entries(&self, base: &Path) -> Result<Vec<DirEntry>, SandboxError> {
        self.check_read_allowed(base)?;
        self.inner.walk_entries(base).await
    }

    async fn search(
        &self,
        pattern: &str,
        base: &Path,
        regex: bool,
        limit: usize,
    ) -> Result<Vec<SearchMatch>, SandboxError> {
        self.check_read_allowed(base)?;
        self.inner.search(pattern, base, regex, limit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEnv {
        _root: tempfile::TempDir,
        fs: RestrictedFs,
    }

    impl TestEnv {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            let canonical_root = dunce::canonicalize(root.path()).unwrap();
            // No extra read roots - only system paths + project root
            let fs = RestrictedFs::new(Arc::new(DirectFs), vec![canonical_root], vec![]);
            Self { _root: root, fs }
        }

        fn root(&self) -> &Path {
            self._root.path()
        }

        /// A path outside all read roots. `/root` is not in SYSTEM_READ_PATHS
        /// and is not a tool root.
        fn outside(&self, name: &str) -> PathBuf {
            PathBuf::from(format!("/root/sandbox_test_outside/{name}"))
        }
    }

    #[tokio::test]
    async fn write_inside_root_succeeds() {
        let env = TestEnv::new();
        let file = env.root().join("test.txt");
        env.fs.write(&file, b"hello").await.unwrap();
        let content = env.fs.read_to_string(&file).await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn write_outside_root_is_denied() {
        let env = TestEnv::new();
        let file = env.outside("evil.txt");
        let result = env.fs.write(&file, b"evil").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SandboxError::PathDenied { .. }
        ));
    }

    #[tokio::test]
    async fn read_inside_root_succeeds() {
        let env = TestEnv::new();
        let file = env.root().join("readme.txt");
        std::fs::write(&file, b"data").unwrap();
        let content = env.fs.read_to_string(&file).await.unwrap();
        assert_eq!(content, "data");
    }

    #[tokio::test]
    async fn extra_read_root_is_readable_but_not_writable() {
        // An extra read root (e.g. a session-state directory) is readable so
        // the agent can read those files back, but writes there stay denied
        // (they are written out-of-band by the host process).
        let project = tempfile::tempdir().unwrap();
        let reads = tempfile::tempdir().unwrap();
        let canonical_project = dunce::canonicalize(project.path()).unwrap();
        let canonical_reads = dunce::canonicalize(reads.path()).unwrap();
        let fs = RestrictedFs::new(
            Arc::new(DirectFs),
            vec![canonical_project],
            vec![canonical_reads.clone()],
        );

        let file = canonical_reads.join("plan.md");
        std::fs::write(&file, b"the plan").unwrap();
        assert_eq!(fs.read_to_string(&file).await.unwrap(), "the plan");

        let result = fs.write(&canonical_reads.join("new.md"), b"x").await;
        assert!(
            matches!(result, Err(SandboxError::PathDenied { .. })),
            "extra read roots must not be writable"
        );
    }

    #[tokio::test]
    async fn read_outside_root_is_denied() {
        let env = TestEnv::new();
        let file = env.outside("secret.txt");
        // Path check happens before filesystem access
        let result = env.fs.read_to_string(&file).await;
        assert!(result.is_err(), "read outside root should be denied");
        assert!(matches!(
            result.unwrap_err(),
            SandboxError::PathDenied { .. }
        ));
    }

    #[tokio::test]
    async fn read_system_path_succeeds() {
        if Path::new("/etc/hostname").exists() {
            let env = TestEnv::new();
            let result = env.fs.read_to_string(Path::new("/etc/hostname")).await;
            assert!(
                result.is_ok(),
                "system path /etc/hostname should be readable"
            );
        }
    }

    #[tokio::test]
    async fn exists_inside_root_returns_true() {
        let env = TestEnv::new();
        let file = env.root().join("exists.txt");
        std::fs::write(&file, b"x").unwrap();
        assert!(env.fs.exists(&file).await);
    }

    #[tokio::test]
    async fn exists_outside_root_returns_false() {
        let env = TestEnv::new();
        let file = env.outside("secret.txt");
        assert!(
            !env.fs.exists(&file).await,
            "exists() should return false for paths outside readable roots"
        );
    }

    #[tokio::test]
    async fn create_dir_inside_root_succeeds() {
        let env = TestEnv::new();
        let dir = env.root().join("sub/nested");
        env.fs.create_dir_all(&dir).await.unwrap();
        assert!(dir.is_dir());
    }

    #[tokio::test]
    async fn create_dir_outside_root_is_denied() {
        let env = TestEnv::new();
        let dir = env.outside("evil_dir");
        let result = env.fs.create_dir_all(&dir).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn remove_file_inside_root_succeeds() {
        let env = TestEnv::new();
        let file = env.root().join("temp.txt");
        std::fs::write(&file, b"x").unwrap();
        env.fs.remove_file(&file).await.unwrap();
        assert!(!file.exists());
    }

    #[tokio::test]
    async fn remove_file_outside_root_is_denied() {
        let env = TestEnv::new();
        let file = env.outside("secret.txt");
        let result = env.fs.remove_file(&file).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn rename_inside_root_succeeds() {
        let env = TestEnv::new();
        let src = env.root().join("old.txt");
        let dst = env.root().join("new.txt");
        std::fs::write(&src, b"data").unwrap();
        env.fs.rename(&src, &dst).await.unwrap();
        assert!(!src.exists());
        assert_eq!(std::fs::read_to_string(&dst).unwrap(), "data");
    }

    #[tokio::test]
    async fn rename_from_outside_root_is_denied() {
        let env = TestEnv::new();
        let src = env.outside("secret.txt");
        let dst = env.root().join("stolen.txt");
        let result = env.fs.rename(&src, &dst).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn rename_to_outside_root_is_denied() {
        let env = TestEnv::new();
        let src = env.root().join("local.txt");
        std::fs::write(&src, b"data").unwrap();
        let dst = env.outside("exfiltrated.txt");
        let result = env.fs.rename(&src, &dst).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn dot_dot_traversal_is_blocked() {
        let env = TestEnv::new();
        let root = env.root();
        let sub = root.join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        let traversal = sub.join("../../evil_dotdot.txt");
        let result = env.fs.write(&traversal, b"evil").await;
        assert!(result.is_err(), "dot-dot traversal should be blocked");
    }

    #[tokio::test]
    async fn dot_dot_traversal_through_missing_component_is_blocked() {
        let env = TestEnv::new();
        let root = env.root();
        let traversal = root.join("no_such_dir/../../../evil.txt");
        let result = env.fs.write(&traversal, b"evil").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn remove_dir_all_inside_root_succeeds() {
        let env = TestEnv::new();
        let dir = env.root().join("removeme");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), b"a").unwrap();
        env.fs.remove_dir_all(&dir).await.unwrap();
        assert!(!dir.exists());
    }

    #[tokio::test]
    async fn remove_dir_all_outside_root_is_denied() {
        let env = TestEnv::new();
        let dir = env.outside("protected");
        let result = env.fs.remove_dir_all(&dir).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn metadata_outside_root_is_denied() {
        let env = TestEnv::new();
        let file = env.outside("secret.txt");
        let result = env.fs.metadata(&file).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_dir_outside_root_is_denied() {
        let env = TestEnv::new();
        let result = env.fs.list_dir(Path::new("/root")).await;
        assert!(result.is_err());
    }
}
