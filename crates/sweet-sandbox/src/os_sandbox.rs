// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! OS-enforced sandbox - bundles an OS-restricted runner + RestrictedFs.

use std::path::PathBuf;
use std::sync::Arc;

use sweet_core::sandbox::{CommandRunner, SandboxPolicy};
use sweet_core::sandbox::{Filesystem, Sandbox, SandboxError};

use crate::restricted_fs::RestrictedFs;

/// A local sandbox that enforces OS-level restrictions.
///
/// On macOS: uses Seatbelt (`sandbox-exec`) for command isolation.
/// On Linux: uses Bubblewrap (`bwrap`) for namespace isolation.
///
/// Both platforms use [`RestrictedFs`] to enforce path boundaries on
/// filesystem operations.
///
/// Policy is fixed at construction time. Neither backend can filter network
/// by host or IP at the kernel layer, so there is no mid-session escape
/// hatch - to change the policy the caller restarts.
pub struct OsSandbox {
    runner: Arc<dyn CommandRunner>,
    fs: Arc<dyn Filesystem>,
}

impl OsSandbox {
    /// Create a new OS-enforced sandbox.
    ///
    /// `project_root` is the only directory where writes are allowed.
    /// `policy` controls sandbox and network restrictions and is fixed for
    /// the lifetime of the sandbox.
    /// `extra_read_roots` are additional directories the agent may read but not
    /// write - e.g. session state outside the project root, or an ancestor
    /// `.cargo` dir cargo discovers by walking up from the working dir. They are
    /// honored by both the in-process filesystem (file tools) and the OS command
    /// runner (sandboxed shell commands like `cargo build`), so the agent can
    /// read those files back even though the home directory is otherwise hidden.
    /// `extra_secret_dirs` lists home-relative directories (e.g. `".myapp"`)
    /// that must never be exposed to the sandbox, on top of the built-in
    /// credential directories. Use it to hide an application's own secrets.
    pub fn new(
        project_root: PathBuf,
        policy: SandboxPolicy,
        extra_read_roots: Vec<PathBuf>,
        extra_secret_dirs: Vec<String>,
    ) -> Result<Self, SandboxError> {
        let canonical_root =
            dunce::canonicalize(&project_root).unwrap_or_else(|_| project_root.clone());

        // Canonicalize the extra read roots once, at the boundary, exactly as we
        // do for `project_root`. The OS command runners match on the resolved
        // path (Seatbelt lists both `/tmp` and `/private/tmp` for this reason),
        // so a symlinked or relative root passed raw would be readable by the
        // in-process file tools (which canonicalize internally) yet denied to
        // sandboxed commands. Canonicalizing here keeps both layers in agreement.
        let extra_read_roots: Vec<PathBuf> = extra_read_roots
            .into_iter()
            .map(|r| dunce::canonicalize(&r).unwrap_or(r))
            .collect();

        let fs: Arc<dyn Filesystem> = Arc::new(RestrictedFs::with_local_fs_and_reads(
            canonical_root.clone(),
            extra_read_roots.clone(),
            extra_secret_dirs.clone(),
        ));

        Self::build(
            canonical_root,
            fs,
            policy,
            extra_read_roots,
            extra_secret_dirs,
        )
    }

    #[cfg(target_os = "macos")]
    fn build(
        canonical_root: PathBuf,
        fs: Arc<dyn Filesystem>,
        policy: SandboxPolicy,
        extra_read_roots: Vec<PathBuf>,
        extra_secret_dirs: Vec<String>,
    ) -> Result<Self, SandboxError> {
        let runner: Arc<dyn CommandRunner> = Arc::new(crate::seatbelt::SeatbeltRunner::new(
            vec![canonical_root],
            extra_read_roots,
            policy,
            extra_secret_dirs,
        ));
        Ok(Self { runner, fs })
    }

    #[cfg(target_os = "linux")]
    fn build(
        canonical_root: PathBuf,
        fs: Arc<dyn Filesystem>,
        policy: SandboxPolicy,
        extra_read_roots: Vec<PathBuf>,
        extra_secret_dirs: Vec<String>,
    ) -> Result<Self, SandboxError> {
        let runner: Arc<dyn CommandRunner> = Arc::new(crate::bubblewrap::BubblewrapRunner::new(
            vec![canonical_root],
            extra_read_roots,
            policy,
            extra_secret_dirs,
        )?);
        Ok(Self { runner, fs })
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn build(
        _canonical_root: PathBuf,
        _fs: Arc<dyn Filesystem>,
        _policy: SandboxPolicy,
        _extra_read_roots: Vec<PathBuf>,
        _extra_secret_dirs: Vec<String>,
    ) -> Result<Self, SandboxError> {
        Err(SandboxError::Backend(
            "OS sandbox is not supported on this platform.".to_string(),
        ))
    }
}

impl Sandbox for OsSandbox {
    fn runner(&self) -> Arc<dyn CommandRunner> {
        self.runner.clone()
    }

    fn fs(&self) -> Arc<dyn Filesystem> {
        self.fs.clone()
    }
}
