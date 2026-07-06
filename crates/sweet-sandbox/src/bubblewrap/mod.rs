// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Linux Bubblewrap (`bwrap`) command runner.
//!
//! Uses `bwrap` to create a lightweight namespace sandbox. Requires `bwrap`
//! to be installed on the system (`apt install bubblewrap` / `dnf install bubblewrap`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use sweet_core::sandbox::{CommandOutput, CommandRunner, SandboxError, SandboxPolicy};

use crate::tool_paths;

/// A command runner that uses Linux Bubblewrap (`bwrap`) for namespace-based
/// sandboxing.
///
/// `bwrap` provides filesystem namespace isolation: only system paths, tool
/// paths (from `$PATH` + known tool dirs), and the project root are mounted.
/// The user's home directory secrets (`~/.ssh`, `~/.aws`, etc.) are **not**
/// accessible. Network isolation is achieved via `--unshare-net`.
///
/// # External dependency
///
/// `bwrap` must be installed and on `$PATH`. A runtime check is performed at
/// construction time with a clear error message if missing.
pub struct BubblewrapRunner {
    /// Directories allowed for write access (the project root).
    allowed_write_roots: Vec<PathBuf>,
    /// Directories allowed for reading only - never write or exec. Used for
    /// paths the agent must read but not modify, e.g. an ancestor `.cargo`
    /// dir cargo discovers by walking up from the project root.
    extra_read_roots: Vec<PathBuf>,
    /// Tool directories resolved from `$PATH` and known locations.
    tool_roots: Vec<PathBuf>,
    /// Policy. Fixed at construction time - bwrap toggles the whole
    /// network namespace with `--unshare-net`, so per-host filtering is not
    /// possible and there is no mid-session escape hatch.
    policy: SandboxPolicy,
}

impl BubblewrapRunner {
    /// Create a new Bubblewrap runner. Checks that `bwrap` is available on
    /// `$PATH` and returns an error with install instructions if not.
    pub fn new(
        allowed_write_roots: Vec<PathBuf>,
        extra_read_roots: Vec<PathBuf>,
        policy: SandboxPolicy,
        extra_secret_dirs: Vec<String>,
    ) -> Result<Self, SandboxError> {
        Self::check_bwrap_available()?;
        let tool_roots = tool_paths::resolve_tool_roots(&extra_secret_dirs);
        Ok(Self {
            allowed_write_roots,
            extra_read_roots,
            tool_roots,
            policy,
        })
    }

    fn check_bwrap_available() -> Result<(), SandboxError> {
        match which::which("bwrap") {
            Ok(_) => Ok(()),
            Err(_) => Err(SandboxError::Backend(
                "bwrap not found on $PATH. Install it with:\n\
                 \n\
                 Debian/Ubuntu: apt install bubblewrap\n\
                 Fedora/RHEL:   dnf install bubblewrap\n\
                 Arch:          pacman -S bubblewrap\n\
                 \n\
                 Or run without --sandbox to disable sandboxing."
                    .to_string(),
            )),
        }
    }

    /// Build the bwrap arguments for a command invocation.
    fn build_args(&self, command: &str, cwd: Option<&Path>) -> Vec<String> {
        let mut args = Vec::new();

        // Start with the full host filesystem read-only as a baseline.
        args.push("--ro-bind".to_string());
        args.push("/".to_string());
        args.push("/".to_string());

        // Hide the user's home directory with an empty tmpfs.
        // Tool paths are re-mounted individually below.
        if let Some(home) = dirs::home_dir().and_then(|h| h.to_str().map(String::from)) {
            args.push("--tmpfs".to_string());
            args.push(home);
        }

        // Mount tool paths read-only on top of the tmpfs.
        // These are under $HOME but are not secret directories.
        for root in &self.tool_roots {
            if let Some(s) = root.to_str() {
                if root.exists() {
                    args.push("--ro-bind".to_string());
                    args.push(s.to_string());
                    args.push(s.to_string());
                }
            }
        }

        // Mount safe individual config files from $HOME (e.g. ~/.gitconfig).
        // These are single files, not directories - only the file itself is visible.
        for file in tool_paths::resolve_safe_config_files() {
            if let Some(s) = file.to_str() {
                if file.exists() {
                    args.push("--ro-bind".to_string());
                    args.push(s.to_string());
                    args.push(s.to_string());
                }
            }
        }

        // Extra read-only roots (e.g. an ancestor `.cargo` dir cargo discovers
        // by walking up from the working dir). Mounted read-only on top of the
        // tmpfs, before the writable project bind so project writability wins
        // on any overlap.
        for root in &self.extra_read_roots {
            if let Some(s) = root.to_str() {
                if root.exists() {
                    args.push("--ro-bind".to_string());
                    args.push(s.to_string());
                    args.push(s.to_string());
                }
            }
        }

        // Writable mount for project root(s)
        for root in &self.allowed_write_roots {
            if let Some(s) = root.to_str() {
                if root.exists() {
                    args.push("--bind".to_string());
                    args.push(s.to_string());
                    args.push(s.to_string());
                }
            }
        }

        // /tmp as tmpfs (writable, needed by compilers, package managers, etc.)
        args.push("--tmpfs".to_string());
        args.push("/tmp".to_string());

        // Dev and proc for basic functionality
        args.push("--dev".to_string());
        args.push("/dev".to_string());
        args.push("--proc".to_string());
        args.push("/proc".to_string());

        // Network isolation. bwrap is binary: share the host's network
        // namespace (Allow) or unshare it (Restricted). No mid-session
        // toggle - bwrap mounts the namespace once per command and the
        // caller would need to construct a new runner to change it.
        match &self.policy {
            SandboxPolicy::Restricted => args.push("--unshare-net".to_string()),
            SandboxPolicy::Sandbox => {}
            SandboxPolicy::Off => {
                unreachable!("BubblewrapRunner must not be constructed with SandboxPolicy::Off")
            }
        }

        // Die on parent death so we don't leak sandboxed processes
        args.push("--die-with-parent".to_string());

        // Set working directory inside the namespace
        if let Some(dir) = cwd {
            if dir.is_absolute() {
                args.push("--chdir".to_string());
                args.push(dir.to_string_lossy().into_owned());
            }
        }

        // Execute the command
        args.push("--".to_string());
        args.push("bash".to_string());
        args.push("-c".to_string());
        args.push(command.to_string());

        args
    }
}

#[async_trait]
impl CommandRunner for BubblewrapRunner {
    async fn run(
        &self,
        command: &str,
        cwd: Option<&Path>,
        env: Option<&HashMap<String, String>>,
    ) -> Result<CommandOutput, SandboxError> {
        let args = self.build_args(command, cwd);

        let mut cmd = tokio::process::Command::new("bwrap");
        cmd.args(&args);

        if let Some(vars) = env {
            for (k, v) in vars {
                cmd.env(k, v);
            }
        }

        // If the future driving us is dropped (turn cancelled), SIGKILL
        // bwrap rather than orphaning it to init.
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
