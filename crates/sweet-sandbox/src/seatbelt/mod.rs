// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! macOS Seatbelt-based command runner.
//!
//! Generates a Seatbelt profile and executes commands via `sandbox-exec(1)`.
//! Seatbelt is always available on macOS - no external dependencies needed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use sweet_core::sandbox::{CommandOutput, CommandRunner, SandboxError, SandboxPolicy};

use crate::tool_paths;

/// System paths that contain no user secrets and are needed for tooling.
/// These are allowed for reading in the Seatbelt profile.
#[cfg(target_os = "macos")]
const SYSTEM_READ_PATHS: &[&str] = &[
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/private/etc",
    "/tmp",
    "/private/tmp",
    "/private/var",
    "/var",
    "/opt",
    "/System",
    "/Library",
    "/Applications",
    "/Developer",
];

/// A command runner that uses macOS Seatbelt (`sandbox-exec`) to enforce
/// filesystem and network restrictions.
///
/// Seatbelt is the native macOS sandboxing technology. It uses profiles
/// (text rules) that are generated per-command. No external tools needed -
/// `sandbox-exec` ships with every macOS installation.
pub struct SeatbeltRunner {
    /// Directories allowed for write access.
    allowed_write_roots: Vec<PathBuf>,
    /// Directories allowed for reading only - never write or exec. Used for
    /// paths the agent must read but not modify, e.g. an ancestor `.cargo`
    /// dir cargo discovers by walking up from the project root.
    extra_read_roots: Vec<PathBuf>,
    /// Tool directories resolved from `$PATH` and known locations.
    tool_roots: Vec<PathBuf>,
    /// Policy. Fixed at construction time - macOS SBPL cannot
    /// filter network by host or IP at the kernel layer (the host token
    /// must be `*` or `localhost`), so per-domain restrictions are not
    /// achievable. The honest model is a one-shot `Sandbox` vs `Restricted`.
    policy: SandboxPolicy,
}

impl SeatbeltRunner {
    pub fn new(
        allowed_write_roots: Vec<PathBuf>,
        extra_read_roots: Vec<PathBuf>,
        policy: SandboxPolicy,
        extra_secret_dirs: Vec<String>,
    ) -> Self {
        let tool_roots = tool_paths::resolve_tool_roots(&extra_secret_dirs);
        Self {
            allowed_write_roots,
            extra_read_roots,
            tool_roots,
            policy,
        }
    }

    /// Generate a Seatbelt profile string.
    fn generate_profile(&self) -> String {
        // Seatbelt uses last-match-wins semantics: the final rule that matches
        // an operation wins. We place (deny default) first so that every
        // specific (allow ...) below it takes precedence.
        //
        // The (literal "/") rule lets bash read the root directory inode
        // itself (not its descendants) - libsystem reads it during dyld
        // initialization and bash aborts with SIGABRT otherwise.
        //
        // (allow sysctl-read) silences the lockdown-mode and bootargs probes
        // bash does at startup. The denials were non-fatal, just noisy.
        let mut rules: Vec<String> = vec![
            "(deny default)".to_string(),
            "(allow file-read-metadata)".to_string(),
            "(allow file-read-data (literal \"/\"))".to_string(),
            "(allow sysctl-read)".to_string(),
        ];

        // Allow reading file contents only from system paths
        for path in SYSTEM_READ_PATHS {
            rules.push(format!(
                "(allow file-read-data (subpath \"{}\"))",
                escape_sbpl(path)
            ));
        }

        // Allow reading the project root(s)
        for root in &self.allowed_write_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow file-read-data (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow reading tool paths (cargo, rustup, node, etc.)
        for root in &self.tool_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow file-read-data (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow reading caller-supplied extra roots (read-only: no write, no
        // exec). These sit outside the project root - e.g. an ancestor
        // `.cargo` dir cargo discovers by walking up from the working dir, or
        // session-state files the agent needs to read back.
        for root in &self.extra_read_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow file-read-data (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow reading safe individual config files from $HOME (e.g. ~/.gitconfig).
        // Uses (literal ...) so only the file itself is exposed, not its directory.
        for file in tool_paths::resolve_safe_config_files() {
            if let Some(s) = file.to_str() {
                rules.push(format!(
                    "(allow file-read-data (literal \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow executing processes from system paths (bash, ls, grep, etc.)
        for path in SYSTEM_READ_PATHS {
            rules.push(format!(
                "(allow process-exec (subpath \"{}\"))",
                escape_sbpl(path)
            ));
        }

        // Allow executing from project root(s) (e.g. node_modules/.bin)
        for root in &self.allowed_write_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow process-exec (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow executing from tool paths (cargo, rustup, node, etc.)
        for root in &self.tool_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow process-exec (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }

        // Allow process forking (needed for bash subshells, pipes, etc.)
        rules.push("(allow process-fork)".to_string());

        // Allow sending signals. Without this, processes inside the sandbox
        // cannot `kill()` their own children - `cargo test`'s test binaries
        // leak background fixture processes (e.g. sweet-mcp-mock-server) on
        // teardown, which then keep the pipeline's stdout pipe open and hang
        // any consumer reading it.
        rules.push("(allow signal)".to_string());

        // Allow mach lookups (needed for DNS resolution, system services)
        rules.push("(allow mach-lookup)".to_string());

        // Allow writes only to specified roots and temp dirs
        for root in &self.allowed_write_roots {
            if let Some(s) = root.to_str() {
                rules.push(format!(
                    "(allow file-write* (subpath \"{}\"))",
                    escape_sbpl(s)
                ));
            }
        }
        rules.push("(allow file-write* (subpath \"/tmp\"))".to_string());
        rules.push("(allow file-write* (subpath \"/private/tmp\"))".to_string());
        rules.push("(allow file-write* (subpath \"/var/folders\"))".to_string());
        rules.push("(allow file-write* (subpath \"/private/var/folders\"))".to_string());

        // Allow read/write on the safe device nodes shells and tools expect.
        // `/dev/null`, `/dev/zero`, `/dev/random`, `/dev/urandom`, `/dev/tty`
        // and the `/dev/fd` / `/dev/std*` shims are routinely used by command
        // pipelines (`> /dev/null`, `</dev/null`, etc.). We deliberately do
        // NOT open `/dev` wholesale because it contains block devices
        // (`/dev/disk*`, `/dev/rdisk*`) that bypass the sandbox.
        for node in &[
            "/dev/null",
            "/dev/zero",
            "/dev/random",
            "/dev/urandom",
            "/dev/tty",
            "/dev/dtracehelper",
        ] {
            rules.push(format!(
                "(allow file-read* file-write* (literal \"{}\"))",
                escape_sbpl(node)
            ));
        }
        // /dev/fd and /dev/std{in,out,err} are symlinks into the per-process
        // file descriptor table; tools like `bash`'s process substitution use
        // them. `(subpath "/dev/fd")` covers both.
        rules.push("(allow file-read* file-write* (subpath \"/dev/fd\"))".to_string());

        // Network rules. SBPL cannot filter by host or IP at the kernel
        // layer, so the policy collapses to a binary allow/deny.
        match &self.policy {
            SandboxPolicy::Restricted => rules.push("(deny network*)".to_string()),
            SandboxPolicy::Sandbox => rules.push("(allow network*)".to_string()),
            SandboxPolicy::Off => {
                unreachable!("SeatbeltRunner must not be constructed with SandboxPolicy::Off")
            }
        }

        format!("(version 1)\n{}\n", rules.join("\n"))
    }
}

/// Escape a string for safe embedding in an SBPL double-quoted string.
fn escape_sbpl(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

#[async_trait]
impl CommandRunner for SeatbeltRunner {
    async fn run(
        &self,
        command: &str,
        cwd: Option<&Path>,
        env: Option<&HashMap<String, String>>,
    ) -> Result<CommandOutput, SandboxError> {
        let profile = self.generate_profile();

        // Use the absolute path to the system bash. PATH-lookup resolves to
        // /opt/homebrew/bin/bash on Apple Silicon Macs, which lives outside
        // the allowed paths and would be denied by Seatbelt.
        let mut cmd = tokio::process::Command::new("sandbox-exec");
        cmd.arg("-p")
            .arg(&profile)
            .arg("--")
            .arg("/bin/bash")
            .arg("-c")
            .arg(command);

        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if let Some(vars) = env {
            for (k, v) in vars {
                cmd.env(k, v);
            }
        }

        // If the future driving us is dropped (turn cancelled), SIGKILL
        // sandbox-exec rather than orphaning it to launchd.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Check whether we are already inside a Seatbelt sandbox.
    ///
    /// `sandbox-exec` cannot nest - launching a profile from within an
    /// existing profile fails with `Operation not permitted`. Tests that
    /// exercise the runner are skipped when this returns `true`.
    fn is_nested_sandbox() -> bool {
        // App Store sandboxed processes get this env var, but command-line
        // `sandbox-exec` profiles do not. Probe by attempting a no-op
        // sandbox-exec - if it fails, we're already sandboxed.
        let output = std::process::Command::new("sandbox-exec")
            .arg("-p")
            .arg("(version 1)(allow default)")
            .arg("--")
            .arg("true")
            .output();
        match output {
            Ok(out) => !out.status.success(),
            Err(_) => true,
        }
    }

    fn runner_with_cwd() -> SeatbeltRunner {
        let cwd = std::env::current_dir().unwrap();
        SeatbeltRunner::new(vec![cwd], Vec::new(), SandboxPolicy::Sandbox, Vec::new())
    }

    #[tokio::test]
    async fn bash_starts_and_runs_simple_command() {
        if is_nested_sandbox() {
            eprintln!("skipping: sandbox-exec cannot nest inside an existing Seatbelt profile");
            return;
        }
        let runner = runner_with_cwd();
        let out = runner.run("echo hello", None, None).await.unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[tokio::test]
    async fn reading_home_is_denied_without_killing_bash() {
        if is_nested_sandbox() {
            eprintln!("skipping: sandbox-exec cannot nest inside an existing Seatbelt profile");
            return;
        }
        // ~ expands to the user's home directory, which is outside the
        // project root. ls should fail with a permission error, but bash
        // itself must not be killed by a signal (would surface as -1).
        let runner = runner_with_cwd();
        let out = runner.run("ls ~", None, None).await.unwrap();
        assert_ne!(out.exit_code, -1, "bash was killed by signal");
        assert_ne!(out.exit_code, 0, "expected ls to fail on $HOME");
    }

    #[test]
    fn profile_uses_no_unsupported_network_filters() {
        // SBPL refuses every host- and IP-pinned form: `(host "...")` is
        // not a valid keyword, and `(remote tcp|ip|udp "...")` requires the
        // host token to be `*` or `localhost`. The generated profile must
        // never use any of those forms.
        let cwd = std::env::current_dir().unwrap();
        for policy in [SandboxPolicy::Sandbox, SandboxPolicy::Restricted] {
            let runner = SeatbeltRunner::new(vec![cwd.clone()], Vec::new(), policy, Vec::new());
            let profile = runner.generate_profile();
            assert!(
                !profile.contains("(host "),
                "profile must not contain (host ...) keyword:\n{profile}"
            );
            for line in profile.lines() {
                if line.contains("network-outbound") || line.contains("network-inbound") {
                    assert!(
                        !line.contains("(remote ip \"")
                            && !line.contains("(remote tcp \"")
                            && !line.contains("(remote udp \""),
                        "profile uses unsupported remote host filter: {line}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn sandboxed_process_can_kill_its_own_child() {
        // Regression test: without `(allow signal)` in the profile, a process
        // inside the sandbox cannot `kill()` its own child. `cargo test`'s
        // test binaries then leak fixture subprocesses (e.g. mock servers),
        // which keep the pipeline's stdout pipe open and hang any consumer
        // reading it.
        if is_nested_sandbox() {
            eprintln!("skipping: sandbox-exec cannot nest inside an existing Seatbelt profile");
            return;
        }
        let runner = runner_with_cwd();
        let out = runner
            .run(
                // Background a long sleep, immediately SIGKILL it. `kill -0`
                // probes whether the PID is still alive without delivering a
                // real signal. Should print `dead` once the kill takes effect.
                "sleep 30 & pid=$!; kill -9 $pid; \
                 for _ in 1 2 3 4 5 6 7 8 9 10; do \
                   if kill -0 $pid 2>/dev/null; then sleep 0.1; else echo dead; exit 0; fi; \
                 done; echo alive",
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            out.stdout.trim(),
            "dead",
            "sandboxed bash failed to kill its own child; stdout={:?} stderr={:?}",
            out.stdout,
            out.stderr,
        );
    }

    #[tokio::test]
    async fn restricted_blocks_network() {
        if is_nested_sandbox() {
            eprintln!("skipping: sandbox-exec cannot nest inside an existing Seatbelt profile");
            return;
        }
        let cwd = std::env::current_dir().unwrap();
        let runner = SeatbeltRunner::new(
            vec![cwd.clone()],
            Vec::new(),
            SandboxPolicy::Restricted,
            Vec::new(),
        );
        let out = runner
            .run(
                "curl -sS --max-time 5 https://example.com -o /dev/null; echo exit=$?",
                Some(&cwd),
                None,
            )
            .await
            .unwrap();
        assert!(
            out.stdout.contains("exit=") && !out.stdout.contains("exit=0"),
            "curl should fail when restricted; stdout: {:?}",
            out.stdout
        );
    }

    #[test]
    fn extra_read_roots_emit_read_only_rules() {
        // An extra read root must get a file-read-data allow, but never a
        // write or exec rule - it is strictly read-only.
        let cwd = std::env::current_dir().unwrap();
        let extra = PathBuf::from("/some/ancestor/.cargo");
        let runner = SeatbeltRunner::new(
            vec![cwd],
            vec![extra.clone()],
            SandboxPolicy::Sandbox,
            Vec::new(),
        );
        let profile = runner.generate_profile();
        let p = extra.to_str().unwrap();
        assert!(
            profile.contains(&format!("(allow file-read-data (subpath \"{p}\"))")),
            "extra read root must get a read-data allow:\n{profile}"
        );
        assert!(
            !profile.contains(&format!("file-write* (subpath \"{p}\")")),
            "extra read root must not be writable:\n{profile}"
        );
        assert!(
            !profile.contains(&format!("(allow process-exec (subpath \"{p}\"))")),
            "extra read root must not be executable:\n{profile}"
        );
    }

    #[tokio::test]
    async fn extra_read_root_outside_project_is_readable() {
        // A directory under $HOME that is not a tool root is hidden by default.
        // Passing it as an extra read root must make its files readable by a
        // sandboxed command (e.g. cargo reading an ancestor `.cargo/config.toml`),
        // while an identical runner without it is still denied.
        if is_nested_sandbox() {
            eprintln!("skipping: sandbox-exec cannot nest inside an existing Seatbelt profile");
            return;
        }
        let home = dirs::home_dir().expect("home directory not found");
        let dir = home.join(format!(".sweet-sandbox-eread-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("config.toml");
        std::fs::write(&file, "ok").unwrap();
        let cwd = std::env::current_dir().unwrap();
        let cmd = format!("cat {}", file.display());

        let denied = SeatbeltRunner::new(
            vec![cwd.clone()],
            Vec::new(),
            SandboxPolicy::Sandbox,
            Vec::new(),
        );
        let denied_out = denied.run(&cmd, None, None).await.unwrap();

        let allowed = SeatbeltRunner::new(
            vec![cwd],
            vec![dir.clone()],
            SandboxPolicy::Sandbox,
            Vec::new(),
        );
        let allowed_out = allowed.run(&cmd, None, None).await.unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_ne!(
            denied_out.exit_code, 0,
            "a dir under $HOME must be denied without an extra read root"
        );
        assert_eq!(
            allowed_out.exit_code, 0,
            "extra read root should permit the read; stderr: {}",
            allowed_out.stderr
        );
        assert_eq!(allowed_out.stdout.trim(), "ok");
    }
}
