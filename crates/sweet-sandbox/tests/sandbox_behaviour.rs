// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end behavioural tests for the platform sandbox.
//!
//! Each test constructs a real [`OsSandbox`] - Seatbelt on macOS, Bubblewrap
//! on Linux - and exercises both the [`CommandRunner`] (sandboxed `bash`)
//! and the [`Filesystem`] (`RestrictedFs`) interfaces, so we can observe what
//! the kernel-level enforcement actually denies. Tests skip gracefully if
//! the platform runner isn't available (e.g. no `bwrap` on a Linux box).
//!
//! Network tests reach `https://example.com` and are skipped if the host has
//! no outbound connectivity.

use std::path::{Path, PathBuf};

use sweet_core::sandbox::{Sandbox, SandboxPolicy};
use sweet_sandbox::{OsSandbox, SandboxRoots};
use tempfile::TempDir;

struct Harness {
    sandbox: OsSandbox,
    project_root: PathBuf,
    outside_root: PathBuf,
    extra_read_root: PathBuf,
    extra_write_root: PathBuf,
    _project: TempDir,
    _outside: TempDir,
    _extra_read: TempDir,
    _extra_write: TempDir,
}

fn try_harness(policy: SandboxPolicy) -> Option<Harness> {
    let home = std::env::var("HOME").ok()?;
    let home = PathBuf::from(home);

    // Place all trees under $HOME so they aren't in the system read-allow
    // list (`/tmp`, `/var/folders/...`) - that keeps "outside the project
    // root" actually outside any sandbox-permitted region on macOS.
    let project = TempDir::new_in(&home).ok()?;
    let outside = TempDir::new_in(&home).ok()?;
    let extra_read = TempDir::new_in(&home).ok()?;
    let extra_write = TempDir::new_in(&home).ok()?;

    let project_root = dunce::canonicalize(project.path()).ok()?;
    let outside_root = dunce::canonicalize(outside.path()).ok()?;
    let extra_read_root = dunce::canonicalize(extra_read.path()).ok()?;
    let extra_write_root = dunce::canonicalize(extra_write.path()).ok()?;

    std::fs::write(project_root.join("inside.txt"), b"INSIDE_MARKER\n").ok()?;
    std::fs::write(outside_root.join("secret.txt"), b"OUTSIDE_SECRET\n").ok()?;
    std::fs::write(extra_read_root.join("config.toml"), b"EXTRA_READ_MARKER\n").ok()?;
    std::fs::write(extra_write_root.join("cache.txt"), b"EXTRA_WRITE_MARKER\n").ok()?;

    // Grant read-only access to `extra_read_root` (the ancestor-`.cargo` case)
    // and read+write access to `extra_write_root` (the `$CARGO_HOME` cache
    // case), but nothing to `outside_root`, so the denial tests still exercise
    // a truly out-of-bounds directory.
    let sandbox = OsSandbox::new(
        project_root.clone(),
        policy,
        SandboxRoots {
            read: vec![extra_read_root.clone()],
            write: vec![extra_write_root.clone()],
        },
        Vec::new(),
    )
    .ok()?;

    Some(Harness {
        sandbox,
        project_root,
        outside_root,
        extra_read_root,
        extra_write_root,
        _project: project,
        _outside: outside,
        _extra_read: extra_read,
        _extra_write: extra_write,
    })
}

macro_rules! harness_or_skip {
    ($net:expr) => {
        match try_harness($net) {
            Some(h) => h,
            None => {
                eprintln!("skipping: OS sandbox unavailable (missing $HOME or platform runner)");
                return;
            }
        }
    };
}

async fn has_outbound_internet() -> bool {
    let probe = tokio::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "5",
            "-o",
            "/dev/null",
            "https://example.com",
        ])
        .output()
        .await;
    matches!(probe, Ok(o) if o.status.success())
}

// ---------------------------------------------------------------------------
// Runner: filesystem reads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_reads_project_file() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("cat inside.txt", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "INSIDE_MARKER");
}

#[tokio::test]
async fn runner_reads_system_binary_path() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "/bin/ls /usr/bin > /dev/null && echo ok",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "ok");
}

#[tokio::test]
async fn runner_denies_reads_outside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let secret = h.outside_root.join("secret.txt");
    let cmd = format!("cat {} 2>&1 | head -1; echo END", secret.to_string_lossy());
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(
        out.stdout.contains("END"),
        "bash should keep running; stderr: {}, stdout: {}",
        out.stderr,
        out.stdout
    );
    assert!(
        !out.stdout.contains("OUTSIDE_SECRET"),
        "file outside project must not be readable; output: {:?}",
        out.stdout
    );
}

#[tokio::test]
async fn runner_reads_extra_read_root() {
    // A directory passed as an extra read root - the ancestor-`.cargo` case -
    // must be readable by a sandboxed command on every platform, even though it
    // sits under $HOME and outside the project root.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let cfg = h.extra_read_root.join("config.toml");
    let cmd = format!("cat {}", cfg.to_string_lossy());
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "EXTRA_READ_MARKER");
}

#[cfg(unix)]
#[tokio::test]
async fn runner_reads_symlinked_extra_read_root() {
    // Regression for the canonicalization in OsSandbox::new: an extra read root
    // passed as a *symlink* must resolve to its real target so the runner's
    // rule/bind matches the actual (canonical) file access. Without it, seatbelt
    // would allow only the symlink's own subpath (no match for the resolved
    // file) and bubblewrap would bind at the symlink path (the real dir stays
    // hidden by the $HOME tmpfs) - either way the read would be denied.
    let home = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h),
        Err(_) => {
            eprintln!("skipping: no $HOME");
            return;
        }
    };
    // Real dir with the file, a symlink pointing at it, and a project root - all
    // under $HOME so they sit outside every default read root.
    let real = match TempDir::new_in(&home) {
        Ok(d) => d,
        Err(_) => return,
    };
    let real_root = dunce::canonicalize(real.path()).unwrap();
    std::fs::write(real_root.join("config.toml"), b"SYMLINK_MARKER\n").unwrap();

    let link_dir = TempDir::new_in(&home).unwrap();
    // Only the final `root` component is a symlink; the parent stays canonical.
    let link = dunce::canonicalize(link_dir.path()).unwrap().join("root");
    std::os::unix::fs::symlink(&real_root, &link).unwrap();

    let project = TempDir::new_in(&home).unwrap();
    let project_root = dunce::canonicalize(project.path()).unwrap();

    // Pass the *symlink* (non-canonical) as the extra read root.
    let sandbox = match OsSandbox::new(
        project_root.clone(),
        SandboxPolicy::Sandbox,
        SandboxRoots {
            read: vec![link.clone()],
            write: Vec::new(),
        },
        Vec::new(),
    ) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("skipping: OS sandbox unavailable");
            return;
        }
    };

    // Read through the *real* (resolved) path - what the kernel actually opens.
    let cfg = real_root.join("config.toml");
    let out = sandbox
        .runner()
        .run(&format!("cat {}", cfg.display()), Some(&project_root), None)
        .await
        .unwrap();
    assert_eq!(
        out.exit_code, 0,
        "a symlinked extra read root must resolve and be readable; stderr: {}",
        out.stderr
    );
    assert_eq!(out.stdout.trim(), "SYMLINK_MARKER");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn runner_denies_listing_home_on_macos() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("ls -1 ~ 2>&1; echo END", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_ne!(
        out.exit_code, -1,
        "bash was killed by signal; stdout={:?} stderr={:?}",
        out.stdout, out.stderr
    );
    assert!(
        out.stdout.contains("END"),
        "bash didn't survive; stdout={:?} stderr={:?}",
        out.stdout,
        out.stderr
    );
    let lower = out.stdout.to_lowercase();
    assert!(
        lower.contains("operation not permitted") || lower.contains("permission denied"),
        "expected denial in output; got: {}",
        out.stdout
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn runner_home_hides_secret_dirs_on_linux() {
    // bwrap masks $HOME with a tmpfs and re-mounts only the tool paths.
    // Secret dirs like .ssh / .aws / .gnupg must never appear.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("ls -a ~/ 2>&1; echo END", Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(out.stdout.contains("END"));
    for secret in &[".ssh", ".aws", ".gnupg"] {
        assert!(
            !out.stdout.split_whitespace().any(|t| t == *secret),
            "secret dir {secret} leaked into sandboxed $HOME: {}",
            out.stdout
        );
    }
}

// ---------------------------------------------------------------------------
// Runner: filesystem writes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_writes_inside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "echo WRITTEN > out.txt && cat out.txt",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "WRITTEN");
    let host_view = std::fs::read_to_string(h.project_root.join("out.txt")).unwrap();
    assert_eq!(host_view.trim(), "WRITTEN");
}

#[tokio::test]
async fn runner_denies_writes_outside_project() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let target = h.outside_root.join("evil.txt");
    let cmd = format!(
        "echo NEW_CONTENT > {} 2>&1; echo END",
        target.to_string_lossy()
    );
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(out.stdout.contains("END"));
    // From the host's perspective the file must not have been overwritten
    // with the sandboxed write.
    let host_view = std::fs::read_to_string(&target).unwrap_or_default();
    assert!(
        !host_view.contains("NEW_CONTENT"),
        "sandboxed write leaked outside project root: {host_view:?}"
    );
}

#[tokio::test]
async fn runner_denies_writes_to_extra_read_root() {
    // Extra read roots are read-only by construction: readable (see
    // `runner_reads_extra_read_root`) but never writable. A sandboxed write
    // must not reach the host file.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let target = h.extra_read_root.join("config.toml");
    let cmd = format!(
        "echo NEW_CONTENT > {} 2>&1; echo END",
        target.to_string_lossy()
    );
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert!(out.stdout.contains("END"));
    let host_view = std::fs::read_to_string(&target).unwrap_or_default();
    assert_eq!(
        host_view.trim(),
        "EXTRA_READ_MARKER",
        "sandboxed write mutated a read-only extra root: {host_view:?}"
    );
}

#[tokio::test]
async fn runner_writes_to_extra_write_root() {
    // An extra write root (the `$CARGO_HOME` cache case) must be writable by a
    // sandboxed command even though it sits under $HOME and outside the project
    // root - this is what lets `cargo build` populate its registry cache.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let target = h.extra_write_root.join("new.txt");
    let cmd = format!("echo WROTE > {} && cat {0}", target.to_string_lossy());
    let out = h
        .sandbox
        .runner()
        .run(&cmd, Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "WROTE");
    let host_view = std::fs::read_to_string(&target).unwrap();
    assert_eq!(host_view.trim(), "WROTE");
}

#[tokio::test]
async fn runner_reads_extra_write_root() {
    // Write roots are folded into the read set, so a pre-existing file under an
    // extra write root is readable too (cargo reads its cached crates back).
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let cfg = h.extra_write_root.join("cache.txt");
    let out = h
        .sandbox
        .runner()
        .run(
            &format!("cat {}", cfg.to_string_lossy()),
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "EXTRA_WRITE_MARKER");
}

// ---------------------------------------------------------------------------
// Runner: process exec
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_can_exec_system_tools() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run("/bin/echo hello-from-bin-echo", Some(&h.project_root), None)
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(out.stdout.trim(), "hello-from-bin-echo");
}

// ---------------------------------------------------------------------------
// Runner: network policy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn network_allow_reaches_internet() {
    if !has_outbound_internet().await {
        eprintln!("skipping: no outbound internet on this host");
        return;
    }
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let out = h
        .sandbox
        .runner()
        .run(
            "curl -sS --max-time 10 https://example.com | head -1",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.to_lowercase().contains("html"),
        "expected HTML body, got: {:?}",
        out.stdout
    );
}

#[tokio::test]
async fn network_restricted_blocks_by_default() {
    let h = harness_or_skip!(SandboxPolicy::Restricted);
    let out = h
        .sandbox
        .runner()
        .run(
            "curl -sS --max-time 5 -o /dev/null https://example.com; echo exit=$?",
            Some(&h.project_root),
            None,
        )
        .await
        .unwrap();
    assert!(
        out.stdout.contains("exit=") && !out.stdout.contains("exit=0"),
        "curl should fail in restricted mode; stdout: {:?}",
        out.stdout
    );
}

// Note: network policy is fixed at sandbox construction. There is no
// runtime escape hatch in either backend - a user who restricted network at
// startup must restart without the deny flag to re-enable it. The two cases
// above (`network_allow_reaches_internet`, `network_restricted_blocks_by_default`)
// cover the entire policy surface.

// ---------------------------------------------------------------------------
// Filesystem (RestrictedFs) - invoked through Sandbox::fs()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fs_reads_project_file() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let bytes = h
        .sandbox
        .fs()
        .read(&h.project_root.join("inside.txt"))
        .await
        .unwrap();
    assert_eq!(bytes, b"INSIDE_MARKER\n");
}

#[tokio::test]
async fn fs_reads_system_path_metadata() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let meta = h.sandbox.fs().metadata(Path::new("/usr/bin")).await;
    assert!(
        meta.is_ok(),
        "/usr/bin should be readable through RestrictedFs: {:?}",
        meta.err()
    );
}

#[tokio::test]
async fn fs_denies_reads_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h
        .sandbox
        .fs()
        .read(&h.outside_root.join("secret.txt"))
        .await;
    assert!(
        result.is_err(),
        "RestrictedFs::read outside project root must fail"
    );
}

#[tokio::test]
async fn fs_reads_extra_read_root() {
    // The in-process filesystem must honor the same extra read root the
    // command runner does (see `runner_reads_extra_read_root`) - the two layers
    // agree on what is readable.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let bytes = h
        .sandbox
        .fs()
        .read(&h.extra_read_root.join("config.toml"))
        .await
        .unwrap();
    assert_eq!(bytes, b"EXTRA_READ_MARKER\n");
}

#[tokio::test]
async fn fs_reads_extra_write_root() {
    // Write roots are folded into the read set, so the in-process filesystem
    // reads back a pre-existing file under an extra write root - completing the
    // matrix alongside `runner_reads_extra_write_root` and the fs write test.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let bytes = h
        .sandbox
        .fs()
        .read(&h.extra_write_root.join("cache.txt"))
        .await
        .unwrap();
    assert_eq!(bytes, b"EXTRA_WRITE_MARKER\n");
}

#[tokio::test]
async fn fs_writes_inside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let path = h.project_root.join("written_via_fs.txt");
    h.sandbox.fs().write(&path, b"hi").await.unwrap();
    let host_view = std::fs::read_to_string(&path).unwrap();
    assert_eq!(host_view, "hi");
}

#[tokio::test]
async fn fs_writes_to_extra_write_root() {
    // The in-process filesystem must honor the same extra write root the command
    // runner does (see `runner_writes_to_extra_write_root`) - the two layers
    // agree on what is writable.
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let path = h.extra_write_root.join("written_via_fs.txt");
    h.sandbox.fs().write(&path, b"hi").await.unwrap();
    let host_view = std::fs::read_to_string(&path).unwrap();
    assert_eq!(host_view, "hi");
}

#[tokio::test]
async fn fs_denies_writes_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h
        .sandbox
        .fs()
        .write(&h.outside_root.join("evil.txt"), b"evil")
        .await;
    assert!(
        result.is_err(),
        "RestrictedFs::write outside root must fail"
    );
    let host_view = std::fs::read_to_string(h.outside_root.join("evil.txt")).unwrap_or_default();
    assert!(
        !host_view.contains("evil"),
        "write should not have reached the host filesystem"
    );
}

#[tokio::test]
async fn fs_denies_dotdot_traversal_out_of_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let outside_name = h.outside_root.file_name().unwrap();
    let traversal = h
        .project_root
        .join("..")
        .join(outside_name)
        .join("secret.txt");
    let result = h.sandbox.fs().read(&traversal).await;
    assert!(
        result.is_err(),
        "dot-dot traversal out of project root must be denied"
    );
}

#[tokio::test]
async fn fs_lists_project_directory() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let entries = h.sandbox.fs().list_dir(&h.project_root).await.unwrap();
    let names: Vec<_> = entries
        .iter()
        .map(|e| {
            e.path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
        .collect();
    assert!(
        names.iter().any(|n| n == "inside.txt"),
        "expected inside.txt in list_dir output: {names:?}"
    );
}

#[tokio::test]
async fn fs_denies_listing_outside_root() {
    let h = harness_or_skip!(SandboxPolicy::Sandbox);
    let result = h.sandbox.fs().list_dir(&h.outside_root).await;
    assert!(
        result.is_err(),
        "RestrictedFs::list_dir outside root must fail"
    );
}
