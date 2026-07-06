// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Local OS-level sandbox implementations.
//!
//! Provides sandboxed command execution and filesystem access using
//! OS-level isolation mechanisms:
//!
//! - **macOS**: Seatbelt (`sandbox-exec`)
//! - **Linux**: Bubblewrap (`bwrap`)
//!
//! Both platforms use [`RestrictedFs`] for filesystem path enforcement.

mod os_sandbox;
mod restricted_fs;
mod tool_paths;

#[cfg(target_os = "macos")]
mod seatbelt;

#[cfg(target_os = "linux")]
mod bubblewrap;

pub use os_sandbox::{OsSandbox, SandboxRoots};
pub use restricted_fs::RestrictedFs;
