// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared utility functions for wire-protocol providers.

use std::time::Instant;

use crate::error::ProviderError;

/// Convert a [`sweet_core::Error`] from a stream-sink callback into a
/// [`ProviderError`]. The sink only fails if the IO layer does — surface the
/// message.
pub(crate) fn provider_error_from_core(err: sweet_core::Error) -> ProviderError {
    ProviderError::Decode(serde::de::Error::custom(err.to_string()))
}

/// Serialize a value to a JSON string for observability logging. Never panics:
/// on serialization failure, returns a descriptive error string instead.
pub(crate) fn json_string<T: serde::Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("observability serialization failed: {e}"))
}

/// Milliseconds elapsed since `started`. Saturates at `u64::MAX` on overflow
/// (which would require ~584 million years of wall time).
pub(crate) fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}
