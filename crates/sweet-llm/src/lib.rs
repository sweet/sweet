// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Inference provider implementations for the sweet framework.
//!
//! Each protocol family lives in its own module behind a Cargo feature so
//! consumers only pay for what they use:
//!
//! - `openai` — [`OpenAIProvider`], speaks `/v1/chat/completions`.
//! - `gemini` — [`GeminiProvider`], speaks Google's native
//!   `/v1beta/models/{model}:generateContent` protocol.
//! - `anthropic` — [`AnthropicProvider`], speaks Anthropic's native
//!   `/v1/messages` protocol.
//!
//! All features are enabled by default. Disable with
//! `default-features = false` and opt back in to just what you need.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;
#[cfg(feature = "gemini")]
pub use gemini::GeminiProvider;
#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod error;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
pub use error::ProviderError;

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod schema;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod sse;
#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod util;
