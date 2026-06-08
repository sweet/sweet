// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Native Google Gemini provider via the Generative Language API.
//!
//! Endpoint: `POST /v1beta/models/{model}:generateContent`  
//! Streaming: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`
//!
//! This module speaks the native Gemini protocol (rather than the
//! OpenAI-compatible endpoint) so it can correctly handle the
//! `thoughtSignature` fields that Gemini 3 models require for multi-turn tool
//! use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

use sweet_core::{
    Message, Model, Result as CoreResult, Role, StreamSink, ToolCall, ToolSpec, SWEET_VERSION,
};

use crate::error::ProviderError;
use crate::schema::sanitize_schema;
use crate::util::provider_error_from_core;

mod wire;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_API_KEY_ENV: &str = "GEMINI_API_KEY";
pub const DEFAULT_MODEL: &str = "gemini-3-flash-preview";
pub const DEFAULT_MAX_TOKENS: usize = 4096;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GeminiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: usize,
    user_agent: String,
    /// Map `tool_call_id → thoughtSignature` so that when the same tool call
    /// is re-injected into a later request (e.g. after the model calls it) we
    /// can echo the signature back exactly as Gemini requires.
    thought_signatures: Arc<Mutex<HashMap<String, String>>>,
    /// Map `tool_call_id → function_name` so that `functionResponse` parts in
    /// history carry the correct `name` field (required by Gemini).
    tool_names: Arc<Mutex<HashMap<String, String>>>,
}

impl GeminiProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.into(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
            tool_names: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Read the API key from the environment variable `GEMINI_API_KEY`.
    pub fn from_env() -> std::result::Result<Self, ProviderError> {
        let key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| ProviderError::MissingApiKey {
            var: DEFAULT_API_KEY_ENV,
        })?;
        Ok(Self::new(key))
    }

    /// Set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL (for proxying or testing).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the maximum number of output tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Replace the underlying HTTP client.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self
    }

    /// Overwrite the User-Agent header sent with every request.
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Prepend a token to the existing User-Agent header.
    pub fn prepend_user_agent(mut self, prefix: impl Into<String>) -> Self {
        self.user_agent = format!("{} {}", prefix.into(), self.user_agent);
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn build_request_body(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<serde_json::Value, ProviderError> {
        let sigs = self
            .thought_signatures
            .lock()
            .expect("gemini thought_signatures mutex poisoned");
        let names = self
            .tool_names
            .lock()
            .expect("gemini tool_names mutex poisoned");
        let (system_instruction, contents) = wire::convert_messages(messages, &sigs, &names);
        drop(sigs);
        drop(names);

        let tools = if tools.is_empty() {
            None
        } else {
            Some(vec![wire::Tool {
                function_declarations: tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        wire::FunctionDeclaration {
                            name: &t.name,
                            description: &t.description,
                            parameters: params,
                        }
                    })
                    .collect(),
            }])
        };

        let req = wire::GenerateContentRequest {
            system_instruction,
            contents,
            tools,
            generation_config: Some(wire::GenerationConfig {
                max_output_tokens: self.max_tokens,
            }),
        };

        Ok(serde_json::to_value(&req)?)
    }

    fn save_meta(&self, thought_sigs: Vec<(String, String)>, tool_names: Vec<(String, String)>) {
        let mut sigs = self
            .thought_signatures
            .lock()
            .expect("gemini thought_signatures mutex poisoned");
        let mut names = self
            .tool_names
            .lock()
            .expect("gemini tool_names mutex poisoned");
        for (id, sig) in thought_sigs {
            sigs.insert(id, sig);
        }
        for (id, name) in tool_names {
            names.insert(id, name);
        }
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let body = self.build_request_body(messages, tools)?;

        let url = format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        );

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-goog-api-key", &self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                body: body_text,
            });
        }

        let gemini_resp: wire::GenerateContentResponse = resp.json().await?;
        let parsed = wire::parse_response(gemini_resp)?;
        self.save_meta(parsed.thought_signatures, parsed.tool_names);
        Ok(parsed.message)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let body = self.build_request_body(messages, tools)?;

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            self.model
        );

        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-goog-api-key", &self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                body: body_text,
            });
        }

        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut text_acc = String::new();
        let mut function_call_parts: Vec<wire::Part> = Vec::new();
        let mut usage: Option<wire::UsageMetadata> = None;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);

            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                let Some(data) = crate::sse::data_lines(&event_text).next() else {
                    continue;
                };

                let gemini_chunk: wire::GenerateContentResponse = serde_json::from_str(data)
                    .map_err(|e| {
                        ProviderError::Decode(serde::de::Error::custom(format!(
                            "invalid JSON in SSE data: {e}. raw: {data}",
                        )))
                    })?;

                for candidate in gemini_chunk.candidates {
                    for part in candidate.content.parts {
                        if let Some(ref text) = part.text {
                            text_acc.push_str(text);
                            sink.on_content_delta(text)
                                .await
                                .map_err(provider_error_from_core)?;
                        }
                        if part.function_call.is_some() {
                            function_call_parts.push(part);
                        }
                    }
                }
                if let Some(u) = gemini_chunk.usage_metadata {
                    usage = Some(u);
                }
            }
        }

        let content = text_acc;
        let mut tool_calls = Vec::new();
        let mut thought_signatures = Vec::new();
        let mut tool_names = Vec::new();

        for part in function_call_parts {
            if let Some(fc) = part.function_call {
                if let Some(sig) = part.thought_signature {
                    thought_signatures.push((fc.id.clone(), sig));
                }
                tool_names.push((fc.id.clone(), fc.name.clone()));
                let call = ToolCall {
                    id: fc.id,
                    name: fc.name,
                    arguments: fc.args,
                };
                sink.on_tool_call(&call)
                    .await
                    .map_err(provider_error_from_core)?;
                tool_calls.push(call);
            }
        }

        let token_count = usage.as_ref().map(|u| u.total_token_count);
        let context_tokens = usage.as_ref().map(|u| u.prompt_token_count);

        self.save_meta(thought_signatures, tool_names);

        Ok(Message {
            role: Role::Assistant,
            content: vec![sweet_core::ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls,
            tool_call_id: None,
            token_count,
            context_tokens,
            compacted: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl Model for GeminiProvider {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> CoreResult<Message> {
        Ok(self.complete_inner(messages, tools).await?)
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> CoreResult<Message> {
        Ok(self.complete_stream_inner(messages, tools, sink).await?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = GeminiProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = GeminiProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = GeminiProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }
}
