// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use sweet_core::message::ToolCall;
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model, Result, ToolSpec, SWEET_VERSION};

use crate::error::ProviderError;
use crate::util::{elapsed_ms, provider_error_from_core};

mod thinking;
pub use thinking::ThinkingConfig;

mod wire;
use wire::{
    convert_messages, message_from_content_blocks, parse_response, ContentBlock, MessagesRequest,
    MessagesResponse, StreamDelta, StreamEvent, WireThinking, WireTool,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
pub const DEFAULT_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
pub const DEFAULT_MAX_TOKENS: usize = 4096;
pub const API_VERSION: &str = "2023-06-01";

/// Inference provider for Anthropic's native `/v1/messages` API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: usize,
    user_agent: String,
    thinking: Option<ThinkingConfig>,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            thinking: None,
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| ProviderError::MissingApiKey {
            var: DEFAULT_API_KEY_ENV,
        })?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    pub fn prepend_user_agent(mut self, prefix: impl Into<String>) -> Self {
        self.user_agent = format!("{} {}", prefix.into(), self.user_agent);
        self
    }

    pub fn with_thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn wire_thinking(&self) -> Option<WireThinking> {
        self.thinking.as_ref().map(|t| match t {
            ThinkingConfig::Enabled { budget_tokens } => WireThinking::Enabled {
                budget_tokens: *budget_tokens,
            },
            ThinkingConfig::Adaptive => WireThinking::Adaptive,
        })
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let (system, anthropic_messages) = convert_messages(messages);

        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| WireTool {
                        name: &t.name,
                        description: &t.description,
                        input_schema: t.parameters_schema.clone(),
                    })
                    .collect(),
            )
        };

        let body = MessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            system,
            messages: anthropic_messages,
            tools: tools_wire,
            stream: false,
            thinking: self.wire_thinking(),
        };

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-api-key", &self.api_key);
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

        let response_body = resp.text().await?;
        let parsed: MessagesResponse = serde_json::from_str(&response_body)?;
        let reply = parse_response(parsed)?;

        tracing::debug!(
            target: "sweet_llm::observability",
            event = "anthropic.complete",
            duration_ms = elapsed_ms(started),
            status = "ok",
            model = %self.model,
            "anthropic complete"
        );

        Ok(reply)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let (system, anthropic_messages) = convert_messages(messages);

        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| WireTool {
                        name: &t.name,
                        description: &t.description,
                        input_schema: t.parameters_schema.clone(),
                    })
                    .collect(),
            )
        };

        let body = MessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            system,
            messages: anthropic_messages,
            tools: tools_wire,
            stream: true,
            thinking: self.wire_thinking(),
        };

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.header("x-api-key", &self.api_key);
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

        let mut block_states: Vec<BlockState> = Vec::new();
        let mut input_tokens: Option<usize> = None;
        let mut output_tokens: Option<usize> = None;
        let mut done = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);
            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                let Some(data) = crate::sse::data_lines(&event_text).last() else {
                    continue;
                };

                let event: StreamEvent = serde_json::from_str(data)?;
                match event {
                    StreamEvent::MessageStart { message } => {
                        if let Some(usage) = message.usage {
                            input_tokens = usage.input_tokens;
                        }
                    }
                    StreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    } => {
                        if block_states.len() <= index {
                            block_states.resize_with(index + 1, || BlockState::Text(String::new()));
                        }
                        match content_block {
                            ContentBlock::Text { .. } => {
                                block_states[index] = BlockState::Text(String::new());
                            }
                            ContentBlock::ToolUse { id, name, .. }
                            | ContentBlock::ServerToolUse { id, name, .. } => {
                                block_states[index] = BlockState::ToolUse {
                                    id,
                                    name,
                                    partial_json: String::new(),
                                };
                            }
                            ContentBlock::Thinking { .. } => {
                                block_states[index] = BlockState::Thinking {
                                    text: String::new(),
                                    signature: String::new(),
                                };
                            }
                            ContentBlock::RedactedThinking { .. } | ContentBlock::Unknown => {
                                block_states[index] = BlockState::Unknown;
                            }
                        }
                    }
                    StreamEvent::ContentBlockDelta { index, delta } => {
                        if index >= block_states.len() {
                            continue;
                        }
                        match delta {
                            StreamDelta::TextDelta { text } => {
                                if let BlockState::Text(ref mut acc) = block_states[index] {
                                    acc.push_str(&text);
                                    sink.on_content_delta(&text)
                                        .await
                                        .map_err(provider_error_from_core)?;
                                }
                            }
                            StreamDelta::InputJsonDelta { partial_json } => {
                                if let BlockState::ToolUse {
                                    partial_json: ref mut acc,
                                    ..
                                } = block_states[index]
                                {
                                    acc.push_str(&partial_json);
                                }
                            }
                            StreamDelta::ThinkingDelta { thinking } => {
                                if let BlockState::Thinking { text, .. } = &mut block_states[index]
                                {
                                    text.push_str(&thinking);
                                    sink.on_thinking_delta(&thinking)
                                        .await
                                        .map_err(provider_error_from_core)?;
                                }
                            }
                            StreamDelta::SignatureDelta { signature } => {
                                if let BlockState::Thinking { signature: sig, .. } =
                                    &mut block_states[index]
                                {
                                    sig.push_str(&signature);
                                }
                            }
                            StreamDelta::Other => {}
                        }
                    }
                    StreamEvent::ContentBlockStop { index } => {
                        if index >= block_states.len() {
                            continue;
                        }
                        match &block_states[index] {
                            BlockState::ToolUse {
                                id,
                                name,
                                partial_json,
                            } => {
                                let input: Value = if partial_json.is_empty() {
                                    Value::Object(serde_json::Map::new())
                                } else {
                                    serde_json::from_str(partial_json)?
                                };
                                let call = ToolCall {
                                    id: id.clone(),
                                    name: name.clone(),
                                    arguments: input,
                                };
                                sink.on_tool_call(&call)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            }
                            BlockState::Text(_) | BlockState::Thinking { .. } => {}
                            BlockState::Unknown => {}
                        }
                    }
                    StreamEvent::MessageDelta { usage, .. } => {
                        if let Some(usage) = usage {
                            output_tokens = usage.output_tokens;
                        }
                    }
                    StreamEvent::MessageStop => {
                        done = true;
                    }
                    StreamEvent::Error { error } => {
                        return Err(ProviderError::Http {
                            status: reqwest::StatusCode::from_u16(529)
                                .unwrap_or(reqwest::StatusCode::SERVICE_UNAVAILABLE),
                            body: format!(
                                "Anthropic streaming error ({}): {}",
                                error.error_type, error.message
                            ),
                        });
                    }
                    StreamEvent::Ping => {}
                }

                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }

        let mut final_blocks = Vec::with_capacity(block_states.len());
        for state in block_states {
            match state {
                BlockState::Text(text) => {
                    final_blocks.push(ContentBlock::Text { text });
                }
                BlockState::ToolUse {
                    id,
                    name,
                    partial_json,
                } => {
                    let input: Value = if partial_json.is_empty() {
                        Value::Object(serde_json::Map::new())
                    } else {
                        serde_json::from_str(&partial_json)?
                    };
                    final_blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                BlockState::Thinking { text, signature } => {
                    final_blocks.push(ContentBlock::Thinking {
                        thinking: text,
                        signature,
                    });
                }
                BlockState::Unknown => {}
            }
        }

        let usage = input_tokens
            .zip(output_tokens)
            .map(|(input, output)| wire::Usage {
                input_tokens: Some(input),
                output_tokens: Some(output),
            });

        let reply = message_from_content_blocks(final_blocks, usage)?;

        tracing::debug!(
            target: "sweet_llm::observability",
            event = "anthropic.complete_stream",
            duration_ms = elapsed_ms(started),
            status = "ok",
            model = %self.model,
            "anthropic stream complete"
        );

        Ok(reply)
    }
}

#[async_trait]
impl Model for AnthropicProvider {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        Ok(self.complete_inner(messages, tools).await?)
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        Ok(self.complete_stream_inner(messages, tools, sink).await?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum BlockState {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        partial_json: String,
    },
    Thinking {
        text: String,
        signature: String,
    },
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = AnthropicProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = AnthropicProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = AnthropicProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }
}
