// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use async_trait::async_trait;
use futures_util::StreamExt;
use sweet_core::message::{Role, ToolCall};
use sweet_core::stream::StreamSink;
use sweet_core::{Message, Model, Result, ToolSpec, SWEET_VERSION};

use crate::error::ProviderError;
use crate::schema::sanitize_schema;
use crate::util::{elapsed_ms, json_string, provider_error_from_core};

mod reasoning;
pub use reasoning::ReasoningContent;

mod thinking;
pub use thinking::ThinkingMode;

mod wire;
use wire::{ChatRequest, StreamChunk, StreamOptions, WireMessage, WireTool, WireToolFunction};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_API_KEY_ENV: &str = "OPENAI_API_KEY";
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    context_window: Option<usize>,
    user_agent: String,
    reasoning_effort: Option<String>,
    thinking: Option<ThinkingMode>,
}

impl OpenAIProvider {
    /// Construct a provider with an explicit API key, using built-in defaults
    /// for everything else.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            context_window: None,
            user_agent: format!("sweet/{}", SWEET_VERSION),
            reasoning_effort: None,
            thinking: None,
        }
    }

    /// Construct a provider by reading the API key from the standard
    /// `OPENAI_API_KEY` environment variable.
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

    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
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

    /// Set the reasoning-effort parameter for thinking-mode models.
    ///
    /// Valid values are provider-specific (e.g. `"high"`, `"max"` for
    /// DeepSeek).  The field is only sent when set.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Configure chain-of-thought reasoning for this request.
    ///
    /// See [`ThinkingMode`] for the field semantics and the
    /// [`ENABLED`](ThinkingMode::ENABLED) / [`DISABLED`](ThinkingMode::DISABLED) /
    /// [`PRESERVED`](ThinkingMode::PRESERVED) presets. When unset, no
    /// thinking field is sent and prior `reasoning_content` is suppressed
    /// from outgoing messages.
    pub fn with_thinking(mut self, mode: ThinkingMode) -> Self {
        self.thinking = Some(mode);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Build the per-request `thinking` config, or `None` when the user has
    /// not configured one.
    fn thinking_config(&self) -> Option<wire::ThinkingConfig> {
        self.thinking.map(|m| wire::ThinkingConfig {
            r#type: if m.enabled { "enabled" } else { "disabled" },
            keep: if m.preserve_history {
                Some("all")
            } else {
                None
            },
        })
    }

    /// Whether to echo prior turns' `reasoning_content` on the wire. Only
    /// true when the user has opted into a thinking-aware backend by setting
    /// one of the reasoning parameters.
    fn echo_reasoning(&self) -> bool {
        self.thinking.is_some() || self.reasoning_effort.is_some()
    }

    fn wire_messages<'a>(&self, messages: &'a [Message]) -> Vec<WireMessage<'a>> {
        let include_reasoning = self.echo_reasoning();
        messages
            .iter()
            .map(|m| WireMessage::new(m, include_reasoning))
            .collect()
    }

    async fn complete_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> std::result::Result<Message, ProviderError> {
        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        WireTool {
                            r#type: "function",
                            function: WireToolFunction {
                                name: &t.name,
                                description: &t.description,
                                parameters: params,
                            },
                        }
                    })
                    .collect(),
            )
        };

        let body = ChatRequest {
            model: &self.model,
            messages: self.wire_messages(messages),
            tools: tools_wire,
            stream: false,
            stream_options: None,
            reasoning_effort: self.reasoning_effort.as_deref(),
            thinking: self.thinking_config(),
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let request_body = json_string(&body);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete.start",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            request_body = %request_body,
            "openai complete start"
        );

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = match req.send().await {
            Ok(resp) => resp,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    error = %err,
                    "openai complete network error"
                );
                return Err(err.into());
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let duration_ms = elapsed_ms(started);
            tracing::debug!(
                target: "sweet_llm::observability",
                event = "openai.complete",
                base_url = %self.base_url,
                endpoint = %url,
                model = %self.model,
                message_count = messages.len(),
                tool_count = tools.len(),
                duration_ms,
                status = "error",
                http_status = status.as_u16(),
                response_body = %body,
                "openai complete http error"
            );
            return Err(ProviderError::Http { status, body });
        }

        let response_body = match resp.text().await {
            Ok(body) => body,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    error = %err,
                    "openai complete response body read error"
                );
                return Err(err.into());
            }
        };
        let parsed: wire::ChatResponse = match serde_json::from_str(&response_body) {
            Ok(parsed) => parsed,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    error = %err,
                    "openai complete decode error"
                );
                return Err(err.into());
            }
        };
        let choice = match parsed.choices.into_iter().next() {
            Some(choice) => choice,
            None => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    error = %ProviderError::EmptyResponse,
                    "openai complete empty response"
                );
                return Err(ProviderError::EmptyResponse);
            }
        };
        let response_message = choice.message;
        let response_message_json = json_string(&response_message);
        let mut reply = match Message::try_from(response_message) {
            Ok(reply) => reply,
            Err(err) => {
                let duration_ms = elapsed_ms(started);
                tracing::debug!(
                    target: "sweet_llm::observability",
                    event = "openai.complete",
                    base_url = %self.base_url,
                    endpoint = %url,
                    model = %self.model,
                    message_count = messages.len(),
                    tool_count = tools.len(),
                    duration_ms,
                    status = "error",
                    http_status = status.as_u16(),
                    response_body = %response_body,
                    response_message = %response_message_json,
                    error = %err,
                    "openai complete response conversion error"
                );
                return Err(err);
            }
        };

        if let Some(usage) = parsed.usage {
            reply.token_count = Some(usage.total_tokens);
            reply.context_tokens = Some(usage.prompt_tokens);
        }

        let duration_ms = elapsed_ms(started);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            duration_ms,
            status = "ok",
            http_status = status.as_u16(),
            response_body = %response_body,
            assistant = %json_string(&reply),
            "openai complete"
        );
        Ok(reply)
    }

    async fn complete_stream_inner(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> std::result::Result<Message, ProviderError> {
        let tools_wire = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| {
                        let mut params = t.parameters_schema.clone();
                        sanitize_schema(&mut params);
                        WireTool {
                            r#type: "function",
                            function: WireToolFunction {
                                name: &t.name,
                                description: &t.description,
                                parameters: params,
                            },
                        }
                    })
                    .collect(),
            )
        };

        let body = ChatRequest {
            model: &self.model,
            messages: self.wire_messages(messages),
            tools: tools_wire,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            reasoning_effort: self.reasoning_effort.as_deref(),
            thinking: self.thinking_config(),
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete_stream.start",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            "openai stream complete start"
        );

        let started = Instant::now();
        let mut req = self
            .http
            .post(&url)
            .header("User-Agent", &self.user_agent)
            .json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let duration_ms = elapsed_ms(started);
            tracing::debug!(
                target: "sweet_llm::observability",
                event = "openai.complete_stream",
                base_url = %self.base_url,
                endpoint = %url,
                model = %self.model,
                duration_ms,
                status = "error",
                http_status = status.as_u16(),
                response_body = %body,
                "openai stream complete http error"
            );
            return Err(ProviderError::Http { status, body });
        }

        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut reasoning_content = String::new();
        // Tracked separately from `reasoning_content.is_empty()` so that an
        // explicit `reasoning_content: ""` from Kimi round-trips as a single
        // empty-text block (matches the non-streaming `TryFrom` path).
        let mut saw_reasoning_field = false;
        let mut tool_call_accums: Vec<ToolCallAccum> = Vec::new();
        let mut total_tokens: Option<usize> = None;
        let mut prompt_tokens: Option<usize> = None;
        let mut done = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);
            while let Some(result) = crate::sse::drain_event(&mut buffer) {
                let event_text = result?;
                for data in crate::sse::data_lines(&event_text) {
                    if data == "[DONE]" {
                        done = true;
                        break;
                    }
                    let chunk: StreamChunk = serde_json::from_str(data)?;
                    if let Some(usage) = chunk.usage {
                        total_tokens = Some(usage.total_tokens);
                        prompt_tokens = Some(usage.prompt_tokens);
                    }
                    for choice in chunk.choices {
                        if let Some(ref rc) = choice.delta.reasoning_content {
                            saw_reasoning_field = true;
                            if !rc.is_empty() {
                                reasoning_content.push_str(rc);
                                sink.on_thinking_delta(rc)
                                    .await
                                    .map_err(provider_error_from_core)?;
                            }
                        }
                        if !choice.delta.content.is_empty() {
                            content.push_str(&choice.delta.content);
                            sink.on_content_delta(&choice.delta.content)
                                .await
                                .map_err(provider_error_from_core)?;
                        }
                        for tc_delta in choice.delta.tool_calls {
                            if tool_call_accums.len() <= tc_delta.index {
                                tool_call_accums
                                    .resize_with(tc_delta.index + 1, ToolCallAccum::default);
                            }
                            let accum = &mut tool_call_accums[tc_delta.index];
                            if let Some(id) = tc_delta.id {
                                accum.id = id;
                            }
                            if let Some(fn_delta) = tc_delta.function {
                                if let Some(name) = fn_delta.name {
                                    accum.name.push_str(&name);
                                }
                                if let Some(args) = fn_delta.arguments {
                                    accum.arguments.push_str(&args);
                                }
                            }
                        }
                    }
                }
                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }

        let mut tool_calls = Vec::with_capacity(tool_call_accums.len());
        for accum in tool_call_accums {
            let arguments: serde_json::Value = if accum.arguments.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&accum.arguments)?
            };
            let call = ToolCall {
                id: accum.id,
                name: accum.name,
                arguments,
            };
            sink.on_tool_call(&call)
                .await
                .map_err(provider_error_from_core)?;
            tool_calls.push(call);
        }

        let mut reply = Message {
            role: Role::Assistant,
            content: vec![sweet_core::ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls,
            tool_call_id: None,
            token_count: total_tokens,
            context_tokens: prompt_tokens,
            compacted: false,
        };
        if saw_reasoning_field {
            reply.set_reasoning_content(reasoning_content);
        }

        let duration_ms = elapsed_ms(started);
        tracing::debug!(
            target: "sweet_llm::observability",
            event = "openai.complete_stream",
            base_url = %self.base_url,
            endpoint = %url,
            model = %self.model,
            duration_ms,
            status = "ok",
            http_status = status.as_u16(),
            assistant = %json_string(&reply),
            "openai stream complete"
        );
        Ok(reply)
    }
}

#[async_trait]
impl Model for OpenAIProvider {
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

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }
}

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_agent_is_sweet_version() {
        let p = OpenAIProvider::new("k");
        assert_eq!(p.user_agent, format!("sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn with_user_agent_overwrites() {
        let p = OpenAIProvider::new("k").with_user_agent("custom/1.0");
        assert_eq!(p.user_agent, "custom/1.0");
    }

    #[test]
    fn prepend_user_agent_prepends_with_space() {
        let p = OpenAIProvider::new("k").prepend_user_agent("app/1.0");
        assert_eq!(p.user_agent, format!("app/1.0 sweet/{}", SWEET_VERSION));
    }

    #[test]
    fn thinking_config_is_none_when_unset() {
        let p = OpenAIProvider::new("k");
        assert!(p.thinking_config().is_none());
    }

    #[test]
    fn enabled_preset_emits_type_enabled_without_keep() {
        let cfg = OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::ENABLED)
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.keep, None);
    }

    #[test]
    fn disabled_preset_emits_type_disabled() {
        let cfg = OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::DISABLED)
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "disabled");
        assert_eq!(cfg.keep, None);
    }

    #[test]
    fn preserved_preset_emits_enabled_with_keep_all() {
        let cfg = OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::PRESERVED)
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "enabled");
        assert_eq!(cfg.keep, Some("all"));
    }

    #[test]
    fn custom_disabled_with_preserve_history_round_trips() {
        let cfg = OpenAIProvider::new("k")
            .with_thinking(ThinkingMode {
                enabled: false,
                preserve_history: true,
            })
            .thinking_config()
            .unwrap();
        assert_eq!(cfg.r#type, "disabled");
        assert_eq!(cfg.keep, Some("all"));
    }

    #[test]
    fn echo_reasoning_off_by_default() {
        let p = OpenAIProvider::new("k");
        assert!(!p.echo_reasoning());
    }

    #[test]
    fn echo_reasoning_on_when_thinking_or_effort_set() {
        assert!(OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::ENABLED)
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::DISABLED)
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_thinking(ThinkingMode::PRESERVED)
            .echo_reasoning());
        assert!(OpenAIProvider::new("k")
            .with_reasoning_effort("high")
            .echo_reasoning());
    }
}
