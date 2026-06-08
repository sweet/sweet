// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors
// SPDX-License-Identifier: Apache-2.0

//! Wire-format DTOs for Google Gemini's native Generative Language API.

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use sweet_core::{ContentBlock, Message, Role, ToolCall};

use crate::error::ProviderError;

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SystemInstruction {
    pub parts: Vec<TextPart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TextPart {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

/// A Gemini content part.  The API treats this as a discriminated union,
/// but `thoughtSignature` can appear as a sibling on the same object as
/// `functionCall`, so we model it as a struct with optional fields.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// Gemini 3 models emit this on `functionCall` parts.  It must be echoed
    /// back verbatim when the `functionCall` is included in history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Inline binary data sent to the model (images, documents).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
}

/// Inline binary data for a Gemini content part (images, documents).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InlineData {
    pub mime_type: String,
    /// Base64-encoded binary data.
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionCall {
    pub name: String,
    pub args: Value,
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Tool<'a> {
    pub function_declarations: Vec<FunctionDeclaration<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionDeclaration<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig {
    pub max_output_tokens: usize,
}

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentResponse {
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Candidate {
    pub content: Content,
    #[serde(default)]
    pub _finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UsageMetadata {
    pub prompt_token_count: usize,
    pub _candidates_token_count: usize,
    pub total_token_count: usize,
}

// ---------------------------------------------------------------------------
// Schema sanitization
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert sweet-core [`Message`]s into Gemini's `contents` + `systemInstruction`.
///
/// - `Role::System` messages are extracted and merged into `systemInstruction`.
/// - Consecutive `Role::Tool` messages are grouped into a single `user` content
///   with multiple `functionResponse` parts.
/// - `Role::Assistant` messages with `tool_calls` are mapped to `model` contents
///   with `functionCall` parts; `thought_signature` values are looked up from
///   the supplied map and injected into the matching parts.
pub(crate) fn convert_messages(
    messages: &[Message],
    thought_signatures: &HashMap<String, String>,
    tool_names: &HashMap<String, String>,
) -> (Option<SystemInstruction>, Vec<Content>) {
    // Extract system prompts.
    let system_parts: Vec<TextPart> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| TextPart {
            text: m.text_content(),
        })
        .collect();

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(SystemInstruction {
            parts: system_parts,
        })
    };

    let mut contents = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == Role::System {
            i += 1;
            continue;
        }

        if msg.role == Role::Tool {
            // Group consecutive tool-result messages into a single user
            // content with multiple functionResponse parts.
            let mut parts = Vec::new();
            while i < messages.len() && messages[i].role == Role::Tool {
                let tool_msg = &messages[i];
                let tc_id = tool_msg.tool_call_id.clone().unwrap_or_default();
                let name = tool_names.get(&tc_id).cloned().unwrap_or_default();
                parts.push(Part {
                    text: None,
                    function_call: None,
                    function_response: Some(FunctionResponse {
                        name,
                        response: serde_json::json!({ "result": tool_msg.text_content() }),
                    }),
                    thought_signature: None,
                    inline_data: None,
                });
                i += 1;
            }
            contents.push(Content {
                role: "user".into(),
                parts,
            });
            continue;
        }

        if msg.role == Role::User {
            if msg.has_attachments() {
                let mut parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            parts.push(Part {
                                text: Some(text.clone()),
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                                inline_data: None,
                            });
                        }
                        ContentBlock::Image { data, media_type }
                        | ContentBlock::File {
                            data, media_type, ..
                        } => {
                            parts.push(Part {
                                text: None,
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                                inline_data: Some(InlineData {
                                    mime_type: media_type.clone(),
                                    data: base64::prelude::BASE64_STANDARD.encode(data),
                                }),
                            });
                        }
                        _ => {}
                    }
                }
                if parts.is_empty() {
                    parts.push(Part {
                        text: Some(String::new()),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                        inline_data: None,
                    });
                }
                contents.push(Content {
                    role: "user".into(),
                    parts,
                });
            } else {
                contents.push(Content {
                    role: "user".into(),
                    parts: vec![Part {
                        text: Some(msg.text_content()),
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                        inline_data: None,
                    }],
                });
            }
            i += 1;
            continue;
        }

        // Role::Assistant
        let mut parts = Vec::new();
        let assistant_text = msg.text_content();
        if !assistant_text.is_empty() {
            parts.push(Part {
                text: Some(assistant_text),
                function_call: None,
                function_response: None,
                thought_signature: None,
                inline_data: None,
            });
        }
        for tc in &msg.tool_calls {
            let thought_signature = thought_signatures.get(&tc.id).cloned();
            parts.push(Part {
                text: None,
                function_call: Some(FunctionCall {
                    name: tc.name.clone(),
                    args: tc.arguments.clone(),
                    id: tc.id.clone(),
                }),
                function_response: None,
                thought_signature,
                inline_data: None,
            });
        }
        if parts.is_empty() {
            // Gemini requires non-empty parts; emit an empty text part.
            parts.push(Part {
                text: Some(String::new()),
                function_call: None,
                function_response: None,
                thought_signature: None,
                inline_data: None,
            });
        }
        contents.push(Content {
            role: "model".into(),
            parts,
        });
        i += 1;
    }

    (system_instruction, contents)
}

/// Metadata extracted from a Gemini response that must be preserved for
/// subsequent turns.
pub(crate) struct ParsedResponse {
    pub message: Message,
    /// `tool_call_id → thoughtSignature`
    pub thought_signatures: Vec<(String, String)>,
    /// `tool_call_id → function_name`
    pub tool_names: Vec<(String, String)>,
}

/// Build a [`Message`] from a Gemini response and extract metadata that must
/// be preserved for subsequent turns.
pub(crate) fn parse_response(
    resp: GenerateContentResponse,
) -> Result<ParsedResponse, ProviderError> {
    let candidate = resp
        .candidates
        .into_iter()
        .next()
        .ok_or(ProviderError::EmptyResponse)?;

    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut thought_signatures = Vec::new();
    let mut tool_names = Vec::new();

    for part in candidate.content.parts {
        if let Some(text) = part.text {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&text);
        }
        if let Some(fc) = part.function_call {
            if let Some(sig) = part.thought_signature {
                thought_signatures.push((fc.id.clone(), sig));
            }
            tool_names.push((fc.id.clone(), fc.name.clone()));
            tool_calls.push(ToolCall {
                id: fc.id,
                name: fc.name,
                arguments: fc.args,
            });
        }
    }

    let token_count = resp.usage_metadata.as_ref().map(|u| u.total_token_count);
    let context_tokens = resp.usage_metadata.as_ref().map(|u| u.prompt_token_count);

    // Gemini assistant responses only carry text in part.text; collapse the
    // joined text into a single `Text` block. If Gemini ever starts returning
    // image content here this would need to be extended.
    Ok(ParsedResponse {
        message: Message {
            role: Role::Assistant,
            content: vec![sweet_core::ContentBlock::text(content)],
            thinking_content: Vec::new(),
            tool_calls,
            tool_call_id: None,
            token_count,
            context_tokens,
            compacted: false,
        },
        thought_signatures,
        tool_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::{ContentBlock, Message, ToolCall};

    // ---------------------------------------------------------------------
    // convert_messages — the Gemini wire format requires tool-result content
    // to be a plain string inside `{"result": "..."}`, not an array of typed
    // content blocks.
    // ---------------------------------------------------------------------

    fn tool_msg(id: &str, content: &str) -> Message {
        Message::tool_result(id, content)
    }

    #[test]
    fn tool_result_response_serializes_as_plain_string() {
        let mut names = HashMap::new();
        names.insert("call_1".to_string(), "echo".to_string());

        let msgs = vec![tool_msg("call_1", "hello world")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &names);
        assert_eq!(contents.len(), 1);
        let part = &contents[0].parts[0];
        let resp = part.function_response.as_ref().unwrap();
        // result must be a JSON string, not an array of content blocks.
        assert_eq!(
            resp.response,
            serde_json::json!({ "result": "hello world" })
        );
        assert_eq!(resp.name, "echo");
    }

    #[test]
    fn tool_result_with_multiple_text_blocks_joins_them() {
        // A tool result built with multiple text blocks should be joined into
        // a single string, matching the behaviour of `Message::text_content`.
        let mut msg = Message::tool_result("call_1", "");
        msg.content = vec![ContentBlock::text("part-a"), ContentBlock::text("part-b")];

        let mut names = HashMap::new();
        names.insert("call_1".to_string(), "echo".to_string());

        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &names);
        let resp = contents[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(
            resp.response,
            serde_json::json!({ "result": "part-apart-b" })
        );
    }

    #[test]
    fn consecutive_tool_results_group_into_single_user_content() {
        let mut names = HashMap::new();
        names.insert("c1".to_string(), "a".to_string());
        names.insert("c2".to_string(), "b".to_string());

        let msgs = vec![tool_msg("c1", "r1"), tool_msg("c2", "r2")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &names);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 2);

        let r1 = contents[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(r1.response, serde_json::json!({ "result": "r1" }));
        assert_eq!(r1.name, "a");

        let r2 = contents[0].parts[1].function_response.as_ref().unwrap();
        assert_eq!(r2.response, serde_json::json!({ "result": "r2" }));
        assert_eq!(r2.name, "b");
    }

    #[test]
    fn tool_result_with_unknown_tool_call_id_uses_empty_name() {
        // Defensive: if the caller didn't supply a name mapping for a
        // tool_call_id we still emit a functionResponse rather than panicking.
        let msgs = vec![tool_msg("orphan", "data")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &HashMap::new());
        let resp = contents[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(resp.name, "");
        assert_eq!(resp.response, serde_json::json!({ "result": "data" }));
    }

    #[test]
    fn full_request_serializes_tool_result_as_plain_string() {
        // End-to-end: serialize a GenerateContentRequest and inspect the
        // raw JSON to lock in the wire shape Gemini expects.
        let mut names = HashMap::new();
        names.insert("call_1".to_string(), "echo".to_string());

        let msgs = vec![tool_msg("call_1", "ok")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &names);

        let req = GenerateContentRequest {
            system_instruction: None,
            contents,
            tools: None,
            generation_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        let result = &json["contents"][0]["parts"][0]["functionResponse"]["response"]["result"];
        assert_eq!(result, &serde_json::Value::String("ok".to_string()));
    }

    #[test]
    fn system_messages_join_into_system_instruction() {
        let msgs = vec![
            Message::system("rule one"),
            Message::system("rule two"),
            Message::user("hi"),
        ];
        let (sys, _) = convert_messages(&msgs, &HashMap::new(), &HashMap::new());
        let sys = sys.expect("system instruction emitted");
        assert_eq!(sys.parts.len(), 2);
        assert_eq!(sys.parts[0].text, "rule one");
        assert_eq!(sys.parts[1].text, "rule two");
    }

    #[test]
    fn system_message_with_multiple_blocks_joins_text() {
        // text_content() concatenates all text blocks — verify the wire
        // conversion preserves that behaviour for system messages.
        let mut sys_msg = Message::system("");
        sys_msg.content = vec![ContentBlock::text("alpha "), ContentBlock::text("beta")];
        let msgs = vec![sys_msg];
        let (sys, _) = convert_messages(&msgs, &HashMap::new(), &HashMap::new());
        let sys = sys.expect("system instruction emitted");
        assert_eq!(sys.parts.len(), 1);
        assert_eq!(sys.parts[0].text, "alpha beta");
    }

    #[test]
    fn assistant_message_with_only_tool_calls_emits_function_call_part() {
        let msg = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            name: "do_thing".to_string(),
            arguments: serde_json::json!({ "x": 1 }),
        }]);
        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &HashMap::new());

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "model");
        // No empty text part should be emitted alongside the function_call.
        assert_eq!(contents[0].parts.len(), 1);
        let fc = contents[0].parts[0].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "do_thing");
        assert_eq!(fc.id, "call_1");
    }

    #[test]
    fn assistant_message_with_text_and_tool_call_emits_both_parts() {
        let mut msg = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            name: "do_thing".to_string(),
            arguments: serde_json::json!({}),
        }]);
        msg.content = vec![ContentBlock::text("about to call")];
        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &HashMap::new());

        assert_eq!(contents[0].parts.len(), 2);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("about to call"));
        assert!(contents[0].parts[1].function_call.is_some());
    }

    #[test]
    fn user_message_emits_single_text_part() {
        let msgs = vec![Message::user("hello")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &HashMap::new());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 1);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn thought_signature_attached_to_matching_function_call() {
        let msg = Message::with_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            name: "do_thing".to_string(),
            arguments: serde_json::json!({}),
        }]);
        let mut sigs = HashMap::new();
        sigs.insert("call_1".to_string(), "sig-abc".to_string());
        let (_, contents) = convert_messages(&[msg], &sigs, &HashMap::new());
        assert_eq!(
            contents[0].parts[0].thought_signature.as_deref(),
            Some("sig-abc")
        );
    }

    #[test]
    fn user_message_with_image_produces_inline_data_part() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("describe this"),
            ContentBlock::Image {
                data: vec![1, 2, 3],
                media_type: "image/png".to_string(),
            },
        ]);
        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &HashMap::new());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 2);

        assert_eq!(contents[0].parts[0].text.as_deref(), Some("describe this"));
        assert!(contents[0].parts[0].inline_data.is_none());

        let inline = contents[0].parts[1].inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/png");
        assert_eq!(
            inline.data,
            base64::prelude::BASE64_STANDARD.encode(vec![1, 2, 3])
        );
        assert!(contents[0].parts[1].text.is_none());
    }

    #[test]
    fn user_message_image_only() {
        let msg = Message::user_blocks(vec![ContentBlock::Image {
            data: vec![0xFF; 10],
            media_type: "image/jpeg".to_string(),
        }]);
        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &HashMap::new());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].parts.len(), 1);
        let inline = contents[0].parts[0].inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/jpeg");
    }

    #[test]
    fn user_message_text_only_no_inline_data() {
        let msgs = vec![Message::user("hello")];
        let (_, contents) = convert_messages(&msgs, &HashMap::new(), &HashMap::new());
        assert_eq!(contents[0].parts.len(), 1);
        assert!(contents[0].parts[0].inline_data.is_none());
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn full_request_with_image_serializes_inline_data() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("what is this"),
            ContentBlock::Image {
                data: vec![4, 5, 6],
                media_type: "image/png".to_string(),
            },
        ]);
        let (_, contents) = convert_messages(&[msg], &HashMap::new(), &HashMap::new());
        let req = GenerateContentRequest {
            system_instruction: None,
            contents,
            tools: None,
            generation_config: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        let inline = &json["contents"][0]["parts"][1]["inlineData"];
        assert_eq!(inline["mimeType"], "image/png");
        assert_eq!(
            inline["data"],
            base64::prelude::BASE64_STANDARD.encode(vec![4, 5, 6])
        );
    }
}
