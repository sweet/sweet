// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors

// SPDX-License-Identifier: Apache-2.0

//! Shared schema sanitization for wire-protocol providers.
//!
//! Strip schemars-specific fields that some providers reject:
//! - Removes `$schema` and `title`.
//! - Converts `type` arrays (e.g. `["string", "null"]` for `Option<T>`)
//!   into a single string, dropping `null`.

/// Strip schemars-specific fields that some providers (e.g. Baidu via
/// OpenRouter, Gemini's protobuf API) reject.
///
/// - Removes `$schema` and `title`.
/// - Converts `type` arrays (e.g. `["string", "null"]` for `Option<T>`)
///   into a single string, dropping `null`.
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value) {
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");

        if let Some(type_val) = obj.get_mut("type") {
            if let Some(arr) = type_val.as_array() {
                let chosen = arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .find(|s| *s != "null")
                    .unwrap_or("string");
                *type_val = serde_json::Value::String(chosen.into());
            }
        }

        for value in obj.values_mut() {
            sanitize_schema(value);
        }
    } else if let Some(arr) = schema.as_array_mut() {
        for item in arr {
            sanitize_schema(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_schema_and_title() {
        let mut v = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Foo",
            "type": "object",
        });
        sanitize_schema(&mut v);
        assert!(v.as_object().unwrap().get("$schema").is_none());
        assert!(v.as_object().unwrap().get("title").is_none());
    }

    #[test]
    fn collapses_type_array() {
        let mut v = serde_json::json!({
            "type": "object",
            "properties": {
                "age": { "type": ["integer", "null"] }
            }
        });
        sanitize_schema(&mut v);
        assert_eq!(v["properties"]["age"]["type"], "integer");
    }

    #[test]
    fn recurses_into_nested_properties() {
        let mut v = serde_json::json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "val": { "type": ["string", "null"] }
                    }
                }
            }
        });
        sanitize_schema(&mut v);
        assert_eq!(
            v["properties"]["nested"]["properties"]["val"]["type"],
            "string"
        );
    }
}
