# sweet

An async, trait-based AI agent framework for Rust.

Sweet provides the building blocks for building autonomous AI agents: a composable agent loop, a pluggable LLM provider layer, typed tool definitions, session management, OS-level sandboxing, and MCP (Model Context Protocol) integration.

## Crates

| Crate | Description |
|-------|-------------|
| `sweet-core` | Core traits and types: `Model`, `Message`, `ToolSpec`, `Session`, `Sandbox` |
| `sweet-tool-derive` | `#[derive(Tool)]` proc-macro for stateless tools |
| `sweet-agent` | Agent loop, hooks, subagents, handoffs, and command routing |
| `sweet-llm` | LLM provider implementations: OpenAI, Gemini, Anthropic |
| `sweet-tools` | Universal built-in tools: HTTP fetch, web search, clock |
| `sweet-session` | Session implementations: in-memory and SQLite-backed |
| `sweet-sandbox` | OS-level sandboxing: macOS Seatbelt and Linux Bubblewrap |
| `sweet-mcp` | MCP tool provider via the official `rmcp` SDK |
| `sweet-mcp-mock-server` | Mock MCP server for hermetic integration tests |

## Quick Start

Add `sweet-core`, `sweet-agent`, and a provider crate to your `Cargo.toml`:

```toml
[dependencies]
sweet-core = { git = "https://github.com/sweet/sweet", branch = "master" }
sweet-agent = { git = "https://github.com/sweet/sweet", branch = "master" }
sweet-llm = { git = "https://github.com/sweet/sweet", branch = "master" }
```

Build and run an agent:

```rust
use sweet_agent::{Agent, TurnResult};
use sweet_llm::OpenAIProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = OpenAIProvider::from_env()?
        .with_model("gpt-4o");
    let mut agent = Agent::new(model)
        .with_instructions("You are a helpful assistant.");

    // drive one turn (non-streaming)
    let reply = match agent.step("Hello!").await? {
        TurnResult::Message(msg) => msg,
        TurnResult::Handoff { .. } => unreachable!(),
    };
    println!("{}", reply.text_content());
    Ok(())
}
```

## Feature Flags

### sweet-llm

| Feature | Default | Description |
|---------|---------|-------------|
| `openai` | yes | OpenAI and OpenAI-compatible providers |
| `gemini` | yes | Google Gemini provider |
| `anthropic` | yes | Anthropic Claude provider |

### sweet-tools

| Feature | Default | Description |
|---------|---------|-------------|
| `http-fetch` | yes | HTTP GET tool |
| `time` | yes | Current timestamp tool |
| `calculator` | no | Basic math tool |
| `web-search` | no | Web search abstraction |
| `brave` | no | Brave Search backend (implies `web-search`) |
| `tavily` | no | Tavily Search backend (implies `web-search`) |

### sweet-session

| Feature | Default | Description |
|---------|---------|-------------|
| `sqlite` | no | SQLite-backed persistent session |

### sweet-agent

| Feature | Default | Description |
|---------|---------|-------------|
| `test-util` | no | `MockModel` and `VecIo` test helpers |

### sweet-sandbox

| Feature | Default | Description |
|---------|---------|-------------|
| `platform-sandbox` | yes | OS-level sandbox runners |

## License

Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Sweet project contributors

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
