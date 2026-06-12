# Agent Guide

Operational brief for coding agents (Claude Code, etc.) working in this repo. Read this before touching anything.

## What lives here

Sweet is an async, trait-based AI agent framework for Rust. It provides:

- **Core abstractions** (`sweet-core`): `Model`, `Message`, `ToolSpec`, `Session`, `Memory`, `Embedder`, `Sandbox`
- **Agent loop** (`sweet-agent`): orchestration, tool dispatch, hooks, subagents, handoffs
- **LLM providers** (`sweet-llm`): OpenAI, Gemini, Anthropic wire-protocol implementations
- **Universal tools** (`sweet-tools`): HTTP fetch, web search, clock
- **Session persistence** (`sweet-session`): in-memory and SQLite-backed
- **Long-term memory** (`sweet-memory`): SQLite store with hybrid FTS5 + embedding recall, memory tools
- **OS sandboxing** (`sweet-sandbox`): macOS Seatbelt, Linux Bubblewrap
- **MCP integration** (`sweet-mcp`): MCP tool provider via the `rmcp` SDK

Dependency direction (one-way — never reverse):
```
sweet-agent    → sweet-core
sweet-llm      → sweet-core
sweet-mcp      → sweet-core
sweet-session  → sweet-core
sweet-memory   → sweet-core
sweet-tools    → sweet-core (with "derive" feature)
sweet-sandbox  → sweet-core
sweet-tool-derive → (standalone proc-macro, no sweet deps)
sweet-mcp-mock-server → rmcp (test fixture only)
```

## Backward compatibility

Sweet is open source and pre-1.0 (`0.x`). The public API is a real surface that downstream crates and external users build on, so don't break it gratuitously. While pre-1.0, breaking changes are still allowed when they make the API genuinely simpler or more correct — but they must be deliberate: update every call site in this workspace, and call the break out in the commit message so consumers can adapt.

Deprecation shims and re-exports for names removed in the same change are still waste — remove them cleanly rather than leave compatibility cruft. The goal is intentional, documented breaks, not silent ones or churn for its own sake.

## Before you write a line of code

### Ask first when

- The task requires an architectural decision or tradeoff (new abstraction, new crate, changing a public API).
- The scope is unclear or the right approach has non-obvious consequences.
- You'd need to change more than one crate's public surface to complete the task.

### Proceed without asking when

- The task is localized (a bug fix, a new provider newtype, a new test).
- The change follows an established pattern (adding a provider, extending a builder, writing a wiremock test).
- The diff is small and obviously reversible.

## Quality bar

In order:

1. **Correctness** — tests must pass, including the full `--workspace` suite.
2. **Simplicity (KISS)** — the simplest solution that works. No defensive complexity, no speculative abstractions.
3. **DRY** — no copy-paste logic. Shared code lives in the right crate.
4. **Cohesion** — every module/struct/fn does one thing.
5. **Test coverage** — new behavior needs tests. New error paths need tests.

Anti-patterns to avoid:
- Half-finished implementations (no TODO stubs committed).
- Abstractions for hypothetical future requirements.
- Comments that describe what the code does rather than why a non-obvious choice was made.
- `unwrap()` in production code paths. Use `?` or a typed error.

## Security — hard rules

- **Never hardcode API keys, tokens, passwords, or private keys.** Use environment variables.
- Read credentials from env at startup (or via `from_env()`) and fail fast with a clear error if they are missing. See `ProviderError::MissingApiKey` for the established pattern.

## Established code patterns

### Error handling

- **Library crates**: use `thiserror`, define typed error enums, expose `pub type Result<T> = std::result::Result<T, Error>`.
- Error conversion between crates: implement `From<ProviderError> for CoreError` and rely on `?` to invoke it.
- No string-typed errors in library code.

### Async

- Runtime is `tokio`. Only binaries depend on it directly.
- Libraries stay runtime-agnostic — no `#[tokio::main]`, no `tokio::spawn` in library code.
- The `Model` trait uses `#[async_trait]` for dyn-compatibility. New async traits should do the same.

### Providers

- Wire-protocol provider implementations live in `sweet-llm`: `OpenAIProvider`, `GeminiProvider`, `AnthropicProvider` — one per protocol.
- An OpenAI-wire-compatible endpoint (Cerebras, OpenRouter, etc.) is `OpenAIProvider` preconfigured with a base URL — not a new provider. Such newtypes belong next to their consumer, not in `sweet-llm`.
- If the new protocol is genuinely different, create `src/<protocol>/mod.rs` and `src/<protocol>/wire.rs` (private serde DTOs in `wire.rs`).
- Gate every provider behind its own Cargo feature. Add `#[cfg(feature = "...")]` to the module declaration in `lib.rs` and `#![cfg(feature = "...")]` at the top of its integration test file.
- Follow the `DEFAULT_BASE_URL`, `DEFAULT_API_KEY_ENV`, `DEFAULT_MODEL` constant naming convention.
- Follow the builder pattern: `new(api_key)`, `from_env()`, `with_base_url()`, `with_model()`, `with_http_client()`.
- Every provider needs hermetic wiremock integration tests in `tests/<protocol>.rs`. No real network calls in tests.

### Tool patterns

- **`ToolSpec` / `ToolHandler`** live in `sweet-core`. Stateless tools use `#[derive(Tool)]` on a struct that also derives `serde::Deserialize` and `schemars::JsonSchema`.
- **Stateful tools** needing injected dependencies use the factory pattern: a `xxx_tool(dep) -> ToolSpec` function creates a private handler struct implementing `ToolHandler` manually.
- **`#[derive(Tool)]`** comes from `sweet-tool-derive`, re-exported through `sweet-core`'s `derive` feature.
- **Universal tools** (useful across domains) live in `sweet-tools`. Each tool is feature-gated.
- Tool errors are stringified into the tool-result message; the inner loop continues.

### Sandbox

All tools that interact with the filesystem or shell go through the `CommandRunner` and `Filesystem` traits in `sweet-core::sandbox`. Concrete implementations:

| Crate | Implementation | Platform |
|-------|---------------|---------|
| `sweet-core` | `DirectRunner` + `DirectFs` | All (unsandboxed) |
| `sweet-sandbox` | `SeatbeltRunner` + `RestrictedFs` | macOS |
| `sweet-sandbox` | `BubblewrapRunner` + `RestrictedFs` | Linux |

The `Sandbox` trait bundles runner + fs together to prevent mixing. `OsSandbox` in `sweet-sandbox` constructs the platform-appropriate pair. `DirectSandbox` in `sweet-core` is the unsandboxed fallback.

### Subagents

- **`SubagentSpec`** converts to a `ToolSpec` via `From`. Register on the parent with `Agent::with_subagent(spec)`.
- **`SubagentHandler::invoke`** builds and runs the child `Agent` once per call.
- **Nesting**: tracked via `tokio::task_local!`. Default cap is `DEFAULT_MAX_DEPTH = 3`. Handlers must not `tokio::spawn` the child Agent — spawning loses task-locals.

### Feature flags

- `test-util` in `sweet-agent` ships `MockModel` and `VecIo`. Downstream crates use `features = ["test-util"]` under `[dev-dependencies]` only. Never enable `test-util` in `[dependencies]`.
- `sweet-core` has a `derive` feature that pulls in `sweet-tool-derive` and enables `#[derive(Tool)]`.

### Tests

- **Unit tests**: `#[cfg(test)] mod tests { ... }` inside the source file.
- **Integration tests**: `crates/<crate>/tests/*.rs`.
- **Env-var tests**: mark `#[serial]` (from `serial_test`). Process env is global state.
- **Provider tests**: use `wiremock`. No live network calls, no real API keys.
- Async tests: `#[tokio::test]`.

## Mandatory pre-commit checklist

Run `./scripts/check.sh`, which mirrors CI (`.github/workflows/ci.yml`) exactly:

```bash
export RUSTFLAGS=-Dwarnings RUSTDOCFLAGS=-Dwarnings
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo check --workspace
cargo build -p sweet-mcp-mock-server
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
```

If you change one, change the other — drift between them is how "passes locally,
fails CI" happens (feature-gated tests skipped, doc warnings not denied).

## Git hygiene

- **Never run `git commit` or `git push` (or open a PR) without the owner's explicit approval.** A green pre-commit checklist is a prerequisite, not authorization.
- Do not amend or rewrite an existing commit on your own initiative — the owner may have pushed it. Default to a follow-up commit; if amending seems like the better call, ask first.
- Do not skip hooks (`--no-verify`). Fix the underlying issue.
- Write commit messages in the imperative mood, ≤ 72 chars for the subject.

## Crate-by-crate quick reference

### sweet-core

Public surface: `Message`, `Role`, `ToolCall`, `ThinkingContent`, `Model` (trait), `Embedder` (trait), `StreamSink`, `NoopSink`, `ToolSpec`, `ToolHandler`, `ToolFn`, `ToolError`, `Session`, `InMemorySession`, `SessionId`, `SessionError`, `MemoryItem`, `SharedSession`, `SharedSessionHandle`, `Memory` (trait), `EphemeralMemory`, `MemoryId`, `MemoryScope`, `MemoryRecord`, `MemoryQuery`, `MemoryHit`, `MemoryError`, `Error`, `Result`, `SWEET_VERSION`, `CommandRunner`, `CommandOutput`, `Filesystem`, `FileMetadata`, `DirEntry`, `SearchMatch`, `Sandbox`, `SandboxPolicy`, `SandboxError`, `DirectRunner`, `DirectFs`, `DirectSandbox`.

- `Message.thinking_content: Vec<ThinkingContent>` carries chain-of-thought blocks.
- `Session` (one conversation's transcript) vs `Memory` (durable records across conversations): both traits live here; `EphemeralMemory` is the Vec-backed default, `SqliteMemory` lives in `sweet-memory`. `MemoryScope` keys (`User`/`Project`/`Session`) are application-chosen — never model-chosen.
- `StreamSink::on_thinking_delta()` receives incremental thinking text during streaming.

### sweet-agent

Public surface: `Agent`, `AgentIo`, `RunOutcome`, `run()`, `TurnResult`, `HandoffSpec`, `HandoffHandler`, `HandoffContext`, `HandoffResult`, `SubagentSpec`, `SubagentHandler`, `SubagentContext`, `DEFAULT_MAX_DEPTH`, `Capability`, `CapabilityProvider`, `Extension`, `ExtensionRegistry`, `ToolCapabilities`, `PromptSpec`, `Activation`, `DynamicPrompt`, `HookEvent`, `HookCapability`, `HookInvocation`, `HookDispatcher`, `ProcedureSpec`, `ProcedureHandler`, `CommandSpec`, `CommandHandler`, `CommandContext`, `CommandRouter`, `MemoryRecall`, `DistillConfig`, `memory_recall_capabilities`, `memory_distill_capabilities`.

- Depends **only on `sweet-core`** — never on `sweet-llm` or any provider crate.
- `Agent::step()` appends user message, fires hooks, calls `model.complete()`, dispatches tool calls (parallel for `ReadOnly`, sequential for writes/dangerous), and returns the final assistant message.
- `ToolCapabilities` is a named bundle of `ToolSpec`s that implements `CapabilityProvider`.
- `Activation::Always` prompts are composed into system instructions every turn; `Activation::ByCommand(name)` prompts are template-only, dispatched by name.
- `DynamicPrompt` re-renders into the system prompt each turn from interior-mutable shared state.
- Long-term memory wiring (`src/memory.rs`) works against the `Memory` trait only: `memory_recall_capabilities` refreshes a `MemoryRecall` dynamic prompt from the latest user message (`BeforeTurn`); `memory_distill_capabilities` periodically extracts durable facts via a model call (`AfterTurn`, watermark-gated, dedup'd). Wire on top-level agents only — never on subagent scratch sessions.

Feature flags:

| Flag | Pulls in |
|------|---------|
| `test-util` | `MockModel`, `MockTool`, `VecIo` |

### sweet-llm

Public surface: `OpenAIProvider`, `GeminiProvider`, `AnthropicProvider`, `OpenAIEmbedder`, `GeminiEmbedder`, `ProviderError`. Per-provider thinking config: `openai::ThinkingMode`, `openai::ReasoningContent`, `anthropic::ThinkingConfig`.

- `OpenAIProvider` supports thinking-mode models via OpenAI-compatible wire format.
- `AnthropicProvider` supports native thinking blocks; configure with `AnthropicProvider::with_thinking(ThinkingConfig)`.
- Embedders follow the provider builder pattern (`new`, `from_env`, `with_base_url`, `with_model`). No Anthropic embedder — Anthropic has no embeddings API (their docs point at Voyage AI).

Feature flags:

| Flag | Pulls in |
|------|---------|
| `openai` | `OpenAIProvider`, `OpenAIEmbedder` |
| `gemini` | `GeminiProvider`, `GeminiEmbedder` |
| `anthropic` | `AnthropicProvider` |
| (default) | all of the above |

### sweet-tools

Public surface: `HttpFetch`, `CurrentTime`, `WebSearch`, `WebSearchBackend`, `SearchResult`, `WebSearchError`, `BraveBackend`, `TavilyBackend`.

Feature flags:

| Flag | Default | Pulls in |
|------|---------|---------|
| `http-fetch` | yes | `HttpFetch` |
| `time` | yes | `CurrentTime` |
| `web-search` | no | `WebSearch` abstraction |
| `brave` | no | `BraveBackend` (implies `web-search`) |
| `tavily` | no | `TavilyBackend` (implies `web-search`) |
| `calculator` | no | `Calculator` |

### sweet-session

Public surface: re-exports `Session`, `InMemorySession`, `SessionId`, `MemoryItem` from `sweet-core`; adds `SqliteSession` behind the `sqlite` feature.

`SqliteSession` keeps compacted-away rows in the same db marked `archived` (invisible to the `Session` trait) instead of deleting them; `full_messages()` / `full_items()` (inherent methods, reach them via `as_any` downcast) return the complete transcript in order. Ordering uses a fractional `position REAL` column.

### sweet-memory

Public surface: re-exports the core memory types; adds `SqliteMemory` behind the `sqlite` feature and `SqliteVecMemory` behind the `sqlite-vec` feature, plus `MemoryToolset` and the `memory_tools` / `memory_save_tool` / `memory_search_tool` / `memory_update_tool` / `memory_delete_tool` factories.

- `SqliteMemory` recall is hybrid: FTS5 bm25 + brute-force cosine over embedded rows, fused with Reciprocal Rank Fusion. Vectors are tagged with `Embedder::id()`; rows from a different embedder stay keyword-searchable only.
- `SqliteVecMemory` is like `SqliteMemory` but uses `sqlite-vec` for vector similarity search (KNN via vec0 virtual table) instead of brute-force cosine. Better suited for large-scale deployments.
- Tool scopes are bound by the application in `MemoryToolset` — search/update/delete refuse records outside `searchable_scopes`.

Feature flags:

| Flag | Pulls in |
|------|---------|
| `sqlite` | `SqliteMemory` (rusqlite, bundled FTS5) |
| `sqlite-vec` | `SqliteVecMemory` (rusqlite, bundled FTS5, sqlite-vec) |

### sweet-mcp

Public surface: `McpConfig`, `McpServerConfig`, `McpProvider`, `ToolFilter`, `McpError`.

MCP tool provider backed by `rmcp` SDK (v1.7). Connects via stdio or Streamable HTTP. Hermetic transport tests in `tests/transports.rs` use the `sweet-mcp-mock-server` fixture.

### sweet-sandbox

Public surface: `OsSandbox`, `RestrictedFs`.

Feature flags:

| Flag | Default | Description |
|------|---------|------------|
| `platform-sandbox` | yes | Platform-specific runners |

### sweet-mcp-mock-server

Test-fixture binary (`publish = false`). Exposes `echo` and `add` tools over stdio and Streamable HTTP. Not shipped — built only during CI as a hermetic test dependency.

## What to update when you change things

| Change | Also update |
|--------|------------|
| New wire-protocol provider | `sweet-llm/src/<protocol>/`, `lib.rs` export + feature gate, `tests/<protocol>.rs`, feature table in README.md and this file |
| New universal tool | `sweet-tools/src/<tool>.rs`, feature-gate in `Cargo.toml` and `lib.rs`, add test |
| New web search backend | `sweet-tools/src/web_search/<name>.rs`, feature flag, tests file |
| Public API change in sweet-core or sweet-agent | All downstream crates that use the changed API |
| New Cargo feature | Feature table in README.md and this file |
| New crate | Root `Cargo.toml` members, this file |
