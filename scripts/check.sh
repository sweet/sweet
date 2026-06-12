#!/usr/bin/env bash
set -euo pipefail

# Mirrors .github/workflows/ci.yml — same flags, same feature selection — so
# a green local check predicts a green CI run. The only intentional
# difference: CI verifies formatting with `--check`, here fmt fixes it.
export RUSTFLAGS="${RUSTFLAGS:--Dwarnings}"
export RUSTDOCFLAGS="${RUSTDOCFLAGS:--Dwarnings}"

cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo check --workspace
cargo build -p sweet-mcp-mock-server
cargo test --workspace --all-features
cargo doc --workspace --no-deps --all-features
