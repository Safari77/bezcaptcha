#!/bin/bash

set -ueo pipefail

cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check
cargo vet
cargo +nightly udeps --all-targets
#cargo semver-checks check-release
cargo test --release --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo build --release
printf '\n################################\ncargo package --list\n'
cargo package --list
printf '\n################################\n'
#cargo publish --dry-run
