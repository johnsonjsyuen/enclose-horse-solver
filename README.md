# enclose-horse-solver

A high-performance solver for the daily [enclose.horse](https://enclose.horse) puzzle. It finds optimal (or near-optimal) wall placements using simulated annealing with SIMD-accelerated flood fill.

Runs both as a **native CLI** and as a **browser-based WASM app** with multi-threading support.

## How it works

[enclose.horse](https://enclose.horse) is a daily puzzle where you place walls on a grid to enclose a horse and maximize your score. The solver uses:

- **Simulated annealing** with multiple parallel restarts to explore the solution space
- **SIMD128 flood fill** (portable\_simd) for fast enclosure evaluation — bitboard-based, processing 128 bits at a time
- **Rayon** for parallel SA restarts (native and WASM via Web Workers)

The WASM frontend fetches the daily puzzle, runs the solver entirely in-browser, and renders the solution on a canvas with pixel-art tile graphics.

## Quick start

### Native CLI

Requires Rust nightly (pinned via `rust-toolchain.toml`).

```bash
# Solve today's puzzle (8 parallel SA restarts)
cargo run --release --bin enclose-horse-solver

# Solve a specific date
cargo run --release --bin enclose-horse-solver -- 2026-03-21

# Benchmark SIMD vs scalar flood fill
cargo run --release --bin enclose-horse-solver -- --bench
```

### Web (WASM)

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) and the `rust-src` component:

```bash
rustup component add rust-src --toolchain nightly
./build-wasm.sh
```

Then serve locally:

```bash
cargo run --release --bin serve
# Opens on http://localhost:8080
```

The built-in server proxies `/api/*` to `https://enclose.horse` (avoiding CORS) and sets the COOP/COEP headers required for SharedArrayBuffer.

### HTTPS (for mobile browsers)

Mobile browsers require HTTPS for `crossOriginIsolated` (needed for WASM threads). The included `run.sh` sets up Caddy as a reverse proxy with TLS:

```bash
./run.sh
# Serves on https://<hostname>:8443
```

If you're on Tailscale, it uses `tailscale cert` for a real Let's Encrypt certificate. The solver gracefully falls back to single-threaded mode if threading isn't available.

## Tests

```bash
cargo test
```
