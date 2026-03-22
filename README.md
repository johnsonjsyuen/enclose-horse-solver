# enclose-horse-solver

A collection of solvers for the daily [enclose.horse](https://enclose.horse) puzzle, ranging from a fast heuristic to a provably optimal MIP formulation, with both CPU and GPU implementations.

## The puzzle

[enclose.horse](https://enclose.horse) is a daily puzzle where you place a limited number of walls on a grid to enclose a horse and maximize your score. The grid contains water, portals, and bonus tiles (cherries, golden apples, gems, bees, skulls) that affect scoring.

## Solvers

| Solver | Approach | Speed | Optimality | Location |
|--------|----------|-------|------------|----------|
| **CPU SA** | Simulated annealing + SIMD flood fill | ~1-3s | Near-optimal (8 restarts) | `src/` |
| **WASM** | Same SA, runs in-browser | ~1-5s | Near-optimal (4 restarts) | `src/lib.rs` + `web/` |
| **GPU SA** | SA on wgpu compute shaders | ~1-15s | Near-optimal (1024 restarts) | `gpu-solver/` |
| **Optimal** | Mixed-Integer Programming (HiGHS) | ~1-60s | Provably optimal | `optimal-solver/` |

### CPU solver (simulated annealing)

The main solver uses simulated annealing with multiple parallel restarts and SIMD-accelerated flood fill:

- **Portable SIMD** (`std::simd`) processes 8 grid rows at once for fast enclosure evaluation
- **Bitboard representation** — one `u32` per row, shifts and masks for flood fill
- **Rayon** for parallel SA restarts

```bash
# Solve today's puzzle (8 parallel SA restarts)
cargo run --release --bin enclose-horse-solver

# Solve a specific date
cargo run --release --bin enclose-horse-solver -- 2026-03-21

# Benchmark SIMD vs scalar flood fill
cargo run --release --bin enclose-horse-solver -- --bench
```

### GPU solver (wgpu + WGSL)

Runs 1024 independent SA restarts on the GPU using wgpu compute shaders. Each GPU thread executes a full SA loop with its own RNG state and bitboard flood fill. The massive parallelism compensates for shorter individual runs, often finding better solutions than the CPU solver.

```bash
# Solve today's puzzle (1024 GPU threads)
cargo run --release -p gpu-solver

# Custom thread count
cargo run --release -p gpu-solver -- --threads=2048
```

Results are verified against the CPU flood fill implementation.

### Optimal solver (MIP)

Finds **provably optimal** solutions by formulating the puzzle as a Mixed-Integer Program with single-commodity flow conservation constraints. Uses [HiGHS](https://highs.dev/). See [optimal-solver/README.md](optimal-solver/README.md) for the full MIP formulation.

```bash
cargo run --release -p optimal-solver
```

### Web (WASM)

The solver also runs entirely in-browser via WASM with Web Workers for multi-threading.

```bash
rustup component add rust-src --toolchain nightly
./build-wasm.sh
cargo run --release --bin serve   # http://localhost:8080
```

The built-in server proxies `/api/*` to `https://enclose.horse` (avoiding CORS) and sets COOP/COEP headers for SharedArrayBuffer. For mobile (HTTPS required), use `./run.sh` which sets up Caddy with TLS.

## Building

Requires Rust nightly (pinned via `rust-toolchain.toml`). Uses [mold](https://github.com/rui314/mold) as the linker for faster native builds.

```bash
cargo build --release                        # all solvers
cargo run --release --bin enclose-horse-solver  # CPU
cargo run --release -p gpu-solver              # GPU (needs Vulkan/Metal/DX12)
cargo run --release -p optimal-solver          # MIP (needs HiGHS)
cargo test                                     # 37 tests
```

## Project structure

```
├── src/
│   ├── main.rs          # CPU CLI entry point
│   ├── lib.rs           # WASM bindings
│   ├── grid.rs          # Tile types, grid parsing, bitboards
│   ├── flood_fill.rs    # SIMD flood fill + scoring
│   ├── solver.rs        # Simulated annealing
│   └── serve.rs         # Dev HTTP server
├── gpu-solver/
│   └── src/
│       ├── main.rs      # wgpu host code
│       └── shader.wgsl  # WGSL compute shader (SA + flood fill)
├── optimal-solver/
│   └── src/main.rs      # HiGHS MIP formulation
└── web/
    └── index.html       # Browser frontend
```
