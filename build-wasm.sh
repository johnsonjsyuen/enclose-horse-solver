#!/bin/bash
set -e

RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128' \
  wasm-pack build --target web \
  -- --no-default-features --features wasm \
  -Z build-std=panic_abort,std

# Copy output to web/pkg/
rm -rf web/pkg
cp -r pkg web/pkg

echo "WASM build complete (threads + SIMD128 enabled)"
