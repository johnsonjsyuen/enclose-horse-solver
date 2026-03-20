#!/bin/bash
set -e

RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 -C link-arg=--shared-memory -C link-arg=--max-memory=1073741824 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base' \
  wasm-pack build --target web \
  -- --no-default-features --features wasm \
  -Z build-std=panic_abort,std

# Copy output to web/pkg/
rm -rf web/pkg
cp -r pkg web/pkg

# Patch workerHelpers.js: browsers can't resolve bare directory imports,
# replace `import('../../..')` with explicit path to the JS entry point
find web/pkg/snippets -name 'workerHelpers.js' -exec \
  sed -i "s|import('\\.\\./\\.\\./\\.\\.')|import('../../../enclose_horse_solver.js')|g" {} \;

echo "WASM build complete (threads + SIMD128 enabled)"
