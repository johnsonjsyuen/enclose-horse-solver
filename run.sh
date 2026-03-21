#!/bin/bash
set -e

cd "$(dirname "$0")"

TS_DOMAIN="johnson-minipc.mermaid-snake.ts.net"

# Download Caddy if not present
CADDY="./caddy"
if [ ! -x "$CADDY" ]; then
  echo "Downloading Caddy..."
  curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o "$CADDY"
  chmod +x "$CADDY"
  echo "Caddy downloaded."
fi

# Refresh Tailscale TLS cert
echo "Fetching Tailscale TLS cert..."
tailscale cert --cert-file ts.crt --key-file ts.key "$TS_DOMAIN"

# Build WASM
echo "Building WASM..."
./build-wasm.sh

# Build and start server
echo "Building server..."
cargo build --release --bin serve --features native

cleanup() {
  echo "Shutting down..."
  kill $SERVER_PID $CADDY_PID 2>/dev/null
  wait $SERVER_PID $CADDY_PID 2>/dev/null
}
trap cleanup EXIT

# Start backend server
echo "Starting backend on :8080..."
./target/release/serve &
SERVER_PID=$!

# Start Caddy (HTTPS on :8443)
echo "Starting Caddy on :8443..."
"$CADDY" run --config Caddyfile &
CADDY_PID=$!

echo ""
echo "============================================"
echo "  https://${TS_DOMAIN}:8443"
echo "============================================"
echo ""

wait
