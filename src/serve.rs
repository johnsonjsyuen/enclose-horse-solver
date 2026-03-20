use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

fn main() {
    let addr = "0.0.0.0:8080";
    let listener = TcpListener::bind(addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!("Serving on http://{}", addr);

    let web_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("web");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Connection error: {}", e);
                continue;
            }
        };

        let mut buf = [0u8; 4096];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request = String::from_utf8_lossy(&buf[..n]);

        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");

        // Proxy API requests to enclose.horse to avoid CORS issues
        if path.starts_with("/api/") {
            let upstream = format!("https://enclose.horse{}", path);
            eprintln!("Proxying: {}", upstream);
            match reqwest::blocking::get(&upstream) {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let body = resp.bytes().unwrap_or_default();
                    let header = format!(
                        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nCross-Origin-Resource-Policy: cross-origin\r\n\r\n",
                        status,
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                }
                Err(e) => {
                    let msg = format!("Proxy error: {}", e);
                    let header = format!(
                        "HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nCross-Origin-Resource-Policy: cross-origin\r\n\r\n",
                        msg.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(msg.as_bytes());
                }
            }
            continue;
        }

        let path = if path == "/" { "/index.html" } else { path };

        // Prevent directory traversal
        let clean = path.trim_start_matches('/');
        if clean.contains("..") {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n");
            continue;
        }

        let file_path = web_dir.join(clean);
        match std::fs::read(&file_path) {
            Ok(contents) => {
                let mime = match file_path.extension().and_then(|e| e.to_str()) {
                    Some("html") => "text/html; charset=utf-8",
                    Some("js") => "application/javascript",
                    Some("wasm") => "application/wasm",
                    Some("json") => "application/json",
                    Some("css") => "text/css",
                    _ => "application/octet-stream",
                };
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\n\r\n",
                    mime,
                    contents.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&contents);
            }
            Err(_) => {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\n\r\n");
            }
        }
    }
}
