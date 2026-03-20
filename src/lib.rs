#![feature(portable_simd)]

pub mod flood_fill;
pub mod grid;
pub mod solver;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn solve_puzzle(json: &str, threaded: bool) -> Result<String, JsValue> {
    let puzzle: grid::PuzzleData = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

    let grid = grid::Grid::from_puzzle(&puzzle);
    let num_restarts = 4;
    let start = now();
    let result = solver::solve(&grid, puzzle.budget, num_restarts, threaded);
    let elapsed_ms = (now() - start) as u64;

    let mut wall_positions: Vec<(usize, usize)> = result
        .walls
        .iter()
        .map(|&idx| grid.pos_to_rc(idx))
        .collect();
    wall_positions.sort();

    let response = serde_json::json!({
        "score": result.score,
        "walls": wall_positions,
        "budget": puzzle.budget,
        "name": puzzle.name,
        "optimal_score": puzzle.optimal_score,
        "elapsed_ms": elapsed_ms,
    });

    Ok(response.to_string())
}
