use std::env;
use std::process;

use chrono::Local;

use enclose_horse_solver::flood_fill;
use enclose_horse_solver::grid::{Grid, PuzzleData};
use enclose_horse_solver::solver;

fn main() {
    // Determine the target date: CLI arg or today.
    let args: Vec<String> = env::args().skip(1).collect();
    let bench = args.iter().any(|a| a == "--bench");
    let date = args.iter().find(|a| *a != "--bench")
        .cloned()
        .unwrap_or_else(|| Local::now().format("%Y-%m-%d").to_string());

    // Fetch the puzzle from the API.
    let url = format!("https://enclose.horse/api/daily/{}", date);
    eprintln!("Fetching puzzle from {}...", url);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Error creating HTTP client: {}", e);
            process::exit(1);
        });

    let response = client.get(&url).send().unwrap_or_else(|e| {
        eprintln!("Error fetching puzzle: {}", e);
        process::exit(1);
    });

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        eprintln!("No puzzle found for date {}", date);
        process::exit(1);
    }

    if !response.status().is_success() {
        eprintln!("API returned status {}", response.status());
        process::exit(1);
    }

    let body = response.text().unwrap_or_else(|e| {
        eprintln!("Error reading response body: {}", e);
        process::exit(1);
    });

    let puzzle: PuzzleData = serde_json::from_str(&body).unwrap_or_else(|e| {
        eprintln!("Error parsing puzzle JSON: {}", e);
        process::exit(1);
    });

    // Parse the grid.
    let grid = Grid::from_puzzle(&puzzle);
    let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);

    eprintln!(
        "Grid: {}x{}, horse at ({},{}), budget: {}, candidates: {}",
        grid.width,
        grid.height,
        horse_col,
        horse_row,
        puzzle.budget,
        grid.grass_indices.len()
    );

    // Benchmark SIMD vs scalar flood fill if --bench flag is passed.
    if bench {
        use std::time::Instant;
        use rand::prelude::*;
        use rand::rngs::SmallRng;

        let mut rng = SmallRng::from_rng(&mut rand::rng());
        let candidates = &grid.grass_indices;
        let iters = 500_000;

        // Generate random wall configs
        let configs: Vec<Vec<usize>> = (0..iters)
            .map(|_| candidates.choose_multiple(&mut rng, puzzle.budget).copied().collect())
            .collect();

        // SIMD
        let mut ff = flood_fill::FloodFillState::new(&grid);
        let start = Instant::now();
        let mut simd_sum = 0i64;
        for walls in &configs {
            simd_sum += ff.evaluate(&grid, walls).score as i64;
        }
        let simd_time = start.elapsed();

        // Scalar
        let start = Instant::now();
        let mut scalar_sum = 0i64;
        for walls in &configs {
            scalar_sum += flood_fill::evaluate_scalar(&grid, walls).score as i64;
        }
        let scalar_time = start.elapsed();

        assert_eq!(simd_sum, scalar_sum, "SIMD and scalar results must match");
        eprintln!("Flood fill benchmark: {} iterations on {}x{} grid, budget={}", iters, grid.width, grid.height, puzzle.budget);
        eprintln!("  SIMD:   {:?} ({:.0} evals/sec)", simd_time, iters as f64 / simd_time.as_secs_f64());
        eprintln!("  Scalar: {:?} ({:.0} evals/sec)", scalar_time, iters as f64 / scalar_time.as_secs_f64());
        eprintln!("  Speedup: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());

        // Also benchmark full SA: 4 sequential restarts (matching WASM config)
        let sa_start = std::time::Instant::now();
        let sa_result = solver::solve(&grid, puzzle.budget, 4, true);
        let sa_time = sa_start.elapsed();
        eprintln!("\nSA benchmark: 4 restarts (parallel+SIMD), score={}", sa_result.score);
        eprintln!("  Time: {:?}", sa_time);
        return;
    }

    // Run the solver.
    let num_restarts = 8;
    eprintln!("Running solver with {} parallel restarts...", num_restarts);
    let result = solver::solve(&grid, puzzle.budget, num_restarts, true);

    // Format wall positions as (row, col).
    let wall_positions: Vec<(usize, usize)> = result
        .walls
        .iter()
        .map(|&idx| grid.pos_to_rc(idx))
        .collect();

    // Print results.
    println!(
        "Puzzle: \"{}\" (Day {}, {})",
        puzzle.name,
        puzzle.day_number.unwrap_or(0),
        puzzle.daily_date.as_deref().unwrap_or("unknown"),
    );
    println!("Budget: {} walls", puzzle.budget);
    println!(
        "Optimal: {}",
        puzzle
            .optimal_score
            .map_or("unknown".to_string(), |s| s.to_string())
    );
    println!();
    println!("Solution found: score = {}", result.score);

    let wall_strs: Vec<String> = wall_positions
        .iter()
        .map(|(r, c)| format!("({},{})", r, c))
        .collect();
    println!("Walls at: {}", wall_strs.join(" "));
    println!();

    // Print the map with W markers for placed walls.
    println!("Map:");
    let wall_set: std::collections::HashSet<usize> =
        result.walls.iter().copied().collect();
    let rows: Vec<&str> = puzzle.map.lines().collect();
    for (r, row_str) in rows.iter().enumerate() {
        let line: String = row_str
            .chars()
            .enumerate()
            .map(|(c, ch)| {
                let idx = r * grid.width + c;
                if wall_set.contains(&idx) {
                    'W'
                } else {
                    ch
                }
            })
            .collect();
        println!("{}", line);
    }
}
