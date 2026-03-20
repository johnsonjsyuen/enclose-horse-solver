use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use crate::flood_fill::FloodFillState;
use crate::grid::Grid;

/// Result of a solver run: the best wall placement found and its score.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Flattened grid indices of wall positions.
    pub walls: Vec<usize>,
    /// Enclosure score (0 if the horse can reach a border).
    pub score: i32,
}

/// SA tuning parameters (from spec Section 3).
const INITIAL_TEMPERATURE: f64 = 50.0;
const COOLING_RATE: f64 = 0.9997;
const MIN_TEMPERATURE: f64 = 0.01;
const ITERATIONS_PER_TEMP: usize = 100;

/// Run `num_restarts` independent simulated annealing runs and return the best
/// result found across all restarts. Uses rayon for parallelism on native targets.
pub fn solve(grid: &Grid, budget: usize, num_restarts: usize, threaded: bool) -> SolveResult {
    if threaded {
        (0..num_restarts)
            .into_par_iter()
            .map(|_| sa_single_run(grid, budget))
            .max_by_key(|r| r.score)
            .unwrap()
    } else {
        (0..num_restarts)
            .map(|_| sa_single_run(grid, budget))
            .max_by_key(|r| r.score)
            .unwrap()
    }
}

/// Perform a single simulated annealing run for wall placement optimization.
///
/// Uses a boolean vec for O(1) wall membership checks and [`SmallRng`] for fast
/// random number generation.
fn sa_single_run(grid: &Grid, budget: usize) -> SolveResult {
    let num_cells = grid.width * grid.height;
    let mut rng = SmallRng::from_rng(&mut rand::rng());
    let mut ff_state = FloodFillState::new(grid);

    // Handle edge case: no walls to place.
    if budget == 0 {
        let result = ff_state.evaluate(grid, &[]);
        return SolveResult {
            walls: Vec::new(),
            score: result.score,
        };
    }

    // Handle edge case: fewer candidates than budget.
    let candidates = &grid.grass_indices;
    if candidates.len() <= budget {
        let result = ff_state.evaluate(grid, candidates);
        return SolveResult {
            walls: candidates.clone(),
            score: result.score,
        };
    }

    // --- Initialization ---
    // Pick `budget` random positions from grass_indices as initial walls.
    let mut walls: Vec<usize> = candidates
        .choose_multiple(&mut rng, budget)
        .copied()
        .collect();

    // Boolean vec indexed by position for O(1) wall membership check.
    let mut is_wall = vec![false; num_cells];
    for &w in &walls {
        is_wall[w] = true;
    }

    // Evaluate initial state.
    let initial_result = ff_state.evaluate(grid, &walls);
    let mut current_score = initial_result.score;
    let mut best_score = current_score;
    let mut best_walls = walls.clone();

    // --- SA Loop ---
    let mut temperature = INITIAL_TEMPERATURE;

    while temperature > MIN_TEMPERATURE {
        for _ in 0..ITERATIONS_PER_TEMP {
            // Pick a random wall to relocate.
            let wall_idx = rng.random_range(0..budget);
            let old_pos = walls[wall_idx];

            // Pick a new position: random candidate tile not currently a wall.
            let candidate = loop {
                let c = candidates[rng.random_range(0..candidates.len())];
                if !is_wall[c] {
                    break c;
                }
            };

            // Apply the move.
            walls[wall_idx] = candidate;
            is_wall[old_pos] = false;
            is_wall[candidate] = true;

            // Evaluate the new configuration.
            let new_result = ff_state.evaluate(grid, &walls);
            let new_score = new_result.score;

            // Metropolis acceptance criterion (higher score is better).
            let delta = new_score - current_score;
            if delta > 0 || rng.random::<f64>() < (delta as f64 / temperature).exp() {
                // Accept the move.
                current_score = new_score;
                if current_score > best_score {
                    best_score = current_score;
                    best_walls = walls.clone();
                }
            } else {
                // Revert the move.
                walls[wall_idx] = old_pos;
                is_wall[candidate] = false;
                is_wall[old_pos] = true;
            }
        }

        temperature *= COOLING_RATE;
    }

    SolveResult {
        walls: best_walls,
        score: best_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::grid::PuzzleData;

    /// Helper to build a PuzzleData from a map string.
    fn puzzle(map: &str, budget: usize) -> PuzzleData {
        PuzzleData {
            id: "test".into(),
            map: map.into(),
            budget,
            name: "Test".into(),
            optimal_score: None,
            has_bonus: None,
            bonus_type: None,
            daily_date: None,
            day_number: None,
        }
    }

    /// 5x5 all-grass grid with horse in the center.
    fn small_grid() -> Grid {
        let data = puzzle(".....\n.....\n..H..\n.....\n.....", 0);
        Grid::from_puzzle(&data)
    }

    #[test]
    fn solve_finds_nonzero_score() {
        let grid = small_grid();
        let result = solve(&grid, 8, 4, true);
        assert!(
            result.score > 0,
            "SA should find an enclosure, got score={}",
            result.score
        );
    }

    #[test]
    fn result_has_correct_wall_count() {
        let grid = small_grid();
        let result = solve(&grid, 6, 2, true);
        assert_eq!(
            result.walls.len(),
            6,
            "Result should have exactly 6 walls, got {}",
            result.walls.len()
        );
    }

    #[test]
    fn walls_are_on_valid_positions() {
        let grid = small_grid();
        let result = solve(&grid, 4, 2, true);
        for &w in &result.walls {
            assert_ne!(w, grid.horse_pos, "Wall placed on horse position");
        }
    }

    #[test]
    fn walls_are_unique() {
        let grid = small_grid();
        let result = solve(&grid, 6, 2, true);
        let mut sorted = result.walls.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            result.walls.len(),
            "Walls should have no duplicates"
        );
    }

    #[test]
    fn zero_budget_returns_valid_result() {
        let grid = small_grid();
        let result = solve(&grid, 0, 2, true);
        assert!(result.walls.is_empty(), "Zero budget should produce no walls");
        assert_eq!(result.score, 0, "Horse should escape with no walls");
    }
}
