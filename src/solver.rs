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

// ---------------------------------------------------------------------------
// Simulated Annealing (SA) Hyperparameters
// ---------------------------------------------------------------------------
// SA is inspired by metallurgical annealing: heating metal and slowly cooling
// it lets atoms settle into a low-energy crystal structure. Analogously, we
// start with high "temperature" (willing to accept bad moves) and gradually
// cool down (becoming pickier), which lets us escape local optima early on
// while still converging to a good solution.
//
// INITIAL_TEMPERATURE: Controls how likely we are to accept worse solutions at
//   the start. Higher values mean the solver explores more broadly in early
//   iterations. A value of 50 means even moves that worsen the score by ~50
//   points have a ~37% chance (1/e) of acceptance at the start.
//
// COOLING_RATE: Multiplicative factor applied to temperature each outer step.
//   Values close to 1.0 (like 0.9997) cool very slowly, giving the solver
//   more iterations to explore at each temperature level. Faster cooling
//   (e.g. 0.99) converges quicker but risks missing the global optimum.
//
// MIN_TEMPERATURE: When to stop. Below this threshold the acceptance
//   probability for any worsening move is essentially zero, so continued
//   iteration would just waste cycles doing pure hill-climbing.
//
// ITERATIONS_PER_TEMP: How many neighbor moves to try at each temperature
//   step before cooling. More iterations per step = finer exploration at
//   that temperature, but slower overall.
// ---------------------------------------------------------------------------
const INITIAL_TEMPERATURE: f64 = 50.0;
const COOLING_RATE: f64 = 0.9997;
const MIN_TEMPERATURE: f64 = 0.01;
const ITERATIONS_PER_TEMP: usize = 100;

/// Run `num_restarts` independent simulated annealing runs and return the best
/// result found across all restarts.
///
/// **Why multi-restart?** SA is stochastic and can converge to different local
/// optima depending on the random initial configuration. Running many
/// independent restarts and keeping the global best dramatically increases our
/// chance of finding the true optimum. Each restart is fully independent (no
/// shared state), making this an "embarrassingly parallel" workload -- perfect
/// for rayon's parallel iterator.
pub fn solve(grid: &Grid, budget: usize, num_restarts: usize, threaded: bool) -> SolveResult {
    if threaded {
        // Rayon distributes restarts across all available CPU cores.
        // Each thread gets its own RNG and FloodFillState (no locks needed).
        (0..num_restarts)
            .into_par_iter()
            .map(|_| sa_single_run(grid, budget))
            .max_by_key(|r| r.score)
            .unwrap()
    } else {
        // Sequential fallback (useful for WASM or single-core targets).
        (0..num_restarts)
            .map(|_| sa_single_run(grid, budget))
            .max_by_key(|r| r.score)
            .unwrap()
    }
}

/// Perform a single simulated annealing run for wall placement optimization.
///
/// The SA algorithm maintains two separate solutions:
/// - **current**: the working solution that SA modifies each iteration.
///   SA is allowed to accept *worse* current solutions (that's the whole point).
/// - **best**: the best solution seen so far across all iterations.
///   Because SA deliberately accepts worse moves to escape local optima,
///   the current solution can drift away from the best one found. We
///   track best separately so we never lose a good solution.
fn sa_single_run(grid: &Grid, budget: usize) -> SolveResult {
    let num_cells = grid.width * grid.height;

    // SmallRng: a fast, non-cryptographic PRNG. We generate millions of
    // random numbers per run; using SmallRng instead of StdRng gives a
    // measurable speedup (~2x) with acceptable randomness quality.
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    // Pre-allocated flood fill buffers. Reusing this across all iterations
    // avoids millions of heap allocations (see flood_fill.rs for details).
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
    // The random starting point is what makes each restart explore a
    // different region of the solution space.
    let mut walls: Vec<usize> = candidates
        .choose_multiple(&mut rng, budget)
        .copied()
        .collect();

    // is_wall: a boolean vec indexed by grid position for O(1) membership
    // checking. Without this, checking "is position X already a wall?" would
    // require scanning the walls vec (O(budget)) on every neighbor generation.
    // With ~10-20 walls and millions of iterations, O(1) vs O(budget) matters.
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
    // The outer loop decreases temperature geometrically: T *= COOLING_RATE.
    // At high T, the solver freely accepts worse moves (exploration).
    // At low T, it only accepts improvements (exploitation / hill-climbing).
    let mut temperature = INITIAL_TEMPERATURE;

    while temperature > MIN_TEMPERATURE {
        for _ in 0..ITERATIONS_PER_TEMP {
            // --- Neighbor generation ---
            // Our neighbor move is simple: relocate one wall to a new position.
            // This is a minimal perturbation that keeps the search space
            // connected (any configuration can reach any other via single
            // wall relocations).
            let wall_idx = rng.random_range(0..budget);
            let old_pos = walls[wall_idx];

            // Rejection sampling: pick random candidate tiles until we find
            // one that isn't already a wall. This is efficient because
            // budget << candidates.len() (typically ~20 walls out of hundreds
            // of grass tiles), so almost every random pick succeeds on the
            // first try. The is_wall vec makes each rejection check O(1).
            let candidate = loop {
                let c = candidates[rng.random_range(0..candidates.len())];
                if !is_wall[c] {
                    break c;
                }
            };

            // Apply the move (mutate in place for speed, revert if rejected).
            walls[wall_idx] = candidate;
            is_wall[old_pos] = false;
            is_wall[candidate] = true;

            // Evaluate the new configuration via flood fill.
            let new_result = ff_state.evaluate(grid, &walls);
            let new_score = new_result.score;

            // --- Metropolis acceptance criterion ---
            // This is the heart of SA. The rule is:
            //   - Always accept improvements (delta > 0).
            //   - Accept worse moves with probability exp(delta / T), where
            //     delta = new_score - current_score (negative for worse moves).
            //
            // Why exp(delta/T)? When T is high, exp(negative/large) is close
            // to 1, so bad moves are accepted often (exploration). When T is
            // low, exp(negative/small) is close to 0, so only improvements
            // pass (exploitation). This smooth transition is what lets SA
            // escape local optima while still converging.
            let delta = new_score - current_score;
            if delta > 0 || rng.random::<f64>() < (delta as f64 / temperature).exp() {
                // Accept the move.
                current_score = new_score;
                if current_score > best_score {
                    best_score = current_score;
                    best_walls = walls.clone();
                }
            } else {
                // Reject: revert the move to restore previous state.
                walls[wall_idx] = old_pos;
                is_wall[candidate] = false;
                is_wall[old_pos] = true;
            }
        }

        // Geometric cooling: temperature decreases exponentially.
        // After k outer steps, T = INITIAL_TEMPERATURE * COOLING_RATE^k.
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
