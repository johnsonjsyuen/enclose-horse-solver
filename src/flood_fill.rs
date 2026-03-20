use std::simd::prelude::*;

use crate::grid::Grid;

/// Maximum rows supported by fixed-size bitboard buffers.
/// Matches the u32-per-row bitboard limit (grids up to 32x32).
const MAX_ROWS: usize = 32;

/// Result of a flood fill evaluation from the horse's position.
#[derive(Debug, Clone)]
pub struct FloodFillResult {
    /// True if the horse cannot reach any border tile.
    pub enclosed: bool,
    /// Number of tiles in the enclosed region (0 if not enclosed).
    pub area: i32,
    /// Sum of bonus tile adjustments within the enclosed region.
    pub bonus: i32,
    /// Total score: area + bonus (or 0 if not enclosed).
    pub score: i32,
    /// Bitboard of all tiles reached by the horse. One u32 per row.
    pub reached: Vec<u32>,
}

/// Pre-allocated buffers for flood fill, designed to be reused across millions
/// of SA evaluations without heap allocation on the hot path.
pub struct FloodFillState {
    passable: [u32; MAX_ROWS],
    reached: [u32; MAX_ROWS],
    height: usize,
    width: usize,
}

impl FloodFillState {
    /// Create a new state pre-sized for the given grid.
    pub fn new(grid: &Grid) -> Self {
        assert!(
            grid.height <= MAX_ROWS,
            "grid height {} exceeds MAX_ROWS {}",
            grid.height,
            MAX_ROWS
        );
        FloodFillState {
            passable: [0u32; MAX_ROWS],
            reached: [0u32; MAX_ROWS],
            height: grid.height,
            width: grid.width,
        }
    }

    /// Evaluate the flood fill score for a given set of wall placements.
    ///
    /// This is the hot-path function called millions of times by SA.
    /// It reuses internal buffers to avoid allocation.
    pub fn evaluate(&mut self, grid: &Grid, walls: &[usize]) -> FloodFillResult {
        let height = self.height;
        let width = self.width;

        // Reset passable from grid's precomputed bitboard.
        self.passable[..height].copy_from_slice(&grid.passable_bitboard);
        self.passable[height..].fill(0);

        // Clear each wall bit in the passable bitboard.
        for &wall_pos in walls {
            let row = wall_pos / width;
            let col = wall_pos % width;
            self.passable[row] &= !(1u32 << col);
        }

        // Initialize reached: only the horse bit is set.
        self.reached = [0u32; MAX_ROWS];
        let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);
        self.reached[horse_row] = 1u32 << horse_col;

        // Collect portal info: (row_a, col_a, row_b, col_b) for passable portal pairs.
        let portal_pairs: Vec<(usize, usize, usize, usize)> = grid
            .portal_pairs
            .iter()
            .filter_map(|&(idx_a, idx_b)| {
                let (row_a, col_a) = grid.pos_to_rc(idx_a);
                let (row_b, col_b) = grid.pos_to_rc(idx_b);
                let a_passable = self.passable[row_a] & (1u32 << col_a) != 0;
                let b_passable = self.passable[row_b] & (1u32 << col_b) != 0;
                if a_passable && b_passable {
                    Some((row_a, col_a, row_b, col_b))
                } else {
                    None
                }
            })
            .collect();

        // Flood fill loop: iterate until convergence.
        loop {
            let changed = flood_fill_step(&mut self.reached, &self.passable, height);

            // Portal propagation: if one end is reached, seed the other.
            let portal_changed = propagate_portals(&mut self.reached, &portal_pairs);

            if !changed && !portal_changed {
                break;
            }
        }

        // Escape check: if any reached tile is on the border, the horse escaped.
        let escaped = check_escape(&self.reached, &grid.border_bitboard, height);

        if escaped {
            return FloodFillResult {
                enclosed: false,
                area: 0,
                bonus: 0,
                score: 0,
                reached: self.reached[..height].to_vec(),
            };
        }

        // Count enclosed area via popcount.
        let area = count_bits(&self.reached, height);

        // Sum bonus values for reached bonus tiles.
        let bonus = compute_bonus(grid, &self.reached, width);

        let score = area + bonus;

        FloodFillResult {
            enclosed: true,
            area,
            bonus,
            score,
            reached: self.reached[..height].to_vec(),
        }
    }
}

/// One step of flood fill expansion using SIMD for batches of 8 rows.
///
/// For each row, the reached set is expanded left, right, up, and down,
/// then masked by passable. Returns true if any row changed.
fn flood_fill_step(
    reached: &mut [u32; MAX_ROWS],
    passable: &[u32; MAX_ROWS],
    height: usize,
) -> bool {
    let mut changed = false;
    let chunks = height / 8;
    let remainder_start = chunks * 8;

    for c in 0..chunks {
        let base = c * 8;

        let r = Simd::<u32, 8>::from_slice(&reached[base..base + 8]);
        let p = Simd::<u32, 8>::from_slice(&passable[base..base + 8]);

        // Expand left and right within each row (bit shifts within each lane).
        let expanded = r | (r << Simd::splat(1)) | (r >> Simd::splat(1));

        // Build up-neighbor vector: for each lane i, the row above is reached[base+i-1].
        // Lane 0 needs reached[base-1] (from previous chunk, or 0 if at top edge).
        let up = Simd::<u32, 8>::from_array([
            if base > 0 { reached[base - 1] } else { 0 },
            reached[base],
            reached[base + 1],
            reached[base + 2],
            reached[base + 3],
            reached[base + 4],
            reached[base + 5],
            reached[base + 6],
        ]);

        // Build down-neighbor vector: for each lane i, the row below is reached[base+i+1].
        // Lane 7 needs reached[base+8] (from next chunk, or 0 if at bottom edge).
        let down = Simd::<u32, 8>::from_array([
            reached[base + 1],
            reached[base + 2],
            reached[base + 3],
            reached[base + 4],
            reached[base + 5],
            reached[base + 6],
            reached[base + 7],
            if base + 8 < height {
                reached[base + 8]
            } else {
                0
            },
        ]);

        let new_r = (expanded | up | down) & p;

        if new_r != r {
            changed = true;
            new_r.copy_to_slice(&mut reached[base..base + 8]);
        }
    }

    // Handle remainder rows with scalar operations.
    for row in remainder_start..height {
        let r = reached[row];
        let mut expanded = r | (r << 1) | (r >> 1);
        if row > 0 {
            expanded |= reached[row - 1];
        }
        if row + 1 < height {
            expanded |= reached[row + 1];
        }
        let new_val = expanded & passable[row];
        if new_val != reached[row] {
            changed = true;
            reached[row] = new_val;
        }
    }

    changed
}

/// Propagate portal teleportation: if one portal in a pair is reached,
/// mark the other as reached too. Returns true if any change occurred.
fn propagate_portals(
    reached: &mut [u32; MAX_ROWS],
    portals: &[(usize, usize, usize, usize)],
) -> bool {
    let mut changed = false;
    for &(row_a, col_a, row_b, col_b) in portals {
        let a_reached = reached[row_a] & (1u32 << col_a) != 0;
        let b_reached = reached[row_b] & (1u32 << col_b) != 0;

        if a_reached && !b_reached {
            reached[row_b] |= 1u32 << col_b;
            changed = true;
        } else if b_reached && !a_reached {
            reached[row_a] |= 1u32 << col_a;
            changed = true;
        }
    }
    changed
}

/// Check whether any reached tile sits on the border (horse escaped).
/// Uses SIMD to process 8 rows at a time.
fn check_escape(reached: &[u32; MAX_ROWS], border_bitboard: &[u32], height: usize) -> bool {
    let chunks = height / 8;
    let remainder_start = chunks * 8;

    for c in 0..chunks {
        let base = c * 8;
        let r = Simd::<u32, 8>::from_slice(&reached[base..base + 8]);
        let b = Simd::<u32, 8>::from_slice(&border_bitboard[base..base + 8]);
        if (r & b) != Simd::splat(0) {
            return true;
        }
    }

    for row in remainder_start..height {
        if reached[row] & border_bitboard[row] != 0 {
            return true;
        }
    }

    false
}

/// Count total set bits across all reached rows.
fn count_bits(reached: &[u32; MAX_ROWS], height: usize) -> i32 {
    let mut total: u32 = 0;
    for row in 0..height {
        total += reached[row].count_ones();
    }
    total as i32
}

/// Compute the bonus score for all reached bonus tiles using the grid's
/// pre-computed `bonus_scores` list. This avoids scanning every cell.
fn compute_bonus(grid: &Grid, reached: &[u32; MAX_ROWS], width: usize) -> i32 {
    let mut bonus = 0i32;
    for &(pos, adjustment) in &grid.bonus_scores {
        let row = pos / width;
        let col = pos % width;
        if reached[row] & (1u32 << col) != 0 {
            bonus += adjustment;
        }
    }
    bonus
}

/// Convenience function for one-off evaluation without pre-allocated state.
/// For hot-path usage, prefer `FloodFillState::evaluate` to avoid allocation.
pub fn evaluate(grid: &Grid, walls: &[usize]) -> FloodFillResult {
    let mut state = FloodFillState::new(grid);
    state.evaluate(grid, walls)
}

/// Scalar-only flood fill step (no SIMD) for benchmarking comparison.
fn flood_fill_step_scalar(
    reached: &mut [u32; MAX_ROWS],
    passable: &[u32; MAX_ROWS],
    height: usize,
) -> bool {
    let mut changed = false;
    for row in 0..height {
        let r = reached[row];
        let mut expanded = r | (r << 1) | (r >> 1);
        if row > 0 {
            expanded |= reached[row - 1];
        }
        if row + 1 < height {
            expanded |= reached[row + 1];
        }
        let new_val = expanded & passable[row];
        if new_val != reached[row] {
            changed = true;
            reached[row] = new_val;
        }
    }
    changed
}

/// Evaluate using scalar-only flood fill (for benchmarking).
pub fn evaluate_scalar(grid: &Grid, walls: &[usize]) -> FloodFillResult {
    let mut state = FloodFillState::new(grid);
    let height = state.height;
    let width = state.width;

    state.passable[..height].copy_from_slice(&grid.passable_bitboard);
    state.passable[height..].fill(0);
    for &wall_pos in walls {
        let row = wall_pos / width;
        let col = wall_pos % width;
        state.passable[row] &= !(1u32 << col);
    }
    state.reached = [0u32; MAX_ROWS];
    let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);
    state.reached[horse_row] = 1u32 << horse_col;

    let portal_pairs: Vec<(usize, usize, usize, usize)> = grid
        .portal_pairs
        .iter()
        .filter_map(|&(idx_a, idx_b)| {
            let (row_a, col_a) = grid.pos_to_rc(idx_a);
            let (row_b, col_b) = grid.pos_to_rc(idx_b);
            let a_passable = state.passable[row_a] & (1u32 << col_a) != 0;
            let b_passable = state.passable[row_b] & (1u32 << col_b) != 0;
            if a_passable && b_passable {
                Some((row_a, col_a, row_b, col_b))
            } else {
                None
            }
        })
        .collect();

    loop {
        let changed = flood_fill_step_scalar(&mut state.reached, &state.passable, height);
        let portal_changed = propagate_portals(&mut state.reached, &portal_pairs);
        if !changed && !portal_changed {
            break;
        }
    }

    let escaped = check_escape(&state.reached, &grid.border_bitboard, height);
    if escaped {
        return FloodFillResult {
            enclosed: false, area: 0, bonus: 0, score: 0,
            reached: state.reached[..height].to_vec(),
        };
    }
    let area = count_bits(&state.reached, height);
    let bonus = compute_bonus(grid, &state.reached, width);
    FloodFillResult {
        enclosed: true, area, bonus, score: area + bonus,
        reached: state.reached[..height].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid, PuzzleData};

    /// Helper to build a Grid from a map string with sensible defaults.
    fn make_grid(map: &str) -> Grid {
        let data = PuzzleData {
            id: "test".into(),
            map: map.into(),
            budget: 0,
            name: "Test".into(),
            optimal_score: None,
            has_bonus: None,
            bonus_type: None,
            daily_date: None,
            day_number: None,
        };
        Grid::from_puzzle(&data)
    }

    /// TC-002: Flood fill on 3x3 all-grass grid with horse in center.
    /// All border tiles are reachable, so the horse escapes.
    #[test]
    fn test_flood_fill_no_walls_3x3() {
        let grid = make_grid("...\n.H.\n...");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(!result.enclosed);
        assert_eq!(result.score, 0);
    }

    /// TC-003: Horse enclosed by walls in a 5x5 grid.
    #[test]
    fn test_flood_fill_enclosed() {
        let grid = make_grid(".....\n.....\n..H..\n.....\n.....");

        // Walls forming a tight ring around horse at (2,2):
        // (1,1) (1,2) (1,3) (2,1) (2,3) (3,1) (3,2) (3,3)
        let walls = vec![
            1 * 5 + 1,
            1 * 5 + 2,
            1 * 5 + 3,
            2 * 5 + 1,
            2 * 5 + 3,
            3 * 5 + 1,
            3 * 5 + 2,
            3 * 5 + 3,
        ];

        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &walls);

        assert!(result.enclosed);
        assert_eq!(result.area, 1); // Only horse tile reachable.
        assert_eq!(result.score, 1);
    }

    /// TC-004: Horse on border with no walls always escapes.
    #[test]
    fn test_flood_fill_horse_on_border_escapes() {
        let grid = make_grid("H..\n...\n...");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(!result.enclosed);
        assert_eq!(result.score, 0);
    }

    /// TC-005: Enclosed area with cherry bonus tiles.
    #[test]
    fn test_flood_fill_with_cherries() {
        // Water border ensures enclosure. Two cherries inside.
        let grid = make_grid("~~~~~\n~C.C~\n~.H.~\n~...~\n~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        assert_eq!(result.area, 9); // 3x3 interior
        assert_eq!(result.bonus, 6); // 2 cherries * 3
        assert_eq!(result.score, 15);
    }

    /// TC-007: Portal flood fill connects disconnected regions.
    #[test]
    fn test_portal_flood_fill() {
        // Horse near portal A, portal B in a separate water-enclosed room.
        let grid = make_grid("~~~~~\n~HP.~\n~~~~~\n~.P.~\n~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        // Row 1: H, P, . => 3 tiles. Row 3: ., P, . => 3 tiles.
        assert_eq!(result.area, 6);
    }

    /// Test that FloodFillState can be reused across multiple evaluations.
    #[test]
    fn test_state_reuse() {
        let grid = make_grid("~~~~~\n~...~\n~.H.~\n~...~\n~~~~~");
        let mut state = FloodFillState::new(&grid);

        let r1 = state.evaluate(&grid, &[]);
        assert!(r1.enclosed);
        assert_eq!(r1.area, 9);

        // Place a wall, reducing area by 1.
        let r2 = state.evaluate(&grid, &[1 * 5 + 1]);
        assert!(r2.enclosed);
        assert_eq!(r2.area, 8);

        // No walls again, should be back to 9.
        let r3 = state.evaluate(&grid, &[]);
        assert!(r3.enclosed);
        assert_eq!(r3.area, 9);
    }

    /// Test bee penalty reduces score.
    #[test]
    fn test_bee_penalty() {
        let grid = make_grid("~~~~~\n~B.B~\n~.H.~\n~...~\n~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        assert_eq!(result.area, 9);
        assert_eq!(result.bonus, -10); // 2 bees * -5
        assert_eq!(result.score, -1);
    }

    /// Test golden apple bonus.
    #[test]
    fn test_golden_apple_bonus() {
        let grid = make_grid("~~~~~\n~A..~\n~.H.~\n~..A~\n~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        assert_eq!(result.area, 9);
        assert_eq!(result.bonus, 20); // 2 apples * 10
        assert_eq!(result.score, 29);
    }

    /// Test with >8 rows to exercise SIMD chunk processing with remainder.
    #[test]
    fn test_large_grid_simd_chunks() {
        // 12 rows, 5 cols. Water border, grass interior, horse in middle.
        let mut rows = Vec::new();
        for r in 0..12 {
            if r == 0 || r == 11 {
                rows.push("~~~~~");
            } else if r == 6 {
                rows.push("~.H.~");
            } else {
                rows.push("~...~");
            }
        }
        let map = rows.join("\n");
        let grid = make_grid(&map);
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        // Inner area: 10 rows * 3 cols = 30 tiles
        assert_eq!(result.area, 30);
    }

    /// Test with exactly 8 rows (no remainder in SIMD processing).
    #[test]
    fn test_exactly_8_rows() {
        let mut rows = Vec::new();
        for r in 0..8 {
            if r == 0 || r == 7 {
                rows.push("~~~~~");
            } else if r == 4 {
                rows.push("~.H.~");
            } else {
                rows.push("~...~");
            }
        }
        let map = rows.join("\n");
        let grid = make_grid(&map);
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        // Inner: 6 rows * 3 cols = 18
        assert_eq!(result.area, 18);
    }

    /// Test with exactly 16 rows (two full SIMD chunks, no remainder).
    #[test]
    fn test_16_rows_two_chunks() {
        let mut rows = Vec::new();
        for r in 0..16 {
            if r == 0 || r == 15 {
                rows.push("~~~~~");
            } else if r == 8 {
                rows.push("~.H.~");
            } else {
                rows.push("~...~");
            }
        }
        let map = rows.join("\n");
        let grid = make_grid(&map);
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        // Inner: 14 rows * 3 cols = 42
        assert_eq!(result.area, 42);
    }

    /// Test the convenience evaluate function.
    #[test]
    fn test_convenience_evaluate() {
        let grid = make_grid("~~~~~\n~...~\n~.H.~\n~...~\n~~~~~");
        let result = evaluate(&grid, &[]);

        assert!(result.enclosed);
        assert_eq!(result.area, 9);
        assert_eq!(result.score, 9);
    }

    /// Test that walls blocking all 4 orthogonal neighbors enclose the horse.
    #[test]
    fn test_walls_immediately_around_horse() {
        let grid = make_grid(".....\n.....\n..H..\n.....\n.....");
        let walls = vec![
            1 * 5 + 2, // above
            3 * 5 + 2, // below
            2 * 5 + 1, // left
            2 * 5 + 3, // right
        ];
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &walls);

        assert!(result.enclosed);
        assert_eq!(result.area, 1);
    }

    /// TC-006: Bitboard row shift operations produce correct results.
    #[test]
    fn test_bitboard_shift_ops() {
        let row: u32 = 0b11011;
        assert_eq!(row << 1, 0b110110);
        assert_eq!(row >> 1, 0b1101);
    }

    /// Test portal on border causes escape.
    #[test]
    fn test_portal_on_border_escapes() {
        // Portal A at (0,0) is on the border. Portal B at (3,2) is reachable from H.
        // Once flood fill reaches portal B, it teleports to portal A on the border.
        let grid = make_grid("P....\n~~~~~\n~.H.~\n~.P.~\n~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(!result.enclosed);
        assert_eq!(result.score, 0);
    }

    /// Test that an unreachable portal pair does not affect the result.
    #[test]
    fn test_unreachable_portal_no_effect() {
        // Portals exist but are blocked by water from the horse.
        let grid = make_grid("~P~~P~\n~~~~~~\n~.H.~~\n~...~~\n~~~~~~");
        let mut state = FloodFillState::new(&grid);
        let result = state.evaluate(&grid, &[]);

        assert!(result.enclosed);
        // Only the inner 2x3 area is reachable: rows 2-3, cols 1-3 = 6 tiles
        assert_eq!(result.area, 6);
    }
}
