// =============================================================================
// GPU Simulated Annealing solver for enclose.horse
// =============================================================================
//
// GPU EXECUTION MODEL OVERVIEW:
// A GPU runs thousands of threads in parallel. Threads are grouped into
// "workgroups" (here, 64 threads each). All threads in a workgroup execute
// on the same compute unit and can share local memory (though we don't use
// shared memory here). The GPU hardware schedules many workgroups across
// its compute units simultaneously.
//
// Our strategy: each thread runs a completely independent SA (Simulated
// Annealing) restart with its own random seed. With 1024+ threads, we
// explore many random starting configurations in parallel and pick the best
// result. This is "embarrassingly parallel" — no inter-thread communication.
// =============================================================================

// Parameters passed from the CPU. This struct layout must exactly match
// the Rust `GpuParams` struct (same field order, sizes, and padding).
struct Params {
    width: u32,
    height: u32,
    horse_row: u32,
    horse_col: u32,
    num_grass: u32,
    num_portals: u32,
    num_bonus: u32,
    budget: u32,          // number of walls we can place
    num_threads: u32,
    iters_per_temp: u32,
    initial_temp: f32,
    cooling_rate: f32,
    min_temp: f32,
    // WGSL uniform buffers must be 16-byte aligned. With 13 x u32/f32 = 52
    // bytes, we pad to 64 bytes (next multiple of 16). Without this padding,
    // the GPU would read garbage for subsequent bindings.
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// --- Buffer bindings ---
//
// `var<uniform>` = small, read-only data broadcast to all threads efficiently.
//   Uniform buffers are cached in special fast-access hardware on the GPU,
//   ideal for parameters that every thread reads identically.
//
// `var<storage, read>` = larger read-only arrays. Storage buffers live in
//   global GPU memory. "read" means the shader promises not to write, which
//   lets the driver skip synchronization overhead.
//
// `var<storage, read_write>` = mutable arrays. "read_write" is needed when
//   the shader writes results back (scores, walls) or updates state (RNG).
//   The driver must ensure writes are visible, so this has more overhead.
//
// @group(0) @binding(N) tells the GPU which buffer slot each variable maps to.
// The Rust host code sets up a "bind group" that connects actual GPU buffers
// to these numbered slots.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> passable_bb: array<u32>;  // bitboard: which tiles the horse can walk through
@group(0) @binding(2) var<storage, read> border_bb: array<u32>;    // bitboard: which tiles are on the grid border
@group(0) @binding(3) var<storage, read> grass: array<u32>;        // flat indices of grass tiles (valid wall locations)
@group(0) @binding(4) var<storage, read> portal_data: array<u32>;  // portal pairs: (row_a, col_a, row_b, col_b) x N
@group(0) @binding(5) var<storage, read> bonus_info: array<i32>;   // bonus tiles: (row, col, score_adjustment) x N
@group(0) @binding(6) var<storage, read_write> rng_buf: array<u32>;  // 2 u32s per thread: (state, increment)
@group(0) @binding(7) var<storage, read_write> out_scores: array<i32>;  // best score per thread
@group(0) @binding(8) var<storage, read_write> out_walls: array<u32>;   // best wall positions per thread (64 slots each)

// ---------------------------------------------------------------------------
// PCG (Permuted Congruential Generator) — a fast, high-quality 32-bit RNG
// ---------------------------------------------------------------------------
// Why PCG on GPU?
//   - We need each thread to have its own independent random stream.
//     If threads shared RNG state, they'd need expensive synchronization.
//   - PCG is small (just 2 u32s of state) and has no branches, making it
//     ideal for GPU where branch divergence kills performance.
//   - The "increment" (stored in rng_buf[idx+1]) selects which of 2^31
//     distinct random streams this thread uses. Each thread gets a unique
//     odd increment, guaranteeing statistically independent sequences.
//
// The algorithm:
//   1. Advance the Linear Congruential Generator: state = state * 747796405 + inc
//   2. Apply a permutation to improve output quality:
//      - Use high bits of state to determine a shift amount
//      - XOR-shift and multiply to scramble the bits
//      This "output permutation" is what makes PCG much better than a raw LCG.

fn rng_next(tid: u32) -> u32 {
    let idx = tid * 2u;                  // each thread owns 2 consecutive u32s
    var state = rng_buf[idx];
    let inc = rng_buf[idx + 1u];         // increment (odd) — selects the random stream
    state = state * 747796405u + inc;    // LCG step
    rng_buf[idx] = state;               // persist updated state for next call
    // Output permutation: use top bits to rotate, then scramble with multiply+xor
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert RNG output to a float in [0.0, 1.0) for Metropolis acceptance check.
// 2.3283064e-10 = 1.0 / 2^32, mapping the full u32 range to [0, 1).
fn rng_float(tid: u32) -> f32 {
    return f32(rng_next(tid)) * 2.3283064e-10;
}

// Return a random value in [0, bound). Uses modulo which has slight bias for
// non-power-of-2 bounds, but the bias is negligible for our use case (grid
// indices up to ~1000) relative to u32 range (4 billion).
fn rng_range(tid: u32, bound: u32) -> u32 {
    return rng_next(tid) % bound;
}

// ---------------------------------------------------------------------------
// Flood fill evaluation
// ---------------------------------------------------------------------------
// This is the core objective function. Given a wall configuration, it answers:
// "How many tiles can the horse reach, and can it escape to the border?"
//
// BITBOARD REPRESENTATION:
// Each row of the grid is encoded as a single u32, where bit i is set if
// column i has a certain property (passable, reached, is a wall, etc.).
// This is perfect for GPU because:
//   - A 32-column grid fits exactly in one u32 (common GPU register size)
//   - Expanding in all 4 directions becomes simple bitwise ops (see below)
//   - Checking membership is a single AND + compare (O(1), no memory lookup)
//   - Counting set bits uses the hardware `countOneBits` instruction
//
// Returns 0 if the horse can escape (bad), otherwise the enclosed area + bonuses.

fn evaluate(tid: u32, walls: ptr<function, array<u32, 64>>) -> i32 {
    let height = params.height;
    let width = params.width;
    let budget = params.budget;

    var passable: array<u32, 32>;   // local copy — we'll punch holes for walls
    var reached: array<u32, 32>;    // flood fill frontier: which tiles the horse can reach

    // Start with the global passable bitboard (trees, water, etc. already excluded)
    for (var r = 0u; r < height; r = r + 1u) {
        passable[r] = passable_bb[r];
    }

    // Remove wall tiles from the passable set. Each wall blocks the horse.
    // Clearing a bit: AND with the complement of the bit mask for that column.
    for (var i = 0u; i < budget; i = i + 1u) {
        let pos = (*walls)[i];
        let row = pos / width;
        let col = pos % width;
        passable[row] = passable[row] & ~(1u << col);
    }

    // Seed the flood fill at the horse's starting position
    reached[params.horse_row] = 1u << params.horse_col;

    // ITERATIVE FLOOD FILL:
    // Each step, we expand the "reached" set by one tile in every direction,
    // but only into passable tiles. We repeat until nothing changes.
    //
    // The bitwise expansion trick for left/right movement:
    //   (r << 1) shifts all bits left  = "can we reach one column to the right?"
    //   (r >> 1) shifts all bits right = "can we reach one column to the left?"
    //   OR them together with the current reached set = expanded neighborhood.
    //
    // For up/down: just OR in the reached set from the adjacent row.
    //
    // Finally, AND with passable[] to mask out walls, trees, etc.
    //
    // This converges in at most max(width, height) steps since the frontier
    // can expand by at most 1 tile per step. The 64-step cap is generous
    // for grids up to 32x32.
    for (var step = 0u; step < 64u; step = step + 1u) {
        var changed = false;

        for (var row = 0u; row < height; row = row + 1u) {
            let r = reached[row];
            // Expand horizontally (left shift = rightward, right shift = leftward)
            // and include current position
            var expanded = r | (r << 1u) | (r >> 1u);
            // Expand vertically by including adjacent rows
            if (row > 0u) {
                expanded = expanded | reached[row - 1u];
            }
            if (row + 1u < height) {
                expanded = expanded | reached[row + 1u];
            }
            // Mask by passable: only keep tiles the horse can actually stand on
            let new_val = expanded & passable[row];
            if (new_val != reached[row]) {
                changed = true;
                reached[row] = new_val;
            }
        }

        // PORTAL PROPAGATION:
        // Portals are pairs of tiles that act as teleporters. If the horse
        // reaches one end, it can instantly travel to the other end.
        // We check each portal pair and propagate reachability through it.
        for (var p = 0u; p < params.num_portals; p = p + 1u) {
            let b = p * 4u;  // 4 u32s per portal: (row_a, col_a, row_b, col_b)
            let ra = portal_data[b];
            let ca = portal_data[b + 1u];
            let rb = portal_data[b + 2u];
            let cb = portal_data[b + 3u];

            // PORTAL FILTERING: if either endpoint has a wall on it, the portal
            // is blocked. A walled portal can't be entered or exited, so we skip
            // it entirely. This is why we check passable[] not just reached[].
            let a_passable = (passable[ra] & (1u << ca)) != 0u;
            let b_passable = (passable[rb] & (1u << cb)) != 0u;
            if (!a_passable || !b_passable) {
                continue;
            }

            let bit_a = reached[ra] & (1u << ca);
            let bit_b = reached[rb] & (1u << cb);

            // If one end is reached but not the other, propagate through the portal
            if (bit_a != 0u && bit_b == 0u) {
                reached[rb] = reached[rb] | (1u << cb);
                changed = true;
            } else if (bit_b != 0u && bit_a == 0u) {
                reached[ra] = reached[ra] | (1u << ca);
                changed = true;
            }
        }

        if (!changed) {
            break;  // flood fill has converged — no new tiles reachable
        }
    }

    // ESCAPE CHECK:
    // If any tile the horse can reach is on the grid border, the horse is not
    // enclosed — it can walk off the edge. This means our wall placement failed
    // to form a complete enclosure, so the score is 0.
    // The border_bb bitboard has bits set for all border tiles, so a single
    // AND per row detects overlap instantly.
    for (var row = 0u; row < height; row = row + 1u) {
        if ((reached[row] & border_bb[row]) != 0u) {
            return 0;
        }
    }

    // COUNT ENCLOSED AREA using hardware popcount.
    // `countOneBits` counts the number of 1-bits in a u32 — this is a single
    // hardware instruction on modern GPUs (extremely fast). Each set bit in
    // reached[row] represents one tile the horse can reach, so summing popcount
    // across all rows gives us the total enclosed area.
    var area: i32 = 0;
    for (var row = 0u; row < height; row = row + 1u) {
        area = area + i32(countOneBits(reached[row]));
    }

    // Add bonus/penalty adjustments for special tiles (gems, skulls, etc.)
    // that fall within the enclosed area.
    var bonus: i32 = 0;
    for (var b = 0u; b < params.num_bonus; b = b + 1u) {
        let idx = b * 3u;  // 3 i32s per bonus tile: (row, col, adjustment)
        let br = u32(bonus_info[idx]);
        let bc = u32(bonus_info[idx + 1u]);
        let adj = bonus_info[idx + 2u];   // positive for gems, negative for skulls
        if ((reached[br] & (1u << bc)) != 0u) {
            bonus = bonus + adj;
        }
    }

    return area + bonus;
}

// ---------------------------------------------------------------------------
// Main SA (Simulated Annealing) kernel
// ---------------------------------------------------------------------------
// @compute marks this as a compute shader entry point.
// @workgroup_size(64) means 64 threads per workgroup. This is a common choice:
//   - GPUs execute threads in "warps" (NVIDIA) or "waves" (AMD) of 32 or 64.
//   - Matching the workgroup size to the warp size avoids wasting lanes.
//   - The host dispatches ceil(num_threads / 64) workgroups to cover all threads.
//
// @builtin(global_invocation_id) gives each thread a unique 3D ID.
// We only use the x dimension since our problem is 1D (independent restarts).

@compute @workgroup_size(64)
fn sa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;  // thread ID: 0, 1, 2, ..., num_threads-1

    // EXCESS THREAD GUARD:
    // We dispatch ceil(num_threads / 64) workgroups, so the last workgroup may
    // have threads beyond num_threads. These excess threads must bail out.
    // We write i32::MIN (-2147483648) so that when the CPU picks the best score
    // via max_by_key, these dummy entries can never win.
    if (tid >= params.num_threads) {
        out_scores[tid] = -2147483648;
        return;
    }

    let budget = params.budget;
    let num_grass = params.num_grass;
    let width = params.width;

    var walls: array<u32, 64>;       // current wall positions (flat indices into grid)
    var best_walls: array<u32, 64>;  // best wall positions found so far
    // WALL MEMBERSHIP BITBOARD:
    // wall_bb[row] has bit col set if there's a wall at (row, col).
    // This gives us O(1) wall membership checks during rejection sampling,
    // instead of scanning the walls array (O(budget) per check).
    var wall_bb: array<u32, 32>;

    // Handle zero budget: just evaluate the empty board
    if (budget == 0u) {
        out_scores[tid] = evaluate(tid, &walls);
        return;
    }

    // INITIALIZATION: randomly place `budget` walls on grass tiles.
    // We use rejection sampling: pick a random grass tile, and if it's not
    // already a wall, place one there. The wall_bb bitboard makes the
    // "already a wall?" check a single bit test.
    //
    // Why rejection sampling? It's simple, branchless-friendly, and the
    // probability of collision is low (budget << num_grass typically).
    // The 10000 attempt cap prevents infinite loops in degenerate cases.
    var placed = 0u;
    for (var attempt = 0u; attempt < 10000u; attempt = attempt + 1u) {
        if (placed >= budget) { break; }
        let idx = rng_range(tid, num_grass);
        let pos = grass[idx];
        let row = pos / width;
        let col = pos % width;
        // O(1) duplicate check via bitboard instead of linear scan
        if ((wall_bb[row] & (1u << col)) == 0u) {
            walls[placed] = pos;
            wall_bb[row] = wall_bb[row] | (1u << col);
            placed = placed + 1u;
        }
    }

    if (placed < budget) {
        out_scores[tid] = 0;
        return;
    }

    // Evaluate the initial random wall configuration
    var current_score = evaluate(tid, &walls);
    var best_score = current_score;
    for (var i = 0u; i < budget; i = i + 1u) {
        best_walls[i] = walls[i];
    }

    // =========================================================================
    // SIMULATED ANNEALING (SA):
    //
    // SA is a metaheuristic for combinatorial optimization. It's inspired by
    // the physical process of annealing in metallurgy (heating then slowly
    // cooling metal to reach a low-energy crystalline state).
    //
    // Key ideas:
    //   - At each step, make a random small change (move one wall).
    //   - If the change improves the score, always accept it.
    //   - If the change makes things worse, accept it with probability
    //     exp(delta / temperature). This allows escaping local optima.
    //   - The temperature starts high (accepting most bad moves) and
    //     gradually decreases (becoming more greedy over time).
    //   - At low temperatures, it behaves almost like hill climbing.
    //
    // The METROPOLIS ACCEPTANCE CRITERION (line ~265):
    //   accept if: delta > 0  OR  random() < exp(delta / temp)
    //   - delta > 0 means improvement: always accept
    //   - delta < 0 means worse: accept with probability exp(delta/temp)
    //     which decreases as |delta| grows or temp shrinks
    //   - This is mathematically equivalent to sampling from a Boltzmann
    //     distribution, which provably converges to the global optimum
    //     given a sufficiently slow cooling schedule.
    //
    // COOLING SCHEDULE:
    //   temp *= cooling_rate each outer iteration (geometric cooling).
    //   cooling_rate < 1.0 (e.g., 0.997) makes temp decrease exponentially.
    //   We do `iters_per_temp` random moves at each temperature level before
    //   cooling, giving the search time to explore at each energy scale.
    // =========================================================================

    var temp = params.initial_temp;

    while (temp > params.min_temp) {
        for (var iter = 0u; iter < params.iters_per_temp; iter = iter + 1u) {
            // NEIGHBOR GENERATION: pick a random wall and move it somewhere else.
            // This is a minimal perturbation that explores nearby configurations.
            let wi = rng_range(tid, budget);
            let old_pos = walls[wi];
            let old_row = old_pos / width;
            let old_col = old_pos % width;

            // REJECTION SAMPLING to find a new position:
            // Pick random grass tiles until we find one that isn't already a wall.
            // With budget << num_grass, most attempts succeed on the first try.
            // The 64-attempt cap prevents hangs if the board is nearly full.
            var new_pos = old_pos;
            var new_row = old_row;
            var new_col = old_col;
            var found = false;
            for (var att = 0u; att < 64u; att = att + 1u) {
                let ci = rng_range(tid, num_grass);
                let cp = grass[ci];
                let cr = cp / width;
                let cc = cp % width;
                // wall_bb gives O(1) check: is this tile already occupied by a wall?
                if ((wall_bb[cr] & (1u << cc)) == 0u) {
                    new_pos = cp;
                    new_row = cr;
                    new_col = cc;
                    found = true;
                    break;
                }
            }

            if (!found) {
                continue;  // couldn't find a free tile; skip this iteration
            }

            // Apply the wall move: update both the walls array and the bitboard
            walls[wi] = new_pos;
            wall_bb[old_row] = wall_bb[old_row] & ~(1u << old_col);  // clear old wall bit
            wall_bb[new_row] = wall_bb[new_row] | (1u << new_col);   // set new wall bit

            let new_score = evaluate(tid, &walls);

            // METROPOLIS ACCEPTANCE CRITERION
            let delta = new_score - current_score;
            if (delta > 0 || rng_float(tid) < exp(f32(delta) / temp)) {
                // Accept: keep the new configuration
                current_score = new_score;
                if (current_score > best_score) {
                    best_score = current_score;
                    for (var i = 0u; i < budget; i = i + 1u) {
                        best_walls[i] = walls[i];
                    }
                }
            } else {
                // Reject: revert the wall move to restore previous state
                walls[wi] = old_pos;
                wall_bb[new_row] = wall_bb[new_row] & ~(1u << new_col);
                wall_bb[old_row] = wall_bb[old_row] | (1u << old_col);
            }
        }

        // Geometric cooling: temperature decreases exponentially
        temp = temp * params.cooling_rate;
    }

    // Write this thread's best result to global memory.
    // Each thread gets a contiguous 64-slot region in out_walls (even if budget < 64).
    out_scores[tid] = best_score;
    let wb = tid * 64u;
    for (var i = 0u; i < budget; i = i + 1u) {
        out_walls[wb + i] = best_walls[i];
    }
}
