// GPU Simulated Annealing solver for enclose.horse
// Each thread (global invocation) runs an independent SA restart.
// Parallelism comes from running thousands of restarts simultaneously.

struct Params {
    width: u32,
    height: u32,
    horse_row: u32,
    horse_col: u32,
    num_grass: u32,
    num_portals: u32,
    num_bonus: u32,
    budget: u32,
    num_threads: u32,
    iters_per_temp: u32,
    initial_temp: f32,
    cooling_rate: f32,
    min_temp: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> passable_bb: array<u32>;
@group(0) @binding(2) var<storage, read> border_bb: array<u32>;
@group(0) @binding(3) var<storage, read> grass: array<u32>;
@group(0) @binding(4) var<storage, read> portal_data: array<u32>;
@group(0) @binding(5) var<storage, read> bonus_info: array<i32>;
@group(0) @binding(6) var<storage, read_write> rng_buf: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_scores: array<i32>;
@group(0) @binding(8) var<storage, read_write> out_walls: array<u32>;

// ---------------------------------------------------------------------------
// PCG-style 32-bit RNG (one state per thread, stored in rng_buf)
// ---------------------------------------------------------------------------

fn rng_next(tid: u32) -> u32 {
    let idx = tid * 2u;
    var state = rng_buf[idx];
    let inc = rng_buf[idx + 1u];
    state = state * 747796405u + inc;
    rng_buf[idx] = state;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rng_float(tid: u32) -> f32 {
    return f32(rng_next(tid)) * 2.3283064e-10; // 1.0 / 2^32
}

fn rng_range(tid: u32, bound: u32) -> u32 {
    return rng_next(tid) % bound;
}

// ---------------------------------------------------------------------------
// Flood fill evaluation
// Returns the enclosure score (0 if horse escapes to border)
// ---------------------------------------------------------------------------

fn evaluate(tid: u32, walls: ptr<function, array<u32, 64>>) -> i32 {
    let height = params.height;
    let width = params.width;
    let budget = params.budget;

    var passable: array<u32, 32>;
    var reached: array<u32, 32>;

    // Copy passable bitboard
    for (var r = 0u; r < height; r = r + 1u) {
        passable[r] = passable_bb[r];
    }

    // Clear wall bits from passable
    for (var i = 0u; i < budget; i = i + 1u) {
        let pos = (*walls)[i];
        let row = pos / width;
        let col = pos % width;
        passable[row] = passable[row] & ~(1u << col);
    }

    // Seed horse position
    reached[params.horse_row] = 1u << params.horse_col;

    // Iterative flood fill until convergence (max 64 steps for 32x32 grid)
    for (var step = 0u; step < 64u; step = step + 1u) {
        var changed = false;

        // Expand reached set: left, right, up, down, masked by passable
        for (var row = 0u; row < height; row = row + 1u) {
            let r = reached[row];
            var expanded = r | (r << 1u) | (r >> 1u);
            if (row > 0u) {
                expanded = expanded | reached[row - 1u];
            }
            if (row + 1u < height) {
                expanded = expanded | reached[row + 1u];
            }
            let new_val = expanded & passable[row];
            if (new_val != reached[row]) {
                changed = true;
                reached[row] = new_val;
            }
        }

        // Portal propagation: if one end reached and both ends passable, seed the other
        for (var p = 0u; p < params.num_portals; p = p + 1u) {
            let b = p * 4u;
            let ra = portal_data[b];
            let ca = portal_data[b + 1u];
            let rb = portal_data[b + 2u];
            let cb = portal_data[b + 3u];

            // Skip portal if either endpoint is not passable (walled over)
            let a_passable = (passable[ra] & (1u << ca)) != 0u;
            let b_passable = (passable[rb] & (1u << cb)) != 0u;
            if (!a_passable || !b_passable) {
                continue;
            }

            let bit_a = reached[ra] & (1u << ca);
            let bit_b = reached[rb] & (1u << cb);

            if (bit_a != 0u && bit_b == 0u) {
                reached[rb] = reached[rb] | (1u << cb);
                changed = true;
            } else if (bit_b != 0u && bit_a == 0u) {
                reached[ra] = reached[ra] | (1u << ca);
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
    }

    // Escape check: if any reached tile is on the border, score = 0
    for (var row = 0u; row < height; row = row + 1u) {
        if ((reached[row] & border_bb[row]) != 0u) {
            return 0;
        }
    }

    // Count enclosed area via popcount
    var area: i32 = 0;
    for (var row = 0u; row < height; row = row + 1u) {
        area = area + i32(countOneBits(reached[row]));
    }

    // Sum bonus adjustments for enclosed bonus tiles
    var bonus: i32 = 0;
    for (var b = 0u; b < params.num_bonus; b = b + 1u) {
        let idx = b * 3u;
        let br = u32(bonus_info[idx]);
        let bc = u32(bonus_info[idx + 1u]);
        let adj = bonus_info[idx + 2u];
        if ((reached[br] & (1u << bc)) != 0u) {
            bonus = bonus + adj;
        }
    }

    return area + bonus;
}

// ---------------------------------------------------------------------------
// Main SA kernel: each invocation runs a full SA restart
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_threads) {
        // Write sentinel so uninitialized memory doesn't win max_by_key
        out_scores[tid] = -2147483648; // i32::MIN
        return;
    }

    let budget = params.budget;
    let num_grass = params.num_grass;
    let width = params.width;

    var walls: array<u32, 64>;
    var best_walls: array<u32, 64>;
    var wall_bb: array<u32, 32>;

    // Handle zero budget
    if (budget == 0u) {
        out_scores[tid] = evaluate(tid, &walls);
        return;
    }

    // Initialize: randomly place budget walls on grass tiles
    var placed = 0u;
    for (var attempt = 0u; attempt < 10000u; attempt = attempt + 1u) {
        if (placed >= budget) { break; }
        let idx = rng_range(tid, num_grass);
        let pos = grass[idx];
        let row = pos / width;
        let col = pos % width;
        if ((wall_bb[row] & (1u << col)) == 0u) {
            walls[placed] = pos;
            wall_bb[row] = wall_bb[row] | (1u << col);
            placed = placed + 1u;
        }
    }

    // Safety: if we couldn't place enough walls, bail
    if (placed < budget) {
        out_scores[tid] = 0;
        return;
    }

    // Evaluate initial configuration
    var current_score = evaluate(tid, &walls);
    var best_score = current_score;
    for (var i = 0u; i < budget; i = i + 1u) {
        best_walls[i] = walls[i];
    }

    // SA temperature loop
    var temp = params.initial_temp;

    while (temp > params.min_temp) {
        for (var iter = 0u; iter < params.iters_per_temp; iter = iter + 1u) {
            // Pick a random wall to relocate
            let wi = rng_range(tid, budget);
            let old_pos = walls[wi];
            let old_row = old_pos / width;
            let old_col = old_pos % width;

            // Find a new grass tile not currently a wall (rejection sampling)
            var new_pos = old_pos;
            var new_row = old_row;
            var new_col = old_col;
            var found = false;
            for (var att = 0u; att < 64u; att = att + 1u) {
                let ci = rng_range(tid, num_grass);
                let cp = grass[ci];
                let cr = cp / width;
                let cc = cp % width;
                if ((wall_bb[cr] & (1u << cc)) == 0u) {
                    new_pos = cp;
                    new_row = cr;
                    new_col = cc;
                    found = true;
                    break;
                }
            }

            if (!found) {
                continue;
            }

            // Apply the move
            walls[wi] = new_pos;
            wall_bb[old_row] = wall_bb[old_row] & ~(1u << old_col);
            wall_bb[new_row] = wall_bb[new_row] | (1u << new_col);

            // Evaluate new configuration
            let new_score = evaluate(tid, &walls);

            // Metropolis acceptance criterion (higher score = better)
            let delta = new_score - current_score;
            if (delta > 0 || rng_float(tid) < exp(f32(delta) / temp)) {
                // Accept move
                current_score = new_score;
                if (current_score > best_score) {
                    best_score = current_score;
                    for (var i = 0u; i < budget; i = i + 1u) {
                        best_walls[i] = walls[i];
                    }
                }
            } else {
                // Revert move
                walls[wi] = old_pos;
                wall_bb[new_row] = wall_bb[new_row] & ~(1u << new_col);
                wall_bb[old_row] = wall_bb[old_row] | (1u << old_col);
            }
        }

        temp = temp * params.cooling_rate;
    }

    // Write results
    out_scores[tid] = best_score;
    let wb = tid * 64u;
    for (var i = 0u; i < budget; i = i + 1u) {
        out_walls[wb + i] = best_walls[i];
    }
}
