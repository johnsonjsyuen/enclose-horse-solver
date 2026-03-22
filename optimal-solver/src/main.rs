use std::collections::{HashMap, HashSet};
use std::env;
use std::process;

use chrono::Local;
use highs::{HighsModelStatus, RowProblem, Sense};

use enclose_horse_solver::flood_fill::FloodFillState;
use enclose_horse_solver::grid::{Grid, PuzzleData};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_passable(grid: &Grid, pos: usize) -> bool {
    let (r, c) = grid.pos_to_rc(pos);
    grid.passable_bitboard[r] & (1u32 << c) != 0
}

fn is_border(grid: &Grid, pos: usize) -> bool {
    let (r, c) = grid.pos_to_rc(pos);
    r == 0 || r == grid.height - 1 || c == 0 || c == grid.width - 1
}

fn neighbors(grid: &Grid, pos: usize) -> Vec<usize> {
    let (r, c) = grid.pos_to_rc(pos);
    let mut result = Vec::with_capacity(4);
    if r > 0 { result.push(grid.rc_to_pos(r - 1, c)); }
    if r + 1 < grid.height { result.push(grid.rc_to_pos(r + 1, c)); }
    if c > 0 { result.push(grid.rc_to_pos(r, c - 1)); }
    if c + 1 < grid.width { result.push(grid.rc_to_pos(r, c + 1)); }
    result
}

// ---------------------------------------------------------------------------
// Puzzle fetching
// ---------------------------------------------------------------------------

fn fetch_puzzle(date: &str) -> PuzzleData {
    let url = format!("https://enclose.horse/api/daily/{}", date);
    eprintln!("Fetching puzzle from {}...", url);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); process::exit(1); });

    let response = client.get(&url).send()
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); process::exit(1); });

    if !response.status().is_success() {
        eprintln!("API returned status {}", response.status());
        process::exit(1);
    }

    let body = response.text()
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); process::exit(1); });

    serde_json::from_str(&body)
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); process::exit(1); })
}

// ---------------------------------------------------------------------------
// MIP formulation (flow-based + propagation for exact reachability)
//
// Variables:
//   w[i] ∈ {0,1}   wall on grass cell i
//   r[i] ∈ {0,1}   cell i reachable from horse
//   f[i→j] ∈ [0,N] flow on directed edge (i,j) — continuous
//   escaped ∈ {0,1}
//
// The flow ensures r[i]=1 only if connected to horse (no islands).
// The propagation ensures r[i]=1 whenever horse CAN reach i (no gaps).
// Together they give r[i] = 1 iff horse can reach i.
//
// Flow constraints (prevents islands):
//   For i ≠ horse: r[i] <= sum(incoming flow)
//   f[i→j] <= r[i] * N         (flow needs sender reachable)
//   f[i→j] <= (1 - w[j]) * N   (flow blocked by wall on receiver)
//
// Propagation constraints (forces spreading):
//   r[i] >= r[j] - w[i]        (if neighbor j reachable & i not walled → i reachable)
//
// N = number of passable cells (max possible flow).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ColIdx {
    col: highs::Col,
    idx: usize,
}

impl ColIdx {
    fn val(self, solution: &[f64]) -> f64 {
        solution[self.idx]
    }
}

fn solve_mip(grid: &Grid, budget: usize) -> (i32, Vec<usize>) {
    let total_cells = grid.width * grid.height;
    let mut pb = RowProblem::default();
    let mut col_count: usize = 0;

    macro_rules! add_binary {
        ($pb:expr, $obj:expr) => {{
            let col = $pb.add_integer_column($obj, 0.0..=1.0);
            let idx = col_count; col_count += 1;
            ColIdx { col, idx }
        }};
    }

    // --- Score map ---
    let mut cell_score: Vec<f64> = (0..total_cells)
        .map(|pos| if is_passable(grid, pos) { 1.0 } else { 0.0 })
        .collect();
    for &(pos, bonus) in &grid.bonus_scores {
        cell_score[pos] += bonus as f64;
    }

    let big_m = (grid.width * grid.height + 200) as f64;

    // --- Passable cells and edges ---
    let passable_cells: Vec<usize> = (0..total_cells)
        .filter(|&pos| is_passable(grid, pos))
        .collect();
    let n_passable = passable_cells.len() as f64;

    // Directed edges (grid adjacency)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for &pos in &passable_cells {
        for nbr in neighbors(grid, pos) {
            if is_passable(grid, nbr) {
                edges.push((pos, nbr));
            }
        }
    }

    // --- Variables ---

    // w[i]: wall (binary)
    let mut w: HashMap<usize, ColIdx> = HashMap::new();
    for &pos in &grid.grass_indices {
        w.insert(pos, add_binary!(pb, 0.0));
    }

    // r[i]: reachable (binary, with score as objective coeff)
    let mut r: HashMap<usize, ColIdx> = HashMap::new();
    for &pos in &passable_cells {
        r.insert(pos, add_binary!(pb, cell_score[pos]));
    }

    // f[i→j]: flow (continuous, [0, N])
    let mut f: HashMap<(usize, usize), ColIdx> = HashMap::new();
    for &(i, j) in &edges {
        let col = pb.add_column(0.0, 0.0..=n_passable);
        let idx = col_count; col_count += 1;
        f.insert((i, j), ColIdx { col, idx });
    }

    // Portal flow: fp[a→b] and fp[b→a] (continuous, [0, N])
    let mut fp: HashMap<(usize, usize), ColIdx> = HashMap::new();
    for &(a, b) in &grid.portal_pairs {
        let col_ab = pb.add_column(0.0, 0.0..=n_passable);
        let idx_ab = col_count; col_count += 1;
        fp.insert((a, b), ColIdx { col: col_ab, idx: idx_ab });

        let col_ba = pb.add_column(0.0, 0.0..=n_passable);
        let idx_ba = col_count; col_count += 1;
        fp.insert((b, a), ColIdx { col: col_ba, idx: idx_ba });
    }

    // escaped (binary)
    let escaped = add_binary!(pb, -big_m);

    let n_binary = w.len() + r.len() + 1;
    let n_continuous = f.len() + fp.len();

    // --- Constraints ---

    // 1. Budget
    {
        let terms: Vec<_> = grid.grass_indices.iter().map(|&pos| (w[&pos].col, 1.0)).collect();
        pb.add_row(..=budget as f64, &terms);
    }

    // 2. Wall blocks reachability: r[i] + w[i] <= 1
    for &pos in &grid.grass_indices {
        if let Some(&ri) = r.get(&pos) {
            pb.add_row(..=1.0, [(ri.col, 1.0), (w[&pos].col, 1.0)]);
        }
    }

    // 3. Horse is reachable
    pb.add_row(1.0..=1.0, [(r[&grid.horse_pos].col, 1.0)]);

    // 4. PROPAGATION (forces r=1 to spread):
    //    r[i] >= r[j] - w[i] for each passable neighbor j of i
    for &i in &passable_cells {
        let ri = r[&i];
        for j in neighbors(grid, i) {
            if !is_passable(grid, j) { continue; }
            let rj = r[&j];
            if let Some(&wi) = w.get(&i) {
                // r[i] - r[j] + w[i] >= 0
                pb.add_row(0.0.., [(ri.col, 1.0), (rj.col, -1.0), (wi.col, 1.0)]);
            } else {
                // r[i] >= r[j]
                pb.add_row(0.0.., [(ri.col, 1.0), (rj.col, -1.0)]);
            }
        }
    }

    // 5. Portal propagation: r[a] >= r[b] and r[b] >= r[a]
    for &(a, b) in &grid.portal_pairs {
        if let (Some(&ra), Some(&rb)) = (r.get(&a), r.get(&b)) {
            pb.add_row(0.0.., [(ra.col, 1.0), (rb.col, -1.0)]);
            pb.add_row(0.0.., [(rb.col, 1.0), (ra.col, -1.0)]);
        }
    }

    // 6. SINGLE-COMMODITY FLOW (prevents islands):
    //    Horse is source. Each reachable cell consumes 1 unit.
    //    Flow conservation: inflow - outflow = r[i] for i ≠ horse
    //    For horse: outflow - inflow = sum(r[i] for all i ≠ horse) ... but that's
    //    hard to express. Instead: for i ≠ horse, inflow - outflow = r[i].
    //    The horse has unlimited supply (no conservation constraint).
    let mut incoming: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut outgoing: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for &(i, j) in &edges {
        incoming.entry(j).or_default().push((i, j));
        outgoing.entry(i).or_default().push((i, j));
    }

    // Also track portal edges for flow conservation
    let mut portal_incoming: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut portal_outgoing: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for &(a, b) in &grid.portal_pairs {
        portal_incoming.entry(b).or_default().push((a, b));
        portal_outgoing.entry(a).or_default().push((a, b));
        portal_incoming.entry(a).or_default().push((b, a));
        portal_outgoing.entry(b).or_default().push((b, a));
    }

    // Flow conservation: for i ≠ horse:
    //   sum(f_in) + sum(fp_in) - sum(f_out) - sum(fp_out) = r[i]
    //   => sum(f_in) + sum(fp_in) - sum(f_out) - sum(fp_out) - r[i] = 0
    for &pos in &passable_cells {
        if pos == grid.horse_pos { continue; }
        let mut terms: Vec<(highs::Col, f64)> = Vec::new();
        // Incoming grid flow
        if let Some(inc) = incoming.get(&pos) {
            for &(j, i) in inc {
                terms.push((f[&(j, i)].col, 1.0));
            }
        }
        // Incoming portal flow
        if let Some(inc) = portal_incoming.get(&pos) {
            for &key in inc {
                if let Some(&fpv) = fp.get(&key) {
                    terms.push((fpv.col, 1.0));
                }
            }
        }
        // Outgoing grid flow
        if let Some(out) = outgoing.get(&pos) {
            for &(i, j) in out {
                terms.push((f[&(i, j)].col, -1.0));
            }
        }
        // Outgoing portal flow
        if let Some(out) = portal_outgoing.get(&pos) {
            for &key in out {
                if let Some(&fpv) = fp.get(&key) {
                    terms.push((fpv.col, -1.0));
                }
            }
        }
        // Consumption: -r[i]
        terms.push((r[&pos].col, -1.0));
        pb.add_row(0.0..=0.0, &terms); // exact equality
    }

    // 7. Flow capacity: f[i→j] <= N * r[i] (only flows through reachable cells)
    for &(i, _j) in &edges {
        pb.add_row(..=0.0, [(f[&(i, _j)].col, 1.0), (r[&i].col, -n_passable)]);
    }

    // 8a. Flow blocked by wall on sender: f[i→j] + N*w[i] <= N
    for &(i, j) in &edges {
        if let Some(&wi) = w.get(&i) {
            pb.add_row(..=n_passable, [(f[&(i, j)].col, 1.0), (wi.col, n_passable)]);
        }
    }

    // 8b. Flow blocked by wall on receiver: f[i→j] + N*w[j] <= N
    for &(i, j) in &edges {
        if let Some(&wj) = w.get(&j) {
            pb.add_row(..=n_passable, [(f[&(i, j)].col, 1.0), (wj.col, n_passable)]);
        }
    }

    // 9. Portal flow capacity: fp[a→b] <= N * r[a]
    for &(a, b) in &grid.portal_pairs {
        if let (Some(&fpv), Some(&ra)) = (fp.get(&(a, b)), r.get(&a)) {
            pb.add_row(..=0.0, [(fpv.col, 1.0), (ra.col, -n_passable)]);
        }
        if let (Some(&fpv), Some(&rb)) = (fp.get(&(b, a)), r.get(&b)) {
            pb.add_row(..=0.0, [(fpv.col, 1.0), (rb.col, -n_passable)]);
        }
    }

    // 10. Border escape: r[b] <= escaped
    for &pos in &passable_cells {
        if is_border(grid, pos) {
            pb.add_row(..=0.0, [(r[&pos].col, 1.0), (escaped.col, -1.0)]);
        }
    }

    // --- Solve ---
    eprintln!(
        "MIP: {} columns ({} binary, {} continuous), solving...",
        col_count, n_binary, n_continuous
    );

    let solved = pb.optimise(Sense::Maximise).solve();
    let status = solved.status();

    if status != HighsModelStatus::Optimal {
        eprintln!("HiGHS returned status: {:?}", status);
        process::exit(1);
    }

    let solution = solved.get_solution().columns().to_vec();
    let horse_escaped = escaped.val(&solution) > 0.5;

    let mut score: f64 = 0.0;
    for &pos in &passable_cells {
        if r[&pos].val(&solution) > 0.5 {
            score += cell_score[pos];
        }
    }
    if horse_escaped { score -= big_m; }

    let walls: Vec<usize> = grid.grass_indices.iter()
        .filter(|&&pos| w[&pos].val(&solution) > 0.5)
        .copied().collect();

    let final_score = score.round() as i32;
    eprintln!("MIP solution: score={}, walls={}, escaped={}", final_score, walls.len(), horse_escaped);

    (final_score, walls)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let date = args.first().cloned()
        .unwrap_or_else(|| Local::now().format("%Y-%m-%d").to_string());

    let puzzle = fetch_puzzle(&date);
    let grid = Grid::from_puzzle(&puzzle);
    let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);

    eprintln!(
        "Grid: {}x{}, horse at ({},{}), budget: {}, candidates: {}",
        grid.width, grid.height, horse_col, horse_row, puzzle.budget, grid.grass_indices.len()
    );

    let solve_start = std::time::Instant::now();
    let (mip_score, walls) = solve_mip(&grid, puzzle.budget);
    let solve_elapsed = solve_start.elapsed();

    // Verify with flood fill.
    let mut ff = FloodFillState::new(&grid);
    let ff_result = ff.evaluate(&grid, &walls);
    let verified_score = ff_result.score;

    if verified_score != mip_score {
        eprintln!(
            "WARNING: MIP score ({}) != flood fill score ({}). Using flood fill result.",
            mip_score, verified_score
        );
    }

    let mut wall_positions: Vec<(usize, usize)> =
        walls.iter().map(|&idx| grid.pos_to_rc(idx)).collect();
    wall_positions.sort();

    println!(
        "Puzzle: \"{}\" (Day {}, {})",
        puzzle.name, puzzle.day_number.unwrap_or(0),
        puzzle.daily_date.as_deref().unwrap_or("unknown"),
    );
    println!("Budget: {} walls", puzzle.budget);
    println!(
        "Optimal: {}",
        puzzle.optimal_score.map_or("unknown".to_string(), |s| s.to_string())
    );
    println!();
    println!("Score: {} (area={}, bonus={}, solved in {:.2?})", verified_score, ff_result.area, ff_result.bonus, solve_elapsed);

    let wall_strs: Vec<String> = wall_positions.iter()
        .map(|(r, c)| format!("({},{})", r, c)).collect();
    println!("Walls ({}): {}", walls.len(), wall_strs.join(" "));
    println!();

    // Build map: show walls, enclosed area, and original tiles
    let wall_set: HashSet<usize> = walls.iter().copied().collect();
    let rows: Vec<&str> = puzzle.map.lines().collect();

    // Column header
    print!("   ");
    for c in 0..grid.width {
        print!("{}", c % 10);
    }
    println!();

    for (r, row_str) in rows.iter().enumerate() {
        print!("{:2} ", r);
        let line: String = row_str.chars().enumerate()
            .map(|(c, ch)| {
                let pos = r * grid.width + c;
                if wall_set.contains(&pos) {
                    '#'
                } else if ff_result.reached[r] & (1u32 << c) != 0 {
                    // Reachable by horse — show original tile
                    ch
                } else if ch == '~' {
                    '~'
                } else {
                    // Not reachable and not water — enclosed area
                    '░'
                }
            }).collect();
        println!("{}", line);
    }
}
