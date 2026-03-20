# Enclose Horse Solver — Implementation Spec

**Type:** Implementation Document
**Status:** Phase 2

## 1. Problem Definition

Given a grid from [enclose.horse](https://enclose.horse), place exactly `budget` walls on grass tiles to maximize the enclosed area around the horse. The horse moves orthogonally and cannot cross walls or water. If the horse can reach any border cell, it escapes and the enclosed area is 0.

**Score** = (number of grass tiles reachable by horse but not reachable to any border) + bonus adjustments.

## 2. Data Acquisition

### API Endpoint

```
GET https://enclose.horse/api/daily/{YYYY-MM-DD}
```

### Response Schema

```json
{
  "id": "string",
  "map": "string (newline-separated rows)",
  "budget": "integer (number of walls to place)",
  "name": "string",
  "optimalScore": "integer",
  "hasBonus": "boolean",
  "bonusType": "string | null (cherries | golden_apples | bees | lovebirds | portals | null)",
  "dailyDate": "string (YYYY-MM-DD)",
  "dayNumber": "integer"
}
```

### Tile Encoding

| Char | Meaning | Passable | Wall-placeable |
|------|---------|----------|----------------|
| `~` | Water | No | No |
| `.` | Grass | Yes | Yes |
| `H` | Horse start | Yes | No |
| `C` | Cherry (+3 bonus) | Yes | Yes |
| `A` | Golden Apple (+10 bonus) | Yes | Yes |
| `B` | Bee Swarm (-5 penalty) | Yes | Yes |
| `L` | Lovebird (scoring TBD) | Yes | Yes |
| `P` | Portal (teleport) | Yes | No |

**Implementation Implication:** Parse map string into a `Grid` struct with enum tiles. Unknown characters should be treated as grass with a warning to stderr.

## 3. Core Algorithm: Simulated Annealing

### State Representation

- **State**: A `Vec<(u8, u8)>` of `budget` wall positions on grass tiles
- **Initial state**: Random valid wall placements
- **Neighbor generation**: Swap one random wall to a random unoccupied grass tile

### Energy Function

`energy = -score` (we minimize energy = maximize score)

Where `score` is computed by:
1. Build grid with current walls placed
2. Flood fill from horse position (BFS, blocked by walls + water)
3. If flood fill reaches any border cell → score = 0
4. Otherwise → score = flood fill area + bonus tile adjustments

### SA Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial temperature | 50.0 | High enough to escape local optima on 17x17 grid |
| Cooling rate | 0.9997 | Slow cooling for thorough exploration |
| Min temperature | 0.01 | Stop threshold |
| Iterations per temp | 100 | Batch size before cooling |
| Restarts | 8 | Multi-start to cover solution space |
| Max total iterations | 5,000,000 | Hard cap for runtime bound |

**Implementation Implication:** Run 8 independent SA runs (one per thread via `rayon`), take best result. Each run gets ~625K iterations.

### Neighbor Function Detail

1. Pick random wall index `i` in `[0, budget)`
2. Remove wall at index `i`
3. Pick random grass tile not in current wall set and not the horse tile
4. Insert as new wall at index `i`
5. Evaluate new score

**Implementation Implication:** Maintain a `HashSet<usize>` of wall positions (flattened index) for O(1) membership check. Maintain a `Vec<usize>` of candidate grass tiles for O(1) random selection.

## 4. SIMD-Accelerated Flood Fill (Bitboard)

### Bitboard Representation

For grids up to 32 columns wide, represent each row as a `u32` bitmask. A grid of height H is `[u32; H]`.

Maintain three bitboards:
- `passable[H]`: 1 = tile is passable (grass/horse/bonus, not water/wall)
- `reached[H]`: 1 = tile reached by flood fill
- `border_mask[H]`: 1 = tile is on grid border

### SIMD Flood Fill Algorithm

Each flood fill iteration expands `reached` by OR-ing with shifted neighbors, masked by `passable`:

```
for each row r:
    expand = reached[r]
    expand |= reached[r] << 1       // right neighbor
    expand |= reached[r] >> 1       // left neighbor
    if r > 0: expand |= reached[r-1] // up neighbor
    if r < H-1: expand |= reached[r+1] // down neighbor
    reached[r] = expand & passable[r]
```

Repeat until no change (fixed point).

### Portable SIMD via `std::simd`

Use `std::simd::Simd<u32, 8>` to process 8 rows simultaneously:
- Load 8 rows of `reached` into a `Simd<u32, 8>`
- Perform shift-left, shift-right via SIMD lane ops
- Neighbor-row propagation via `rotate_elements_left/right` on the SIMD vector
- AND with `passable` mask via `&` operator

**Implementation Implication:** Use `std::simd` portable SIMD (requires nightly or Rust 1.93+ with `feature(portable_simd)`). No platform-specific intrinsics needed — the compiler maps to AVX2 on this i9-12950HX when built with `-C target-cpu=native`.

### Score Extraction with SIMD

After flood fill converges:
1. Check escape: `(reached & border_mask) != Simd::splat(0)` lane-wise, reduce with `any()`
2. Count enclosed: per-lane `count_ones()` then `reduce_sum()`
3. All portable, no `std::arch` needed

## 5. Bonus Tile Handling

After flood fill, scan enclosed tiles for bonus characters:

| Bonus Type | Per-tile Effect | Detection |
|------------|----------------|-----------|
| cherries | score += 3 | Tile == `C` and tile is in enclosed set |
| golden_apples | score += 10 | Tile == `A` and tile is in enclosed set |
| bees | score -= 5 | Tile == `B` and tile is in enclosed set |
| lovebirds | Unknown — treat as +0 | Log warning |
| portals | Horse can teleport between portal pairs | Flood fill must propagate through portals |

**Implementation Implication:** Portal handling requires: when flood fill reaches a portal tile, also seed the paired portal tile. Store portal pairs during parsing.

## 6. Output Format

```
Puzzle: "Fortress" (Day 81, 2026-03-20)
Budget: 9 walls
Optimal: 80

Solution found: score = 80
Walls at: (3,2) (4,1) (5,0) (7,5) (8,3) (8,4) (9,3) (10,3) (11,5)

Map:
~~~~..~..~.~~~~~~
~~..........~~~~~
~..W...~..~..~~~~
.W..~........~~~~
...~~~...~.....~~
W.~~~~~.....~...~
.~~~~~~~...~~~...
......~W..~~~~~..
....~W~WH~~~~~~~.
..~W~.~...~...~..
..~W~.....~.~....
......~.........W
..~...~~.~~~.....
..~...~..........
..~...........~..
..~~~~~~~~~~~~~..
~~~~~~~~~~~~~~~~~
```

Where `W` marks placed walls. Print to stdout.

## 7. Build Configuration

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
target-cpu = "native"
```

Build command: `RUSTFLAGS="-C target-cpu=native" cargo build --release`

This enables AVX2/AVX-VNNI on the host i9-12950HX.

## 8. Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `reqwest` | 0.12 | HTTP client (blocking) |
| `serde` | 1.0 | JSON deserialization |
| `serde_json` | 1.0 | JSON parsing |
| `rand` | 0.9 | RNG for SA |
| `rayon` | 1.10 | Parallel SA restarts |
| `chrono` | 0.4 | Today's date formatting |

## 9. Module Structure

```
src/
  main.rs          — CLI entry, fetch puzzle, run solver, print result
  grid.rs          — Grid parsing, tile types, bitboard construction
  solver.rs        — Simulated annealing loop
  flood_fill.rs    — SIMD-accelerated flood fill + score computation
```

---

## ANTI-PATTERNS (DO NOT)

| Don't | Do Instead | Why |
|-------|-----------|-----|
| Re-flood-fill from scratch each neighbor eval | Incrementally update only if wall moved affects reach | Performance: flood fill is the hot loop |
| Use `Vec<Vec<Tile>>` for grid | Use flat `[Tile; MAX_CELLS]` + bitboards | Cache locality and SIMD compatibility |
| Place walls on water or horse tile | Filter candidate list during init | Invalid states waste iterations |
| Use f64 for temperature on hot path | Use f64 only for SA control, not score | Score is integer, avoid float conversion |
| Allocate per iteration | Pre-allocate all buffers, clone/reset | Allocation is the #1 SA performance killer |
| Use async reqwest | Use `reqwest::blocking` | Single request, async adds complexity for no gain |
| Ignore portal pairs | Implement portal teleportation in flood fill | Portals change reachability fundamentally |

## TEST CASE SPECIFICATIONS

### Unit Tests

| Test ID | Component | Input | Expected Output | Edge Cases |
|---------|-----------|-------|-----------------|------------|
| TC-001 | Grid parse | "~.H\n..~" | 3x2 grid, horse at (2,0) | Empty map, single cell |
| TC-002 | Flood fill (no walls) | 3x3 all grass, horse center | All 9 tiles reached | Horse on edge |
| TC-003 | Flood fill (enclosed) | 5x5, horse center, walls surrounding | Inner tiles only | Diagonal gap (should NOT leak) |
| TC-004 | Flood fill (escaped) | Horse at (0,0), no walls | Score = 0 | Border horse always escapes without walls |
| TC-005 | Score with cherries | Enclosed area=10, 2 cherries inside | Score = 16 | Cherry on wall position |
| TC-006 | Bitboard row ops | Row 0b11011, shift left | 0b10110 | Overflow bits masked |
| TC-007 | Portal flood fill | Two portals, horse near portal A | Reaches portal B's area | Portal on border |

### Integration Tests

| Test ID | Flow | Setup | Verification | Teardown |
|---------|------|-------|--------------|----------|
| IT-001 | Full solve known puzzle | Hardcoded small 5x5 puzzle, budget=2, known optimal=8 | Score >= 8 | None |
| IT-002 | API fetch + parse | Mock or live fetch today's puzzle | Grid dimensions match, horse found, budget > 0 | None |
| IT-003 | SA convergence | Run SA on known puzzle 10 times | All runs find optimal | None |

## ERROR HANDLING MATRIX

| Error Type | Detection | Response | Fallback | Logging |
|------------|-----------|----------|----------|---------|
| API unreachable | reqwest timeout (10s) | Exit with error message | None — puzzle data required | stderr |
| API 404 | HTTP 404 | "No puzzle for date {date}" | Try yesterday's date | stderr |
| Invalid map chars | Unknown char in map | Treat as grass | Log warning per unknown char | stderr |
| No horse in map | Post-parse check | Exit with error | None | stderr |
| SA finds score 0 | All restarts return 0 | Report "no enclosure possible" | None | stderr |
| SIMD not available | `cfg` check at compile time | Fall back to scalar flood fill | Scalar implementation | None (compile-time) |

## REFERENCES

| Topic | Location |
|-------|----------|
| enclose.horse API | `GET https://enclose.horse/api/daily/{date}` |
| Existing solver (Python/ASP) | [EncloseHorseBreaker](https://github.com/pierreeurope/EncloseHorseBreaker) |
| ILP formulation | [dynomight.substack.com/p/horse](https://dynomight.substack.com/p/horse) |
| Graph theory analysis | [Menger's Horse Enclosure](https://buttondown.com/jaffray/archive/mengers-horse-enclosure/) |
