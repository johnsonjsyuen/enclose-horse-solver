# optimal-solver

A provably optimal solver for [enclose.horse](https://enclose.horse) puzzles using Mixed-Integer Programming (MIP) via the [HiGHS](https://highs.dev/) solver.

Unlike the simulated annealing solver in the parent crate, this finds **guaranteed optimal** solutions by formulating the puzzle as an integer linear program.

## How it works

The solver models the puzzle as a MIP with:

- **Binary variables** for wall placement (`w[i]`) and cell reachability (`r[i]`)
- **Single-commodity flow conservation** to prevent fictitious disconnected islands — each reachable cell must consume exactly 1 unit of flow originating from the horse
- **Propagation constraints** to force reachability to spread through unblocked neighbors
- **Portal constraints** for bidirectional teleportation
- **Border escape penalty** with big-M to ensure the horse is fully enclosed

The objective maximizes enclosed area plus bonus tile scores (cherries +3, golden apples +10, gems +10, bees -5, skulls -5) while penalizing any path to the grid border.

Results are verified against the library's flood-fill implementation.

## Usage

```bash
# Solve today's puzzle
cargo run --release -p optimal-solver

# Solve a specific date
cargo run --release -p optimal-solver -- 2026-03-21
```

### Example output

```
Puzzle: "Eruption" (Day 82, 2026-03-21)
Budget: 10 walls
Optimal: 97

Score: 97 (area=97, bonus=0)
Walls (10): (0,1) (0,8) (0,9) (4,14) (5,0) (10,2) (10,14) (11,13) (12,12) (13,6)

   012345678901234
 0 ~#~~~~~~##~~░░░
 1 ~.........~~~░░
 2 ~..~~~.~...~~~░
 3 ~.~~~..~~~..~~░
 4 ~.~.....~~~...#
 5 #........~~.~~~
 6 ~~...~......~~~
 7 ~~~~.........~~
 8 ~..~~...H..~.~░
 9 ~~..~........~░
10 ░░#...........#
11 ~░░~..~~.....#░
12 ~░░░~..~~~..#░░
13 ~░~░░~#░░~~~░░░
14 ~░~~░░░░░░░~~~░
15 ~░░~~░~~~~░░░~~
```

Legend: `#` = wall, `░` = enclosed area, `~` = water, `H` = horse

## Dependencies

- [HiGHS](https://highs.dev/) via the [`highs`](https://crates.io/crates/highs) crate — high-performance open-source MIP solver
- `enclose-horse-solver` parent crate for grid parsing and flood-fill verification
