use serde::Deserialize;

// ---------------------------------------------------------------------------
// Tile enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Water,
    Grass,
    Horse,
    Cherry,
    GoldenApple,
    Bee,
    Lovebird,
    Portal,
}

impl Tile {
    /// Parse a single map character into a `Tile`.
    /// Unknown characters are treated as `Grass` with a warning on stderr.
    fn from_char(ch: char) -> Self {
        match ch {
            '~' => Tile::Water,
            '.' => Tile::Grass,
            'H' => Tile::Horse,
            'C' => Tile::Cherry,
            'A' => Tile::GoldenApple,
            'B' => Tile::Bee,
            'L' => Tile::Lovebird,
            'P' => Tile::Portal,
            other => {
                eprintln!(
                    "warning: unknown tile character '{}', treating as Grass",
                    other
                );
                Tile::Grass
            }
        }
    }

    /// Whether the horse can move through this tile (when no wall is placed).
    fn is_passable(self) -> bool {
        self != Tile::Water
    }

    /// Whether a wall can be placed on this tile.
    /// Walls go on grass and bonus tiles, but NOT on water, the horse start,
    /// or portals.
    fn is_wall_placeable(self) -> bool {
        matches!(
            self,
            Tile::Grass | Tile::Cherry | Tile::GoldenApple | Tile::Bee | Tile::Lovebird
        )
    }

    /// The bonus score adjustment when this tile is enclosed, if any.
    fn bonus_score(self) -> Option<i32> {
        match self {
            Tile::Cherry => Some(3),
            Tile::GoldenApple => Some(10),
            Tile::Bee => Some(-5),
            Tile::Lovebird => Some(0),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// PuzzleData (deserialized from the enclose.horse API)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PuzzleData {
    pub id: String,
    pub map: String,
    pub budget: usize,
    pub name: String,
    pub optimal_score: Option<i32>,
    pub has_bonus: Option<bool>,
    pub bonus_type: Option<String>,
    pub daily_date: Option<String>,
    pub day_number: Option<u32>,
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Parsed puzzle grid with pre-computed bitboards for fast flood fill.
#[derive(Debug, Clone)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// Flat array of tiles in row-major order.
    pub tiles: Vec<Tile>,
    /// Flattened index of the horse starting position.
    pub horse_pos: usize,
    /// Indices of tiles where walls can be placed (grass, cherry, apple, bee,
    /// lovebird -- NOT water, horse, or portal).
    pub grass_indices: Vec<usize>,
    /// One `u32` per row. Bit `j` is set if column `j` is passable (not water).
    pub passable_bitboard: Vec<u32>,
    /// One `u32` per row. Bit `j` is set if the cell is on the grid border AND
    /// is passable.
    pub border_bitboard: Vec<u32>,
    /// Portal positions, paired in the order they appear in the map.
    pub portal_pairs: Vec<(usize, usize)>,
    /// `(position, score_adjustment)` for every bonus tile on the grid.
    pub bonus_scores: Vec<(usize, i32)>,
}

impl Grid {
    /// Build a `Grid` from API puzzle data.
    ///
    /// # Panics
    ///
    /// Panics if the map contains no horse tile, if any row width is
    /// inconsistent, or if the grid exceeds 32 columns (bitboard limit).
    pub fn from_puzzle(data: &PuzzleData) -> Self {
        let rows: Vec<&str> = data.map.split('\n').collect();
        let height = rows.len();
        assert!(height > 0, "map must have at least one row");

        let width = rows[0].len();
        assert!(
            width <= 32,
            "grid width ({}) exceeds 32 columns (bitboard limit)",
            width
        );

        let mut tiles = Vec::with_capacity(width * height);
        let mut horse_pos: Option<usize> = None;
        let mut portal_positions: Vec<usize> = Vec::new();
        let mut grass_indices: Vec<usize> = Vec::new();
        let mut bonus_scores: Vec<(usize, i32)> = Vec::new();
        let mut passable_bitboard: Vec<u32> = Vec::with_capacity(height);
        let mut border_bitboard: Vec<u32> = Vec::with_capacity(height);

        for (r, row_str) in rows.iter().enumerate() {
            assert_eq!(
                row_str.len(),
                width,
                "row {} has width {} but expected {}",
                r,
                row_str.len(),
                width,
            );

            let mut passable_bits: u32 = 0;
            let mut border_bits: u32 = 0;

            for (c, ch) in row_str.chars().enumerate() {
                let tile = Tile::from_char(ch);
                let pos = r * width + c;

                if tile == Tile::Horse {
                    assert!(horse_pos.is_none(), "multiple horse tiles found in map");
                    horse_pos = Some(pos);
                }

                if tile == Tile::Portal {
                    portal_positions.push(pos);
                }

                if tile.is_wall_placeable() {
                    grass_indices.push(pos);
                }

                if let Some(bonus) = tile.bonus_score() {
                    bonus_scores.push((pos, bonus));
                }

                if tile.is_passable() {
                    passable_bits |= 1u32 << c;
                }

                let on_border = r == 0 || r == height - 1 || c == 0 || c == width - 1;
                if on_border && tile.is_passable() {
                    border_bits |= 1u32 << c;
                }

                tiles.push(tile);
            }

            passable_bitboard.push(passable_bits);
            border_bitboard.push(border_bits);
        }

        let horse_pos = horse_pos.expect("map must contain exactly one horse tile (H)");

        // Pair portals in the order they appear in the map.
        assert!(
            portal_positions.len().is_multiple_of(2),
            "portals must come in pairs, found {}",
            portal_positions.len(),
        );
        let portal_pairs: Vec<(usize, usize)> = portal_positions
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect();

        Grid {
            width,
            height,
            tiles,
            horse_pos,
            grass_indices,
            passable_bitboard,
            border_bitboard,
            portal_pairs,
            bonus_scores,
        }
    }

    /// Convert a flattened index to `(row, col)`.
    #[inline]
    pub fn pos_to_rc(&self, pos: usize) -> (usize, usize) {
        (pos / self.width, pos % self.width)
    }

    /// Convert `(row, col)` to a flattened index.
    #[inline]
    pub fn rc_to_pos(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a `PuzzleData` from a map string with sensible defaults.
    fn puzzle(map: &str, budget: usize) -> PuzzleData {
        PuzzleData {
            id: "test".into(),
            map: map.into(),
            budget,
            name: "Test Puzzle".into(),
            optimal_score: None,
            has_bonus: None,
            bonus_type: None,
            daily_date: None,
            day_number: None,
        }
    }

    #[test]
    fn tc001_basic_parse() {
        let data = puzzle("~.H\n..~", 0);
        let grid = Grid::from_puzzle(&data);

        assert_eq!(grid.width, 3);
        assert_eq!(grid.height, 2);
        assert_eq!(grid.horse_pos, 2);
        assert_eq!(grid.tiles[0], Tile::Water);
        assert_eq!(grid.tiles[1], Tile::Grass);
        assert_eq!(grid.tiles[2], Tile::Horse);
        assert_eq!(grid.tiles[3], Tile::Grass);
        assert_eq!(grid.tiles[4], Tile::Grass);
        assert_eq!(grid.tiles[5], Tile::Water);
    }

    #[test]
    fn grass_indices_exclude_water_horse_portal() {
        let data = puzzle("..HP\n..P~", 0);
        let grid = Grid::from_puzzle(&data);

        // Placeable: (0,0)=Grass pos=0, (0,1)=Grass pos=1, (1,0)=Grass pos=4, (1,1)=Grass pos=5
        // Not placeable: Horse(pos=2), Portal(pos=3), Portal(pos=6), Water(pos=7)
        assert_eq!(grid.grass_indices, vec![0, 1, 4, 5]);
    }

    #[test]
    fn passable_bitboard_marks_non_water() {
        let data = puzzle("~.H\n..~", 0);
        let grid = Grid::from_puzzle(&data);

        // Row 0: col0=Water(0), col1=Grass(1), col2=Horse(1) -> bits 1,2 set -> 0b110
        assert_eq!(grid.passable_bitboard[0], 0b110);
        // Row 1: col0=Grass(1), col1=Grass(1), col2=Water(0) -> bits 0,1 set -> 0b011
        assert_eq!(grid.passable_bitboard[1], 0b011);
    }

    #[test]
    fn border_bitboard_marks_passable_border_only() {
        // 3x3 grid -- every cell is on the border except (1,1)
        let data = puzzle("...\n.H.\n...", 0);
        let grid = Grid::from_puzzle(&data);

        assert_eq!(grid.border_bitboard[0], 0b111); // top row, all passable
        assert_eq!(grid.border_bitboard[1], 0b101); // middle, only col 0 and 2
        assert_eq!(grid.border_bitboard[2], 0b111); // bottom row, all passable
    }

    #[test]
    fn border_bitboard_excludes_water_on_border() {
        let data = puzzle("~..\n.H.\n..~", 0);
        let grid = Grid::from_puzzle(&data);

        // Row 0: col0=Water (border but not passable), col1,2 passable+border
        assert_eq!(grid.border_bitboard[0], 0b110);
        // Row 2: col0,1 passable+border, col2=Water
        assert_eq!(grid.border_bitboard[2], 0b011);
    }

    #[test]
    fn portal_pairing() {
        let data = puzzle("P..P\n.H..\n..P.\n.P..", 0);
        let grid = Grid::from_puzzle(&data);

        // Portals at pos 0, 3, 10, 13 in reading order
        assert_eq!(grid.portal_pairs, vec![(0, 3), (10, 13)]);
    }

    #[test]
    fn bonus_scores_computed() {
        let data = puzzle("C.A\n.H.\nB.L", 0);
        let grid = Grid::from_puzzle(&data);

        assert_eq!(
            grid.bonus_scores,
            vec![(0, 3), (2, 10), (6, -5), (8, 0)],
        );
    }

    #[test]
    fn pos_rc_roundtrip() {
        let data = puzzle("...\n.H.\n...", 0);
        let grid = Grid::from_puzzle(&data);

        assert_eq!(grid.pos_to_rc(4), (1, 1));
        assert_eq!(grid.rc_to_pos(1, 1), 4);
        assert_eq!(grid.pos_to_rc(grid.rc_to_pos(2, 2)), (2, 2));
    }

    #[test]
    #[should_panic(expected = "map must contain exactly one horse tile")]
    fn panics_without_horse() {
        let data = puzzle("...\n...\n...", 0);
        Grid::from_puzzle(&data);
    }

    #[test]
    #[should_panic(expected = "portals must come in pairs")]
    fn panics_on_odd_portals() {
        let data = puzzle("P..\n.H.\n...", 0);
        Grid::from_puzzle(&data);
    }

    #[test]
    fn unknown_char_treated_as_grass() {
        let data = puzzle("X.H", 0);
        let grid = Grid::from_puzzle(&data);

        assert_eq!(grid.tiles[0], Tile::Grass);
        assert!(grid.grass_indices.contains(&0));
    }
}
