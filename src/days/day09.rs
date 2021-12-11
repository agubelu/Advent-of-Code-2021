use itertools::Itertools;
use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

struct HeightMap {
    pub heights: Vec<u8>,
    pub visited: Vec<bool>,
    pub rows: usize,
    pub columns: usize
}

pub fn solve() -> SolutionPair {
    let s = read_to_string("input/day09.txt").unwrap();
    let mut height_data = HeightMap::from_str(&s);

    let mut sol1 = 0;
    let mut basins = vec![];

    for pos in 0..height_data.heights.len() {
        let basin_size = get_basin_size(&mut height_data, pos);
        if basin_size > 0 {
            basins.push(basin_size);
            sol1 += height_data.heights[pos] as u64 + 1;
        }
    }

    basins.sort_unstable_by(|a, b| b.cmp(a));
    let sol2 = basins[..3].iter().product();

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn get_basin_size(height_data: &mut HeightMap, pos: usize) -> u64 {
    let val = height_data.heights[pos];
    if val == 9 || height_data.visited[pos] {
        return 0;
    }

    let nearby_unvisited = height_data.nearby(pos).into_iter().filter(|&i| !height_data.visited[i]).collect_vec();
    let is_min = nearby_unvisited.iter().all(|&pos| height_data.heights[pos] >= val);

    if is_min {
        height_data.visited[pos] = true;
        1 + nearby_unvisited.iter().map(|&i| get_basin_size(height_data, i)).sum::<u64>()
    } else {
        0
    }
}

impl HeightMap {
    pub fn from_str(s: &str) -> Self {
        let lines = s.lines().collect_vec();
        let rows = lines.len();
        let columns = lines[0].len();
        let heights = lines.into_iter().flat_map(|line| line.chars().map(|ch| ch as u8 - '0' as u8)).collect_vec();
        Self { visited: vec![false; heights.len()], rows, columns, heights }
    }

    pub fn nearby(&self, pos: usize) -> Vec<usize> {
        let x = pos % self.columns;
        let y = pos / self.columns;

        match (x, y) {
            (0, 0) => vec![pos + 1, pos + self.columns], //top-left
            (i, 0) if i == self.columns - 1 => vec![pos -1, pos + self.columns], // top-right
            (0, j) if j == self.rows - 1 => vec![pos + 1, pos - self.columns], //bottom-left
            (i, j) if i == self.columns - 1 && j == self.rows - 1 => vec![pos - 1, pos - self.columns], //bottom-right
            (_, 0) => vec![pos + 1, pos -1, pos + self.columns], // top
            (0, _) => vec![pos + 1, pos + self.columns, pos - self.columns], // left
            (i, _) if i == self.columns - 1 => vec![pos - 1, pos + self.columns, pos - self.columns], // right
            (_, j) if j == self.rows - 1 => vec![pos - 1, pos + 1, pos - self.columns], //bottom
            _ => vec![pos + 1, pos - 1, pos + self.columns, pos - self.columns] // any other
        }
    }
}
