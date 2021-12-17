use itertools::Itertools;

use Part::*;
use crate::{Solution, SolutionPair};
use std::{fs::read_to_string, collections::BinaryHeap};
use rustc_hash::FxHashMap;

///////////////////////////////////////////////////////////////////////////////

type CoordItem = i32;
type Coord = (CoordItem, CoordItem);
const MOVE_DIRS: [Coord; 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)]; 

struct CaveInfo {
    data: Vec<u32>,
    rows: usize,
    cols: usize,
    rows_p2: usize,
    cols_p2: usize,
}

#[derive(Copy, Clone, Eq)]
struct AStarState {
    pub cost: u32,
    pub node: Coord,
}

#[derive(Copy, Clone)]
enum Part { Part1, Part2 }

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day15.txt").unwrap();
    let cave_data = CaveInfo::from_string(&input);

    let start = (0, 0);
    let (target_p1, target_p2) = cave_data.get_targets();

    let sol1 = a_star(start, target_p1, &cave_data, Part1) as u64;
    let sol2 = a_star(start, target_p2, &cave_data, Part2) as u64;

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn a_star(from: Coord, to: Coord, cave_info: &CaveInfo, part: Part) -> u32 {
    let mut to_visit = BinaryHeap::new();
    let mut visited_costs = FxHashMap::default();

    to_visit.push(AStarState { cost: 0, node: from });

    while let Some(AStarState { cost, node }) = to_visit.pop() {
        if node == to {
            return cost;
        }

        for mv in MOVE_DIRS {
            let new_node = (node.0 + mv.0, node.1 + mv.1);
            if !cave_info.coord_is_ok(new_node, part) {
                continue;
            }

            let new_cost = cost + cave_info.get_value(new_node);

            let prev_cost = match visited_costs.get(&new_node) {
                Some(val) => *val,
                None => u32::MAX,
            };

            if new_cost < prev_cost {
                visited_costs.insert(new_node, new_cost);
                to_visit.push(AStarState {
                    cost: new_cost,
                    node: new_node
                });
            }
        }
    }

    u32::MAX
}

///////////////////////////////////////////////////////////////////////////////

impl CaveInfo {
    pub fn from_string(input: &str) -> Self {
        let data = input.lines().flat_map(|line| {
            line.chars().map(|ch| ch as u32 - '0' as u32)
        }).collect_vec();

        let cols = input.lines().next().unwrap().len();
        let rows = data.len() / cols;

        Self { data, rows, cols, rows_p2: rows * 5, cols_p2: cols * 5 }
    }

    pub fn coord_is_ok(&self, (x, y): Coord, part: Part) -> bool {
        let (max_x, max_y) = match part {
            Part1 => (self.cols, self.rows),
            Part2 => (self.cols_p2, self.rows_p2),
        };
        x >= 0 && x < max_x as i32 && y >= 0 && y < max_y as i32
    }

    pub fn get_targets(&self) -> (Coord, Coord) {
        ((self.cols as CoordItem - 1, self.rows as CoordItem - 1),
         (self.cols_p2 as CoordItem - 1, self.rows_p2 as CoordItem - 1))
    }

    pub fn get_value(&self, (x, y): Coord) -> u32 {
        let (actual_x, ovfx) = (x as usize % self.cols, x as usize / self.cols);
        let (actual_y, ovfy) = (y as usize % self.rows, y as usize / self.rows);
        let val = self.data[actual_x as usize + actual_y as usize * self.cols] + ovfx as u32 + ovfy as u32;
        val % 10 + val / 10
    }
}

impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.cmp(&self.cost)
    }
}