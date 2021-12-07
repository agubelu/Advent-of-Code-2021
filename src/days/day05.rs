use std::fs::read_to_string;
use std::cmp::{Ordering::*, max};
use rustc_hash::FxHashMap;
use itertools::Itertools;

use crate::{Solution, SolutionPair};

///////////////////////////////////////////////////////////////////////////////

type CoordType = i16;

trait PointCounter {
    fn update(&mut self, point: Coord, increment: Coord);
    fn count(&self) -> Coord;
}

pub fn solve() -> SolutionPair {
    let coords = read_to_string("input/day05.txt").unwrap()
        .lines()
        .map(CoordPair::from_line)
        .collect_vec();
        
    let max = coords.iter().map(|x| x.max()).max().unwrap();

    let (sol1, sol2) = if max <= 1000 {
        solve_with(&coords, VecCounter::new(max))
    } else {
        solve_with(&coords, MapCounter::new())
    };

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn solve_with(coords: &[CoordPair], mut solver: impl PointCounter) -> (u64, u64) {
    coords.iter().for_each(|coord| {
        let inc_1 = if coord.is_not_diagonal() {1} else {0};
        let increment = Coord {x: inc_1, y: 1};

        coord.points_iter().for_each(|p| solver.update(p, increment));
    });

    let counters = solver.count();
    (counters.x as u64, counters.y as u64)
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Coord {
    pub x: CoordType,
    pub y: CoordType
}

#[derive(Copy, Clone)]
struct CoordPair {
    pub start: Coord,
    pub end: Coord
}

#[derive(Copy, Clone)]
struct PointIter {
    current: Coord,
    target: Coord,
    step: Coord,
}

struct VecCounter {
    row_size: usize,
    counter: Vec<Coord>
}

struct MapCounter {
    counter: FxHashMap<Coord, Coord>
}

///////////////////////////////////////////////////////////////////////////////

impl VecCounter {
    pub fn new(max: CoordType) -> Self {
        let row_size = max as usize + 1;
        let counter = vec![Coord::zero(); row_size * row_size];
        Self { row_size, counter }
    }
}

impl PointCounter for VecCounter {
    fn update(&mut self, point: Coord, increment: Coord) {
        self.counter[point.y as usize * self.row_size + point.x as usize].increment(increment);
    }

    fn count(&self) -> Coord {
        self.counter.iter().fold(Coord::zero(), |acc, count| {
            let x = if count.x >= 2 {acc.x + 1} else {acc.x};
            let y = if count.y >= 2 {acc.y + 1} else {acc.y};
            Coord { x, y }
        })
    }
}

impl MapCounter {
    pub fn new() -> Self {
        Self { counter: FxHashMap::default() }
    }
}

impl PointCounter for MapCounter {
    fn update(&mut self, point: Coord, increment: Coord) {
        self.counter.entry(point).or_insert_with(Coord::zero).increment(increment);
    }

    fn count(&self) -> Coord {
        self.counter.values().fold(Coord::zero(), |acc, count| {
            let x = if count.x >= 2 {acc.x + 1} else {acc.x};
            let y = if count.y >= 2 {acc.y + 1} else {acc.y};
            Coord { x, y }
        })
    }
}

///////////////////////////////////////////////////////////////////////////////

impl CoordPair {
    pub fn from_line(line: &str) -> Self {
        let mut spl = line.split(" -> ");
        let start = Coord::from_str(spl.next().unwrap());
        let end = Coord::from_str(spl.next().unwrap());
        Self { start, end }
    }

    pub fn is_not_diagonal(&self) -> bool {
        self.start.x == self.end.x || self.start.y == self.end.y
    }

    pub fn points_iter(&self) -> PointIter {
        let step = Coord{ x: diff(self.start.x, self.end.x), y: diff(self.start.y, self.end.y) };
        let target = Coord {x: self.end.x + step.x, y: self.end.y + step.y};
        PointIter { current: self.start, target, step }
    }

    pub fn max(&self) -> CoordType {
        max(self.start.max(), self.end.max())
    }
}

impl Iterator for PointIter {
    type Item = Coord;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            point if point == self.target => None,
            point => {
                self.current.increment(self.step);
                Some(point)
            }
        }
    }
}

impl Coord {
    pub fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    pub fn from_str(s: &str) -> Self {
        let mut spl = s.split(',');
        let x = spl.next().unwrap().parse().unwrap();
        let y = spl.next().unwrap().parse().unwrap();
        Self { x, y }
    }

    pub fn increment(&mut self, diff: Coord) {
        self.x += diff.x;
        self.y += diff.y;
    }

    pub fn max(&self) -> CoordType {
        max(self.x, self.y)
    }
}

fn diff(a: CoordType, b: CoordType) -> CoordType {
    match a.cmp(&b) {
        Less => 1,
        Equal => 0,
        Greater => -1,
    }
}
