use std::fs::read_to_string;
use std::cmp::Ordering::*;
use rustc_hash::FxHashMap;

use crate::{Solution, SolutionPair};

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let mut points_count = FxHashMap::default();

    read_to_string("input/day05.txt").unwrap()
        .lines()
        .map(CoordPair::from_line)
        .for_each(|coord| {
            let points = coord.points_iter();
            let cnt = (if coord.is_not_diagonal() {1} else {0}, 1);

            points.for_each(|p| {
                let x = points_count.entry(p).or_insert((0, 0));
                *x = (x.0 + cnt.0, x.1 + cnt.1);
            });
        });


    let (sol1, sol2) = count_shared_points(&points_count);
    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn count_shared_points(map: &FxHashMap<CoordTuple, (u32, u32)>) -> (u64, u64) {
    map.values().fold((0, 0), |(sol1, sol2), (cnt1, cnt2)| {
        let next1 = if *cnt1 >= 2 {sol1 + 1} else {sol1};
        let next2 = if *cnt2 >= 2 {sol2 + 1} else {sol2};
        (next1, next2)
    })
}

///////////////////////////////////////////////////////////////////////////////

type CoordType = i16;
type CoordTuple = (CoordType, CoordType);

#[derive(Copy, Clone, PartialEq)]
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
}

impl Iterator for PointIter {
    type Item = CoordTuple;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            point if point == self.target => None,
            point => {
                self.current.x += self.step.x;
                self.current.y += self.step.y;
                Some((point.x, point.y))
            }
        }
    }
}

impl Coord {
    pub fn from_str(s: &str) -> Self {
        let mut spl = s.split(',');
        let x = spl.next().unwrap().parse().unwrap();
        let y = spl.next().unwrap().parse().unwrap();
        Self { x, y }
    }
}

fn diff(a: CoordType, b: CoordType) -> CoordType {
    match a.cmp(&b) {
        Less => 1,
        Equal => 0,
        Greater => -1,
    }
}
