use itertools::Itertools;

use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use std::cmp::{max, min};

///////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
struct Cuboid {
    on: bool,
    x_min: i64, x_max: i64,
    y_min: i64, y_max: i64,
    z_min: i64, z_max: i64,
}

struct SignedCuboid {
    cuboid: Cuboid,
    sign: i64
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day22.txt").unwrap();
    let cuboids_all = input.lines().map(Cuboid::from_line).collect_vec();
    let cuboids_p1 = cuboids_all.iter().copied().filter(|c| c.use_part_1()).collect_vec();

    let sol1 = apply_cuboids(&cuboids_p1);
    let sol2 = apply_cuboids(&cuboids_all);

    (Solution::Int(sol1), Solution::Int(sol2))
}

fn apply_cuboids(ls: &[Cuboid]) -> i64 {
    let mut processed: Vec<SignedCuboid> = vec![];

    for cuboid in ls {
        let mut to_add = vec![];
        for other in &processed {
            if let Some(inter) = cuboid.intersection(&other.cuboid) {
                to_add.push(SignedCuboid { cuboid: inter, sign: -other.sign});
            }
        }

        if cuboid.on {
            to_add.push(SignedCuboid { cuboid: *cuboid, sign: 1});
        }

        processed.extend(to_add);
    }

    processed.iter().map(|x| x.sign * x.cuboid.volume()).sum()
}

///////////////////////////////////////////////////////////////////////////////

impl Cuboid {
    fn from_line(line: &str) -> Self {
        let spl = line.split(' ').collect_vec();

        let on = spl[0] == "on";

        let mut coords = spl[1].split(',');
        let (x_min, x_max) = read_coords(coords.next().unwrap());
        let (y_min, y_max) = read_coords(coords.next().unwrap());
        let (z_min, z_max) = read_coords(coords.next().unwrap());

        Self { x_min, x_max, y_min, y_max, z_min, z_max, on }
    }

    fn use_part_1(&self) -> bool {
        self.x_min >= -50 && self.x_max <= 50 && 
        self.y_min >= -50 && self.y_max <= 50 && 
        self.z_min >= -50 && self.z_max <= 50
    }

    fn collides_with(&self, other: &Cuboid) -> bool {
        !(self.x_min > other.x_max || self.x_max < other.x_min ||
          self.y_min > other.y_max || self.y_max < other.y_min ||
          self.z_min > other.z_max || self.z_max < other.z_min)
    }

    fn volume(&self) -> i64 {
        (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1) * (self.z_max - self.z_min + 1)
    }

    fn intersection(&self, other: &Cuboid) -> Option<Cuboid> {
        if !self.collides_with(other) {
            None
        } else {
            let x_min = max(self.x_min, other.x_min);
            let x_max = min(self.x_max, other.x_max);
            let y_min = max(self.y_min, other.y_min);
            let y_max = min(self.y_max, other.y_max);
            let z_min = max(self.z_min, other.z_min);
            let z_max = min(self.z_max, other.z_max);

            Some(Self { x_min, x_max, y_min, y_max, z_min, z_max, on: self.on })
        }
    }
}

fn read_coords(s: &str) -> (i64, i64) {
    s[2..].split("..").map(|x| x.parse().unwrap()).collect_tuple().unwrap()
}