use itertools::Itertools;

use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

type CoordElem = i16;

#[derive(Copy, Clone)]
struct Coord {
    x: CoordElem,
    y: CoordElem,
}

struct TargetZone {
    top_left: Coord,
    bottom_right: Coord
}

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day17.txt").unwrap();
    let target = TargetZone::from_string(&input);
    let initial_pos = Coord::new(0, 0);

    let min_xspeed = (-1 + (1.0 + 8.0 * target.top_left.x as f32).sqrt() as CoordElem) / 2;
    let max_xspeed = target.bottom_right.x;
    let min_yspeed = target.bottom_right.y;
    let max_yspeed = target.bottom_right.x / 2;

    let speeds = (min_xspeed..max_xspeed + 1).into_iter()
        .flat_map(move |xs| (min_yspeed..max_yspeed + 1).into_iter().map(move |ys| (xs, ys)))
        .filter_map(|(xs, ys)| {
            let speed = Coord::new(xs, ys);
            if initial_pos.hits_target(&speed, &target) { Some(speed) } else { None }
        })
        .collect_vec();

    let sol2 = speeds.len();
    let sol1 = speeds.into_iter()
        .filter_map(|Coord {y, ..}| if y > 0 { Some(y * (y+1) / 2) } else { None })
        .max().unwrap();

    (Solution::Int(sol1 as i64), Solution::UInt(sol2 as u64))
}

///////////////////////////////////////////////////////////////////////////////

impl TargetZone {
    pub fn from_string(s: &str) -> Self {
        let mut spl = s.split(' ');
        let xs = spl.nth(2).unwrap();
        let (x1, x2) = read_coords(&xs[2..xs.len() - 1]);
        let ys = spl.next().unwrap();
        let (y1, y2) = read_coords(&ys[2..]);
        Self { top_left: Coord::new(x1, y2), bottom_right: Coord::new(x2, y1)}
    }
}

impl Coord {
    pub fn new(x: CoordElem, y: CoordElem) -> Self {
        Self { x, y }
    }

    pub fn hits_target(&self, speed: &Coord, target: &TargetZone) -> bool {
        let (mut speedx, mut speedy) = (speed.x, speed.y);
        let (mut posx, mut posy) = (self.x, self.y);

        while posx + speedx <= target.bottom_right.x && posy + speedy >= target.bottom_right.y {
            posx += speedx;
            posy += speedy;
            if speedx > 0 {
                speedx -= 1;
            }
            speedy -= 1;
        }

        posx >= target.top_left.x && posx <= target.bottom_right.x &&
        posy <= target.top_left.y && posy >= target.bottom_right.y
    }
}

fn read_coords(s: &str) -> (CoordElem, CoordElem) {
    let mut spl = s.split("..");
    (spl.next().unwrap().parse().unwrap(), spl.next().unwrap().parse().unwrap())
}