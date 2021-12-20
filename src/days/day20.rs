use rustc_hash::FxHashSet;

use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use std::str::Lines;
use std::cmp::{max, min};

///////////////////////////////////////////////////////////////////////////////

type CoordType = i16;
type Coord = (CoordType, CoordType);

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day20.txt").unwrap();
    let mut lines = input.lines();

    let lookup = read_lookup(lines.next().unwrap());
    let mut image = read_image(&mut lines);

    let [mut min_x, mut max_x, mut min_y, mut max_y] = get_bounds(&image);
    let mut default = false;
    
    for _ in 0..2 {
        image = enhance(&image, &lookup, [min_x, max_x, min_y, max_y], default);
        default ^= lookup[0];
        min_x -= 1; min_y -= 1;
        max_x += 1; max_y += 1;
    }

    let sol1 = image.len() as u64;

    for _ in 0..48 {
        image = enhance(&image, &lookup, [min_x, max_x, min_y, max_y], default);
        default ^= lookup[0];
        min_x -= 1; min_y -= 1;
        max_x += 1; max_y += 1;
    }

    let sol2 = image.len() as u64;

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn enhance(image: &FxHashSet<Coord>, lookup: &[bool], bounds: [CoordType; 4], default: bool) -> FxHashSet<Coord> {
    let [min_x, max_x, min_y, max_y] = bounds;
    let mut new_img = FxHashSet::default();

    for x in min_x-1..=max_x+1 {
        for y in min_y-1..=max_y+1 {
            if new_pixel((x, y), image, lookup, bounds, default) {
                new_img.insert((x, y));
            }
        }
    }

    new_img
}

fn new_pixel(pos: Coord, image: &FxHashSet<Coord>, lookup: &[bool], bounds: [CoordType; 4], default: bool) -> bool {
    let mut index = 0;
    neighbors(pos).into_iter().for_each(|point| index = index << 1 | (image.contains(&point) || out_of_bounds(point, bounds) && default) as usize);
    lookup[index]
}

fn out_of_bounds((x, y): Coord, [min_x, max_x, min_y, max_y]: [CoordType; 4]) -> bool {
    x < min_x || x > max_x || y < min_y || y > max_y
}

fn get_bounds(image: &FxHashSet<Coord>) -> [CoordType; 4] {
    image.iter().fold([CoordType::MAX, CoordType::MIN, CoordType::MAX, CoordType::MIN], 
        |[min_x, max_x, min_y, max_y], (x, y)| [min(min_x, *x), max(max_x, *x), min(min_y, *y), max(max_y, *y)]
    )
}

fn neighbors((x, y): Coord) -> [Coord; 9] {
    [(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
}

fn read_lookup(s: &str) -> Vec<bool> {
    s.chars().map(|ch| ch == '#').collect()
}

fn read_image(lines: &mut Lines) -> FxHashSet<Coord> {
    let mut set = FxHashSet::default();

    for (y, line) in lines.skip(1).enumerate() {
        for (x, ch) in line.chars().enumerate() {
            if ch == '#' {
                set.insert((x as CoordType, y as CoordType));
            }
        }
    }

    set
}