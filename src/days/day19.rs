use rustc_hash::{FxHashMap, FxHashSet};
use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use ndarray::{arr2, Array2};
use itertools::Itertools;
use rayon::prelude::*;

///////////////////////////////////////////////////////////////////////////////

#[cfg(windows)]
const SPLITTER: &str = "\r\n\r\n";
#[cfg(not(windows))]
const SPLITTER: &str = "\n\n";

type CoordType = i32;
type CoordMat = Array2<CoordType>;
type Coord = [CoordType; 3];

struct Scanner {
    rel_coords: Option<Coord>,
    readings: CoordMat
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day19.txt").unwrap();
    let mut scanners = input.split(SPLITTER).map(Scanner::from_string).collect_vec();
    
    scanners[0].rel_coords = Some([0, 0, 0]);
    let mut resolved_beacons = scanners[0].readings.clone();
    let mut resolved_set = FxHashSet::default();

    let rotation_matrices = calculate_rot_mat();

    for row in resolved_beacons.rows() {
        resolved_set.insert([row[0], row[1], row[2]]);
    }


        
    while !scanners.iter().all(|s| s.rel_coords.is_some()) {
        for scanner in scanners.iter_mut() {
            if scanner.rel_coords.is_some() {
                continue;
            }

            let coords_find = rotation_matrices.par_iter().find_map_any(|rot_mat| {
                let readings = scanner.readings.dot(rot_mat);
                find_shared_coord(&resolved_beacons, &readings).map(move |coords| (coords, readings))
            });

            if let Some((coords, readings)) = coords_find {
                scanner.rel_coords = Some(coords);
                update_beacons(&mut resolved_beacons, &mut resolved_set, &(readings + &arr2(&[coords])));
                break;
            }
        }
    }
    
    let sol2 = scanners.iter().tuple_combinations().map(|(s1, s2)| s1.dist_to(s2)).max().unwrap();

    (Solution::UInt(resolved_set.len() as u64), Solution::UInt(sol2))
}

fn update_beacons(resolved_mat: &mut CoordMat, resolved_set: &mut FxHashSet<Coord>, new: &CoordMat) {
    for row in new.rows() {
        let row_arr = [row[0], row[1], row[2]];
        if !resolved_set.contains(&row_arr) {
            resolved_set.insert(row_arr);
            resolved_mat.push_row(row).unwrap();
        }
    }
}

fn find_shared_coord(resolved: &CoordMat, readings: &CoordMat) -> Option<Coord> {
    let mut map = FxHashMap::default();
    for row in readings.rows() {
        let diffs = resolved - &row;
        for row_diff in diffs.rows() {
            let cnt = map.entry(row_diff.into_owned()).or_insert(0);
            if *cnt == 11 {
                return Some([row_diff[0], row_diff[1], row_diff[2]]);
            } else {
                *cnt += 1;
            }
        }
    }

    None
}

fn calculate_rot_mat() -> Vec<CoordMat> {
    // thanks to https://stackoverflow.com/a/50546727 and https://stackoverflow.com/a/16453299
    let rots = vec!["I", "X", "Y", "XX", "XY", "YX", "YY", "XXX", "XXY", "XYX", "XYY", "YXX", "YYX", "YYY", "XXXY", "XXYX", "XXYY", "XYXX", "XYYY", "YXXX", "YYYX", "XXXYX", "XYXXX", "XYYYX"];
    let i = arr2(&[[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    let x = arr2(&[[1, 0, 0], [0, 0, -1], [0, 1, 0]]);
    let y = arr2(&[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]);

    rots.into_iter().map(|s| s.chars().fold(i.clone(), |acc, ch| {
        let mat = match ch {
            'I' => &i,
            'X' => &x,
            'Y' => &y,
             _  => unreachable!()
        };
        acc.dot(mat)
    })).collect_vec()
}

impl Scanner {
    pub fn from_string(s: &str) -> Self {
        let rel_coords = None;
        let readings = s.lines().skip(1)
            .map(|line| {
                let mut spl = line.split(',');
                [spl.next().unwrap().parse().unwrap(), spl.next().unwrap().parse().unwrap(), spl.next().unwrap().parse().unwrap()]
            }).collect_vec();
        Self { rel_coords, readings: arr2(&readings) }
    }

    pub fn dist_to(&self, other: &Self) -> u64 {
        let [x1, y1, z1] = self.rel_coords.unwrap();
        let [x2, y2, z2] = other.rel_coords.unwrap();

        ((x1 - x2).abs() + (y1 - y2).abs() + (z1 - z2).abs()) as u64
    }
}