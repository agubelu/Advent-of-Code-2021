use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use itertools::Itertools;
use rustc_hash::FxHashMap;

///////////////////////////////////////////////////////////////////////////////

type ConnectivityMap<'a> = FxHashMap<&'a str, Vec<&'a str>>;

pub fn solve() -> SolutionPair {
    let input_data = read_to_string("input/day12.txt").unwrap();
    let connects = load_connectivity(&input_data);
    let sol1 = n_paths_between(&connects, "start", "end", false);
    let sol2 = n_paths_between(&connects, "start", "end", true);
    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn n_paths_between(conn: &ConnectivityMap, from: &str, to: &str, part2: bool) -> u64 {
    pathfinder(conn, from, to, &mut vec![], part2, false)
}

fn pathfinder<'a, 'b>(
    conn: &'a ConnectivityMap, 
    from: &'a str, 
    to: &'a str, 
    visited_small_caves: &'b mut Vec<&'a str>,
    part2: bool,
    mut small_visited_twice: bool,
) -> u64 {
    let mut paths = 0;
    let mut current_small = false;

    if from.chars().next().unwrap().is_ascii_lowercase() {
        current_small = true;
        visited_small_caves.push(from);

        if part2 && !small_visited_twice && visited_small_caves.iter().filter(|&&cave| cave == from).count() == 2 {
            small_visited_twice = true;
        }
    }

    let caves_to_visit = conn[from].iter().filter(|cave| !visited_small_caves.contains(cave) || part2 && !small_visited_twice).collect_vec();

    for next_cave in caves_to_visit {
        if *next_cave == to {
            paths += 1;
        } else {
            paths += pathfinder(conn, next_cave, to, visited_small_caves, part2, small_visited_twice);
        }
    }

    if current_small {
        visited_small_caves.pop();
    }

    paths
}

fn load_connectivity(input: &str) -> ConnectivityMap {
    let mut map: ConnectivityMap = FxHashMap::default();

    for line in input.lines() {
        let mut spl = line.split('-');
        let c1 = spl.next().unwrap();
        let c2 = spl.next().unwrap();

        // Don't add paths that go back to the start
        if c2 != "start" {
            map.entry(c1).or_default().push(c2);
        }
        if c1 != "start" {
            map.entry(c2).or_default().push(c1);
        }
    }

    map
}
