use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use rustc_hash::FxHashMap;

///////////////////////////////////////////////////////////////////////////////

struct CaveIter {
    val: u64
}

pub fn solve() -> SolutionPair {
    let input_data = read_to_string("input/day12.txt").unwrap();
    let (connects, start, end, small_caves) = load_info(&input_data);
    let sol1 = count_paths_between(&connects, small_caves, start, end, false);
    let sol2 = count_paths_between(&connects, small_caves, start, end, true);
    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn count_paths_between(conn: &[u64], small_caves: u64, from: u64, to: u64, part2: bool) -> u64 {
    pathfinder(conn, small_caves, from, to, 0, part2, false)
}

fn pathfinder(conn: &[u64], small_caves: u64, from: u64, to: u64, visited: u64, part2: bool, mut small_visited_twice: bool) -> u64 {
    let mut paths = 0;

    if part2 && !small_visited_twice && (from & small_caves & visited) != 0 {
        small_visited_twice = true;
    }

    paths += get_cave_ids(conn[from.trailing_zeros() as usize])
        .filter(|cave| (cave & small_caves & visited) == 0 || part2 && !small_visited_twice)
        .map(|next_cave| match next_cave {
            x if x == to => 1,
            _ => pathfinder(conn, small_caves, next_cave, to, visited | from, part2, small_visited_twice)
        })
        .sum::<u64>();

    paths
}

fn load_info(input: &str) -> (Vec<u64>, u64, u64, u64) {
    let (mut start, mut end, mut small) = (0, 0, 0);
    let mut corresps = vec![];

    let mut id_map: FxHashMap<&str, u64> = FxHashMap::default();
    let mut i = 0;

    for line in input.lines() {
        let mut spl = line.split('-');
        let c1 = spl.next().unwrap();
        let c2 = spl.next().unwrap();

        let c1_id = match id_map.get(c1) {
            Some(id) => *id,
            None => {
                let id = 1 << i;
                id_map.insert(c1, id);
                corresps.push(0);
                i += 1;
                id
            }
        };

        let c2_id = match id_map.get(c2) {
            Some(id) => *id,
            None => {
                let id = 1 << i;
                id_map.insert(c2, id);
                corresps.push(0);
                i += 1;
                id
            }
        };

        if c1.chars().next().unwrap().is_ascii_lowercase() {
            small |= c1_id;
        }

        if c2.chars().next().unwrap().is_ascii_lowercase() {
            small |= c2_id;
        }

        match c1 {
            "start" => start = c1_id,
            x => {
                corresps[c2_id.trailing_zeros() as usize] |= c1_id;
                if x == "end" {
                    end = c1_id;
                }
            }
        }

        match c2 {
            "start" => start = c2_id,
            x => {
                corresps[c1_id.trailing_zeros() as usize] |= c2_id;
                if x == "end" {
                    end = c2_id;
                }
            }
        }
    }

    (corresps, start, end, small)
}

fn get_cave_ids(val: u64) -> CaveIter {
    CaveIter { val }
}


impl Iterator for CaveIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        match self.val {
            0 => None,
            x => {
                self.val = x & (x - 1);
                Some(1 << x.trailing_zeros())
            }
        }
    }
}