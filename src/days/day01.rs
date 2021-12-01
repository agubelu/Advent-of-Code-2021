use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let numbers: Vec<u32> = read_to_string("input/day01.txt").unwrap()
        .lines()
        .map(|x| x.parse().unwrap())
        .collect();

    let sol1 = get_sol(&numbers, 1);
    let sol2 = get_sol(&numbers, 3);

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn get_sol(ls: &[u32], diff: usize) -> u64 {
    ls.iter().zip(ls[diff..].iter()).filter(|(a, b)| b > a).count() as u64
}