use crate::{Solution, SolutionPair};
use itertools::Itertools;
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let numbers: Vec<u32> = read_to_string("input/day01.txt").unwrap()
        .lines()
        .map(|x| x.parse().unwrap())
        .collect();

    let sol1 = numbers.windows(2).filter(|x| x[1] > x[0]).count() as u64;
 
    let sol2 = numbers.windows(3).map(|x| x.iter().sum())
        .tuple_windows::<(u32, u32)>()
        .filter(|(a, b)| b > a)
        .count() as u64;

    (Solution::UInt(sol1), Solution::UInt(sol2))
}
