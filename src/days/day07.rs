use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let mut crabs: Vec<i32> = read_to_string("input/day07.txt").unwrap()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();

    crabs.sort_unstable();
    let median = crabs[crabs.len() / 2];
    let sol1 = get_sol1_for_pos(&crabs, median);

    let avg = (crabs.iter().sum::<i32>() as f32 / crabs.len() as f32).round() as i32;
    let sol2 = get_sol2_for_pos(&crabs, avg as i32);
   
    (Solution::Int32(sol1), Solution::Int32(sol2))
}

fn get_sol1_for_pos(crabs: &[i32], pos: i32) -> i32 {
    crabs.iter().map(|x| (x-pos).abs()).sum()
}

fn get_sol2_for_pos(crabs: &[i32], pos: i32) -> i32 {
    crabs.iter().map(|x| {
        let diff = (x-pos).abs();
        diff * (diff + 1) / 2
    }).sum()
}
