use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {

    let (hor, _, depth1, depth2) = read_to_string("input/day02.txt").unwrap()
        .lines()
        .map(|line| {
            let mut spl = line.split(' ');
            let op = spl.next().unwrap();
            let val: i64 = spl.next().unwrap().parse().unwrap();
            match op {
                "forward" => (val, 0),
                "up" => (0, -val),
                "down" => (0, val),
                _ => unreachable!()
            }
        }).fold((0, 0, 0, 0), |(hor, aim, depth1, depth2), mv| {
            match mv {
                (x, 0) => (hor + x, aim, depth1, depth2 + aim * x),
                (0, y) => (hor, aim + y, depth1 + y, depth2),
                _ => unreachable!()
            }
        });

    (Solution::Int(hor * depth1), Solution::Int(hor * depth2))
}
