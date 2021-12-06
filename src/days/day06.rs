use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

struct FishTank {
    fish: [u128; 9]
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day06.txt").unwrap();
    let mut tank = FishTank::from_string(&input);

    tank.advance(80);
    let sol1 = tank.count_fish();

    tank.advance(256 - 80);
    let sol2 = tank.count_fish();

    (Solution::BigUInt(sol1), Solution::BigUInt(sol2))
}

impl FishTank {
    pub fn from_string(s: &str) -> Self {
        let mut fish = [0; 9];
        s.trim_end().split(',').for_each(|x| fish[x.parse::<usize>().unwrap()] += 1);
        Self { fish }
    }

    pub fn count_fish(&self) -> u128 {
        self.fish.iter().sum()
    }

    pub fn advance(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    fn step(&mut self) {
        let zeros = self.fish[0];
        for i in 1..9 {
            self.fish[i-1] = self.fish[i];
        }
        self.fish[8] = zeros;
        self.fish[6] += zeros;
    }
}
