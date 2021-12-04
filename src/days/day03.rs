use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

// Note: if the numbers have more than 16 bits,
// we must modify these
type Num = u16;
const TYPE_BITS: usize = 16;

pub fn solve() -> SolutionPair {
    let mut aux = 0; // We use this to calculate the number of bits in the input numbers
    let numbers: Vec<Num> = read_to_string("input/day03.txt").unwrap()
        .lines()
        .map(|x| {
            let val = Num::from_str_radix(x, 2).unwrap();
            aux |= val;
            val
        })
        .collect();
    
    let num_bits = TYPE_BITS - aux.leading_zeros() as usize;
    
    // Construct both numbers for part 1
    let mut gamma: Num = 0;
    (0..num_bits).into_iter().for_each(|i| gamma |= get_most_common_bit(&numbers, i, false) << i);
    let epsilon = !gamma & Num::MAX >> (TYPE_BITS - num_bits);

    // Find the numbers for part 2
    let oxy = get_part2_number(&numbers, true, num_bits);
    let co2 = get_part2_number(&numbers, false, num_bits);

    (Solution::UInt(epsilon as u64 * gamma as u64), 
     Solution::UInt(oxy as u64 * co2 as u64))
}

fn get_part2_number(numbers: &[Num], most_common: bool, bits: usize) -> Num {
    let mut numbers: Vec<Num> = numbers.iter().copied().collect();
    let mut i = bits - 1;

    while numbers.len() > 1 {
        let mask = 1 << i;
        let target = get_most_common_bit(&numbers, i, !most_common) << i;
        numbers = numbers.into_iter().filter(|x| x & mask == target).collect();
        i -= 1;
    }

    numbers[0]
}

fn get_most_common_bit(numbers: &[Num], i: usize, reverse: bool) -> Num {
    let count = numbers.iter().copied().filter(|x| x & (1 << i) != 0).count();
    let more_ones = count * 2 >= numbers.len();
    match (more_ones, reverse) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 1,
        (false, false) => 0,
    }
}