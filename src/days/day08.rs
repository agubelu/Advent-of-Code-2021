use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

struct SegmentDecoder {
    digits: [u8; 10],
    four: u8,
    seven: u8,
    reading: Vec<u8>,
}

pub fn solve() -> SolutionPair {
    let (sol1, sol2) = read_to_string("input/day08.txt").unwrap()
        .lines()
        .map(|line| {
            let decoder = SegmentDecoder::from_line(line);
            let corresp = decoder.resolve();
            decoder.get_answers(corresp)
        })
        .reduce(|(a1, a2), (b1, b2)| (a1 + b1, a2 + b2))
        .unwrap();

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

impl SegmentDecoder {
    pub fn from_line(line: &str) -> Self {
        let mut digits = [0; 10];
        let mut seven = 0;
        let mut four = 0;

        let mut spl = line.split(' ');
        let mut i = 0;
        let mut chunk;

        while {chunk = spl.next().unwrap(); chunk != "|"} {
            let n = str_to_u8(chunk);
            let bits = n.count_ones();
            if bits == 3 {
                seven = n;
            } else if bits == 4 {
                four = n;
            }
            digits[i] = n;
            i += 1;
        }

        let reading = spl.map(str_to_u8).collect();
        Self { digits, seven, four, reading }
    }

    pub fn resolve(&self) -> [u8; 7] {
        let mut resolved = 0;
        let mut corresp = [0; 7];

        // We can resolve segments 1, 4 and 5 just from how many times they appear
        for i in 0..8 {
            let mask = 1 << i;
            let sum = self.digits.iter().filter(|x| *x & mask != 0).count();
            let val = match sum {
                4 => Some(4),
                6 => Some(1),
                9 => Some(5),
                _ => None
            };

            if let Some(x) = val {
                corresp[x] = mask;
                resolved |= mask;
            }
        }

        // The rest can be obtained by process of elimination
        corresp[3] = self.four & !self.seven & !resolved;
        resolved |= corresp[3];
        corresp[2] = self.four & !resolved;
        resolved |= corresp[2];
        corresp[0] = self.seven & !resolved;
        resolved |= corresp[0];
        corresp[6] = !resolved & 0x7F;

        corresp
    }

    pub fn get_answers(&self, corresp: [u8; 7]) -> (u64, u64) {
        let seven = self.seven;
        let four = self.four ;
        let one = seven ^ corresp[0];
        let three = seven | corresp[3] | corresp[6];
        let eight = three | corresp[1] | corresp[4];
        let two = eight ^ corresp[1] ^ corresp[5];
        let five = eight ^ corresp[2] ^ corresp[4];
        let zero = eight ^ corresp[3];
        let six = eight ^ corresp[2];
        let nine = eight ^ corresp[4];

        let (mut part1, mut part2) = (0, 0);

        for val in self.reading.iter().copied() {
            let number = match val {
                x if x == zero => 0,
                x if x == one => 1,
                x if x == two => 2,
                x if x == three => 3,
                x if x == four => 4,
                x if x == five => 5,
                x if x == six => 6,
                x if x == seven => 7,
                x if x == eight => 8,
                x if x == nine => 9,
                _ => unreachable!(),
            };

            part2 = part2 * 10 + number;
            if number == 1 || number == 4 || number == 7 || number == 8 {
                part1 += 1;
            }
        
        }

        (part1, part2)
    }
}

fn str_to_u8(s: &str) -> u8 {
    s.chars().map(|x| 1 << (x as u16 - 97)).reduce(|a, b| a | b).unwrap() as u8
}