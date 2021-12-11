use crate::{Solution, SolutionPair};
use std::fs::read_to_string;


///////////////////////////////////////////////////////////////////////////////

enum LineResult {
    Corrupted { score: u64 },
    Incomplete { score: u64 },
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day10.txt").unwrap();
    let mut sol1 = 0;
    let mut sol2_scores = Vec::with_capacity(100);

    input.lines().for_each(|line| match analyze_line(line) {
        LineResult::Corrupted{score} => sol1 += score,
        LineResult::Incomplete{score} => sol2_scores.push(score),
    });

    sol2_scores.sort_unstable();
    let sol2 = sol2_scores[sol2_scores.len() / 2];

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn analyze_line(line: &str) -> LineResult {
    let mut closers = Vec::with_capacity(100);

    for ch in line.chars() {
        match ch {
            '{' => closers.push('}'),
            '[' => closers.push(']'),
            '(' => closers.push(')'),
            '<' => closers.push('>'),
            '}' | ']' | ')' | '>' => {
                if closers.pop().unwrap() != ch {
                    let score = match ch {
                        ')' => 3,
                        ']' => 57,
                        '}' => 1197,
                        '>' => 25137,
                         _  => unreachable!()
                    };
                    return LineResult::Corrupted{ score };
                }
            },
            _ => unreachable!()
        }
    }

    get_completion_score(&closers)
}

fn get_completion_score(closers: &[char]) -> LineResult {
    let mut score = 0;
    for ch in closers.iter().rev() {
        score = score * 5 + match ch {
            ')' => 1,
            ']' => 2,
            '}' => 3,
            '>' => 4,
             _  => unreachable!()
        };
    }
    LineResult::Incomplete{score}
}
