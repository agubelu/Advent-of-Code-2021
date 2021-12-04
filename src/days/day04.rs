use itertools::Itertools;

use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use std::cmp::min;

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day04.txt").unwrap();
    let mut spl = input.split("\n\n");

    // Construct the number to position vec
    // This will allow us to quickly look up when a number appears in the bingo
    let numbers: Vec<usize> = spl.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
    let mut indices = vec![0; numbers.len()];
    for (i, val) in numbers.iter().copied().enumerate() {
        indices[val] = i;
    }

    // Construct out boards, and keep track of the minimum row value
    // of each one, so we known which one wins first and last
    let mut boards = Vec::with_capacity(50);
    let mut winning_row_pos = usize::MAX;
    let mut losing_row_pos = 0;
    let mut winning_board_index = 0;
    let mut losing_board_index = 0;

    for (i, s) in spl.enumerate() {
        let (board, min) = BingoBoard::new(s, &indices);
        boards.push(board);

        if min < winning_row_pos {
            winning_row_pos = min;
            winning_board_index = i;
        } else if min > losing_row_pos {
            losing_row_pos = min;
            losing_board_index = i;
        }
    }

    let sol1 = boards[winning_board_index].get_score(winning_row_pos, &numbers);
    let sol2 = boards[losing_board_index].get_score(losing_row_pos, &numbers);

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

///////////////////////////////////////////////////////////////////////////////

// Aux struct to hold the numbers in a bingo board
struct BingoBoard {
    numbers: Vec<usize>,
}

impl BingoBoard {
    // Constructs a new board from a string representing the rows
    // of numbers, returns the lowest maximum found in a single row or column
    pub fn new(string: &str, indices: &[usize]) -> (Self, usize) {
        let n_lines = string.lines().count();
        let numbers = string.lines().into_iter().flat_map(
            |line| line.split_ascii_whitespace().map(|val| indices[val.parse::<usize>().unwrap()])
        ).collect_vec();
        let line_len = numbers.len() / n_lines;
        let min_rows = numbers.chunks(line_len).map(|line| line.iter().copied().max().unwrap()).min().unwrap();
        let min_columns = (0..line_len).map(|c| (0..n_lines).map(|r| numbers[c + r * line_len]).max().unwrap()).min().unwrap();

        (Self { numbers }, min(min_rows, min_columns))
    } 

    // Gets the score after the nth number has been called, which results
    // in a completed row. It is assumed that this will be called on the
    // winning board
    pub fn get_score(&self, pos: usize, numbers: &[usize]) -> u64 {
        let sum: usize = self.numbers.iter().copied().filter_map(|v| {
            if v > pos {
                Some(numbers[v])
            } else {
                None
            }
        }).sum();

        let n = numbers[pos];
        (sum * n) as u64
    }
}