use crate::{Solution, SolutionPair};
use std::fs::read_to_string;

const GRID_SIZE: usize = 10;
type Grid = [[u8; GRID_SIZE]; GRID_SIZE];

const WINDOW_STEPS: [(i8, i8); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day11.txt").unwrap();
    let mut grid = read_input(&input);
    let mut flashes = 0;

    for _ in 0..100 {
        flashes += do_step(&mut grid);
    }

    let mut steps = 101;
    while do_step(&mut grid) != GRID_SIZE * GRID_SIZE {
        steps += 1;
    }

    (Solution::UInt(flashes as u64), Solution::UInt(steps))
}

// Performs one step and returns the number of flashes
fn do_step(grid: &mut Grid) -> usize {
    let mut n_flashes = 0;
    let mut flash_pos = Vec::with_capacity(100);

    grid.iter_mut().enumerate().for_each(|(row, arr)| {
        arr.iter_mut().enumerate().for_each(|(col, val)| {
            *val = (*val + 1) % 10;
            if *val == 0 {
                flash_pos.push((row, col));
            }
        })
    });

    for (row, col) in flash_pos {
        n_flashes += flash(grid, row, col);
    }

    n_flashes
}

// Flashes one position, increments the nearby ones and returns the number
// of total flashes as a consequence
fn flash(grid: &mut Grid, row: usize, col: usize) -> usize {
    let mut total_flashes = 1;

    for (dx, dy) in WINDOW_STEPS {
        let new_row = row as i8 + dx;
        let new_col = col as i8 + dy;
        if new_row < 0 || new_row == GRID_SIZE as i8 || new_col < 0 || new_col == GRID_SIZE as i8 {
            continue;
        }

        match &mut grid[new_row as usize][new_col as usize] {
            0 => {},
            v if *v == 9 => {
                *v = 0;
                total_flashes += flash(grid, new_row as usize, new_col as usize);
            },
            v => *v += 1,
        };
    }

    total_flashes
}

fn read_input(input: &str) -> Grid {
    let mut res = [[0; GRID_SIZE]; GRID_SIZE];
    
    input.lines().enumerate().for_each(|(row, line)| {
        line.chars().enumerate().for_each(|(col, ch)| {
            res[row][col] = ch as u8 - b'0';
        })
    });

    res
}
