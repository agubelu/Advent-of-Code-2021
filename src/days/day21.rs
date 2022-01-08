use crate::{Solution, SolutionPair};
use std::{fs::read_to_string, ops::Range, iter::Cycle, cmp::{min, max}};

const N_SPACES: usize = 10;
const WIN_SCORE: usize = 21;
const N_STATES: usize = N_SPACES * N_SPACES * WIN_SCORE * WIN_SCORE;

const ROLLS_CHANCE: [u64; 7] = [1, 3, 6, 7, 6, 3, 1];

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day21.txt").unwrap();
    let (pos1, pos2) = read_positions(&input);

    let sol1 = game1(pos1, pos2);
    let sol2 = game2(pos1, pos2);

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn game1(p1: usize, p2: usize) -> u64 {
    let (mut pos1, mut pos2) = (p1, p2);

    let mut dice = (1..101).into_iter().cycle();
    let (mut score1, mut score2, mut rolls) = (0, 0, 0);

    loop {
        pos1 = update_pos(pos1, &mut dice, &mut rolls);
        score1 += pos1;
        if score1 >= 1000 { break; }

        pos2 = update_pos(pos2, &mut dice, &mut rolls);
        score2 += pos2;
        if score2 >= 1000 { break; }
    }

    (min(score1, score2) * rolls) as u64
}

fn game2(pos1: usize, pos2: usize) -> u64 {
    let (wins1, wins2) = count_wins(pos1, 0, pos2, 0, &mut vec![None; N_STATES]);
    max(wins1, wins2)
}

fn count_wins(pos1: usize, score1: usize, pos2: usize, score2: usize, mem: &mut [Option<(u64, u64)>]) -> (u64, u64) {
    let i = (pos1 - 1) + (pos2 - 1) * 10 + score1 * 100 + score2 * 2100;
    match mem[i] {
        Some(val) => val,
        None => {
            let (mut wins1, mut wins2) = (0, 0);
            ROLLS_CHANCE.iter().enumerate().for_each(|(inc_0, count)| {
                let inc = inc_0 + 3;
                let next_pos = next_pos(pos1, inc);
                let next_score = score1 + next_pos;

                if next_score >= WIN_SCORE {
                    wins1 += count;
                } else {
                    let wins_rec = count_wins(pos2, score2, next_pos, next_score, mem);
                    wins1 += wins_rec.1 * count;
                    wins2 += wins_rec.0 * count;
                }
            });

            mem[i] = Some((wins1, wins2));
            (wins1, wins2)
        }
    }
}

fn next_pos(cur_pos: usize, inc: usize) -> usize {
    (cur_pos - 1 + inc) % 10 + 1
}

fn update_pos(cur_pos: usize, dice: &mut Cycle<Range<usize>>, rolls: &mut usize) -> usize {
    *rolls += 3;
    let inc = dice.take(3).sum::<usize>() % 10;
    next_pos(cur_pos, inc)
}

fn read_positions(input: &str) -> (usize, usize) {
    let mut lines = input.lines();
    let pos1 = lines.next().unwrap()[28..].parse().unwrap();
    let pos2 = lines.next().unwrap()[28..].parse().unwrap();
    (pos1, pos2)
}