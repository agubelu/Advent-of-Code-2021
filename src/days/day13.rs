use crate::{Solution, SolutionPair};
use std::{fs::read_to_string, cmp::min};
use itertools::Itertools;
use Fold::*;

///////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum Fold {
    YFold { y: usize },
    XFold { x: usize },
}

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
struct Point {
    pub x: usize,
    pub y: usize
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day13.txt").unwrap();
    let (points, folds) = read_data(&input);

    let new_points = points.into_iter().map(|p| p.apply_fold(folds[0])).unique().collect_vec();
    let sol1 = new_points.len() as u64;
    
    let (cols, rows) = get_rows_cols(&folds);
    let mut chars = vec![' '; rows as usize * cols as usize];
    new_points.into_iter()
        .map(|p| p.apply_fold_list(&folds[1..]))
        .for_each(|point| {
            let pos = point.y * cols + point.x;
            chars[pos] = '█';
        });

    let sol2 = "\n".to_owned() + &chars.chunks(cols)
        .map(String::from_iter)
        .join("\n");

    (Solution::UInt(sol1), Solution::Str(sol2))
}

fn get_rows_cols(folds: &[Fold]) -> (usize, usize) {
    folds.iter().map(|fold| match fold {
        XFold{x} => (*x, usize::MAX),
        YFold{y} => (usize::MAX, *y),
    }).reduce(|(x1, y1), (x2, y2)| (min(x1, x2), min(y1, y2))).unwrap()
}

fn read_data(input: &str) -> (Vec<Point>, Vec<Fold>) {
    let (mut points, mut folds) = (vec![], vec![]);
    let mut reading_points = true;

    for line in input.lines() {
        if line.is_empty() {
            reading_points = false;
            continue;
        }

        if reading_points {
            let mut spl = line.split(',');
            let x = spl.next().unwrap().parse().unwrap();
            let y = spl.next().unwrap().parse().unwrap();
            points.push(Point {x, y});
        } else {
            let val = line[13..].parse().unwrap();
            match line.chars().nth(11).unwrap() {
                'x' => folds.push(XFold{x: val}),
                'y' => folds.push(YFold{y: val}),
                _ => unreachable!()
            };
        }
    }

    (points, folds)
}

impl Point {
    pub fn apply_fold(&self, fold: Fold) -> Self {
        let (x, y) = match fold {
            XFold{x} => (if self.x > x {2 * x - self.x} else {self.x}, self.y),
            YFold{y} => (self.x, if self.y > y {2 * y - self.y} else {self.y}),
        };

        Self {x, y}
    }

    pub fn apply_fold_list(&self, folds: &[Fold]) -> Self {
        let mut x = self.x;
        let mut y = self.y;

        folds.iter().for_each(|fold| {
            match fold {
                XFold{x: x_fold} if *x_fold < x => x = 2 * *x_fold - x,
                YFold{y: y_fold} if *y_fold < y => y = 2 * *y_fold - y,
                _ => {},
            }
        });

        Self {x, y}
    }
}