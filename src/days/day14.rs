use crate::{Solution, SolutionPair};
use std::hash::Hash;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

struct IdAssigner<T: Hash + Eq + Copy> {
    i: usize,
    map: FxHashMap<T, usize>
}

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day14.txt").unwrap();
    let (pairs, recipes, char_map) = read_input(&input);

    let sol1 = do_n_steps(&pairs, &recipes, &char_map, 10);
    let sol2 = do_n_steps(&pairs, &recipes, &char_map, 40);

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn do_n_steps(pairs: &[usize], recipes: &[(usize, usize)], chars: &[(usize, usize)], steps: u32) -> u64 {
    let n_recipes = recipes.len();
    let mut count = vec![0; n_recipes];
    for &p in pairs {
        count[p] += 1;
    }

    for _ in 0..steps {
        let mut new_count = vec![0; n_recipes];

        for (i, val) in count.into_iter().enumerate() {
            let (pair1, pair2) = recipes[i];
            new_count[pair1] += val;
            new_count[pair2] += val;
        }

        count = new_count;
    }

    let mut char_count = vec![0; (n_recipes as f32).sqrt() as usize];
    for (i, val) in count.into_iter().enumerate() {
        let (char1, char2) = chars[i];
        char_count[char1] += val;
        char_count[char2] += val;
    }

    (char_count.iter().max().unwrap() - char_count.iter().min().unwrap()) / 2u64
}

fn read_input(input: &str) -> (Vec<usize>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut polymer_pairs = Vec::with_capacity(25);
    let mut recipes_map = FxHashMap::default();
    let mut chars_map = FxHashMap::default();

    let mut id_assigner_pairs = IdAssigner::new();
    let mut id_assigner_chars = IdAssigner::new();
    let mut lines = input.lines();

    for (c1, c2) in lines.next().unwrap().chars().tuple_windows() {
        polymer_pairs.push(id_assigner_pairs.get_id(&(c1, c2)));
    }

    for line in lines.skip(1) {
        let chars = line.chars().collect_vec();
        let (c1, c2, res) = (chars[0], chars[1], chars[6]);
        let id_pair = id_assigner_pairs.get_id(&(c1, c2));
        let id_res1 = id_assigner_pairs.get_id(&(c1, res));
        let id_res2 = id_assigner_pairs.get_id(&(res, c2));

        let id_c1 = id_assigner_chars.get_id(&c1);
        let id_c2 = id_assigner_chars.get_id(&c2);
        recipes_map.insert(id_pair, (id_res1, id_res2));
        chars_map.insert(id_pair, (id_c1, id_c2));
    }

    let mut recipes_vec = vec![(0, 0); recipes_map.len()];
    for (k, v) in recipes_map.into_iter() {
        recipes_vec[k] = v;
    }

    let mut chars_vec = vec![(0, 0); chars_map.len()];
    for (k, v) in chars_map.into_iter() {
        chars_vec[k] = v;
    }

    (polymer_pairs, recipes_vec, chars_vec)
}

impl<T: Hash + Eq + Copy> IdAssigner<T> {
    pub fn new() -> Self {
        Self { i: 0, map: FxHashMap::default() }
    }

    pub fn get_id(&mut self, elem: &T) -> usize {
        match self.map.get(elem) {
            Some(id) => *id,
            None => {
                let id = self.i;
                self.i += 1;
                self.map.insert(*elem, id);
                id
            }
        }
    }
}