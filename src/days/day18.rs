use itertools::Itertools;

use crate::{Solution, SolutionPair};
use std::cell::RefCell;
use std::fs::read_to_string;

///////////////////////////////////////////////////////////////////////////////

type MutBox<T> = RefCell<Box<T>>;

#[derive(Clone)]
enum Node {
    Value { val: MutBox<u64> },
    Pair { left: MutBox<Node>, right: MutBox<Node> }
}

enum Dir { Right, Left }

#[derive(Debug, PartialEq)]
enum ProcessResult {
    Changed,
    Noop,
    MustExplode { right: u64, left: u64 },
    MustAddRight { val: u64 },
    MustAddLeft { val: u64 }
}

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day18.txt").unwrap();
    let nodes = input.lines()
        .map(|line| parse_node(line).0)
        .collect_vec();

    let sol1 = nodes.iter().cloned()
        .reduce(|a, b| {
            let node = Node::new(Node::Pair { left: a, right: b });
            process_node(&node);
            node
        }).unwrap().borrow().magnitude();

    let sol2 = nodes.iter().combinations(2)
        .flat_map(|nodes| {
            let node1 = Node::new(Node::Pair { left: nodes[0].clone(), right: nodes[1].clone() });
            let node2 = Node::new(Node::Pair { left: nodes[1].clone(), right: nodes[0].clone() });
            process_node(&node1);
            process_node(&node2);
            vec![node1, node2].into_iter()
        })
        .map(|node| node.borrow().magnitude())
        .max().unwrap();

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn process_node(node: &MutBox<Node>) {
    while {
        while do_explodes(node, 0) != ProcessResult::Noop {}
        do_splits(node) != ProcessResult::Noop
    } {}
}

fn do_explodes(node: &MutBox<Node>, depth: u32) -> ProcessResult {
    // Deal with the base cases first
    if depth >= 4 && node.borrow().is_perfect_pair() {
        // We must explode this node
        let (left, right) = node.borrow().get_val_pair();
        node.borrow_mut().set_zero();
        return ProcessResult::MustExplode { left, right }
    } else if node.borrow().is_value() {
        // We don't have to do anything to this node
        return ProcessResult::Noop;
    }

    // This node is not a stop condition. Process the left side first
    let resl = do_explodes(node.borrow().get_node(Dir::Left), depth + 1);

    match resl {
        ProcessResult::Changed => return resl,
        ProcessResult::MustExplode { left, right } => {
            // The left child node just exploded. We must increment the rightmost
            // value, which we have access to, and ask the parent caller to
            // increment the leftmost one
            node.borrow().get_node(Dir::Right).borrow().increment_leftmost(right);
            return ProcessResult::MustAddLeft{ val: left}
        },
        ProcessResult::MustAddRight { val } => {
            // We received a petition from the left child to increment the rightmost value
            // We can do that, since we have access to the right child
            node.borrow().get_node(Dir::Right).borrow().increment_leftmost(val);
            return ProcessResult::Changed;
        },
        ProcessResult::MustAddLeft { .. } => {
            // We received a petition from the left child to increment the leftmost value
            // We can't do that, so we pass it on to the caller
            return resl;
        },
        ProcessResult::Noop => {},
    };
    
    // If nothing interesting happened, process the right child
    let resr = do_explodes(node.borrow().get_node(Dir::Right), depth + 1);

    match resr {
        ProcessResult::Changed => resr,
        ProcessResult::MustExplode { left, right } => {
            // The right child node just exploded. We must increment the leftmost
            // value, which we have access to, and ask the parent caller to
            // increment the rightmost one
            node.borrow().get_node(Dir::Left).borrow().increment_rightmost(left);
            ProcessResult::MustAddRight{ val: right}
        },
        ProcessResult::MustAddRight { .. } => {
            // We received a petition from the right child to increment the rightmost value
            // We can''t do that so we pass on the problem to the caller
            resr
        },
        ProcessResult::MustAddLeft { val } => {
            // We received a petition from the right child to increment the leftmost value
            // We can do that! :D
            node.borrow().get_node(Dir::Left).borrow().increment_rightmost(val);
            return ProcessResult::Changed
        },
        ProcessResult::Noop => resr,
    }
}

fn do_splits(node: &MutBox<Node>) -> ProcessResult {
    // Deal with the base cases first
    if node.borrow().can_split() {
        node.borrow_mut().split();
        return ProcessResult::Changed;
    } else if node.borrow().is_value() {
        // We don't have to do anything to this node
        return ProcessResult::Noop;
    }

    // This node is not a stop condition. Process the left side first
    let resl = do_splits(node.borrow().get_node(Dir::Left));
    if let ProcessResult::Changed = resl {
        return resl;
    }
    
    // If nothing interesting happened, process the right child
    do_splits(node.borrow().get_node(Dir::Right))
}

fn parse_node(s: &str) -> (MutBox<Node>, usize) {
    let mut pos = 1;

    let left = match s.chars().nth(pos) {
        Some('[') => {
            let (node, len) = parse_node(&s[pos..]);
            pos += len;
            node
        },
        _ => {
            let (node, len) = parse_val(&s[pos..]);
            pos += len;
            node
        }
    };

    pos += 1; //  comma

    let right = match s.chars().nth(pos) {
        Some('[') => {
            let (node, len) = parse_node(&s[pos..]);
            pos += len;
            node
        },
        _ => {
            let (node, len) = parse_val(&s[pos..]);
            pos += len;
            node
        }
    };

    let node_res = Node::Pair { left, right };
    (Node::new(node_res), pos + 1)
}

fn parse_val(s: &str) -> (MutBox<Node>, usize) {
    let mut i = 0;
    let mut val = 0;

    for ch in s.chars() {
        if let Some(x) = ch.to_digit(10) {
            val = val * 10 + x as u64;
            i += 1;
        } else {
            break;
        }
    }

    (Node::new(Node::Value { val: RefCell::new(Box::new(val)) }), i)
}

impl Node {
    pub fn new(node: Self) -> MutBox<Self> {
        RefCell::new(Box::new(node))
    }

    pub fn is_perfect_pair(&self) -> bool {
        match self {
            Node::Value { .. } => false,
            Node::Pair { left, right } => left.borrow().is_value() && right.borrow().is_value()
        }
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Node::Value{..})
    }

    pub fn get_val(&self) -> u64 {
        match self {
            Node::Value { val } => *val.borrow().as_ref(),
            Node::Pair { .. } => unreachable!(),
        }
    }

    pub fn can_split(&self) -> bool {
        match self {
            Node::Value { val } => *val.borrow().as_ref() >= 10,
            Node::Pair { .. } => false,
        }
    }

    pub fn get_val_pair(&self) -> (u64, u64) {
        match self {
            Node::Pair { left, right } => (left.borrow().get_val(), right.borrow().get_val()),
            _ => unreachable!()
        }
    }

    pub fn increment_rightmost(&self, inc: u64) {
        match self {
            Node::Pair { right, .. } => right.borrow().increment_rightmost(inc),
            Node::Value { val } => *val.borrow_mut().as_mut() += inc,
        }
    }

    pub fn increment_leftmost(&self, inc: u64) {
        match self {
            Node::Pair { left, .. } => left.borrow().increment_leftmost(inc),
            Node::Value { val } => *val.borrow_mut().as_mut() += inc,
        }
    }

    pub fn set_zero(&mut self) {
        *self = Node::Value { val: RefCell::new(Box::new(0)) }
    }

    pub fn split(&mut self) {
        match self {
            Node::Value { val } => {
                let f = *val.borrow_mut().as_mut() as f32 / 2.0;
                let left = Node::Value { val: RefCell::new(Box::new(f.floor() as u64)) };
                let right = Node::Value { val: RefCell::new(Box::new(f.ceil() as u64)) };
                *self = Node::Pair { left: Node::new(left), right: Node::new(right) }
            },
            Node::Pair { .. } => unreachable!(),
        }
    }

    pub fn get_node(&self, dir: Dir) -> &MutBox<Node> {
        match self {
            Node::Pair { left, right } => match dir {
                Dir::Right => right,
                Dir::Left => left,
            },
            Node::Value { .. } => unreachable!(),
        }
    }

    pub fn magnitude(&self) -> u64 {
        match self {
            Node::Value { val } => *val.borrow().as_ref(),
            Node::Pair { left, right } => 3 * left.borrow().magnitude() + 2 * right.borrow().magnitude(),
        }
    }
}
