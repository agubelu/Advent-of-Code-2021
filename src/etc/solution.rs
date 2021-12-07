use std::fmt::{Display, Formatter, Result};
use Solution::*;

pub enum Solution {
    Int(i64),
    Int32(i32),
    UInt(u64),
    BigUInt(u128),
    Str(String)
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Int(i) => i.fmt(f),
            Int32(i) => i.fmt(f),
            UInt(u) => u.fmt(f),
            BigUInt(u) => u.fmt(f),
            Str(s) => s.fmt(f),
        }
    }
}