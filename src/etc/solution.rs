use std::fmt::{Display, Formatter, Result};
use Solution::*;

pub enum Solution {
    Int(i64),
    UInt(u64),
    Str(String)
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Int(i) => i.fmt(f),
            UInt(u) => u.fmt(f),
            Str(s) => s.fmt(f),
        }
    }
}