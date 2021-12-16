use crate::{Solution, SolutionPair};
use std::fs::read_to_string;
use bitvec::prelude::*;

///////////////////////////////////////////////////////////////////////////////

type Bits = BitSlice<Msb0, u8>;

enum Packet {
    Literal { version: u64, value: u64 },
    Operation { version: u64, type_id: u64, payload: Vec<Packet> }
}

enum LengthType { BitLength, NPackets }

///////////////////////////////////////////////////////////////////////////////

pub fn solve() -> SolutionPair {
    let input = read_to_string("input/day16.txt").unwrap();
    let bits = BitVec::<Msb0, _>::from_slice(&hex::decode(input).unwrap()).unwrap();
    
    let (root_pkt, _) = parse_packet(&bits);
    let sol1 = root_pkt.sum_versions();
    let sol2 = root_pkt.get_value();

    (Solution::UInt(sol1), Solution::UInt(sol2))
}

fn parse_packet(bits: &Bits) -> (Packet, usize) {
    let version = bits[..3].load_be();
    let type_id = bits[3..6].load_be();

    match type_id {
        4 => {
            let (value, payload_len) = parse_literal(&bits[6..]);
            (Packet::Literal { version, value }, 6 + payload_len)
        },
        _ => {
            let (payload, payload_len) = parse_operation(&bits[6..]);
            (Packet::Operation { version, type_id, payload }, 6 + payload_len)
        }
    }
}

fn parse_literal(bits: &Bits) -> (u64, usize) {
    let (mut val, mut pos) = (0, 0);
    let mut keep_reading = true;

    while keep_reading {
        keep_reading = bits[pos];
        val <<= 4;
        val |= bits[pos+1..pos+5].load_be::<u64>();
        pos += 5;
    }

    (val, pos)
}

fn parse_operation(bits: &Bits) -> (Vec<Packet>, usize) {
    let mut payload = vec![];
    let (len_type, mut pos, len) = if bits[0] {
        (LengthType::NPackets, 12, bits[1..12].load_be())
    } else {
        (LengthType::BitLength, 16, bits[1..16].load_be())
    };

    let mut keep_reading = true;
    while keep_reading {
        let (pkt, size) = parse_packet(&bits[pos..]);
        payload.push(pkt);
        pos += size;
        keep_reading = match len_type {
            LengthType::NPackets => payload.len() < len,
            LengthType::BitLength => pos - 16 < len,
        }
    };

    (payload, pos)
}

///////////////////////////////////////////////////////////////////////////////

impl Packet {
    fn sum_versions(&self) -> u64 {
        match &self {
            Packet::Literal { version, .. } => *version,
            Packet::Operation { version, payload, .. } => *version + payload.iter().map(|pkt| pkt.sum_versions()).sum::<u64>(),
        }
    }

    fn get_value(&self) -> u64 {
        match &self {
            Packet::Literal { value, .. } => *value,
            Packet::Operation { type_id, payload, .. } => match type_id {
                0 => payload.iter().map(|pkt| pkt.get_value()).sum(),
                1 => payload.iter().map(|pkt| pkt.get_value()).product(),
                2 => payload.iter().map(|pkt| pkt.get_value()).min().unwrap(),
                3 => payload.iter().map(|pkt| pkt.get_value()).max().unwrap(),
                5 => (payload[0].get_value() > payload[1].get_value()) as u64,
                6 => (payload[0].get_value() < payload[1].get_value()) as u64,
                7 => (payload[0].get_value() == payload[1].get_value()) as u64,
                _ => unreachable!()
            },
        }
    }
}

