# AoC 2021 journal

As stated, my goal is to minimize the runtime of all days without deviating *too much* from idiomatic Rust. The extremely scientific methodology I'll follow will be to run each day a bunch of times in my computer and write down the [minimal](https://stackoverflow.com/a/43939840) runtime I get.

I'll keep updating this file with some commentary and discussion every day.

Of course, expect spoilers if you keep reading!

---
* **[Day 1](#day-1):** 0.1367 ms
* **[Day 2](#day-2):** 0.1269 ms
* **[Day 3](#day-3):** 0.1215 ms
* **[Day 4](#day-4):** 0.1856 ms

---

# Day 1

First day! As usual, the first challenge is pretty simple. In this case, it involves sliding through a list of numbers with different window sizes.

My first intuition for part 1 was to `.zip()` both the numbers vector `numbers` and `numbers[1..]` to create the size 2 windows. While not technically incorrect, upon reading part 2, I learned that, conveniently, Rust already includes a `windows()` method that creates sliding windows over a slice.

Great! In that case, Part 1 is a simple one-liner:

```rust
let sol1 = numbers.windows(2).filter(|x| x[1] > x[0]).count() as u64;
```

In this case, the `u64` cast occurs because the type that I use to represent numeric solutions is `u64`, while `count()` returns `usize`. Anyways, I reckon in 64-bit systems this should be a no-op.

We can also use `windows()` for the bigger windows of part 2. In this case, my idea was to first get the windows, then map them to their respective sums, and then `window()` again over them to filter and count the pairs. This way, I avoid having to sum all 3-windows twice.

However, `windows()` cannot be used on map iterators. Thankfully, the great [itertools crate](https://docs.rs/itertools/latest/itertools/) provides `tuple_windows()` on any iterator:

```rust
let sol2 = numbers.windows(3).map(|x| x.iter().sum())
        .tuple_windows::<(u32, u32)>()
        .filter(|(a, b)| b > a)
        .count() as u64;
```

The result is 0.1374ms, and as expected, most of it is spent reading the file.

![Day 1 results](imgs/d01.png)

Can we do better? Yes. Should we? Well, the main optimization here that comes to mind is doing all operations as we are reading the file. In that case, we can arrive to both answers as soon as we finish reading the last line, by manually keeping track of the windows and previous values. Something like...

```rust
let mut win3 = [0, 0, 0];
let mut prev_win = 0;
let mut prev = 0;
let mut sol1 = 0;
let mut sol2 = 0;

read_to_string("input/day01.txt").unwrap().lines()
    .enumerate()
    .for_each(|(i, line)| {
        let val = line.parse::<u32>().unwrap();

        // Update the counter for part 1
        if i > 0 {
            if val > prev {
                sol1 += 1;
            }
            prev = val;
        }

        // Update the sliding window
        win3[i % 3] = val;
        let sum = win3.iter().sum();

        // If we have at least two full windows, compare it with the previous one
        if i > 3 && sum > prev_win {
            sol2 += 1;
        }

        prev_win = sum;
    });
```

This very verbose solution runs at around 0.11ms. While faster, I think we can sacrify 0.02 milliseconds for the sake of more concise and easy to read code.

EDIT: It has been pointed out to me on Twitter that part 2 can be simplified, since you only need to compare the number that changes when the window slides. So, we dont need to actually sum the numbers in the window, and both parts can be simplified to a generalization, where you compare pairs of numbers in the input separated by a certain distance.

This new solution is very elegant:
```rust
fn get_sol(ls: &[u32], n: usize) -> u64 {
    ls.windows(n+1).filter(|x| x[n] > x[0]).count() as u64
}

let sol1 = get_sol(&numbers, 1);
let sol2 = get_sol(&numbers, 3);
```

The runtime didn't change too much, since the bottleneck is clearly reading the file from disk, but it provided a tiny improvement:

![Day 1 results](imgs/d01_1.png)

Kudos to [@ajiiisai](https://github.com/ajiiisai) for discovering this neat trick!

# Day 2

Today is one of those "follow these pseudo-instructions and update some values" days. We have to move a submarine up, down and forward, and the twist in part 2 is that the depth is calculated differently, using some "aim" that you update when you go up and down.

At first, I implemented this declarative version to have some sort of baseline to compare against:

```rust
let (mut hor, mut aim, mut depth1, mut depth2) = (0, 0, 0, 0);
read_to_string("input/day02.txt").unwrap()
    .lines()
    .for_each(|line| {
        let spl: Vec<&str> = line.split(' ').collect();
        let val: i64 = spl[1].parse().unwrap();

        match spl[0] {
            "forward" => {
                hor += val;
                depth2 += aim * val;
            },
            "down" => {
                depth1 += val;
                aim += val;
            },
            "up" => {
                depth1 -= val;
                aim -= val;
            },
            _ => unreachable!()
        }
    });
```

The previous code does both parts at once, by updating whatever is necessary each step. This clocks in at around 0.256ms.

So, to try to make it both faster and prettier, I tried using iterators. If there's something I learned about iterators in Rust, it is that 1) they are your friends, and 2) they go brrrrr.

In that case, we can first convert each instruction to some kind of (x, y) pair, where moving forward is (x, 0) and moving up/down is (0, y). Then we apply the previous algorithm in a more functional way, by folding the iterator and updating the accumulated values:

```rust
let (hor, _, depth1, depth2) = read_to_string("input/day02.txt").unwrap()
    .lines()
    .map(|line| {
        let mut spl = line.split(' ');
        let op = spl.next().unwrap();
        let val: i64 = spl.next().unwrap().parse().unwrap();
        match op {
            "forward" => (val, 0),
            "up" => (0, -val),
            "down" => (0, val),
            _ => unreachable!()
        }
    }).fold((0, 0, 0, 0), |(hor, aim, depth1, depth2), mv| {
        match mv {
            (x, 0) => (hor + x, aim, depth1, depth2 + aim * x),
            (0, y) => (hor, aim + y, depth1 + y, depth2),
            _ => unreachable!()
        }
    });
```

I'm pretty happy with this solution. It's shorter, looks prettier, and indeed, iterators go brrrr:

![Day 2 results](imgs/d02.png)

# Day 3

Binary numbers today. We're tasked with finding the most common bit in each position for a whole bunch of binary number, and then constructing a new number that is composed of those majority bits.

To me, it seemed like it's one of those problems where there's some crazy bit operation that you can do which results in the most common bits, but I couldn't think of anything and I didn't find anything online. So, let's do it ourselves:

```rust
fn get_most_common_bit(numbers: &[Num], i: usize, reverse: bool) -> Num {
    let count = numbers.iter().copied().filter(|x| x & (1 << i) != 0).count();
    let more_ones = count * 2 >= numbers.len();
    match (more_ones, reverse) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 1,
        (false, false) => 0,
    }
}
```

`Num` is a type alias to `u16` that I defined, to fiddle with the size of the integers used. This will give 1 or 0 depending on the most common bit in the position `i` for all the numbers. The `reverse` flag will come in handy for part 2.

(Side note: initially, `more_ones` was `count >= numbers.len() / 2)`, however, this was problematic for an odd amount of numbers. If there are 7 numbers and 3 of them had a `1` in the desired position, `more_ones` should be false, since 4 numbers have a 0. However, `7 / 2 = 3` in integer division, so it was `true` instead).

We can then get the gamma and epsilon values:

```rust
let mut aux = 0; // We use this to calculate the number of bits in the input numbers
let numbers: Vec<Num> = read_to_string("input/day03.txt").unwrap()
    .lines()
    .map(|x| {
        let val = Num::from_str_radix(x, 2).unwrap();
        aux |= val;
        val
    })
    .collect();

let num_bits = TYPE_BITS - aux.leading_zeros() as usize;
    
// Construct both numbers for part 1
let mut gamma: Num = 0;
(0..num_bits).into_iter().for_each(|i| gamma |= get_most_common_bit(&numbers, i, false) << i);
let epsilon = !gamma & Num::MAX >> (TYPE_BITS - num_bits);
```

It is obviously important that we know the length in bits of the number we're given. Since I decided to take the bitwise operations route and not work with strings, one quick way to do it is to `or` all the numbers, and then count the number of leading zeros in an unsigned type of known size.

Then, we simply construct `gamma` by setting the most common bit to each position through a simple combination of logical ors and right shifts. `epsilon` is `gamma` with its bits flipped, but we must be careful not to flip the rest of the bits in the `u16` that don't belong to the number.

For part 2, we can use a function that keeps filtering the numbers, and returns the last one remaining:

```rust
fn get_part2_number(numbers: &[Num], most_common: bool, bits: usize) -> Num {
    let mut numbers: Vec<Num> = numbers.iter().copied().collect();
    let mut i = bits - 1;

    while numbers.len() > 1 {
        let mask = 1 << i;
        let target = get_most_common_bit(&numbers, i, !most_common) << i;
        numbers = numbers.into_iter().filter(|x| x & mask == target).collect();
        i -= 1;
    }

    numbers[0]
}
```

For each position (keep in mind that the *first* position is the *most* significant bit), we get the most common bits of the remaining numbers, and we filter out those whose bit is not the same. This is done by `and`ing the number with a mask, whose only set bit is that of the current position.

The code for this day is a bit longer than I would have liked, but pretty fast:

![Day 3 results](imgs/d03.png)

# Day 4

Time to play bingo! We're given a whole bunch of bingo boards, and the order in which the numbers are called.

I think this one is all about finding a nice way to represent the problem, and prevent the complexity from blowing up every time a number appears. The thought of looping through all the columns and rows in every board every single time makes me wanna cry.

I realized that knowing the order of the numbers beforehand gives us a huge advantage. Instead of storing the numbers *themselves* in the boards, we can replace them with *the order in which they appear*. For example, if 9 is the first number to be called, we replace 13 with 0. If the second one is 37, we replace 37 with 1, and so on.

This way, we can know precisely on which turn a row or column will be completed: it is the **maximum** value in said row or column. Then, we can also know the turn in which the board wins: the first row or column to be completed, that is, the **minimum** out of the maximums of all rows and columns.

This makes things much faster! We know when a board wins **as soon as we are done creating it**. Let's create a struct to represent our board:

```rust
struct BingoBoard {
    numbers: Vec<usize>,
}
```

I decided against using a 2D representation because it's a bit less flexible, it may be less cache friendly, and I didn't want to make any assumptions on the size of the boards. Treating a 1D representation as 2D isn't too hard anyway.

Then, let's see how to create a board from the string representation in the input:

```rust
impl BingoBoard {
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
}
```

`indices` is an aux vector which maps a number to the moment in which it appears. We then turn each line of the board into the corresponding numbers, and concatenate them all through a flatmap to construct the vector. Then it's just a matter of finding the maximum values of all rows and columns, and selecting the minimum of them all. This is the moment in which this board wins.

Finally, it's just a matter of reading the input numbers into the `indices` array...

```rust
let input = read_to_string("input/day04.txt").unwrap();
let mut spl = input.split("\n\n");

// Construct the number to position vec
// This will allow us to quickly look up when a number appears in the bingo
let numbers: Vec<usize> = spl.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
let mut indices = vec![0; numbers.len()];
for (i, val) in numbers.iter().copied().enumerate() {
    indices[val] = i;
}
```

...and building the boards, keeping track of which one is the first to win and lose, along which the corresponding turn when this happens...

```rust
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
```

Finally, we simply have to have a way to get the score of a board. The numbers to sum will be those that are greater to the turn in which the board wins. We also must have a way to convert back from *turn* numbers to their actual values, thanks to the `numbers` vector previously constructed:

```rust
impl BingoBoard {
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
```

And the solutions are...

```rust
let sol1 = boards[winning_board_index].get_score(winning_row_pos, &numbers);
let sol2 = boards[losing_board_index].get_score(losing_row_pos, &numbers);
```

For a total of 0.1856ms. The slowest part is apparently parsing spaces in the string (`.split_ascii_whitespace()`). `.split(' ')` doesn't work because, in order to be prettier, the input sometimes adds more than one space between the numbers. Splitting and then filtering for non-empty strings is roughly the same.

![Day 4 results](imgs/d04.png)