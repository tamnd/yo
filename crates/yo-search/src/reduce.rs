//! The reducers a `GROUPBY` folds a group with.
//!
//! Twelve of them, and they divide into three families. Five read the property
//! as a number and answer a number, three answer something about how many
//! different values there were, two answer a list of values and two answer one
//! value picked out of the group. Nothing in here knows about the wire or about
//! an index, so every one of them is a fold over whatever the caller hands it:
//! the value of the property the reducer was pointed at, once per document in
//! the group, and the value it was told to order by when it was told to order
//! by one.
//!
//! Three rules run through the numeric family and none of them is the obvious
//! guess. A document that does not hold the property at all is not part of the
//! fold. A document that holds it as something that is not a number is not part
//! of it either, so a group of three numbers and one word sums to the three.
//! And a fold with no number in it at all answers `nan` rather than nought,
//! which is measured: `SUM` over a property nothing in the group holds answers
//! `nan` and `STDDEV` over the same group answers `0`.

use yo_common::Rng;

/// What a reducer answers once the group has been walked.
///
/// A number is written by the caller, because the twelve significant digit form
/// the wire wants belongs to the wire and not here.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// A number, which is what nine of the twelve answer.
    Number(f64),
    /// A value taken out of the group as it was found, which is what
    /// `FIRST_VALUE` answers.
    Text(Box<[u8]>),
    /// Several of them, which is what `TOLIST` and `RANDOM_SAMPLE` answer.
    List(Vec<Box<[u8]>>),
    /// Nothing, which is what `FIRST_VALUE` answers when the document it picked
    /// does not hold the property.
    Nil,
}

/// Which reducer, with whatever it was told beside the property name.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// How many documents were in the group, which is the one reducer that does
    /// not look at a property at all.
    Count,
    /// How many different values the property held.
    Distinct,
    /// The same question answered from a sketch rather than from a set, which
    /// is the reference's own approximation and is exact here. Divergence D-69.
    Distinctish,
    /// The numbers added up.
    Sum,
    /// The smallest of them.
    Min,
    /// The largest of them.
    Max,
    /// The mean of them.
    Avg,
    /// The sample standard deviation of them, which divides by one less than
    /// the count and answers nought for a group with fewer than two numbers.
    Stddev,
    /// The value at a fraction of the way through them, sorted.
    Quantile(f64),
    /// Every different value, in the order they were first seen.
    ToList,
    /// The value from the first document once the group is put in order of
    /// another property, or of nothing at all, in which case the first document
    /// is the first one that arrived.
    First(Order),
    /// Up to this many values, picked without favouring any of them.
    Sample(usize),
}

/// How `FIRST_VALUE` puts the group in order before it takes the front of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Order {
    /// Whether there is a property to order by at all. Without one the group
    /// stays in the order the documents arrived in.
    pub by: bool,
    /// Whether the order runs backwards.
    pub desc: bool,
    /// Whether the property being ordered by is held as a number, which is what
    /// decides between comparing two values as numbers and comparing them as
    /// bytes. It is the field and not the value that decides: a schema that
    /// holds a field as `NUMERIC` orders `9` before `10`, and a value loaded
    /// off the key with no type behind it orders `10` before `9`.
    pub numeric: bool,
}

/// The document a `FIRST_VALUE` is holding on to: the value it was judged on
/// and the value it answers, either of which the document may not have had.
type Best = Option<(Option<Box<[u8]>>, Option<Box<[u8]>>)>;

/// The seed every `RANDOM_SAMPLE` starts from.
///
/// A fixed one, so the same query over the same group answers the same sample
/// every time. A real server seeds from the process and answers a different
/// sample on every call, which is not something a test can be written against.
const SEED: u64 = 0x5265_6475_6365_7221;

/// One reducer part way through a group.
pub struct Fold {
    kind: Kind,
    /// Every document in the group, whether or not it held the property.
    seen: usize,
    /// The numbers among the values, which is what the numeric family folds.
    count: usize,
    sum: f64,
    /// The running mean and sum of squared differences from it, which is how a
    /// standard deviation is taken in one pass without losing the low bits.
    mean: f64,
    square: f64,
    least: f64,
    most: f64,
    /// Every different value in the order they were first seen, which two
    /// reducers count and one answers.
    different: Vec<Box<[u8]>>,
    /// Every number, kept because a quantile cannot be taken without sorting.
    numbers: Vec<f64>,
    /// The reservoir a `RANDOM_SAMPLE` fills, and how many values have gone
    /// past it.
    reservoir: Vec<Box<[u8]>>,
    rng: Rng,
    /// The best document so far by whatever `FIRST_VALUE` was told to order by,
    /// as the key it was judged on and the value it answers.
    best: Best,
}

impl Fold {
    /// A reducer with nothing in it yet.
    #[must_use]
    pub fn new(kind: Kind) -> Fold {
        Fold {
            kind,
            seen: 0,
            count: 0,
            sum: 0.0,
            mean: 0.0,
            square: 0.0,
            least: f64::INFINITY,
            most: f64::NEG_INFINITY,
            different: Vec::new(),
            numbers: Vec::new(),
            reservoir: Vec::new(),
            rng: Rng::new(SEED),
            best: None,
        }
    }

    /// One document of the group.
    ///
    /// The value is what the document holds at the property the reducer was
    /// pointed at, and the key is what it holds at the property it was told to
    /// order by. Both are `None` for a document that does not hold them.
    pub fn add(&mut self, value: Option<&[u8]>, key: Option<&[u8]>) {
        self.seen += 1;
        match &self.kind {
            Kind::Count => {}
            Kind::Distinct | Kind::Distinctish | Kind::ToList => {
                if let Some(value) = value
                    && !self.different.iter().any(|held| &**held == value)
                {
                    self.different.push(value.into());
                }
            }
            Kind::Sum | Kind::Min | Kind::Max | Kind::Avg | Kind::Stddev => {
                if let Some(number) = value.and_then(number) {
                    self.count += 1;
                    self.sum += number;
                    // Welford, so a group of large numbers close together does
                    // not lose the difference between them to the subtraction.
                    let step = number - self.mean;
                    self.mean += step / self.count as f64;
                    self.square += step * (number - self.mean);
                    self.least = self.least.min(number);
                    self.most = self.most.max(number);
                }
            }
            Kind::Quantile(_) => {
                if let Some(number) = value.and_then(number) {
                    self.numbers.push(number);
                }
            }
            Kind::First(order) => self.first(*order, value, key),
            Kind::Sample(want) => self.reserve(*want, value),
        }
    }

    /// Keeps the document that sorts first, or the first one that arrived when
    /// there is nothing to sort by.
    ///
    /// A document that does not hold the key sorts last, so it only wins a
    /// group where nothing holds it. That is a rule this side settles and the
    /// reference does not: its comparison treats a missing key as neither
    /// smaller nor larger than a value, so which document wins depends on the
    /// order the sort happened to walk them in. Divergence D-70.
    fn first(&mut self, order: Order, value: Option<&[u8]>, key: Option<&[u8]>) {
        let key = key.map(Box::<[u8]>::from);
        let value = value.map(Box::<[u8]>::from);
        let Some((held, _)) = &self.best else {
            self.best = Some((key, value));
            return;
        };
        if !order.by {
            return;
        }
        let better = match (held, &key) {
            (None, None) | (Some(_), None) => false,
            (None, Some(_)) => true,
            (Some(held), Some(key)) => match order.desc {
                false => compare(key, held, order.numeric).is_lt(),
                true => compare(key, held, order.numeric).is_gt(),
            },
        };
        if better {
            self.best = Some((key, value));
        }
    }

    /// Fills the reservoir and then replaces from it, which is the sampling
    /// that gives every value the same chance of being in the answer whatever
    /// the size of the group turns out to be.
    ///
    /// The first `want` values go in in the order they arrived, so a sample as
    /// wide as the group is the whole group in document order and there is
    /// nothing random about it.
    fn reserve(&mut self, want: usize, value: Option<&[u8]>) {
        let Some(value) = value else {
            return;
        };
        self.count += 1;
        if self.reservoir.len() < want {
            self.reservoir.push(value.into());
            return;
        }
        let at = self.rng.below(self.count);
        if at < want {
            self.reservoir[at] = value.into();
        }
    }

    /// What the group came to.
    #[must_use]
    pub fn done(self) -> Answer {
        match self.kind {
            Kind::Count => Answer::Number(self.seen as f64),
            Kind::Distinct | Kind::Distinctish => Answer::Number(self.different.len() as f64),
            // A fold with no number in it answers `nan` rather than nought,
            // which is only visible on a group where nothing held the property
            // or nothing held it as a number.
            Kind::Sum => Answer::Number(match self.count {
                0 => f64::NAN,
                _ => self.sum,
            }),
            // The two ends keep the value they started from, so a group with no
            // number in it is smallest at positive infinity and largest at
            // negative infinity. That reads like a bug and is what a real
            // server answers.
            Kind::Min => Answer::Number(self.least),
            Kind::Max => Answer::Number(self.most),
            Kind::Avg => Answer::Number(self.sum / self.count as f64),
            Kind::Stddev => Answer::Number(match self.count {
                0 | 1 => 0.0,
                _ => (self.square / (self.count - 1) as f64).sqrt(),
            }),
            Kind::Quantile(want) => Answer::Number(quantile(self.numbers, want)),
            Kind::ToList => Answer::List(self.different),
            Kind::First(_) => match self.best.and_then(|(_, value)| value) {
                Some(value) => Answer::Text(value),
                None => Answer::Nil,
            },
            Kind::Sample(_) => Answer::List(self.reservoir),
        }
    }
}

/// The value a fraction of the way through a sorted group.
///
/// The index is one below the fraction of the count rounded up, held at nought
/// from below, so a median over two numbers is the smaller of them and a median
/// over three is the middle one. That is measured over group sizes from two to
/// three hundred and it is not the interpolating quantile most libraries write:
/// nothing is ever averaged, the answer is always one of the numbers that was
/// there.
fn quantile(mut numbers: Vec<f64>, want: f64) -> f64 {
    if numbers.is_empty() {
        return f64::NAN;
    }
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let reach = want * numbers.len() as f64;
    let at = (reach.ceil() as usize).saturating_sub(1);
    numbers[at.min(numbers.len() - 1)]
}

/// Two values in the order the property they came from is held in.
fn compare(a: &[u8], b: &[u8], numeric: bool) -> core::cmp::Ordering {
    if numeric && let (Some(a), Some(b)) = (number(a), number(b)) {
        return a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal);
    }
    a.cmp(b)
}

/// The number a value holds, or nothing when it holds something else.
///
/// The whole value or nothing, the same rule the numeric index reads a field
/// by, so `2.5` and `1e3` are numbers and `7x` and an empty value are not.
#[must_use]
pub fn number(raw: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(raw).ok()?;
    let number: f64 = text.parse().ok()?;
    (!number.is_nan()).then_some(number)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folds a group of values, none of them missing, and answers what came out.
    fn fold(kind: Kind, values: &[&[u8]]) -> Answer {
        let mut fold = Fold::new(kind);
        for value in values {
            fold.add(Some(value), None);
        }
        fold.done()
    }

    #[test]
    fn a_number_fold_leaves_out_what_is_not_a_number() {
        let mixed: &[&[u8]] = &[b"10", b"xx", b"20", b""];
        assert_eq!(fold(Kind::Sum, mixed), Answer::Number(30.0));
        assert_eq!(fold(Kind::Avg, mixed), Answer::Number(15.0));
        assert_eq!(fold(Kind::Min, mixed), Answer::Number(10.0));
        assert_eq!(fold(Kind::Max, mixed), Answer::Number(20.0));
        // The count is over the values and not over the numbers among them.
        assert_eq!(fold(Kind::Count, mixed), Answer::Number(4.0));
        assert_eq!(fold(Kind::Distinct, mixed), Answer::Number(4.0));
    }

    #[test]
    fn a_fold_with_no_number_in_it_answers_the_ends_it_started_from() {
        let words: &[&[u8]] = &[b"red", b"blue"];
        let Answer::Number(sum) = fold(Kind::Sum, words) else {
            panic!("a sum is a number");
        };
        assert!(sum.is_nan());
        assert_eq!(fold(Kind::Min, words), Answer::Number(f64::INFINITY));
        assert_eq!(fold(Kind::Max, words), Answer::Number(f64::NEG_INFINITY));
        assert_eq!(fold(Kind::Stddev, words), Answer::Number(0.0));
    }

    #[test]
    fn a_deviation_divides_by_one_less_than_the_count() {
        let values: &[&[u8]] = &[b"10", b"20", b"60"];
        let Answer::Number(got) = fold(Kind::Stddev, values) else {
            panic!("a deviation is a number");
        };
        assert!((got - 26.457_513_110_645_9).abs() < 1e-9, "{got}");
        assert_eq!(fold(Kind::Stddev, &[b"7"]), Answer::Number(0.0));
    }

    #[test]
    fn a_quantile_rounds_the_reach_up_and_steps_back_one() {
        let five: &[&[u8]] = &[b"1", b"2", b"3", b"4", b"5"];
        assert_eq!(fold(Kind::Quantile(0.0), five), Answer::Number(1.0));
        assert_eq!(fold(Kind::Quantile(0.25), five), Answer::Number(2.0));
        assert_eq!(fold(Kind::Quantile(0.5), five), Answer::Number(3.0));
        assert_eq!(fold(Kind::Quantile(1.0), five), Answer::Number(5.0));
        // Two numbers and a median, which is the case that tells the rounding
        // up from the rounding down: the smaller one and not the larger.
        assert_eq!(
            fold(Kind::Quantile(0.5), &[b"10", b"20"]),
            Answer::Number(10.0)
        );
    }

    #[test]
    fn a_list_keeps_the_order_the_values_were_first_seen_in() {
        assert_eq!(
            fold(Kind::ToList, &[b"b", b"a", b"b", b"c"]),
            Answer::List(vec![
                b"b".to_vec().into(),
                b"a".to_vec().into(),
                b"c".to_vec().into()
            ])
        );
    }

    #[test]
    fn a_sample_as_wide_as_the_group_is_the_group_in_order() {
        assert_eq!(
            fold(Kind::Sample(4), &[b"a", b"b", b"c"]),
            Answer::List(vec![
                b"a".to_vec().into(),
                b"b".to_vec().into(),
                b"c".to_vec().into()
            ])
        );
        let Answer::List(some) = fold(Kind::Sample(2), &[b"a", b"b", b"c", b"d"]) else {
            panic!("a sample is a list");
        };
        assert_eq!(some.len(), 2);
    }

    #[test]
    fn a_first_value_orders_by_the_key_it_was_given() {
        let plain = Order {
            by: false,
            desc: false,
            numeric: false,
        };
        let mut fold = Fold::new(Kind::First(plain));
        for (value, key) in [(b"one", b"3"), (b"two", b"1"), (b"tre", b"2")] {
            fold.add(Some(value), Some(key));
        }
        assert_eq!(fold.done(), Answer::Text(b"one".to_vec().into()));

        let up = Order { by: true, ..plain };
        let mut fold = Fold::new(Kind::First(up));
        for (value, key) in [(b"one", b"3"), (b"two", b"1"), (b"tre", b"2")] {
            fold.add(Some(value), Some(key));
        }
        assert_eq!(fold.done(), Answer::Text(b"two".to_vec().into()));

        let down = Order { desc: true, ..up };
        let mut fold = Fold::new(Kind::First(down));
        for (value, key) in [(b"one", b"3"), (b"two", b"1"), (b"tre", b"2")] {
            fold.add(Some(value), Some(key));
        }
        assert_eq!(fold.done(), Answer::Text(b"one".to_vec().into()));
    }

    #[test]
    fn a_numeric_key_orders_nine_before_ten_and_a_plain_one_does_not() {
        for (numeric, want) in [(true, b"nine".as_slice()), (false, b"ten".as_slice())] {
            let mut fold = Fold::new(Kind::First(Order {
                by: true,
                desc: false,
                numeric,
            }));
            fold.add(Some(b"nine"), Some(b"9"));
            fold.add(Some(b"ten"), Some(b"10"));
            assert_eq!(fold.done(), Answer::Text(want.to_vec().into()));
        }
    }

    #[test]
    fn a_document_with_no_key_only_wins_a_group_where_nothing_has_one() {
        let up = Order {
            by: true,
            desc: false,
            numeric: false,
        };
        let mut fold = Fold::new(Kind::First(up));
        fold.add(Some(b"one"), None);
        fold.add(Some(b"two"), Some(b"9"));
        assert_eq!(fold.done(), Answer::Text(b"two".to_vec().into()));

        let mut fold = Fold::new(Kind::First(up));
        fold.add(None, None);
        fold.add(Some(b"two"), None);
        assert_eq!(fold.done(), Answer::Nil);
    }
}
