//! Reading a series back, a sample at a time or a bucket at a time.
//!
//! A read names a span of time and optionally a bucket width. Without a width
//! it answers the stored samples. With one it cuts the span into buckets and
//! answers a reduction of each, one number per reduction asked for.
//!
//! Two things about this are worth knowing before reading the code.
//!
//! A reading that is not a number is not the same as a bucket with nothing in
//! it. Every reduction except `countnan` and `countall` skips a NaN, so a
//! bucket holding nothing but NaN reduces to nothing and is left out unless the
//! read asks for the empty buckets. `countall` counts every reading and
//! `countnan` counts only the ones that are not numbers, so a bucket the others
//! call empty is a bucket those two have an answer for. The difference shows up
//! again in `last`, which carries the previous reading into a bucket that has
//! none: a bucket with no readings at all takes the reading before the gap
//! whichever way the read runs, and a bucket whose readings are all not numbers
//! takes whatever the bucket before it in the reading direction answered, so
//! that one is not the same forwards and backwards.
//!
//! Filling the empty buckets only fills the gaps between readings. A run of
//! empty buckets before the first reading in the series, or after the last one,
//! is left out, because otherwise a read that ends at the largest timestamp
//! there is would ask for more buckets than there are atoms to build them from.
//! The gap has to have a reading on each side, and the readings that count are
//! the ones in the whole series rather than the ones inside the span, so a gap
//! that runs off the end of the span is still a gap.

use crate::sample::Sample;
use crate::series::Series;

/// The most rows one read will build.
///
/// Only a read that fills its empty buckets can get near this, because every
/// other read is bounded by how many samples are stored. A gap filled at a fine
/// enough bucket width asks for more rows than there is memory to hold them in,
/// and the reference module will try, so the ceiling is here instead.
pub const MAX_ROWS: usize = 8 << 20;

/// What a bucket of readings is reduced to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Agg {
    /// The smallest reading.
    Min,
    /// The largest.
    Max,
    /// All of them added up.
    Sum,
    /// The mean.
    Avg,
    /// The mean weighted by how long each reading stood for.
    Twa,
    /// The last reading.
    Last,
    /// How many readings there were.
    Count,
    /// The largest minus the smallest.
    Range,
    /// The first reading.
    First,
    /// The population standard deviation.
    StdP,
    /// The sample standard deviation.
    StdS,
    /// The population variance.
    VarP,
    /// The sample variance.
    VarS,
    /// How many readings were not numbers.
    CountNan,
    /// How many readings there were of any kind.
    CountAll,
}

impl Agg {
    /// Read one off the wire, in any case.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Self> {
        let mut lower = [0u8; 8];
        if word.len() > lower.len() {
            return None;
        }
        for (slot, byte) in lower.iter_mut().zip(word) {
            *slot = byte.to_ascii_lowercase();
        }
        Some(match &lower[..word.len()] {
            b"min" => Self::Min,
            b"max" => Self::Max,
            b"sum" => Self::Sum,
            b"avg" => Self::Avg,
            b"twa" => Self::Twa,
            b"last" => Self::Last,
            b"count" => Self::Count,
            b"range" => Self::Range,
            b"first" => Self::First,
            b"std.p" => Self::StdP,
            b"std.s" => Self::StdS,
            b"var.p" => Self::VarP,
            b"var.s" => Self::VarS,
            b"countnan" => Self::CountNan,
            b"countall" => Self::CountAll,
            _ => return None,
        })
    }

    /// What it is called back, which is always lower case.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
            Self::Avg => "avg",
            Self::Twa => "twa",
            Self::Last => "last",
            Self::Count => "count",
            Self::Range => "range",
            Self::First => "first",
            Self::StdP => "std.p",
            Self::StdS => "std.s",
            Self::VarP => "var.p",
            Self::VarS => "var.s",
            Self::CountNan => "countnan",
            Self::CountAll => "countall",
        }
    }

    /// Whether a reading is one this reduction has any use for.
    fn takes(self, value: f64) -> bool {
        match self {
            Self::CountNan => value.is_nan(),
            Self::CountAll => true,
            _ => !value.is_nan(),
        }
    }

    /// What it answers for a bucket it found nothing in. `Last` is not here
    /// because it carries the reading before the gap forward instead, and `Twa`
    /// is not either because it interpolates across the gap.
    fn nothing(self) -> f64 {
        match self {
            Self::Sum | Self::Count | Self::CountNan | Self::CountAll => 0.0,
            _ => f64::NAN,
        }
    }
}

/// Where in a bucket the timestamp it is reported under sits.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Stamp {
    /// At the start of it, which is what a bucket gets unless asked otherwise.
    #[default]
    Start,
    /// Halfway through.
    Mid,
    /// At the start of the one after it.
    End,
}

/// How a read is cut into buckets.
#[derive(Clone, Debug)]
pub struct Buckets {
    /// One reduction per number each row carries, in the order asked for.
    pub aggs: Vec<Agg>,
    /// How wide a bucket is, in milliseconds. Always above zero.
    pub delta: i64,
    /// The timestamp the bucket edges are lined up against.
    pub align: i64,
    /// Whether to fill the gaps between readings with empty buckets.
    pub empty: bool,
    /// Which end of a bucket its timestamp comes from.
    pub stamp: Stamp,
}

/// Everything a read needs to know.
#[derive(Clone, Debug)]
pub struct Query {
    /// The oldest timestamp wanted.
    pub from: i64,
    /// The newest.
    pub to: i64,
    /// Whether the rows come back newest first.
    pub reverse: bool,
    /// How many rows at most, counted from whichever end comes first.
    pub count: Option<usize>,
    /// Keep only readings on one of these timestamps. Sorted, no repeats.
    pub by_ts: Option<Vec<i64>>,
    /// Keep only readings between these two, both ends included.
    pub by_value: Option<(f64, f64)>,
    /// The bucketing, if the read asked for any.
    pub buckets: Option<Buckets>,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            from: 0,
            to: i64::MAX,
            reverse: false,
            count: None,
            by_ts: None,
            by_value: None,
            buckets: None,
        }
    }
}

/// What a read answers with.
///
/// The numbers are laid out row by row, [`Rows::width`] of them per timestamp,
/// so a read of one reduction is one number a row and a read of five is five.
#[derive(Clone, Debug, Default)]
pub struct Rows {
    /// One per row, in the order they go on the wire.
    pub stamps: Vec<i64>,
    /// `width` numbers per row, row after row.
    pub values: Vec<f64>,
    /// How many numbers a row carries. Always at least one.
    pub width: usize,
}

impl Rows {
    /// How many rows there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    /// Whether the read found anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    /// The numbers row `i` carries.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.width..(i + 1) * self.width]
    }

    /// Turn the rows around, which is all a read backwards is.
    fn flip(&mut self) {
        self.stamps.reverse();
        if self.width == 1 {
            self.values.reverse();
            return;
        }
        let rows = self.stamps.len();
        for i in 0..rows / 2 {
            let (head, tail) = self.values.split_at_mut((rows - 1 - i) * self.width);
            head[i * self.width..(i + 1) * self.width].swap_with_slice(&mut tail[..self.width]);
        }
    }

    /// Drop everything past the first `n` rows.
    fn keep(&mut self, n: usize) {
        if n < self.stamps.len() {
            self.stamps.truncate(n);
            self.values.truncate(n * self.width);
        }
    }
}

/// Why a read answered nothing at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unread {
    /// Filling the gaps would have built more than [`MAX_ROWS`] rows.
    TooWide,
}

/// Which bucket a timestamp falls in.
fn bucket_of(at: i64, delta: i64, align: i64) -> i64 {
    let start = at - (at - align).rem_euclid(delta);
    // A bucket that starts before the epoch is reported as starting on it,
    // because no reading can live in the part that hangs off the front.
    start.max(0)
}

/// The timestamp a bucket is reported under.
fn stamp_of(stamp: Stamp, start: i64, delta: i64) -> i64 {
    match stamp {
        Stamp::Start => start,
        Stamp::Mid => start + delta / 2,
        Stamp::End => start + delta,
    }
}

/// The variance, written the way the module writes it so the last digit agrees.
fn variance(sum: f64, sum_2: f64, count: f64) -> f64 {
    (sum_2 - 2.0 * sum * sum / count + (sum / count).powi(2) * count) / count
}

impl Series {
    /// Read the series back the way `query` asks for it.
    ///
    /// # Errors
    ///
    /// [`Unread::TooWide`] when the read asks for the empty buckets and the gap
    /// it would have to fill runs to more than [`MAX_ROWS`] of them.
    pub fn read(&self, query: &Query) -> Result<Rows, Unread> {
        // Anything older than the retention window is gone as far as a reader
        // is concerned, whether or not it has been trimmed off the front yet.
        let mut from = query.from;
        if let (Some(last), retention) = (self.last(), self.retention())
            && retention > 0
            && last > retention
        {
            from = from.max(last - retention);
        }
        let kept: Vec<Sample> = self
            .range(from, query.to)
            .filter(|s| self.wanted(query, *s))
            .collect();

        let mut rows = match &query.buckets {
            None => plain(&kept),
            Some(buckets) => self.bucketed(query, buckets, &kept)?,
        };
        if query.reverse {
            rows.flip();
        }
        if let Some(n) = query.count {
            rows.keep(n);
        }
        Ok(rows)
    }

    /// Whether a reading survives the two filters.
    fn wanted(&self, query: &Query, s: Sample) -> bool {
        if let Some((min, max)) = query.by_value
            && !(s.value >= min && s.value <= max)
        {
            return false;
        }
        match &query.by_ts {
            Some(list) => list.binary_search(&s.at).is_ok(),
            None => true,
        }
    }

    /// The samples the whole series holds either side of a timestamp, up to two
    /// each way, skipping the readings that are not numbers and honouring the
    /// two filters. This is what an empty bucket interpolates from and what
    /// decides whether a run of empty buckets is a gap or an edge.
    fn neighbours(&self, query: &Query, at: i64) -> (Vec<Sample>, Vec<Sample>) {
        let mut left: Vec<Sample> = Vec::with_capacity(2);
        if at > 0 {
            for s in self.range(0, at - 1) {
                if !s.value.is_nan() && self.wanted(query, s) {
                    if left.len() == 2 {
                        left[1] = left[0];
                        left[0] = s;
                    } else {
                        left.insert(0, s);
                    }
                }
            }
        }
        let mut right = Vec::with_capacity(2);
        for s in self.range(at, i64::MAX) {
            if !s.value.is_nan() && self.wanted(query, s) {
                right.push(s);
                if right.len() == 2 {
                    break;
                }
            }
        }
        (left, right)
    }

    /// The bucketed read.
    fn bucketed(&self, query: &Query, buckets: &Buckets, kept: &[Sample]) -> Result<Rows, Unread> {
        let width = buckets.aggs.len();
        let mut rows = Rows {
            width,
            ..Rows::default()
        };
        // One flag per number written, saying whether it is a `last` waiting on
        // the bucket before it in the reading direction. See [`carry`].
        let mut marks: Vec<bool> = Vec::new();
        // Every run of samples that shares a bucket, in order.
        let mut groups: Vec<(i64, usize, usize)> = Vec::new();
        for (i, s) in kept.iter().enumerate() {
            let at = bucket_of(s.at, buckets.delta, buckets.align);
            match groups.last_mut() {
                Some(group) if group.0 == at => group.2 = i + 1,
                _ => groups.push((at, i, i + 1)),
            }
        }

        let first = groups.first().map(|g| g.0);
        let last = groups.last().map(|g| g.0);
        if buckets.empty {
            let head = bucket_of(query.from.max(0), buckets.delta, buckets.align);
            let tail = first.map_or_else(
                || bucket_of(query.to, buckets.delta, buckets.align),
                |b| b - buckets.delta,
            );
            self.fill(query, buckets, head, tail, true, &mut rows, &mut marks)?;
        }

        for (i, &(start, lo, hi)) in groups.iter().enumerate() {
            let before = i.checked_sub(1).map(|j| groups[j]);
            let group = Group {
                start,
                samples: &kept[lo..hi],
                before,
                kept,
                after: hi,
            };
            self.emit(query, buckets, &group, &mut rows, &mut marks)?;
            if buckets.empty
                && let Some(&(next, _, _)) = groups.get(i + 1)
            {
                self.fill(
                    query,
                    buckets,
                    start + buckets.delta,
                    next - buckets.delta,
                    false,
                    &mut rows,
                    &mut marks,
                )?;
            }
        }

        if buckets.empty
            && let Some(last) = last
        {
            let end = bucket_of(query.to, buckets.delta, buckets.align);
            self.fill(
                query,
                buckets,
                last + buckets.delta,
                end,
                true,
                &mut rows,
                &mut marks,
            )?;
        }
        carry(&mut rows, &marks, query.reverse);
        Ok(rows)
    }

    /// One bucket that has samples in it.
    fn emit(
        &self,
        query: &Query,
        buckets: &Buckets,
        group: &Group<'_>,
        rows: &mut Rows,
        marks: &mut Vec<bool>,
    ) -> Result<(), Unread> {
        // A bucket whose readings none of the reductions has a use for is a
        // bucket with nothing in it, and is left out unless the read asked for
        // those too.
        let anything = buckets
            .aggs
            .iter()
            .any(|agg| group.samples.iter().any(|s| agg.takes(s.value)));
        if !anything && !buckets.empty {
            return Ok(());
        }
        if rows.len() == MAX_ROWS {
            return Err(Unread::TooWide);
        }

        // The reading that closes the bucket before this one and the one that
        // opens the bucket after it, which is all the weighted mean needs to
        // know about its neighbours.
        let prev = group
            .before
            .and_then(|(_, lo, hi)| group.kept[lo..hi].iter().rev().find(|s| !s.value.is_nan()))
            .copied();
        let next = group
            .kept
            .get(group.after)
            .copied()
            .filter(|s| !s.value.is_nan());

        rows.stamps
            .push(stamp_of(buckets.stamp, group.start, buckets.delta));
        for &agg in &buckets.aggs {
            let taken: Vec<f64> = group
                .samples
                .iter()
                .filter(|s| agg.takes(s.value))
                .map(|s| s.value)
                .collect();
            let value = if agg == Agg::Twa {
                if taken.is_empty() {
                    self.interpolate(query, buckets, group.start)
                } else {
                    weighted(query, buckets, group.start, group.samples, prev, next)
                }
            } else if taken.is_empty() {
                agg.nothing()
            } else {
                reduce(agg, &taken)
            };
            marks.push(agg == Agg::Last && taken.is_empty());
            rows.values.push(value);
        }
        Ok(())
    }

    /// A run of buckets with nothing in them, `head` through `tail`.
    ///
    /// A run at either edge of the readings is dropped rather than filled, so
    /// `edge` says whether this is one of those and the run has to prove it has
    /// a reading on both sides before anything is written.
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        query: &Query,
        buckets: &Buckets,
        head: i64,
        tail: i64,
        edge: bool,
        rows: &mut Rows,
        marks: &mut Vec<bool>,
    ) -> Result<(), Unread> {
        if head > tail {
            return Ok(());
        }
        let (left, right) = if edge {
            let ta = head.max(query.from);
            let (left, right) = self.neighbours(query, ta);
            if left.is_empty() || right.is_empty() {
                return Ok(());
            }
            (left, right)
        } else {
            // An interior run has a reading on each side by construction, and
            // the two scans it would take to prove it are not worth paying for.
            self.neighbours(query, head.max(query.from))
        };
        let span = (tail - head) / buckets.delta + 1;
        let span = usize::try_from(span).map_err(|_| Unread::TooWide)?;
        if rows.len() + span > MAX_ROWS {
            return Err(Unread::TooWide);
        }
        // The reading carried into a gap is the last one before it, and it is
        // the same one for every bucket in the run.
        let carried = left.first().map_or(f64::NAN, |s| s.value);
        for i in 0..span {
            let start = head + buckets.delta * i64::try_from(i).unwrap_or(i64::MAX);
            rows.stamps
                .push(stamp_of(buckets.stamp, start, buckets.delta));
            for &agg in &buckets.aggs {
                let value = match agg {
                    Agg::Twa => {
                        let ta = start.max(query.from);
                        let tb = (start + buckets.delta).min(query.to);
                        gap_value(ta, tb, &left, &right)
                    }
                    Agg::Last => carried,
                    other => other.nothing(),
                };
                marks.push(false);
                rows.values.push(value);
            }
        }
        Ok(())
    }

    /// The weighted mean of a bucket that has nothing in it, read off the
    /// readings either side of it.
    fn interpolate(&self, query: &Query, buckets: &Buckets, start: i64) -> f64 {
        let ta = start.max(query.from);
        let tb = (start + buckets.delta).min(query.to);
        let (left, _) = self.neighbours(query, ta);
        let (_, right) = self.neighbours(query, tb);
        gap_value(ta, tb, &left, &right)
    }
}

/// One bucket's worth of samples and enough of what is around it for the
/// weighted mean to reach across its edges.
struct Group<'s> {
    /// Where the bucket starts.
    start: i64,
    /// The samples inside it.
    samples: &'s [Sample],
    /// The bucket before this one, if the read found any samples in it.
    before: Option<(i64, usize, usize)>,
    /// Every sample the read kept, which is what the two above index into.
    kept: &'s [Sample],
    /// Where the samples after this bucket start.
    after: usize,
}

/// Carry the last reading into the buckets that had none of their own.
///
/// A bucket holding readings that are all not numbers takes whatever the bucket
/// before it in the reading direction answered, so this runs in that direction
/// rather than in timestamp order, and a bucket at the near end of the read has
/// nothing to take. A bucket with no readings at all is a different thing: that
/// one takes the reading before the gap whichever way the read runs, and is
/// already filled in by the time this sees it.
fn carry(rows: &mut Rows, marks: &[bool], reverse: bool) {
    if !marks.contains(&true) {
        return;
    }
    let mut held = vec![f64::NAN; rows.width];
    for step in 0..rows.len() {
        let row = if reverse { rows.len() - 1 - step } else { step };
        for (column, slot) in held.iter_mut().enumerate() {
            let at = row * rows.width + column;
            if marks[at] {
                rows.values[at] = *slot;
            } else {
                *slot = rows.values[at];
            }
        }
    }
}

/// A read with no bucketing, which is the samples as they were stored.
fn plain(kept: &[Sample]) -> Rows {
    Rows {
        stamps: kept.iter().map(|s| s.at).collect(),
        values: kept.iter().map(|s| s.value).collect(),
        width: 1,
    }
}

/// Everything but the weighted mean, over the readings the reduction takes.
fn reduce(agg: Agg, taken: &[f64]) -> f64 {
    let n = taken.len() as f64;
    match agg {
        Agg::Count | Agg::CountNan | Agg::CountAll => n,
        Agg::First => taken[0],
        Agg::Last => taken[taken.len() - 1],
        Agg::Min => taken.iter().copied().fold(f64::MAX, f64::min),
        Agg::Max => taken.iter().copied().fold(-f64::MAX, f64::max),
        Agg::Range => {
            taken.iter().copied().fold(-f64::MAX, f64::max)
                - taken.iter().copied().fold(f64::MAX, f64::min)
        }
        Agg::Sum => taken.iter().sum(),
        Agg::Avg => mean(taken),
        Agg::VarP => variance(taken.iter().sum(), taken.iter().map(|v| v * v).sum(), n),
        Agg::StdP => variance(taken.iter().sum(), taken.iter().map(|v| v * v).sum(), n).sqrt(),
        Agg::VarS => sample_variance(taken, n),
        Agg::StdS => sample_variance(taken, n).sqrt(),
        Agg::Twa => unreachable!("the weighted mean is worked out from the whole bucket"),
    }
}

/// The mean, which adds the readings up unless doing so would run past the
/// largest number there is and folds them one at a time when it would.
fn mean(taken: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut cnt = 0.0f64;
    let mut folded = false;
    for &value in taken {
        cnt += 1.0;
        if folded || ((sum < 0.0) == (value < 0.0) && sum.abs() > f64::MAX - value.abs()) {
            let mut running = sum / cnt;
            if folded {
                running *= cnt - 1.0;
            }
            sum = running + value / cnt;
            folded = true;
        } else {
            sum += value;
        }
    }
    if folded { sum } else { sum / cnt }
}

/// The variance over a sample rather than a population, which is the population
/// one stretched by one over `n - 1`.
fn sample_variance(taken: &[f64], n: f64) -> f64 {
    if taken.len() == 1 {
        return 0.0;
    }
    variance(taken.iter().sum(), taken.iter().map(|v| v * v).sum(), n) * n / (n - 1.0)
}

/// The mean of a bucket weighted by how long each reading stood for.
///
/// The readings inside the bucket give the area under the line joining them.
/// The reading that closed the bucket before this one and the one that opens
/// the bucket after it are used to work out where the line crosses the two
/// edges, so a bucket with a neighbour on each side is measured across its
/// whole width and one on the end of the series only across its readings.
fn weighted(
    query: &Query,
    buckets: &Buckets,
    start: i64,
    group: &[Sample],
    prev: Option<Sample>,
    next: Option<Sample>,
) -> f64 {
    let inside: Vec<Sample> = group
        .iter()
        .copied()
        .filter(|s| !s.value.is_nan())
        .collect();
    let mut area = 0.0f64;
    let opens = inside[0];
    let closes = inside[inside.len() - 1];

    let first_ts = match prev {
        Some(p) => {
            let edge = start;
            let at_edge = cross(p, opens, edge);
            area += (at_edge + opens.value) * (opens.at - edge) as f64 / 2.0;
            edge
        }
        None => opens.at,
    };
    for pair in inside.windows(2) {
        area += (pair[0].value + pair[1].value) * (pair[1].at - pair[0].at) as f64 / 2.0;
    }
    let mut last_ts = closes.at;
    if let Some(n) = next {
        let edge = (start + buckets.delta).min(query.to);
        let at_edge = cross(closes, n, edge);
        area += (at_edge + closes.value) * (edge - closes.at) as f64 / 2.0;
        last_ts = edge;
    }
    if last_ts == first_ts {
        closes.value
    } else {
        area / (last_ts - first_ts).abs() as f64
    }
}

/// Where the line through two readings sits at a given moment.
fn cross(a: Sample, b: Sample, at: i64) -> f64 {
    a.value + ((at - a.at) as f64 * (b.value - a.value)) / (b.at - a.at) as f64
}

/// The weighted mean of a bucket with nothing in it, from the readings around
/// it. With one on each side the line between them is read at both edges and
/// halved. With readings on one side only, the bucket takes the nearest one if
/// it sits within half a step of the bucket and is left as not a number if it
/// does not, because past that there is nothing to say the reading still held.
fn gap_value(ta: i64, tb: i64, left: &[Sample], right: &[Sample]) -> f64 {
    let mut real = false;
    if left.len() > 1 {
        let step = left[0].at - left[1].at;
        real |= left[0].at + step > ta;
    }
    if right.len() > 1 {
        let step = right[1].at - right[0].at;
        real |= tb + step > right[0].at;
    }
    let both = !left.is_empty() && !right.is_empty();
    if both {
        let (a, b) = (left[0], right[0]);
        return (cross(a, b, ta) + cross(a, b, tb)) / 2.0;
    }
    if !real {
        return f64::NAN;
    }
    if right.len() > 1 {
        let step = right[1].at - right[0].at;
        if tb + step / 2 <= right[0].at {
            f64::NAN
        } else {
            right[0].value
        }
    } else {
        let step = left[0].at - left[1].at;
        if left[0].at + step / 2 <= ta {
            f64::NAN
        } else {
            left[0].value
        }
    }
}
