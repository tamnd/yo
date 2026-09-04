//! A whole series: the chunks, and everything the commands can set on it.

use crate::chunk::{Chunk, Encoding};
use crate::query::Agg;
use crate::sample::Sample;

/// How much room a chunk gets when nobody says otherwise.
pub const DEFAULT_CHUNK_BYTES: usize = 4096;

/// What happens to a sample that lands on a timestamp already in the series.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    /// Refuse it, which is what a series gets unless it asks otherwise.
    Block,
    /// Keep what is already there.
    First,
    /// Take the new value.
    Last,
    /// Keep whichever of the two is smaller.
    Min,
    /// Keep whichever is larger.
    Max,
    /// Keep the two added together.
    Sum,
}

impl Policy {
    /// The word `TS.INFO` reports this as, which is also the word that sets it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::First => "first",
            Self::Last => "last",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }

    /// The policy `word` names, whatever case it is written in.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Self> {
        [
            Self::Block,
            Self::First,
            Self::Last,
            Self::Min,
            Self::Max,
            Self::Sum,
        ]
        .into_iter()
        .find(|policy| word.eq_ignore_ascii_case(policy.name().as_bytes()))
    }
}

/// Why a sample was turned away.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refused {
    /// It is far enough behind the newest sample that retention has already
    /// gone past it, so storing it would only mean dropping it again.
    Old,
    /// There is already a sample on that timestamp and the policy in force will
    /// not have it replaced.
    Duplicate,
}

/// A standing instruction to fold one series into another.
///
/// The rule lives on the series being read from and names the one being written
/// to. Both directions are stored, the source keeping a rule each and the
/// destination keeping the name of the one series allowed to feed it, because
/// `TS.INFO` reports both and neither side can work the other out on its own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    /// The key the folded readings are written to.
    pub dest: Vec<u8>,
    /// How wide a bucket is, in whatever unit the timestamps are.
    pub delta: i64,
    /// The reduction each bucket is folded through.
    pub agg: Agg,
    /// The timestamp the bucket edges line up against.
    pub align: i64,
    /// Which bucket the rule is filling and has not written down yet, or `None`
    /// when it has not been given a sample since it was made.
    ///
    /// A rule does not go back over what the source held before it existed, so
    /// this is what tells the two apart: the buckets before this one were
    /// written as they closed and this one is the only one still moving.
    pub open: Option<i64>,
    /// The oldest timestamp counted into the open bucket, which is the first
    /// sample the rule was given after it started that bucket.
    pub start: i64,
}

/// A run of samples with the settings the commands hang off it.
#[derive(Clone, Debug)]
pub struct Series {
    /// How far back samples are kept, in whatever unit the timestamps are, or
    /// zero to keep all of them.
    retention: i64,
    /// How much room each chunk gets.
    chunk_bytes: usize,
    /// How the chunks store what they hold.
    encoding: Encoding,
    /// What to do about a repeated timestamp, or `None` to fall back on
    /// refusing it.
    policy: Option<Policy>,
    /// The name and value pairs this series can be found by.
    labels: Vec<(Vec<u8>, Vec<u8>)>,
    /// How close in time a sample has to be to the last one to be a candidate
    /// for being dropped as uninteresting.
    ignore_time: i64,
    /// How close in value.
    ignore_value: f64,
    /// The one series allowed to write folded readings here, if this one is the
    /// destination of a rule.
    source: Option<Vec<u8>>,
    /// The series this one is folded into, none or several.
    rules: Vec<Rule>,
    /// The samples. There is always at least one chunk, empty or not.
    chunks: Vec<Chunk>,
    /// How many samples are in those chunks.
    total: usize,
}

impl Default for Series {
    fn default() -> Self {
        Self::new()
    }
}

impl Series {
    /// A series with nothing in it and every setting left at its default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            retention: 0,
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            encoding: Encoding::Compressed,
            policy: None,
            labels: Vec::new(),
            ignore_time: 0,
            ignore_value: 0.0,
            source: None,
            rules: Vec::new(),
            chunks: vec![Chunk::new(Encoding::Compressed, DEFAULT_CHUNK_BYTES)],
            total: 0,
        }
    }

    /// How far back samples are kept.
    #[must_use]
    pub fn retention(&self) -> i64 {
        self.retention
    }

    /// Sets how far back samples are kept. Zero keeps all of them.
    pub fn set_retention(&mut self, retention: i64) {
        self.retention = retention;
        self.trim();
    }

    /// How much room each chunk gets.
    #[must_use]
    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    /// Sets how much room each chunk gets, from the next chunk on. What is
    /// already stored is left where it is, the same way the module leaves it.
    pub fn set_chunk_bytes(&mut self, bytes: usize) {
        self.chunk_bytes = bytes;
        if self.total == 0 {
            self.chunks = vec![Chunk::new(self.encoding, bytes)];
        }
    }

    /// How the chunks store what they hold.
    #[must_use]
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Sets the encoding, which only a series that has not been written to yet
    /// should be asked to do.
    pub fn set_encoding(&mut self, encoding: Encoding) {
        self.encoding = encoding;
        if self.total == 0 {
            self.chunks = vec![Chunk::new(encoding, self.chunk_bytes)];
        }
    }

    /// What happens to a repeated timestamp here, if this series has said.
    #[must_use]
    pub fn policy(&self) -> Option<Policy> {
        self.policy
    }

    /// Sets what happens to a repeated timestamp.
    pub fn set_policy(&mut self, policy: Policy) {
        self.policy = Some(policy);
    }

    /// The name and value pairs this series can be found by.
    #[must_use]
    pub fn labels(&self) -> &[(Vec<u8>, Vec<u8>)] {
        &self.labels
    }

    /// Replaces the labels with `labels`.
    pub fn set_labels(&mut self, labels: Vec<(Vec<u8>, Vec<u8>)>) {
        self.labels = labels;
    }

    /// The value of the label named `name`.
    #[must_use]
    pub fn label(&self, name: &[u8]) -> Option<&[u8]> {
        self.labels
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_slice())
    }

    /// How close in time and value a sample has to be to the last one to be
    /// dropped as uninteresting.
    #[must_use]
    pub fn ignore(&self) -> (i64, f64) {
        (self.ignore_time, self.ignore_value)
    }

    /// Sets that window.
    pub fn set_ignore(&mut self, time: i64, value: f64) {
        self.ignore_time = time;
        self.ignore_value = value;
    }

    /// The series that folds readings into this one, if there is one.
    #[must_use]
    pub fn source(&self) -> Option<&[u8]> {
        self.source.as_deref()
    }

    /// Names the series that folds readings into this one.
    pub fn set_source(&mut self, key: Option<Vec<u8>>) {
        self.source = key;
    }

    /// The series this one is folded into.
    #[must_use]
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Adds a rule, which the caller has already checked is allowed.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// The rule writing to `dest`, to move its open bucket along.
    pub fn rule_mut(&mut self, dest: &[u8]) -> Option<&mut Rule> {
        self.rules.iter_mut().find(|rule| rule.dest == dest)
    }

    /// Drops the rule that writes to `dest`, and says whether there was one.
    pub fn drop_rule(&mut self, dest: &[u8]) -> bool {
        let before = self.rules.len();
        self.rules.retain(|rule| rule.dest != dest);
        self.rules.len() != before
    }

    /// Keeps only the rules writing to one of `dest`, which is how a rule whose
    /// destination has been deleted stops being reported.
    pub fn keep_rules(&mut self, dest: &[Vec<u8>]) {
        self.rules.retain(|rule| dest.contains(&rule.dest));
    }

    /// How many samples are stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total
    }

    /// Whether nothing is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// How many chunks the samples are spread over.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The oldest timestamp stored.
    #[must_use]
    pub fn first(&self) -> Option<i64> {
        self.chunks.iter().find_map(Chunk::first)
    }

    /// The newest one.
    #[must_use]
    pub fn last(&self) -> Option<i64> {
        self.chunks.iter().rev().find_map(Chunk::last)
    }

    /// The newest sample, value and all.
    #[must_use]
    pub fn last_sample(&self) -> Option<Sample> {
        self.chunks
            .iter()
            .rev()
            .find(|chunk| !chunk.is_empty())
            .and_then(|chunk| chunk.samples().last())
    }

    /// Roughly what the samples take up.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let chunks: usize = self.chunks.iter().map(Chunk::memory_bytes).sum();
        let labels: usize = self
            .labels
            .iter()
            .map(|(key, value)| key.len() + value.len() + 2 * size_of::<Vec<u8>>())
            .sum();
        let rules: usize = self
            .rules
            .iter()
            .map(|rule| rule.dest.len() + size_of::<Rule>())
            .sum();
        let source = self.source.as_ref().map_or(0, Vec::len);
        size_of::<Self>()
            + self.chunks.len() * size_of::<Chunk>()
            + chunks
            + labels
            + rules
            + source
    }

    /// Stores `sample`, and answers the timestamp it went in on.
    ///
    /// The answer is not always `sample.at`. A sample close enough to the last
    /// one to be uninteresting is dropped and the last timestamp comes back
    /// instead, which is what the module does and what lets a client tell the
    /// two cases apart.
    ///
    /// `over` is a policy for this one sample, which beats the series setting.
    pub fn add(&mut self, sample: Sample, over: Option<Policy>) -> Result<i64, Refused> {
        let policy = over.or(self.policy).unwrap_or(Policy::Block);
        if let Some(last) = self.last_sample() {
            if self.retention != 0 && sample.at < last.at && self.retention < last.at - sample.at {
                return Err(Refused::Old);
            }
            if policy == Policy::Last && self.uninteresting(sample, last) {
                return Ok(last.at);
            }
            if sample.at <= last.at {
                return self.upsert(sample, policy);
            }
        }
        self.append(sample);
        Ok(sample.at)
    }

    /// Drops every sample from `from` to `to`, both ends included, and answers
    /// how many that was.
    pub fn delete(&mut self, from: i64, to: i64) -> usize {
        if from > to {
            return 0;
        }
        let mut gone = 0;
        let mut at = 0;
        while at < self.chunks.len() {
            let (Some(first), Some(last)) = (self.chunks[at].first(), self.chunks[at].last())
            else {
                at += 1;
                continue;
            };
            if last < from || first > to {
                at += 1;
                continue;
            }
            if first >= from && last <= to {
                gone += self.chunks[at].len();
                if self.chunks.len() > 1 {
                    self.chunks.remove(at);
                } else {
                    self.chunks[at] = Chunk::new(self.encoding, self.chunk_bytes);
                    at += 1;
                }
                continue;
            }
            let kept: Vec<Sample> = self.chunks[at]
                .samples()
                .filter(|s| s.at < from || s.at > to)
                .collect();
            gone += self.chunks[at].len() - kept.len();
            self.rewrite(at, &kept);
            at += 1;
        }
        self.total -= gone;
        gone
    }

    /// Every sample from `from` to `to`, both ends included, in order.
    pub fn range(&self, from: i64, to: i64) -> impl Iterator<Item = Sample> + '_ {
        self.chunks
            .iter()
            .filter(move |chunk| match (chunk.first(), chunk.last()) {
                (Some(first), Some(last)) => first <= to && last >= from,
                _ => false,
            })
            .flat_map(Chunk::samples)
            .filter(move |s| s.at >= from && s.at <= to)
    }

    /// Whether `sample` is close enough to `last` in both time and value that
    /// the ignore window says not to bother storing it.
    ///
    /// A reading that is not a number is never uninteresting, because there is
    /// no sense in which it is close to anything.
    fn uninteresting(&self, sample: Sample, last: Sample) -> bool {
        !sample.value.is_nan()
            && !last.value.is_nan()
            && sample.at >= last.at
            && sample.at - last.at <= self.ignore_time
            && (sample.value - last.value).abs() <= self.ignore_value
    }

    /// Puts a sample somewhere other than the end.
    fn upsert(&mut self, sample: Sample, policy: Policy) -> Result<i64, Refused> {
        let at = self.chunk_for(sample.at);
        let mut samples: Vec<Sample> = self.chunks[at].samples().collect();
        match samples.binary_search_by(|s| s.at.cmp(&sample.at)) {
            Ok(hit) => {
                let old = samples[hit].value;
                let new = sample.value;
                // Adding a reading that is not a number to one that is, or
                // asking which of the two is smaller, has no answer worth
                // storing, so those three refuse rather than guess. Two
                // readings that are both not numbers are not a mismatch and go
                // through.
                let ranked = matches!(policy, Policy::Min | Policy::Max | Policy::Sum);
                if policy == Policy::Block || (ranked && old.is_nan() != new.is_nan()) {
                    return Err(Refused::Duplicate);
                }
                samples[hit].value = if old.is_nan() || new.is_nan() {
                    // Whichever of the two is a number wins whatever the policy
                    // says, and if neither is one the stored reading stays as
                    // it was.
                    if new.is_nan() { old } else { new }
                } else {
                    match policy {
                        Policy::First => old,
                        Policy::Min => old.min(new),
                        Policy::Max => old.max(new),
                        Policy::Sum => old + new,
                        Policy::Block | Policy::Last => new,
                    }
                };
            }
            Err(spot) => {
                samples.insert(spot, sample);
                self.total += 1;
            }
        }
        self.rewrite(at, &samples);
        Ok(sample.at)
    }

    /// Puts a sample after everything already stored.
    fn append(&mut self, sample: Sample) {
        let last = self.chunks.last_mut().expect("a series always has a chunk");
        if last.has_room() {
            last.append(sample);
        } else {
            let mut chunk = Chunk::new(self.encoding, self.chunk_bytes);
            chunk.append(sample);
            self.chunks.push(chunk);
        }
        self.total += 1;
        self.trim();
    }

    /// Which chunk a timestamp belongs in, which is the last one that starts at
    /// or before it.
    fn chunk_for(&self, at: i64) -> usize {
        self.chunks
            .iter()
            .rposition(|chunk| chunk.first().is_some_and(|first| first <= at))
            .unwrap_or(0)
    }

    /// Writes `samples` over the chunk at `at`, spilling into fresh chunks
    /// behind it if they no longer fit in one.
    fn rewrite(&mut self, at: usize, samples: &[Sample]) {
        let mut left = self.chunks[at].rewrite(samples);
        let mut spot = at;
        while !left.is_empty() {
            spot += 1;
            let mut chunk = Chunk::new(self.encoding, self.chunk_bytes);
            left = chunk.rewrite(left);
            self.chunks.insert(spot, chunk);
        }
        if self.chunks[at].is_empty() && self.chunks.len() > 1 {
            self.chunks.remove(at);
        }
    }

    /// Drops whatever retention has gone past.
    fn trim(&mut self) {
        if self.retention == 0 {
            return;
        }
        let Some(last) = self.last() else {
            return;
        };
        let cut = last - self.retention;
        while self.chunks.len() > 1 && self.chunks[0].last().is_some_and(|end| end < cut) {
            self.total -= self.chunks[0].len();
            self.chunks.remove(0);
        }
        if self.chunks[0].first().is_some_and(|start| start < cut) {
            let kept: Vec<Sample> = self.chunks[0].samples().filter(|s| s.at >= cut).collect();
            self.total -= self.chunks[0].len() - kept.len();
            self.rewrite(0, &kept);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(series: &Series) -> Vec<(i64, f64)> {
        series
            .range(i64::MIN, i64::MAX)
            .map(|s| (s.at, s.value))
            .collect()
    }

    #[test]
    fn a_new_series_is_empty_and_still_has_a_chunk() {
        let series = Series::new();
        assert!(series.is_empty());
        assert_eq!(series.chunk_count(), 1);
        assert_eq!(series.first(), None);
        assert_eq!(series.last(), None);
    }

    #[test]
    fn samples_come_back_in_order_however_they_went_in() {
        let mut series = Series::new();
        for point in [300, 100, 400, 200] {
            series.add(Sample::new(point, point as f64), None).unwrap();
        }
        assert_eq!(
            at(&series),
            [(100, 100.0), (200, 200.0), (300, 300.0), (400, 400.0)]
        );
        assert_eq!(series.len(), 4);
        assert_eq!(series.first(), Some(100));
        assert_eq!(series.last(), Some(400));
    }

    #[test]
    fn every_duplicate_policy_does_what_it_says() {
        let cases = [
            (Policy::First, Some(5.0)),
            (Policy::Last, Some(3.0)),
            (Policy::Min, Some(3.0)),
            (Policy::Max, Some(5.0)),
            (Policy::Sum, Some(8.0)),
            (Policy::Block, None),
        ];
        for (policy, want) in cases {
            let mut series = Series::new();
            series.set_policy(policy);
            series.add(Sample::new(100, 5.0), None).unwrap();
            let out = series.add(Sample::new(100, 3.0), None);
            match want {
                Some(value) => {
                    assert_eq!(out, Ok(100), "{policy:?}");
                    assert_eq!(at(&series), [(100, value)], "{policy:?}");
                }
                None => assert_eq!(out, Err(Refused::Duplicate), "{policy:?}"),
            }
            assert_eq!(series.len(), 1, "{policy:?}");
        }
    }

    #[test]
    fn a_reading_that_is_not_a_number_lands_where_the_module_lands_it() {
        // A number arriving on top of one that is not a number, and the other
        // way round. The three that rank or add refuse the pair, and the rest
        // keep whichever of the two is a number.
        let cases = [
            (Policy::First, 5.0, f64::NAN, Some(5.0)),
            (Policy::Last, 5.0, f64::NAN, Some(5.0)),
            (Policy::First, f64::NAN, 5.0, Some(5.0)),
            (Policy::Last, f64::NAN, 5.0, Some(5.0)),
            (Policy::Min, 5.0, f64::NAN, None),
            (Policy::Max, f64::NAN, 5.0, None),
            (Policy::Sum, 5.0, f64::NAN, None),
        ];
        for (policy, first, second, want) in cases {
            let mut series = Series::new();
            series.set_policy(policy);
            series.add(Sample::new(100, first), None).unwrap();
            let out = series.add(Sample::new(100, second), None);
            match want {
                Some(value) => {
                    assert_eq!(out, Ok(100), "{policy:?}");
                    assert_eq!(at(&series), [(100, value)], "{policy:?}");
                }
                None => assert_eq!(out, Err(Refused::Duplicate), "{policy:?}"),
            }
        }

        // Two readings that are both not numbers are not a mismatch, so even
        // the ranking policies take the pair and store one of them.
        for policy in [Policy::Min, Policy::Max, Policy::Sum, Policy::Last] {
            let mut series = Series::new();
            series.set_policy(policy);
            series.add(Sample::new(100, f64::NAN), None).unwrap();
            assert_eq!(series.add(Sample::new(100, f64::NAN), None), Ok(100));
            assert!(series.last_sample().unwrap().value.is_nan(), "{policy:?}");
        }
    }

    #[test]
    fn a_policy_on_the_command_beats_the_one_on_the_series() {
        let mut series = Series::new();
        series.add(Sample::new(100, 5.0), None).unwrap();
        assert_eq!(
            series.add(Sample::new(100, 7.0), Some(Policy::Max)),
            Ok(100)
        );
        assert_eq!(at(&series), [(100, 7.0)]);
        assert_eq!(
            series.add(Sample::new(100, 1.0), Some(Policy::Min)),
            Ok(100)
        );
        assert_eq!(at(&series), [(100, 1.0)]);
    }

    #[test]
    fn retention_keeps_the_window_and_nothing_older() {
        let mut series = Series::new();
        series.set_retention(5000);
        for i in 1..=20 {
            series.add(Sample::new(i * 1000, 1.0), None).unwrap();
        }
        assert_eq!(series.len(), 6);
        assert_eq!(series.first(), Some(15_000));
        assert_eq!(series.last(), Some(20_000));
        assert_eq!(
            series.add(Sample::new(1000, 1.0), None),
            Err(Refused::Old),
            "a sample retention has already passed by is turned away"
        );
        assert_eq!(series.add(Sample::new(15_500, 1.0), None), Ok(15_500));
    }

    #[test]
    fn the_ignore_window_only_bites_under_the_last_policy() {
        let mut series = Series::new();
        series.set_ignore(100, 0.5);
        series.add(Sample::new(1000, 10.0), None).unwrap();
        assert_eq!(
            series.add(Sample::new(1050, 10.2), None),
            Ok(1050),
            "the window does nothing while duplicates are blocked"
        );

        let mut series = Series::new();
        series.set_policy(Policy::Last);
        series.set_ignore(100, 0.5);
        series.add(Sample::new(1000, 10.0), None).unwrap();
        assert_eq!(series.add(Sample::new(1050, 10.2), None), Ok(1000));
        assert_eq!(at(&series), [(1000, 10.0)]);
        assert_eq!(
            series.add(Sample::new(1100, 10.9), None),
            Ok(1100),
            "far enough in value to be worth storing"
        );
        assert_eq!(
            series.add(Sample::new(1300, 11.0), None),
            Ok(1300),
            "far enough in time to be worth storing"
        );
    }

    #[test]
    fn deleting_takes_out_a_span_and_leaves_the_rest() {
        let mut series = Series::new();
        for point in [100, 200, 300, 400] {
            series
                .add(Sample::new(point, point as f64 / 100.0), None)
                .unwrap();
        }
        assert_eq!(series.delete(150, 350), 2);
        assert_eq!(at(&series), [(100, 1.0), (400, 4.0)]);
        assert_eq!(series.delete(i64::MIN, i64::MAX), 2);
        assert!(series.is_empty());
        assert_eq!(
            series.chunk_count(),
            1,
            "a series that has been emptied still has its chunk"
        );
        assert_eq!(series.first(), None);
    }

    #[test]
    fn a_series_spills_into_more_chunks_as_it_grows() {
        let mut series = Series::new();
        series.set_chunk_bytes(48);
        for i in 0..200 {
            series.add(Sample::new(i * 1000, i as f64), None).unwrap();
        }
        assert!(series.chunk_count() > 1, "{}", series.chunk_count());
        assert_eq!(series.len(), 200);
        let back: Vec<i64> = series.range(i64::MIN, i64::MAX).map(|s| s.at).collect();
        assert_eq!(back, (0..200).map(|i| i * 1000).collect::<Vec<_>>());
    }

    #[test]
    fn a_backfill_into_a_full_chunk_still_reads_back_in_order() {
        let mut series = Series::new();
        series.set_chunk_bytes(48);
        for i in 0..100 {
            series.add(Sample::new(i * 10, i as f64), None).unwrap();
        }
        for i in 0..100 {
            let point = i * 10 + 5;
            assert_eq!(series.add(Sample::new(point, 1.0), None), Ok(point));
        }
        assert_eq!(series.len(), 200);
        let back: Vec<i64> = series.range(i64::MIN, i64::MAX).map(|s| s.at).collect();
        let want: Vec<i64> = (0..200).map(|i| i * 5).collect();
        assert_eq!(back, want);
    }

    #[test]
    fn a_range_only_answers_what_was_asked_for() {
        let mut series = Series::new();
        series.set_chunk_bytes(48);
        for i in 0..100 {
            series.add(Sample::new(i, i as f64), None).unwrap();
        }
        let back: Vec<i64> = series.range(30, 40).map(|s| s.at).collect();
        assert_eq!(back, (30..=40).collect::<Vec<_>>());
        assert_eq!(series.range(1000, 2000).count(), 0);
    }

    #[test]
    fn policies_are_named_the_way_the_commands_name_them() {
        assert_eq!(Policy::parse(b"BLOCK"), Some(Policy::Block));
        assert_eq!(Policy::parse(b"sUm"), Some(Policy::Sum));
        assert_eq!(Policy::parse(b"nope"), None);
        assert_eq!(Policy::Min.name(), "min");
    }
}
