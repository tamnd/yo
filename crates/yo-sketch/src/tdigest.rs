//! The t digest, which answers what the quantiles of a stream were.
//!
//! A digest is a list of centroids kept in order of their mean, where a centroid
//! is a mean and the number of samples that went into it. A sample arriving is
//! appended to a buffer at the back of the same array, and when the array fills
//! the whole thing is sorted and swept once from the small end, merging each
//! neighbour into the one being built while the merged weight still fits under a
//! bound that depends on where in the distribution the pair sits. The bound is
//! tight at the two ends and slack in the middle, so the tails end up described
//! by centroids holding one or two samples each and the middle by centroids
//! holding thousands. That is what makes the extreme quantiles accurate to parts
//! per million on a structure whose size does not depend on how much went into
//! it.
//!
//! ```
//! use yo_sketch::tdigest::TDigest;
//!
//! let mut t = TDigest::new(100).expect("a digest that small always fits");
//! for i in 1..=1000 {
//!     t.add(f64::from(i), 1).expect("a thousand samples cannot overflow");
//! }
//! let mut out = [0.0; 2];
//! t.quantiles(&[0.5, 0.99], &mut out);
//! assert!((out[0] - 500.0).abs() < 5.0);
//! assert!((out[1] - 990.0).abs() < 5.0);
//! assert_eq!(t.min(), 1.0);
//! assert_eq!(t.max(), 1000.0);
//! ```
//!
//! # Why this one and not a better one
//!
//! This is a port of the C digest RedisBloom carries, down to the sort and the
//! order the floating point is done in, and there are newer designs that beat it
//! on both accuracy and speed. The reason to copy it anyway is that a quantile
//! is a number a client compares against a threshold. Two servers holding the
//! same samples and answering different p99s is a page for whoever owns the
//! alert, and there is no way to look at the two numbers and say which server is
//! wrong, because both are estimates and neither is the true quantile. So the
//! arithmetic is the arithmetic in `deps/t-digest-c`, including the parts that
//! look like accidents.
//!
//! The sort is one of those parts. It is an introsort with three way
//! partitioning, a median of three pivot and a heapsort fallback, and it is not
//! stable, so a run of equal means comes out in an order that depends on the
//! partitioning rather than on how the samples arrived. That is invisible while
//! every sample weighs one, which is the case for anything added through
//! `TDIGEST.ADD`, and it becomes visible after a merge, where centroids of equal
//! mean and unequal weight can be sorted against each other. The sort is
//! therefore copied rather than replaced by a stable one.
//!
//! # Where this differs from the module
//!
//! The capacity is capped, which is D-52. Everything else in here is the
//! reference's, and the wire layer says which of its answers are not.

/// The largest compression a digest is allowed here.
///
/// The reference stops at the point where the capacity would no longer fit an
/// `int`, which is a digest of about six billion bytes, and it only finds out by
/// asking `calloc` for them. This stops earlier, at [`MAX_BYTES`], for the
/// reason D-52 gives.
pub const MAX_COMPRESSION: i64 = (i32::MAX as i64 - 10) / 6;

/// How many bytes one digest's two node arrays are allowed, which is a gibibyte.
pub const MAX_BYTES: u64 = 1 << 30;

/// The bytes a node costs: a mean and a weight, eight each.
const NODE_BYTES: u64 = 16;

/// The bytes the header costs on the reference, which `TDIGEST.INFO` reports.
const HEADER_BYTES: u64 = 80;

/// Below this many nodes a range is finished with insertion sort.
const INSORT_THRESHOLD: i64 = 16;

/// What went wrong, which is only ever that a weight stopped fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    /// A weight or a total weight no longer fits, so nothing was changed.
    Weight,
    /// The value offered was not a finite number.
    NotFinite,
}

/// A t digest.
///
/// The node arrays are one allocation each and never grow. `merged_nodes` at the
/// front are the digest proper, in order of mean, and `unmerged_nodes` behind
/// them are the buffer of what has arrived since the last sweep.
#[derive(Debug)]
pub struct TDigest {
    compression: f64,
    min: f64,
    max: f64,
    cap: usize,
    merged_nodes: usize,
    unmerged_nodes: usize,
    total_compressions: i64,
    merged_weight: i64,
    unmerged_weight: i64,
    means: Box<[f64]>,
    weights: Box<[i64]>,
}

impl TDigest {
    /// A digest at this compression, or `None` if it would not fit.
    ///
    /// The capacity is six times the compression plus ten, which is the
    /// reference's formula and is visible through `TDIGEST.INFO`.
    #[must_use]
    pub fn new(compression: i64) -> Option<Self> {
        if compression <= 0 || compression > MAX_COMPRESSION {
            return None;
        }
        let cap = 6 * compression + 10;
        #[allow(clippy::cast_sign_loss)]
        let cap = cap as u64;
        if cap * NODE_BYTES > MAX_BYTES {
            return None;
        }
        let cap = usize::try_from(cap).ok()?;
        #[allow(clippy::cast_precision_loss)]
        let compression = compression as f64;
        Some(Self {
            compression,
            min: f64::MAX,
            max: -f64::MAX,
            cap,
            merged_nodes: 0,
            unmerged_nodes: 0,
            total_compressions: 0,
            merged_weight: 0,
            unmerged_weight: 0,
            means: vec![0.0; cap].into_boxed_slice(),
            weights: vec![0; cap].into_boxed_slice(),
        })
    }

    /// The compression the digest was made with.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn compression(&self) -> i64 {
        self.compression as i64
    }

    /// How many nodes the arrays hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// How many nodes are merged, which is the digest proper.
    #[must_use]
    pub fn merged_nodes(&self) -> usize {
        self.merged_nodes
    }

    /// How many nodes are waiting in the buffer.
    #[must_use]
    pub fn unmerged_nodes(&self) -> usize {
        self.unmerged_nodes
    }

    /// The weight in the merged nodes.
    #[must_use]
    pub fn merged_weight(&self) -> i64 {
        self.merged_weight
    }

    /// The weight in the buffer.
    #[must_use]
    pub fn unmerged_weight(&self) -> i64 {
        self.unmerged_weight
    }

    /// How many sweeps the digest has been through.
    #[must_use]
    pub fn compressions(&self) -> i64 {
        self.total_compressions
    }

    /// The weight of everything in the digest.
    #[must_use]
    pub fn size(&self) -> i64 {
        self.merged_weight + self.unmerged_weight
    }

    /// The smallest sample the digest has seen, which is `f64::MAX` if none.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// The largest sample the digest has seen, which is `-f64::MAX` if none.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// What the digest costs here, which is the struct and both node arrays.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>() + self.cap * (size_of::<f64>() + size_of::<i64>())
    }

    /// What `TDIGEST.INFO` says the digest costs.
    ///
    /// The reference's number rather than this one's, because it is a field in a
    /// reply a client reads and compares across servers. The two are within a
    /// few bytes of each other anyway: both are the node arrays plus a header,
    /// and the header is the same nine fields either way.
    #[must_use]
    pub fn reported_bytes(&self) -> u64 {
        HEADER_BYTES + self.cap as u64 * NODE_BYTES
    }

    /// Empty the digest without giving up its arrays.
    pub fn reset(&mut self) {
        self.min = f64::MAX;
        self.max = -self.min;
        self.merged_nodes = 0;
        self.merged_weight = 0;
        self.unmerged_nodes = 0;
        self.unmerged_weight = 0;
        self.total_compressions = 0;
    }

    /// Add a sample of this weight, sweeping first if the buffer is full.
    ///
    /// # Errors
    ///
    /// [`Overflow::NotFinite`] if the value is not finite, and
    /// [`Overflow::Weight`] if the weight would take a total past what a signed
    /// sixty four bit number or a double can carry, in which case nothing moved.
    pub fn add(&mut self, mean: f64, weight: i64) -> Result<(), Overflow> {
        if !mean.is_finite() {
            return Err(Overflow::NotFinite);
        }
        if self.merged_nodes + self.unmerged_nodes >= self.cap - 1 {
            self.compress()?;
        }
        let pos = self.merged_nodes + self.unmerged_nodes;
        if pos >= self.cap {
            return Err(Overflow::Weight);
        }
        let unmerged = self
            .unmerged_weight
            .checked_add(weight)
            .ok_or(Overflow::Weight)?;
        let total = unmerged
            .checked_add(self.merged_weight)
            .ok_or(Overflow::Weight)?;
        #[allow(clippy::cast_precision_loss)]
        check_overflow(unmerged as f64, total as f64)?;
        if mean < self.min {
            self.min = mean;
        }
        if mean > self.max {
            self.max = mean;
        }
        self.means[pos] = mean;
        self.weights[pos] = weight;
        self.unmerged_nodes += 1;
        self.unmerged_weight = unmerged;
        Ok(())
    }

    /// Sweep the buffer into the digest, merging neighbours where they fit.
    ///
    /// # Errors
    ///
    /// [`Overflow::Weight`] if the total weight has grown past what the scale
    /// function can be computed at, in which case nothing moved.
    #[allow(clippy::cast_precision_loss)]
    pub fn compress(&mut self) -> Result<(), Overflow> {
        if self.unmerged_nodes == 0 {
            return Ok(());
        }
        let n = self.merged_nodes + self.unmerged_nodes;
        sort(&mut self.means, &mut self.weights, 0, n as i64 - 1);
        let total_weight = self.merged_weight as f64 + self.unmerged_weight as f64;
        check_overflow(self.unmerged_weight as f64, total_weight)?;
        if total_weight <= 1.0 {
            // One sample or none, so there is nothing to merge and the buffer
            // becomes the digest as it stands.
            self.merged_nodes += self.unmerged_nodes;
            self.merged_weight = total_weight as i64;
            self.unmerged_nodes = 0;
            self.unmerged_weight = 0;
            self.total_compressions += 1;
            return Ok(());
        }
        let denom = 2.0 * std::f64::consts::PI * total_weight * total_weight.ln();
        if denom.is_infinite() {
            return Err(Overflow::Weight);
        }
        let normalizer = self.compression / denom;
        if normalizer.is_infinite() {
            return Err(Overflow::Weight);
        }
        let mut cur = 0usize;
        let mut weight_so_far = 0.0f64;
        for i in 1..n {
            let proposed = self.weights[cur] as f64 + self.weights[i] as f64;
            let z = proposed * normalizer;
            let q0 = weight_so_far / total_weight;
            let q2 = (weight_so_far + proposed) / total_weight;
            // The k scale in the form the reference tests it in: a pair may
            // merge only if the weight it would end up with is under the bound
            // at both ends of the span it would cover.
            if z <= q0 * (1.0 - q0) && z <= q2 * (1.0 - q2) {
                self.weights[cur] += self.weights[i];
                let delta = self.means[i] - self.means[cur];
                let weighted = (delta * self.weights[i] as f64) / self.weights[cur] as f64;
                self.means[cur] += weighted;
            } else {
                weight_so_far += self.weights[cur] as f64;
                cur += 1;
                self.weights[cur] = self.weights[i];
                self.means[cur] = self.means[i];
            }
            if cur != i {
                self.weights[i] = 0;
                self.means[i] = 0.0;
            }
        }
        self.merged_nodes = cur + 1;
        self.merged_weight = total_weight as i64;
        self.unmerged_nodes = 0;
        self.unmerged_weight = 0;
        self.total_compressions += 1;
        Ok(())
    }

    /// The merged centroids as pairs of mean and weight.
    ///
    /// Only meaningful straight after a [`TDigest::compress`], which is how a
    /// merge reads its source.
    #[must_use]
    pub fn centroids(&self) -> Vec<(f64, i64)> {
        let n = self.merged_nodes + self.unmerged_nodes;
        (0..n).map(|i| (self.means[i], self.weights[i])).collect()
    }

    /// The fraction of the samples that are at or below `val`.
    ///
    /// NaN when the digest is empty, which is what the reference answers.
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    pub fn cdf(&mut self, val: f64) -> f64 {
        let _ = self.compress();
        if self.merged_nodes == 0 {
            return f64::NAN;
        }
        if val < self.min {
            return 0.0;
        }
        if val > self.max {
            return 1.0;
        }
        if self.merged_nodes == 1 {
            let width = self.max - self.min;
            if val - self.min <= width {
                return 0.5;
            }
            return (val - self.min) / width;
        }
        let n = self.merged_nodes;
        let left_mean = self.means[0];
        let left_weight = self.weights[0] as f64;
        let merged = self.merged_weight as f64;
        if val < left_mean {
            let width = left_mean - self.min;
            if width > 0.0 {
                if val == self.min {
                    return 0.5 / merged;
                }
                return (1.0 + (val - self.min) / width * (left_weight / 2.0 - 1.0)) / merged;
            }
            return 0.0;
        }
        let right_mean = self.means[n - 1];
        let right_weight = self.weights[n - 1] as f64;
        if val > right_mean {
            let width = self.max - right_mean;
            if width > 0.0 {
                if val == self.max {
                    return 1.0 - 0.5 / merged;
                }
                let dq = (1.0 + (self.max - val) / width * (right_weight / 2.0 - 1.0)) / merged;
                return 1.0 - dq;
            }
            return 1.0;
        }
        // At least two centroids and the value sits between the two end means,
        // so either one or more centroids are exactly at it or a pair brackets
        // it.
        let mut weight_so_far = 0.0f64;
        let mut it = 0usize;
        while it < n - 1 {
            if self.means[it] == val {
                let mut dw = 0.0;
                while it < n && self.means[it] == val {
                    dw += self.weights[it] as f64;
                    it += 1;
                }
                return (weight_so_far + dw / 2.0) / merged;
            } else if self.means[it] <= val && val < self.means[it + 1] {
                let node_weight = self.weights[it] as f64;
                let node_weight_next = self.weights[it + 1] as f64;
                let node_mean = self.means[it];
                let node_mean_next = self.means[it + 1];
                if node_mean_next - node_mean > 0.0 {
                    // A centroid holding one sample holds it exactly at its
                    // mean, so its weight is kept out of the interpolation.
                    let mut left_excluded = 0.0;
                    let mut right_excluded = 0.0;
                    if node_weight == 1.0 {
                        if node_weight_next == 1.0 {
                            return (weight_so_far + 1.0) / merged;
                        }
                        left_excluded = 0.5;
                    } else if node_weight_next == 1.0 {
                        right_excluded = 0.5;
                    }
                    let dw = (node_weight + node_weight_next) / 2.0;
                    let dw_no_singleton = dw - left_excluded - right_excluded;
                    let base = weight_so_far + node_weight / 2.0 + left_excluded;
                    return (base
                        + dw_no_singleton * (val - node_mean) / (node_mean_next - node_mean))
                        / merged;
                }
                let dw = (node_weight + node_weight_next) / 2.0;
                return (weight_so_far + dw) / merged;
            }
            weight_so_far += self.weights[it] as f64;
            it += 1;
        }
        1.0 - 0.5 / merged
    }

    /// The values at these quantiles, written into `out`.
    ///
    /// The walk over the centroids is carried from one quantile to the next, so
    /// a run of quantiles that does not decrease costs one pass over the digest
    /// between them all. A quantile that goes backwards therefore reads from
    /// wherever the walk had got to, which is the reference's behaviour and is
    /// why its wire layer splits the client's list into runs first.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `quantiles`.
    #[allow(clippy::cast_precision_loss)]
    pub fn quantiles(&mut self, quantiles: &[f64], out: &mut [f64]) {
        let _ = self.compress();
        let n = self.merged_nodes;
        if n == 0 {
            out[..quantiles.len()].fill(f64::NAN);
            return;
        }
        if n == 1 {
            for (slot, &q) in out.iter_mut().zip(quantiles) {
                // Again the reference's test rather than a range, so a NaN
                // quantile answers the one mean the digest has.
                *slot = if q < 0.0 || q > 1.0 {
                    f64::NAN
                } else {
                    self.means[0]
                };
            }
            return;
        }
        let left_weight = self.weights[0] as f64;
        let mut weight_so_far = left_weight / 2.0;
        let mut node_pos = 0usize;
        for (slot, &q) in out.iter_mut().zip(quantiles) {
            let index = q * self.merged_weight as f64;
            *slot = self.walk_to_index(index, left_weight, n, &mut weight_so_far, &mut node_pos);
        }
    }

    /// The value at one quantile, which is [`TDigest::quantiles`] of one.
    #[allow(clippy::cast_precision_loss)]
    pub fn quantile(&mut self, q: f64) -> f64 {
        let _ = self.compress();
        if q < 0.0 || q > 1.0 || self.merged_nodes == 0 {
            return f64::NAN;
        }
        if self.merged_nodes == 1 {
            return self.means[0];
        }
        let index = q * self.merged_weight as f64;
        if index < 1.0 {
            return self.min;
        }
        let n = self.merged_nodes;
        let left_weight = self.weights[0] as f64;
        let mut weight_so_far = left_weight / 2.0;
        let mut node_pos = 0usize;
        self.walk_to_index(index, left_weight, n, &mut weight_so_far, &mut node_pos)
    }

    /// The sample at the given offset into the sorted stream, interpolated.
    #[allow(clippy::cast_precision_loss)]
    fn walk_to_index(
        &self,
        index: f64,
        left_weight: f64,
        total_centroids: usize,
        weight_so_far: &mut f64,
        node_pos: &mut usize,
    ) -> f64 {
        if left_weight > 1.0 && index < left_weight / 2.0 {
            // One sample sits exactly at min, so the first centroid's span is
            // interpolated against a weight one short.
            return self.min
                + (index - 1.0) / (left_weight / 2.0 - 1.0) * (self.means[0] - self.min);
        }
        if index > self.merged_weight as f64 - 1.0 {
            return self.max;
        }
        let right_weight = self.weights[total_centroids - 1] as f64;
        let right_mean = self.means[total_centroids - 1];
        if right_weight > 1.0 && self.merged_weight as f64 - index <= right_weight / 2.0 {
            return self.max
                - (self.merged_weight as f64 - index - 1.0) / (right_weight / 2.0 - 1.0)
                    * (self.max - right_mean);
        }
        while *node_pos < total_centroids - 1 {
            let i = *node_pos;
            let node_weight = self.weights[i] as f64;
            let node_weight_next = self.weights[i + 1] as f64;
            let node_mean = self.means[i];
            let node_mean_next = self.means[i + 1];
            let dw = (node_weight + node_weight_next) / 2.0;
            if *weight_so_far + dw > index {
                let mut left_unit = 0.0;
                if node_weight == 1.0 {
                    if index - *weight_so_far < 0.5 {
                        return node_mean;
                    }
                    left_unit = 0.5;
                }
                let mut right_unit = 0.0;
                if node_weight_next == 1.0 {
                    if *weight_so_far + dw - index <= 0.5 {
                        return node_mean_next;
                    }
                    right_unit = 0.5;
                }
                let z1 = index - *weight_so_far - left_unit;
                let z2 = *weight_so_far + dw - index - right_unit;
                return weighted_average(node_mean, z2, node_mean_next, z1);
            }
            *weight_so_far += dw;
            *node_pos += 1;
        }
        let z1 = index - self.merged_weight as f64 - right_weight / 2.0;
        let z2 = right_weight / 2.0 - z1;
        weighted_average(right_mean, z1, self.max, z2)
    }

    /// The mean of what is left after cutting both tails at these fractions.
    #[allow(clippy::cast_precision_loss)]
    pub fn trimmed_mean(&mut self, low: f64, high: f64) -> f64 {
        let _ = self.compress();
        // Written the way the reference writes it rather than as a range, which
        // is not the same test: a NaN cut fails both comparisons and carries on
        // into the arithmetic instead of being refused here.
        if self.merged_nodes == 0 || low < 0.0 || low > 1.0 || high < 0.0 || high > 1.0 {
            return f64::NAN;
        }
        if self.merged_nodes == 1 {
            return self.means[0];
        }
        let leftmost = (self.merged_weight as f64 * low).floor();
        let rightmost = (self.merged_weight as f64 * high).ceil();
        let mut count_done = 0.0f64;
        let mut trimmed_sum = 0.0f64;
        let mut trimmed_count = 0.0f64;
        for i in 0..self.merged_nodes {
            let n_weight = self.weights[i] as f64;
            let mut count_add = n_weight;
            count_add -= smaller(larger(0.0, leftmost - count_done), count_add);
            count_add = smaller(larger(0.0, rightmost - count_done), count_add);
            count_done += n_weight;
            trimmed_sum += self.means[i] * count_add;
            trimmed_count += count_add;
            if count_done >= rightmost {
                break;
            }
        }
        trimmed_sum / trimmed_count
    }
}

/// The mean of two values by weight, held between the two of them.
fn weighted_average(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    if x1 <= x2 {
        weighted_average_sorted(x1, w1, x2, w2)
    } else {
        weighted_average_sorted(x2, w2, x1, w1)
    }
}

fn weighted_average_sorted(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    let x = (x1 * w1 + x2 * w2) / (w1 + w2);
    larger(x1, smaller(x, x2))
}

/// The larger of two, the way C's ternary answers it.
///
/// `f64::max` is not this function: it answers the number when one side is NaN,
/// where the reference's macro answers whichever side the comparison sends it
/// to, which for a NaN is always the second. Nothing on the wire can put a NaN
/// in here, because the module refuses one at every argument it parses, so this
/// is about keeping the arithmetic the same shape rather than about a case a
/// client can reach.
fn larger(x: f64, y: f64) -> f64 {
    if x > y { x } else { y }
}

/// The smaller of two, the way C's ternary answers it.
fn smaller(x: f64, y: f64) -> f64 {
    if x < y { x } else { y }
}

/// Whether a weight that big can still be worked with.
fn check_overflow(unmerged: f64, total: f64) -> Result<(), Overflow> {
    if unmerged.is_infinite() || total.is_infinite() {
        return Err(Overflow::Weight);
    }
    let denom = 2.0 * std::f64::consts::PI * total * total.ln();
    if denom.is_infinite() {
        return Err(Overflow::Weight);
    }
    Ok(())
}

/// Sort the two arrays together, keyed on the means.
///
/// This is the reference's introsort and not the standard library's sort, for
/// the reason the module doc gives: which of two equal means comes first decides
/// which weights end up merged together, and the reference's answer to that is
/// whatever its partitioning does.
fn sort(means: &mut [f64], weights: &mut [i64], lo: i64, hi: i64) {
    if lo >= hi {
        return;
    }
    let mut depth_limit = 0;
    let mut t = hi - lo + 1;
    while t > 1 {
        depth_limit += 2;
        t >>= 1;
    }
    introsort(means, weights, lo, hi, depth_limit);
}

fn introsort(means: &mut [f64], weights: &mut [i64], lo: i64, hi: i64, depth_limit: i32) {
    let (mut lo, mut hi, mut depth_limit) = (lo, hi, depth_limit);
    while hi - lo > INSORT_THRESHOLD {
        if depth_limit == 0 {
            heap_sort(means, weights, lo, hi);
            return;
        }
        depth_limit -= 1;
        let mid = lo + (hi - lo) / 2;
        let pivot = median3(means, weights, lo, mid, hi);
        // While the scan runs: [lo, lt) is under the pivot, [lt, i) is equal to
        // it and (gt, hi] is over it.
        let mut lt = lo;
        let mut i = lo;
        let mut gt = hi;
        while i <= gt {
            let v = means[i as usize];
            if v < pivot {
                swap(means, weights, i, lt);
                lt += 1;
                i += 1;
            } else if pivot < v {
                swap(means, weights, i, gt);
                gt -= 1;
            } else {
                i += 1;
            }
        }
        let left_size = lt - lo;
        let right_size = hi - gt;
        // Recurse into the smaller side and loop on the larger, which is what
        // holds the stack to the log of the length.
        if left_size < right_size {
            if left_size > 1 {
                introsort(means, weights, lo, lt - 1, depth_limit);
            }
            lo = gt + 1;
        } else {
            if right_size > 1 {
                introsort(means, weights, gt + 1, hi, depth_limit);
            }
            hi = lt - 1;
        }
    }
    insertion_sort(means, weights, lo, hi);
}

fn insertion_sort(means: &mut [f64], weights: &mut [i64], lo: i64, hi: i64) {
    for i in lo + 1..=hi {
        let m = means[i as usize];
        let w = weights[i as usize];
        let mut j = i - 1;
        while j >= lo && m < means[j as usize] {
            means[(j + 1) as usize] = means[j as usize];
            weights[(j + 1) as usize] = weights[j as usize];
            j -= 1;
        }
        means[(j + 1) as usize] = m;
        weights[(j + 1) as usize] = w;
    }
}

fn heap_sort(means: &mut [f64], weights: &mut [i64], lo: i64, hi: i64) {
    let n = hi - lo + 1;
    for i in (0..n / 2).rev() {
        sift_down(means, weights, lo, i, n);
    }
    for end in (1..n).rev() {
        swap(means, weights, lo, lo + end);
        sift_down(means, weights, lo, 0, end);
    }
}

fn sift_down(means: &mut [f64], weights: &mut [i64], lo: i64, i: i64, n: i64) {
    let mut i = i;
    while i < n / 2 {
        let mut child = 2 * i + 1;
        if child + 1 < n && means[(lo + child) as usize] < means[(lo + child + 1) as usize] {
            child += 1;
        }
        if means[(lo + child) as usize] <= means[(lo + i) as usize] {
            break;
        }
        swap(means, weights, lo + i, lo + child);
        i = child;
    }
}

/// Put the median of the three into `mid` and answer it.
fn median3(means: &mut [f64], weights: &mut [i64], lo: i64, mid: i64, hi: i64) -> f64 {
    if means[mid as usize] < means[lo as usize] {
        swap(means, weights, lo, mid);
    }
    if means[hi as usize] < means[lo as usize] {
        swap(means, weights, lo, hi);
    }
    if means[hi as usize] < means[mid as usize] {
        swap(means, weights, mid, hi);
    }
    means[mid as usize]
}

fn swap(means: &mut [f64], weights: &mut [i64], i: i64, j: i64) {
    means.swap(i as usize, j as usize);
    weights.swap(i as usize, j as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(compression: i64, values: &[f64]) -> TDigest {
        let mut t = TDigest::new(compression).expect("small enough");
        for &v in values {
            t.add(v, 1).expect("no overflow");
        }
        t
    }

    #[test]
    fn a_new_digest_is_empty_and_sized_by_its_compression() {
        let t = TDigest::new(100).expect("small enough");
        assert_eq!(t.capacity(), 610);
        assert_eq!(t.reported_bytes(), 9840);
        assert_eq!(t.size(), 0);
        assert_eq!(t.compression(), 100);
        assert_eq!(t.merged_nodes(), 0);
    }

    #[test]
    fn a_compression_outside_the_range_is_refused() {
        assert!(TDigest::new(0).is_none());
        assert!(TDigest::new(-1).is_none());
        assert!(TDigest::new(MAX_COMPRESSION).is_none());
        assert!(TDigest::new(i64::MAX).is_none());
        assert!(TDigest::new(1).is_some());
    }

    #[test]
    fn samples_stay_in_the_buffer_until_a_query_asks() {
        let mut t = filled(100, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.merged_nodes(), 0);
        assert_eq!(t.unmerged_nodes(), 4);
        assert_eq!(t.size(), 4);
        assert_eq!(t.quantile(0.5), 3.0);
        assert_eq!(t.merged_nodes(), 4);
        assert_eq!(t.unmerged_nodes(), 0);
        assert_eq!(t.compressions(), 1);
    }

    #[test]
    fn the_ends_of_the_digest_are_the_ends_of_the_stream() {
        let mut t = filled(100, &[5.0, -2.0, 9.5, 0.0]);
        assert_eq!(t.min(), -2.0);
        assert_eq!(t.max(), 9.5);
        assert_eq!(t.quantile(0.0), -2.0);
        assert_eq!(t.quantile(1.0), 9.5);
    }

    #[test]
    fn the_cdf_of_a_small_digest_is_the_reference_arithmetic() {
        let mut t = filled(100, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.cdf(0.0), 0.0);
        assert_eq!(t.cdf(1.0), 0.125);
        assert_eq!(t.cdf(2.0), 0.375);
        assert_eq!(t.cdf(2.5), 0.5);
        assert_eq!(t.cdf(9.0), 1.0);
    }

    #[test]
    fn an_empty_digest_answers_nan_everywhere() {
        let mut t = TDigest::new(100).expect("small enough");
        assert!(t.cdf(1.0).is_nan());
        assert!(t.quantile(0.5).is_nan());
        assert!(t.trimmed_mean(0.1, 0.9).is_nan());
    }

    #[test]
    fn a_run_of_quantiles_is_walked_once() {
        let mut t = filled(100, &[1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 3];
        t.quantiles(&[0.0, 0.1, 1.0], &mut out);
        assert_eq!(out, [1.0, 1.0, 4.0]);
    }

    #[test]
    fn a_trimmed_mean_drops_the_tails() {
        let mut t = filled(100, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.trimmed_mean(0.0, 1.0), 2.5);
        assert_eq!(t.trimmed_mean(0.1, 0.9), 2.5);
        let mut t = filled(100, &[1.0, 2.0, 3.0, 100.0]);
        assert_eq!(t.trimmed_mean(0.0, 0.75), 2.0);
    }

    #[test]
    fn a_thousand_samples_compress_to_a_bounded_number_of_centroids() {
        let mut t = TDigest::new(100).expect("small enough");
        for i in 0..100_000 {
            t.add(f64::from(i % 1000), 1).expect("no overflow");
        }
        t.compress().expect("no overflow");
        assert!(t.merged_nodes() < t.capacity());
        assert_eq!(t.size(), 100_000);
        assert_eq!(t.merged_weight(), 100_000);
        let q = t.quantile(0.5);
        assert!((q - 500.0).abs() < 20.0, "median came out at {q}");
    }

    #[test]
    fn the_sort_puts_the_means_in_order_whatever_they_arrive_in() {
        let mut means: Vec<f64> = (0..200).map(|i| f64::from((i * 37) % 200)).collect();
        let mut weights: Vec<i64> = (0..200).collect();
        sort(&mut means, &mut weights, 0, 199);
        assert!(means.windows(2).all(|w| w[0] <= w[1]));
        let mut sorted = weights.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..200).collect::<Vec<i64>>());
    }

    #[test]
    fn a_reset_digest_is_a_new_one() {
        let mut t = filled(100, &[1.0, 2.0, 3.0]);
        t.reset();
        assert_eq!(t.size(), 0);
        assert_eq!(t.merged_nodes(), 0);
        assert_eq!(t.compressions(), 0);
        assert_eq!(t.min(), f64::MAX);
        assert!(t.quantile(0.5).is_nan());
    }

    #[test]
    fn a_weight_that_does_not_fit_leaves_the_digest_alone() {
        let mut t = TDigest::new(100).expect("small enough");
        t.add(1.0, i64::MAX).expect("the first one fits");
        assert_eq!(t.add(2.0, i64::MAX), Err(Overflow::Weight));
        assert_eq!(t.size(), i64::MAX);
        assert_eq!(t.add(f64::NAN, 1), Err(Overflow::NotFinite));
        assert_eq!(t.add(f64::INFINITY, 1), Err(Overflow::NotFinite));
    }
}
