using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs found by OptunaSharp (SciSharp/OptunaSharp, the Optuna port built on NumSharp 0.80.0) while it moved
    ///     its hand-ported NumPy math onto NumSharp's <c>np.*</c> calls, pinned on journey5 at the owner's requests: a
    ///     result (2026-11-11, "Create on K:/source/NumSharp an openbug reproducing this") and two costs of
    ///     <c>np.sum(axis=-1)</c> (2026-11-12, "create OpenBugs at journey5 in K:/source/NumSharp for the slowness or memory
    ///     overallocation"). Each test asserts NumPy 2.4.2's result, or a cost NumPy's own same-process ratio bounds, and
    ///     fails until NumSharp gives it.
    /// </summary>
    /// <remarks>
    ///     <para>OptunaSharp tracks the result finding as its parity item QA-85
    ///     (<c>PrunerTests.NanPercentile_TiedSignedZeros_KeepNumpysZeroSign</c>, <c>[OpenBugs]</c> there too). Everything
    ///     else that round moved onto NumSharp matched NumPy bit for bit: the scalar <c>log1p</c> / <c>expm1</c> /
    ///     <c>logaddexp</c> models, <c>np.dot</c> of two vectors, <c>np.var</c>, <c>np.mean</c> / <c>np.std</c> over axis 0,
    ///     <c>np.linalg.norm</c> over an axis and <c>np.linspace</c>.</para>
    ///     <para>The cost findings come from its QA-84: the TPE sampler now sums its continuous log-density terms with
    ///     <c>np.sum(terms, axis: -1, @out: sums)</c>, which equals NumPy bit for bit at every row length from 1 to 300
    ///     but made OptunaSharp's 1,000-trial TPE studies 27-66 % slower. The memory half of the request found nothing
    ///     to pin: <c>np.sum(axis=-1)</c> allocates a constant 1.7-1.9 KB of managed memory per call, fresh or into
    ///     <c>out=</c>, from 24,000 to 2,400,000 outputs (measured warm).</para>
    /// </remarks>
    [TestClass]
    public class OpenBugsOptunaSharpConsumerTests : TestClass
    {
        // ============================================================================
        //  BUG: the linear quantile family returns +0.0 for [0.0, -0.0, -0.0, -1.0] at q = 0.5; NumPy returns -0.0.
        //
        //  Found by OptunaSharp's PercentilePruner, whose cutoff is np.nanpercentile of the trials' reports: over 600
        //  seeded cases NumSharp's np.nanpercentile matched NumPy bit for bit, except on this tie of signed zeros. No
        //  comparison observes a zero's sign, so no prune decision changes; the bits are what parity pins.
        //
        //  Root cause, the one ParityQb_QB43_MultiKthPartition_KeepsIntroselectsChoiceAmongTies (OpenBugs.ParityQb.cs)
        //  pins for np.partition, np.argpartition and method='lower': NumPy's _quantile partitions at
        //  np.unique([0, -1, prev, next]), here [-1, 0, 1, 2], and several kth run the scalar introselect_ on every host
        //  (introselect_noarg hands only ONE kth to x86-simd-sort). Its dumb_select moves each window's FIRST strict
        //  minimum to the window's front, which leaves [-1.0, -0.0, -0.0, 0.0] (np.argpartition: [3, 1, 2, 0]), so the
        //  'linear' lerp runs between two -0.0. NumSharp's Utilities/QuickSelect.cs insertion-sorts a window of at most 16
        //  elements instead (IntroSelectBlock; the quantile kernels' PartitionAtMany(int) sorts a line that short whole):
        //  a stable order, [-1.0, 0.0, -0.0, -0.0] (np.argpartition: [3, 0, 1, 2]), so the lerp runs between 0.0 and
        //  -0.0, and b - (b - a) * (1 - t) = -0.0 - (-0.0) * 0.5 is +0.0. On this input NumSharp's 'lower' and
        //  'midpoint' are +0.0 as well, 'higher' and 'nearest' already -0.0; NumPy's five methods all give -0.0.
        //  np.median gives +0.0 on both sides: its middle pair is -0.0, -0.0 in NumPy too, but np.mean of it is +0.0.
        //
        //  The fix: QB43's - keep NumPy's dumb_select / last-maximum shortcuts for short windows as well (drop the
        //  insertion-sort window, or make it reproduce dumb_select's swaps), so ties land where NumPy puts them.
        // ============================================================================

        /// <summary>
        ///     <c>np.percentile(x, 50)</c>, <c>np.nanpercentile(x, 50)</c>, <c>np.quantile(x, 0.5)</c> and
        ///     <c>np.nanquantile(x, 0.5)</c> of <c>[0.0, -0.0, -0.0, -1.0]</c> are <c>-0.0</c> in NumPy 2.4.2 (its partition
        ///     places the two <c>-0.0</c> at positions 1 and 2, the interpolated neighbours). NumSharp returns <c>+0.0</c> for
        ///     all four, because its short-window insertion sort leaves <c>0.0</c> at position 1.
        /// </summary>
        /// <remarks>
        ///     The value is checked first, so a fix that changes the result rather than its sign cannot pass; then the sign
        ///     bits of all four calls are collected, so a failure names every call still returning <c>+0.0</c>. Each call
        ///     gets a fresh input array (the family does not write its input, but the test does not depend on that).
        /// </remarks>
        [TestMethod, OpenBugs(IssueUrl = "consumer-optunasharp-quantile-zero-sign")]
        public void LinearQuantiles_OfTiedSignedZeros_KeepNumPysNegativeZero()
        {
            var calls = new (string Call, Func<NDArray, NDArray> Run)[]
            {
                ("np.percentile(x, 50)", x => np.percentile(x, 50.0)),
                ("np.nanpercentile(x, 50)", x => np.nanpercentile(x, 50.0)),
                ("np.quantile(x, 0.5)", x => np.quantile(x, 0.5)),
                ("np.nanquantile(x, 0.5)", x => np.nanquantile(x, 0.5)),
            };

            var positiveZeros = new List<string>();
            foreach (var (call, run) in calls)
            {
                using var x = np.array(new[] { 0.0, -0.0, -0.0, -1.0 });
                using var result = run(x);
                double value = result.GetDouble();
                // The control: a zero on both sides (0.0 == -0.0), so only the sign below is in question.
                value.Should().Be(0.0, $"{call} is a zero in NumPy 2.4.2 and here");
                if (BitConverter.DoubleToInt64Bits(value) != BitConverter.DoubleToInt64Bits(-0.0))
                    positiveZeros.Add(call);
            }

            // The list goes into the message: the assertion's own text names only the first item.
            positiveZeros.Should().BeEmpty(
                "NumPy 2.4.2 returns -0.0 for each: its introselect_ (several kth) leaves [-1.0, -0.0, -0.0, 0.0] and " +
                "interpolates between the two -0.0, where NumSharp's insertion-sorted window leaves " +
                "[-1.0, 0.0, -0.0, -0.0] (the root cause of ParityQb_QB43); +0.0 from " +
                string.Join(", ", positiveZeros));
        }

        // ============================================================================
        //  BUG: np.sum along a short contiguous axis costs about 18 ns per output row, more than twice NumPy's.
        //
        //  Found by OptunaSharp's TPE sampler (its QA-84, 2026-11-12). Optuna adds each candidate's continuous log density
        //  as _truncnorm.logpdf(...).sum(axis=-1) over a fresh C-contiguous (candidates, kernels, parameters) array, and
        //  OptunaSharp now gets NumPy's pairwise order by calling np.sum(terms, axis: -1, @out: sums) on that layout. The
        //  sums are NumPy's bit for bit; the cost is not. The same 24,000 float64 values summed along the last axis as
        //  2,400 rows of 10 ([24,100,10]) and as 24 rows of 1,000 ([24,1000]), on an i9-13900K, NumPy 2.4.2 against
        //  journey5 at 1d45cbd4, the fastest of 15 alternating batches (BestOfAlternatingBatches):
        //                    2,400 rows of 10    24 rows of 1,000    ratio
        //      NumSharp          42-48 us           5.7-7.9 us        6.1-7.4   (test host, 22 runs on both frameworks;
        //                                                                        a slow stretch moves both: 68 and 10.9 us)
        //      NumPy             26-27 us           7.8-8.0 us        3.3-3.5   (5 runs)
        //  At 24,000 rows (pinned to two P-cores) the axis sum costs NumSharp 432 us over rows of 10 and 460 us over rows
        //  of 40, NumPy 250 and 426 us: per output row 17.6 ns + 0.04 ns per element in NumSharp against 7.9 ns + 0.25 ns
        //  per element in NumPy. NumSharp's elements are cheaper (its pairwise fold is vectorized), its fixed cost per row
        //  is 2.2 times NumPy's, so rows up to about 40 elements lose. The arithmetic is not the cost: a console loop with
        //  NumPy's eight accumulators sums the 2,400 rows in 12-16 us. ExecuteAxisReductionNDIter (Default.Reduction.Add.cs)
        //  runs NDIterRef.NewReduce(arr, result, axis) and ForEach, which calls the PINNED pairwise inner loop once per
        //  output row (count 10 here); NumPy's reduction iterator also calls its inner loop once per row, at less than half
        //  the fixed cost.
        //  The fix (a direction, not profiled): sum every row of a contiguous last axis in one kernel call, in the same
        //  pairwise order, or make the per-row path through ForEach and the PINNED kernel cheaper.
        // ============================================================================

        /// <summary>
        ///     <c>np.sum(a, axis=-1)</c> over 2,400 rows of 10 (a C-contiguous float64 <c>[24, 100, 10]</c> array) costs a
        ///     few sums of the same 24,000 values laid out as 24 rows of 1,000, as in NumPy, where it costs 3.3-3.5 of them.
        ///     Today it costs 6.1-7.4: 42-48 us where NumPy takes 26-27 us, although the long rows cost NumSharp no more than
        ///     NumPy (5.7-7.9 us against 7.8-8.0 us).
        /// </summary>
        /// <remarks>
        ///     Two routes through the same call over the same values, timed in one process with their batches alternating
        ///     (<see cref="BestOfAlternatingBatches"/>), so the host's speed cancels out and only the row count differs: the
        ///     ratio is the cost of 2,376 more rows. The 5.0 limit sits between NumPy's own speed (its 26.5 us scores 3.4-4.7
        ///     against NumSharp's long rows) and today's 6.1 and up. The flat sum is no reference: NumSharp's takes 4-5 us in
        ///     the test host and 7.3 us in a console process, which moves a flat-sum ratio with the process. The sums are
        ///     checked first, against NumPy's pairwise order, so a fast fix that adds the elements in another order cannot
        ///     pass. A wall-clock assertion, so it runs alone (<c>[DoNotParallelize]</c>). Calibrated in Release (CI's
        ///     configuration); a Debug build of NumSharp.Core measured 7.3-8.1, failing too.
        /// </remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-optunasharp-axis-sum-row-cost")]
        public void SumOverShortRows_CostsAFewLongRowSums()
        {
            // OptunaSharp's layout: 24 candidates x 100 kernels x 10 continuous parameters, one row per (candidate, kernel);
            // the reference is the same buffer as 24 rows of 1,000.
            const int rowLength = 10;
            double[] terms = LogDensityTerms(24 * 100 * rowLength, seed: 5);
            using var flat = np.array(terms);
            using var a = flat.reshape(24, 100, rowLength);
            using var longRows = flat.reshape(24, 100 * rowLength);

            // The control: every row's sum is NumPy's (the identity 0.0 plus the row's pairwise sum), and that order differs
            // from a left-to-right sum on some rows, so a fix that adds in another order fails here, not in the timing.
            using (var rows = np.sum(a, axis: -1))
            {
                double[] sums = rows.ToArray<double>();
                int notNumPys = 0, orderMatters = 0;
                for (int row = 0; row < sums.Length; row++)
                {
                    var values = new ReadOnlySpan<double>(terms, row * rowLength, rowLength);
                    double numPys = 0.0 + NumPyPairwiseSum(values);
                    if (BitConverter.DoubleToInt64Bits(sums[row]) != BitConverter.DoubleToInt64Bits(numPys))
                        notNumPys++;
                    double leftToRight = 0.0;
                    foreach (double value in values)
                        leftToRight += value;
                    if (BitConverter.DoubleToInt64Bits(leftToRight) != BitConverter.DoubleToInt64Bits(numPys))
                        orderMatters++;
                }
                notNumPys.Should().Be(0, "np.sum(axis=-1) is NumPy's pairwise sum of each row, bit for bit");
                orderMatters.Should().BePositive("the check above must tell NumPy's order from a left-to-right sum");
            }

            var (longRowSums, shortRowSums) = BestOfAlternatingBatches(
                () => { using var s = np.sum(longRows, axis: -1); },
                () => { using var s = np.sum(a, axis: -1); });
            (shortRowSums / longRowSums).Should().BeLessThan(5.0,
                $"np.sum(a, axis=-1) over 2,400 rows of 10 should cost a few sums of the same values in 24 rows of 1,000 " +
                $"({longRowSums * 1000:F1} us; NumPy's ratio is 3.3-3.5); it took {shortRowSums * 1000:F1} us, " +
                $"{shortRowSums * 1e6 / 2400:F1} ns per row");
        }

        // ============================================================================
        //  BUG: np.sum(..., out=) computes a fresh result and copies it into out.
        //
        //  Found with the bug above: OptunaSharp reduces into a reused out= buffer, and where TPE samples a parameter on
        //  its own (Optuna's default, independent sampling: a (24, kernels, 1) array) the out= form costs about twice the
        //  fresh one. ReduceIntoOut (Default.Reduction.Out.cs) runs the whole reduction without out
        //  (ReduceAdd(arr, axis, keepdims, loop)), copies the fresh result into out with NDIter.Copy (np.copyto's unsafe
        //  cast) and disposes it; NumPy's add.reduce writes into out itself. The copy is a second pass as long as the
        //  reduction wherever the reduction is cheap, as along a length-1 axis, whose sum is identity + x. float64
        //  [24,3000,1] along -1 (a larger study's kernels; [24,1000,1]'s 7 us calls timed less steadily in console probes,
        //  1.5-3.1), the fastest of 15 alternating batches:
        //      NumSharp  fresh 18-22 us   out= 34-46 us   out=/fresh 1.84-2.19   (test host, 12 runs on both frameworks)
        //      NumPy     fresh 29-32 us   out= 30-32 us   out=/fresh 1.02-1.04   (5 runs)
        //  Rows long enough to dominate hide it ([24,1000,10]: out=/fresh 1.02-1.07 in NumSharp too).
        //  The fix: reduce into out directly when out can take the loop's results as they are (its dtype is the loop's, it
        //  is writeable, aligned and does not overlap the input), keeping the copy for a cast or an overlap.
        // ============================================================================

        /// <summary>
        ///     <c>np.sum(a, axis=-1, out=o)</c> costs about what the fresh <c>np.sum(a, axis=-1)</c> costs, as in NumPy
        ///     (1.02-1.07 times there). Today it costs 1.84-2.19 times as much for a float64 <c>[24, 3000, 1]</c> array,
        ///     because the result is computed fresh and then copied into <c>o</c>.
        /// </summary>
        /// <remarks>
        ///     The same two-route timing as <see cref="SumOverShortRows_CostsAFewLongRowSums"/>, so the host's speed cancels
        ///     out; the 1.25 limit leaves a direct write (about 1.0: it also skips the result's allocation) a quarter of room,
        ///     and today's ratio misses it by half again. The call is checked first: it returns <c>o</c> itself, holding the
        ///     fresh call's sums. Calibrated in Release (CI's configuration) only: in a Debug build of NumSharp.Core the fresh
        ///     sum's own code slows 2.5 times and the copy's share shrinks, so six Debug runs measured 1.05-1.90 and a Debug
        ///     run can pass with the bug present (the category audit keeps the tag while Release fails).
        /// </remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-optunasharp-sum-out-copy")]
        public void SumIntoOut_CostsAboutAFreshSum()
        {
            // One continuous parameter's terms: 24 candidates x 3,000 kernels x 1.
            double[] terms = LogDensityTerms(24 * 3000, seed: 7);
            using var flat = np.array(terms);
            using var a = flat.reshape(24, 3000, 1);
            using var o = np.zeros(new Shape(24, 3000));
            using (var fresh = np.sum(a, axis: -1))
            {
                NDArray returned = np.sum(a, axis: -1, @out: o);
                ReferenceEquals(returned, o).Should().BeTrue("np.sum returns out itself, as NumPy does");
                o.ToArray<double>().Should().Equal(fresh.ToArray<double>(), "out= holds the fresh call's sums");
            }

            var (freshSum, intoOut) = BestOfAlternatingBatches(
                () => { using var s = np.sum(a, axis: -1); },
                () => np.sum(a, axis: -1, @out: o));
            (intoOut / freshSum).Should().BeLessThan(1.25,
                $"np.sum(a, axis=-1, out=o) should cost about the fresh sum ({freshSum * 1000:F1} us), as in NumPy; it " +
                $"took {intoOut * 1000:F1} us because ReduceIntoOut computes a fresh result and copies it into out");
        }

        /// <summary>
        ///     The fastest of several timed batches of each of two routes, their batches alternating, after a warm-up that
        ///     lets the tiered JIT compile both.
        /// </summary>
        /// <param name="first">The first route; it must not depend on earlier calls.</param>
        /// <param name="second">The second route, timed in the batches between <paramref name="first"/>'s.</param>
        /// <returns>Each route's fastest batch, in milliseconds per call.</returns>
        /// <remarks>
        ///     <see cref="OpenBugsConsumerTests"/>' BestOfBatches times one route after the other. On a hybrid CPU (the
        ///     i9-13900K these pins were measured on has P- and E-cores) the test thread can spend one route's batches on a
        ///     slower stretch: timed that way, a first version of <see cref="SumOverShortRows_CostsAFewLongRowSums"/> (against
        ///     the flat sum) passed once on net8.0 with the bug present, and instrumented runs timed that flat sum at
        ///     4.0-11.2 us. Alternating the batches gives both routes the same conditions, so a slow stretch raises both
        ///     minimums or neither: one run met one in both routes (68 and 10.9 us) and still measured 6.2, inside the
        ///     6.1-7.4 of 22 runs. The warm-up is BestOfBatches': two rounds of 20 calls with a pause between them, as one
        ///     round only queues the tier-1 compile.
        /// </remarks>
        private static (double First, double Second) BestOfAlternatingBatches(Action first, Action second)
        {
            for (int warm = 0; warm < 2; warm++)
            {
                for (int i = 0; i < 20; i++)
                {
                    first();
                    second();
                }
                System.Threading.Thread.Sleep(150);
            }

            // 15 rounds: more chances that both routes meet the fastest core, at a few milliseconds per test.
            const int calls = 10;
            double bestFirst = double.MaxValue, bestSecond = double.MaxValue;
            for (int round = 0; round < 15; round++)
            {
                var watch = Stopwatch.StartNew();
                for (int i = 0; i < calls; i++)
                    first();
                bestFirst = System.Math.Min(bestFirst, watch.Elapsed.TotalMilliseconds / calls);
                watch.Restart();
                for (int i = 0; i < calls; i++)
                    second();
                bestSecond = System.Math.Min(bestSecond, watch.Elapsed.TotalMilliseconds / calls);
            }
            return (bestFirst, bestSecond);
        }

        /// <summary>
        ///     Values shaped like OptunaSharp's log-density terms: negative and a few units large (<c>-4u - 0.3</c> for a
        ///     uniform <c>u</c>), never zero, so no sum meets a signed zero.
        /// </summary>
        /// <param name="count">How many values.</param>
        /// <param name="seed">The seed of the <see cref="System.Random"/> they come from.</param>
        /// <returns>A new array of <paramref name="count"/> values.</returns>
        /// <exception cref="OverflowException"><paramref name="count"/> is negative.</exception>
        private static double[] LogDensityTerms(int count, int seed)
        {
            // System.Random spelled out: NumSharp.Tests has a Random namespace of its own.
            var random = new System.Random(seed);
            var values = new double[count];
            for (int i = 0; i < values.Length; i++)
                values[i] = -random.NextDouble() * 4 - 0.3;
            return values;
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>pairwise_sum</c> (<c>numpy/_core/src/umath/loops_utils.h.src</c>) over contiguous values:
        ///     below 8 values a left-to-right sum from <c>-0.0</c>; up to 128, eight accumulators combined as
        ///     <c>((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))</c>, then the rest left to right; past 128, the two
        ///     halves (the first rounded down to a multiple of 8) summed apart and added.
        /// </summary>
        /// <param name="values">The values, in memory order.</param>
        /// <returns>
        ///     The pairwise sum without the reduction's start: <c>np.add.reduce</c> adds it to the identity
        ///     <c>0.0</c>, so a test compares <c>0.0 + NumPyPairwiseSum(row)</c> with a row's <c>np.sum</c>.
        /// </returns>
        private static double NumPyPairwiseSum(ReadOnlySpan<double> values)
        {
            int n = values.Length;
            if (n < 8)
            {
                // NumPy starts from -0.0 so that a sum of only -0.0 stays -0.0 (0.0 + -0.0 is 0.0).
                double res = -0.0;
                for (int i = 0; i < n; i++)
                    res += values[i];
                return res;
            }
            if (n <= 128)
            {
                double r0 = values[0], r1 = values[1], r2 = values[2], r3 = values[3];
                double r4 = values[4], r5 = values[5], r6 = values[6], r7 = values[7];
                int i = 8;
                for (; i < n - n % 8; i += 8)
                {
                    r0 += values[i]; r1 += values[i + 1]; r2 += values[i + 2]; r3 += values[i + 3];
                    r4 += values[i + 4]; r5 += values[i + 5]; r6 += values[i + 6]; r7 += values[i + 7];
                }
                double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++)
                    res += values[i];
                return res;
            }
            // Halves on a multiple of the unroll factor, as NumPy splits them.
            int n2 = n / 2;
            n2 -= n2 % 8;
            return NumPyPairwiseSum(values[..n2]) + NumPyPairwiseSum(values[n2..]);
        }
    }
}
