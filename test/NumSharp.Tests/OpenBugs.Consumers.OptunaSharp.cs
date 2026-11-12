using System;
using System.Collections.Generic;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs found by OptunaSharp (SciSharp/OptunaSharp, the Optuna port built on NumSharp 0.80.0) while it moved
    ///     its hand-ported NumPy math onto NumSharp's <c>np.*</c> calls (2026-11-11), pinned on journey5 at the owner's
    ///     request ("Create on K:/source/NumSharp an openbug reproducing this"). Each test asserts NumPy 2.4.2's result and
    ///     fails until NumSharp gives it.
    /// </summary>
    /// <remarks>
    ///     OptunaSharp tracks the same finding as its parity item QA-85 (<c>PrunerTests.NanPercentile_TiedSignedZeros_KeepNumpysZeroSign</c>,
    ///     <c>[OpenBugs]</c> there too). Everything else that round moved onto NumSharp matched NumPy bit for bit: the scalar
    ///     <c>log1p</c> / <c>expm1</c> / <c>logaddexp</c> models, <c>np.dot</c> of two vectors, <c>np.var</c>,
    ///     <c>np.mean</c> / <c>np.std</c> over axis 0, <c>np.linalg.norm</c> over an axis and <c>np.linspace</c>.
    /// </remarks>
    [TestClass]
    public class OpenBugsOptunaSharpConsumerTests : TestClass
    {
        // ============================================================================
        //  FIXED 2026-11-12 (QB43): the linear quantile family returned +0.0 for [0.0, -0.0, -0.0, -1.0] at q = 0.5;
        //  NumPy returns -0.0.
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
        //  'linear' lerp runs between two -0.0. NumSharp's Utilities/QuickSelect.cs insertion-sorted a window of at most
        //  16 elements instead, and partitioned at the neighbours alone: a stable order, [-1.0, 0.0, -0.0, -0.0], so the
        //  lerp ran between 0.0 and -0.0, and b - (b - a) * (1 - t) = -0.0 - (-0.0) * 0.5 is +0.0.
        //
        //  The fix: several kth run a statement-for-statement port of introselect_ (QuickSelect.Introselect.cs), and the
        //  quantile family partitions float data at NumPy's own kth list.
        // ============================================================================

        /// <summary>
        ///     Regression pin: <c>np.percentile(x, 50)</c>, <c>np.nanpercentile(x, 50)</c>, <c>np.quantile(x, 0.5)</c> and
        ///     <c>np.nanquantile(x, 0.5)</c> of <c>[0.0, -0.0, -0.0, -1.0]</c> are <c>-0.0</c> in NumPy 2.4.2 (its partition
        ///     places the two <c>-0.0</c> at positions 1 and 2, the interpolated neighbours). Guards the multi-kth
        ///     introselect_ port and the quantile family's kth list; the short-window insertion sort they replaced left
        ///     <c>0.0</c> at position 1 and returned <c>+0.0</c> for all four.
        /// </summary>
        /// <remarks>
        ///     The value is checked first, so a fix that changes the result rather than its sign cannot pass; then the sign
        ///     bits of all four calls are collected, so a failure names every call returning <c>+0.0</c>. Each call gets a
        ///     fresh input array (the family does not write its input, but the test does not depend on that).
        /// </remarks>
        [TestMethod]
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
    }
}
