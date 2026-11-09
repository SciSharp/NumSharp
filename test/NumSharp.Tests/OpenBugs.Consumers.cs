using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs found by upgrading NumSharp's consumers to 0.80.0 (SciSharp/Bigtree.Algorithm, 2026-11-08/09, and
    ///     SciSharp/SharpCV, 2026-11-09), pinned and deferred by the owner on the consumer PRs' design pages ("Defer all"
    ///     and "Create an openbug on journey5", 2026-11-09). The consumer works around each one today; every test asserts
    ///     the behavior a fix must give, and fails until then.
    /// </summary>
    /// <remarks>
    ///     The third finding of that round, <c>[NDScoped]</c> refusing a <c>double[]</c> / <c>int[]</c> / <c>string[]</c>
    ///     return with NDW003, is an analyzer test: <c>GateAnalyzerTests.ManagedArrayReturns_AreNotGated</c> in
    ///     <c>NumSharp.Tests.Build.Analyzer</c>.
    /// </remarks>
    [TestClass]
    public class OpenBugsConsumerTests : TestClass
    {
        // ============================================================================
        //  BUG: np.array(string[]) discovers its dtype one value at a time.
        //
        //  NumPy's np.array of a list of str gives <U{longest, in code points}. NumSharp's FromManagedText hands a flat
        //  string[] to Build -> Discover, which runs DiscoverDTypeFromPyObject + FindScalarDescriptor + PromoteTypes for
        //  EVERY value before the copy: 65-160 ns per string (steady state, 0.80.0) where the same array built from a
        //  sized request (np.array(values, np.dtype("<U2")), whose Discover returns the request at once) costs 7-18 ns.
        //  Bigtree.Algorithm measures the width itself and passes the sized dtype (its NDHelpers.Text).
        //  The fix: a flat string[] without nulls takes the longest code-point length in one pass and builds that
        //  descriptor (a null keeps the general route, which refuses None).
        // ============================================================================

        /// <summary>
        ///     <c>np.array(string[])</c> costs about what the same array built from a sized <c>&lt;U</c> request costs: the
        ///     dtype NumPy discovers for a list of <c>str</c> is the longest value's width, which one length pass finds.
        ///     Today the discovering route is about 10 times slower (50,000 two-character strings: 3.9 ms against
        ///     0.34 ms).
        /// </summary>
        /// <remarks>
        ///     The test compares two routes in one process (the best of several batches each, after warm-up), so the host's
        ///     speed cancels out: a regression of either route moves every batch, an interruption only one. Both routes
        ///     must give the same dtype first, or the comparison would not be like for like. A wall-clock assertion, so it
        ///     runs alone (<c>[DoNotParallelize]</c>): sibling tests on the other workers would load the cores unevenly
        ///     between the two routes' batches.
        /// </remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-bigtree-np-array-strings")]
        public void ArrayOfStrings_CostsAboutASizedRequest()
        {
            // System.Random and System.Math spelled out: NumSharp.Tests has Random and Math namespaces of its own.
            var random = new System.Random(3);
            string[] values = Enumerable.Range(0, 50_000).Select(_ => "v" + random.Next(4)).ToArray();
            DType sized = np.dtype("<U2");
            using (var discovered = np.array(values))
            using (var requested = np.array(values, sized))
            {
                discovered.dtype.str.Should().Be("<U2", "NumPy discovers <U2 for these strings");
                requested.dtype.str.Should().Be("<U2");
            }

            double discovering = BestOfBatches(() => { using var a = np.array(values); });
            double sizedRequest = BestOfBatches(() => { using var a = np.array(values, sized); });

            (discovering / sizedRequest).Should().BeLessThan(2.0,
                $"np.array(string[]) should cost about a sized request ({sizedRequest:F3} ms); it took {discovering:F3} ms " +
                "because it discovers the dtype value by value");
        }

        /// <summary>
        ///     The fastest of several timed batches of one action, after a warm-up that lets the tiered JIT compile it.
        /// </summary>
        /// <param name="action">The work to time; it must not depend on earlier calls.</param>
        /// <returns>The fastest batch's time per call, in milliseconds.</returns>
        /// <remarks>
        ///     Two warm-up rounds with a pause between them: one round only queues the tier-1 compile. The minimum over
        ///     batches discards interruptions (a collection, another process taking the core), which can only add time.
        /// </remarks>
        private static double BestOfBatches(Action action)
        {
            for (int warm = 0; warm < 2; warm++)
            {
                for (int i = 0; i < 20; i++)
                    action();
                System.Threading.Thread.Sleep(150);
            }

            const int calls = 10;
            double best = double.MaxValue;
            for (int batch = 0; batch < 7; batch++)
            {
                var watch = Stopwatch.StartNew();
                for (int i = 0; i < calls; i++)
                    action();
                best = System.Math.Min(best, watch.Elapsed.TotalMilliseconds / calls);
            }
            return best;
        }

        // ============================================================================
        //  BUG: 17 reductions take one axis only; NumPy takes a tuple of axes.
        //
        //  np.sum(a, axis=(1, 2)) reduces both axes at once in NumPy, and so do np.prod, np.mean, np.std, np.var,
        //  np.max / np.min / np.amax / np.amin, the nan-reductions and np.count_nonzero. NumSharp's forms take int? (or
        //  int) only; a consumer reshapes first (Bigtree.Algorithm's first Baum-Welch port did). Ten reductions already
        //  have the multi-axis form, as an int[] overload beside the int? one: np.all, np.any, np.average, np.median,
        //  np.nanmedian, np.percentile, np.nanpercentile, np.quantile, np.nanquantile and np.ptp.
        // ============================================================================

        /// <summary>
        ///     Every NumPy reduction that takes a tuple of axes has a multi-axis form in NumSharp: an overload whose
        ///     <c>axis</c> parameter is neither <see cref="int"/> nor <see cref="Nullable{Int32}"/> (the ten that have
        ///     one use <c>int[] axis</c>).
        /// </summary>
        /// <remarks>
        ///     Reflection over <see cref="np"/>'s public static methods, so the test compiles today and turns green when
        ///     the forms are added, whatever type they take. NumPy 2.4.2 documents <c>axis : None or int or tuple of
        ///     ints</c> for each name below.
        /// </remarks>
        [TestMethod, OpenBugs(IssueUrl = "consumer-bigtree-tuple-axis")]
        public void Reductions_TakeATupleOfAxes()
        {
            string[] reductions =
            {
                "sum", "prod", "mean", "std", "var", "max", "min", "amax", "amin", "nansum", "nanprod", "nanmean",
                "nanstd", "nanvar", "nanmax", "nanmin", "count_nonzero",
            };
            // The control first: the probe finds the multi-axis forms that exist, so the failure below names missing
            // forms, never a probe that cannot see any.
            new[] { "all", "any", "average", "median", "percentile", "quantile", "ptp" }
                .Where(name => !HasMultiAxisForm(name)).Should().BeEmpty("these already take int[] axis");

            string[] withoutTuple = reductions.Where(name => !HasMultiAxisForm(name)).ToArray();
            withoutTuple.Should().BeEmpty(
                $"NumPy 2.4.2 takes a tuple of axes for every one of them; one axis only: {string.Join(", ", withoutTuple)}");
        }

        // ============================================================================
        //  BUG: copy() of a view with a reversed axis gathers element by element.
        //
        //  Found upgrading SciSharp/SharpCV (2026-11-09): cv2 copies a reversed image view before OpenCV sees it (a
        //  cv::Mat holds no negative stride), and SharpCV made that copy with copy('C'). A 1080p uint8 frame, journey5
        //  against NumPy 2.4.2, both pinned to the same two CPUs, best of 7 batches:
        //      contiguous                      NumSharp  0.34 ms    NumPy  1.48 ms
        //      a[::-1]        rows reversed    NumSharp 10.03 ms    NumPy  1.73 ms
        //      a[:, ::-1]     columns          NumSharp  9.90 ms    NumPy  9.56 ms
        //      a[..., ::-1]   channels         NumSharp  9.48 ms    NumPy 17.50 ms
        //  The reversed rows are a NumPy-parity gap: every row is contiguous, and NumPy copies the view as one block per
        //  row. The reversed inner axes are a speed target beyond NumPy (the SharpCV owner asked for the pin): OpenCV
        //  makes the same copies with flip and its channel swap in 0.25-0.5 ms, which SharpCV now calls instead
        //  (its ReversedLayoutCopy).
        //  The fix: walk a negative-stride outer axis backwards over its contiguous inner blocks, and give a short
        //  reversed inner axis (3 or 4 channels, reversed columns) a reversal kernel instead of the element gather.
        // ============================================================================

        /// <summary>
        ///     Copying a view whose ROWS are reversed (<c>a[::-1]</c>, every row contiguous) costs about a contiguous copy,
        ///     as in NumPy (1.73 ms against 1.48 ms for a 1080p uint8 frame). Today it is an element gather: 10.03 ms
        ///     against 0.34 ms, 29 times NumSharp's own contiguous copy and 5.8 times NumPy's copy of the same view.
        /// </summary>
        /// <remarks>
        ///     Two routes over the same bytes, timed in one process (as <see cref="ArrayOfStrings_CostsAboutASizedRequest"/>
        ///     does), so the host's speed cancels out; the 3x limit leaves the row loop the room NumPy needs (1.17x). The copy
        ///     is checked first, so a fast but wrong fix cannot pass.
        /// </remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-sharpcv-reversed-rows-copy")]
        public void CopyOfReversedRows_CostsAboutAContiguousCopy()
        {
            using var frame = Frame1080();
            using var reversed = frame["::-1"];
            using (var copy = reversed.copy('C'))
                np.array_equal(copy, reversed).Should().BeTrue("the copy is right; only its speed is pinned");

            double contiguous = BestOfBatches(() => { using var c = frame.copy('C'); });
            double rows = BestOfBatches(() => { using var c = reversed.copy('C'); });
            (rows / contiguous).Should().BeLessThan(3.0,
                $"a[::-1].copy() should cost about a contiguous copy ({contiguous:F3} ms), as in NumPy; it took {rows:F3} ms " +
                "because a negative-stride view is copied element by element");
        }

        /// <summary>
        ///     Copying a view whose COLUMNS or CHANNELS are reversed (<c>a[:, ::-1]</c>, and <c>a[..., ::-1]</c>, the BGR /
        ///     RGB idiom) stays within a few contiguous copies, as OpenCV's flip and channel swap do (0.25-0.5 ms for a
        ///     1080p uint8 frame). Today both gather element by element: 9.90 ms and 9.48 ms against 0.34 ms. NumPy is no
        ///     faster (9.56 ms and 17.50 ms), so this pins a speed target beyond parity.
        /// </summary>
        /// <remarks>The same two-route ratio as <see cref="CopyOfReversedRows_CostsAboutAContiguousCopy"/>; the limit is
        ///     4x a contiguous copy, which a reversal kernel meets with room to spare (OpenCV's are 0.7-1.5x).</remarks>
        [TestMethod, DoNotParallelize, OpenBugs(IssueUrl = "consumer-sharpcv-reversed-inner-copy")]
        public void CopyOfReversedColumnsOrChannels_CostsAFewContiguousCopies()
        {
            using var frame = Frame1080();
            using var columns = frame[":, ::-1"];
            using var channels = frame["..., ::-1"];
            foreach (var view in new[] { columns, channels })
            {
                using var copy = view.copy('C');
                np.array_equal(copy, view).Should().BeTrue("the copy is right; only its speed is pinned");
            }

            double contiguous = BestOfBatches(() => { using var c = frame.copy('C'); });
            double reversedColumns = BestOfBatches(() => { using var c = columns.copy('C'); });
            double reversedChannels = BestOfBatches(() => { using var c = channels.copy('C'); });
            System.Math.Max(reversedColumns, reversedChannels).Should().BeLessThan(4 * contiguous,
                $"a[:, ::-1] took {reversedColumns:F3} ms and a[..., ::-1] {reversedChannels:F3} ms against a contiguous " +
                $"copy's {contiguous:F3} ms: a short reversed inner axis is copied element by element");
        }

        /// <summary>A 1080 x 1920 x 3 uint8 frame of varied values: the image these copies were measured on.</summary>
        /// <returns>A new C-contiguous array the caller owns.</returns>
        private static NDArray Frame1080()
        {
            using var values = np.arange(1080L * 1920 * 3);
            using var wrapped = values % 251;
            using var shaped = wrapped.reshape(1080, 1920, 3);
            return shaped.astype(np.uint8);
        }

        /// <summary>Whether a public static <see cref="np"/> method of this name takes more than one axis.</summary>
        /// <param name="name">The method name, NumPy's spelling.</param>
        /// <returns>
        ///     <c>true</c> when some overload has an <c>axis</c> parameter whose type is neither <see cref="int"/> nor
        ///     <see cref="Nullable{Int32}"/>.
        /// </returns>
        private static bool HasMultiAxisForm(string name)
            => typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(method => method.Name == name)
                .SelectMany(method => method.GetParameters())
                .Any(parameter => parameter.Name == "axis" && parameter.ParameterType != typeof(int) &&
                                  parameter.ParameterType != typeof(int?));
    }
}
