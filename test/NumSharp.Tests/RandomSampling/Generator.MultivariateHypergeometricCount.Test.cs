using System;
using System.Linq;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     The count method of <c>Generator.multivariate_hypergeometric</c> over its two representations of NumPy's work
    ///     array: the dense <c>sum(colors)</c>-entry array NumPy allocates, and the sparse one a large, lightly sampled urn
    ///     uses instead (only the positions the shuffle has written are stored).
    /// </summary>
    /// <remarks>
    ///     Both representations run one shuffle loop with the same draws, so they must agree on the variates AND on the
    ///     generator's state afterwards. The NumPy cases are urns of a million items or more sampled lightly, which take the
    ///     sparse array by the size rule; their expected values and the three <c>integers(0, 2**62)</c> draws that follow
    ///     are NumPy 2.4.2's <c>default_rng(seed)</c> output.
    /// </remarks>
    [TestClass]
    public class GeneratorMultivariateHypergeometricCountTests
    {
        /// <summary><c>2**62</c>: the exclusive bound of the draws that check the generator's state after the call.</summary>
        private const long StateProbeBound = 1L << 62;

        /// <summary>
        ///     Large urns sampled lightly — the sparse array by the size rule — give NumPy's variates and leave the
        ///     generator where NumPy's does.
        /// </summary>
        /// <param name="seed">The <c>default_rng</c> seed.</param>
        /// <param name="colors">The items per color.</param>
        /// <param name="nsample">The items drawn per variate.</param>
        /// <param name="size">The number of variates.</param>
        /// <param name="expected">NumPy's variates, flattened.</param>
        /// <param name="next">NumPy's next three <c>integers(0, 2**62)</c> draws.</param>
        [TestMethod]
        [DataRow(42L, new long[] { 600000, 300000, 100000, 50000 }, 10L, 20,
            "6, 4, 0, 0, 6, 4, 0, 0, 5, 3, 1, 1, 6, 3, 1, 0, 6, 3, 1, 0, 5, 1, 2, 2, 5, 3, 0, 2, 4, 5, 1, 0, 6, 3, 0, 1, 5, 3, 1, 1, 7, 2, 1, 0, 5, 2, 3, 0, 5, 1, 2, 2, 5, 5, 0, 0, 5, 3, 2, 0, 4, 3, 2, 1, 4, 4, 0, 2, 7, 3, 0, 0, 8, 2, 0, 0, 5, 4, 1, 0",
            "4033205535977457616, 905895001410355513, 1431115343500401259")]
        // More than half the urn: the complement of four swaps per variate.
        [DataRow(7L, new long[] { 1048576, 3, 0, 5 }, 1048580L, 2,
            "1048572, 3, 0, 5, 1048572, 3, 0, 5",
            "24281931447823132, 3787247614991040951, 3675833940291716695")]
        [DataRow(3L, new long[] { 1000000000, 1 }, 1L, 3,
            "1, 0, 1, 0, 1, 0",
            "3695226248115223309, 2684748522177266121, 434091743353603011")]
        [DataRow(5L, new long[] { 123456789, 987654, 0, 4242 }, 33L, 4,
            "33, 0, 0, 0, 33, 0, 0, 0, 33, 0, 0, 0, 33, 0, 0, 0",
            "1754632947151793719, 3289321319516117972, 2824739793507013772")]
        public void LargeLightlySampledUrns_MatchNumPy_AndLeaveNumPysState(long seed, long[] colors, long nsample, int size,
                                                                         string expected, string next)
        {
            var g = np.random.default_rng(seed);
            using var variates = g.multivariate_hypergeometric(colors, nsample, new Shape(size), method: "count");
            using var after = g.integers(0, StateProbeBound, new Shape(3));

            string.Join(", ", variates.ToArray<long>()).Should().Be(expected);
            string.Join(", ", after.ToArray<long>()).Should().Be(next);
        }

        /// <summary>
        ///     Forcing each representation in turn, every urn of a seeded sweep — zero-item colors, one color, the
        ///     complement side, a sample of the whole urn, one and many variates — gives the same variates, and the
        ///     generator's next draws agree.
        /// </summary>
        [TestMethod]
        public void SparseAndDenseWorkArrays_DrawTheSameVariates_AndLeaveTheSameState()
        {
            var rnd = new System.Random(20261016);
            int compared = 0;
            for (int urn = 0; urn < 120; urn++)
            {
                int numColors = 1 + rnd.Next(6);
                var colors = new long[numColors];
                for (int c = 0; c < numColors; c++)
                    colors[c] = rnd.Next(4) == 0 ? 0 : rnd.Next(1, 400);
                long total = colors.Sum();
                // Every regime of the sample size: none, one, a third, either side of half, all but one, all.
                long[] samples = { 0, 1, total / 3, total / 2, total / 2 + 1, total - 1, total };
                foreach (long nsample in samples.Where(s => s >= 0 && s <= total).Distinct())
                    foreach (int size in new[] { 1, 3, 17 })
                    {
                        long seed = rnd.Next();
                        var (denseVariates, denseNext) = Draw(seed, colors, nsample, size, sparse: false);
                        var (sparseVariates, sparseNext) = Draw(seed, colors, nsample, size, sparse: true);
                        string context = $"colors [{string.Join(", ", colors)}], nsample {nsample}, size {size}, seed {seed}";
                        sparseVariates.Should().Equal(denseVariates, context);
                        sparseNext.Should().Equal(denseNext, context);
                        compared++;
                    }
            }
            compared.Should().BeGreaterThan(1000, "the sweep must cover the regimes it lists");
        }

        /// <summary>
        ///     An urn past 2**32 items draws 64-bit words for its first swaps — a branch NumPy's dense array (8 bytes an
        ///     item, here 40 GB) leaves out of reach of any test — and still gives a sample: each variate draws exactly
        ///     <c>nsample</c> items, no color more than it holds, and the same seed gives the same variates.
        /// </summary>
        [TestMethod]
        public void UrnBeyondThirtyTwoBits_IsSampled_WithTheSparseArray()
        {
            var colors = new long[] { 5_000_000_000, 7, 0, 3 };
            long total = colors.Sum();
            foreach (long nsample in new[] { 5L, total - 2 })
            {
                var first = Draw(11, colors, nsample, 4, sparse: null).Variates;
                var again = Draw(11, colors, nsample, 4, sparse: null).Variates;
                again.Should().Equal(first, "the same seed draws the same variates");
                for (int row = 0; row < 4; row++)
                {
                    var counts = first.Skip(row * colors.Length).Take(colors.Length).ToArray();
                    counts.Sum().Should().Be(nsample, $"variate {row} of nsample {nsample}");
                    for (int c = 0; c < colors.Length; c++)
                        counts[c].Should().BeInRange(0, colors[c], $"variate {row}, color {c}, nsample {nsample}");
                }
            }
        }

        /// <summary>
        ///     Draws one call's variates with the given representation forced (or the size rule), then three state-probe
        ///     words from the same generator.
        /// </summary>
        /// <param name="seed">The <c>default_rng</c> seed.</param>
        /// <param name="colors">The items per color.</param>
        /// <param name="nsample">The items drawn per variate.</param>
        /// <param name="size">The number of variates.</param>
        /// <param name="sparse"><c>true</c> forces the sparse array, <c>false</c> the dense one, <c>null</c> the size rule.</param>
        /// <returns>The flattened variates and the three words drawn after them.</returns>
        private static (long[] Variates, long[] Next) Draw(long seed, long[] colors, long nsample, int size, bool? sparse)
        {
            // The hook is [ThreadStatic]: it applies to this thread's calls only, so parallel tests are unaffected.
            Generator.CountChoicesSparseOverride = sparse;
            try
            {
                var g = np.random.default_rng(seed);
                using var variates = g.multivariate_hypergeometric(colors, nsample, new Shape(size), method: "count");
                using var next = g.integers(0, StateProbeBound, new Shape(3));
                return (variates.ToArray<long>(), next.ToArray<long>());
            }
            finally
            {
                Generator.CountChoicesSparseOverride = null;
            }
        }
    }
}
