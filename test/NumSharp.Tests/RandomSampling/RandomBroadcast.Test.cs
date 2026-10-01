using System;
using System.Linq;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     The array-valued (broadcast) distribution parameters of both random APIs — the parts the oracle corpus
    ///     (random_parity / generator_parity, <c>params["bargs"]</c>) cannot pin: legacy outcomes that depend on the width of
    ///     C <c>long</c>, RandomState's cached Gaussian carried into the next call, the read-ahead edge where a position draws
    ///     nothing, the <c>size=()</c> contract, the overloads C# must resolve — plus the exact closed-range and sign scans
    ///     behind the constraint checks (vector body and tail), the draw-free thresholds NumPy never returns from (legacy
    ///     <c>vonmises</c> from <c>2^511</c>, legacy <c>zipf</c> from 1025), the bit-neutrality of the setups a run of equal
    ///     parameters shares, NumPy's allocate-before-broadcast error order for sizes too big to allocate, and which
    ///     parameter layouts are used in place.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every expected value is NumPy 2.4.2's own output (seed 42), compared BIT for bit, including the stream position
    ///     after the call (the next draw). The legacy LP64 pins come from Linux NumPy — NumSharp's RandomState models C
    ///     <c>long</c> as 64-bit — because Windows NumPy's 32-bit long rejects, truncates or (for HRUA's popsize overflow)
    ///     never returns on those inputs; everything else is win-amd64 output, which Linux NumPy reproduces.
    ///     </para>
    ///     <para>
    ///     The systematic matrix — every sampler × layouts × dtypes × every constraint and broadcast error × long mixed
    ///     streams — lives in the oracle corpus (gen_oracle._gen_random_broadcast); these tests pin what a Windows-authored
    ///     corpus cannot, plus the edges a mutation of the read-ahead predicates would break.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class RandomBroadcastTests
    {
        // ---------------------------------------------------------------- harness

        /// <summary>A float64 array's raw bits, element by element in C order.</summary>
        /// <param name="a">The array (float64).</param>
        /// <returns>The IEEE bit patterns.</returns>
        private static ulong[] Bits(NDArray a)
        {
            var r = new ulong[a.size];
            for (long i = 0; i < a.size; i++)
                r[i] = (ulong)BitConverter.DoubleToInt64Bits(a.GetAtIndex<double>(i));
            return r;
        }

        /// <summary>Asserts a float64 result's dtype, shape and exact bits.</summary>
        /// <param name="actual">The result.</param>
        /// <param name="shape">The expected dimensions (empty for 0-d).</param>
        /// <param name="bits">The expected IEEE bit patterns, C order.</param>
        private static void AssertBits(NDArray actual, long[] shape, params ulong[] bits)
        {
            Assert.AreEqual(NPTypeCode.Double, actual.typecode);
            CollectionAssert.AreEqual(shape, actual.Shape.dimensions ?? Array.Empty<long>());
            CollectionAssert.AreEqual(bits, Bits(actual));
        }

        /// <summary>Asserts an int64 result's dtype, shape and values.</summary>
        /// <param name="actual">The result.</param>
        /// <param name="shape">The expected dimensions.</param>
        /// <param name="values">The expected values, C order.</param>
        private static void AssertInt64(NDArray actual, long[] shape, params long[] values)
        {
            Assert.AreEqual(NPTypeCode.Int64, actual.typecode);
            CollectionAssert.AreEqual(shape, actual.Shape.dimensions ?? Array.Empty<long>());
            var got = new long[actual.size];
            for (long i = 0; i < actual.size; i++)
                got[i] = actual.GetAtIndex<long>(i);
            CollectionAssert.AreEqual(values, got);
        }

        /// <summary>The next legacy <c>random_sample()</c> draw's bits — where the call left RandomState's stream.</summary>
        /// <param name="rs">The RandomState.</param>
        /// <returns>The bits.</returns>
        private static ulong NextSample(NumPyRandom rs) => (ulong)BitConverter.DoubleToInt64Bits(rs.random_sample().GetAtIndex<double>(0));

        /// <summary>The next Generator <c>random()</c> draw's bits — where the call left the PCG64 stream.</summary>
        /// <param name="g">The Generator.</param>
        /// <returns>The bits.</returns>
        private static ulong NextRandom(Generator g) => (ulong)BitConverter.DoubleToInt64Bits(g.random().GetAtIndex<double>(0));

        // ---------------------------------------------------------------- legacy C long = 64-bit (LP64 NumPy)

        /// <summary>
        ///     Legacy HRUA with <c>ngood + nbad</c> past <c>2^31</c> draws LP64 NumPy's values and leaves the stream
        ///     where LP64 NumPy does — a 32-bit popsize (Windows NumPy) overflows and never returns, so a regression to
        ///     32-bit arithmetic shows as different draws or a hang.
        /// </summary>
        [TestMethod]
        public void Legacy_Hypergeometric_PopulationPast2Pow31_IsLP64NumPy()
        {
            // Windows NumPy's legacy HRUA computes popsize = good + bad in a 32-bit long: 3.5e9 overflows and its
            // rejection loop never returns. LP64 NumPy (and NumSharp) draw it.
            var rs = np.random.RandomState(42);
            using var good = np.array(new long[] { 2_000_000_000, 7 });
            using var bad = np.array(new long[] { 1_500_000_000, 9 });
            using var sample = NDArray.Scalar(12L);
            using var r = rs.hypergeometric(good, bad, sample);
            AssertInt64(r, new long[] { 2 }, 6, 4);
            Assert.AreEqual(0x3F95141A07387E80UL, NextSample(rs));
        }

        /// <summary>
        ///     A legacy binomial count past <c>2^31</c> is legal and draws what LP64 NumPy draws:
        ///     <c>LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG</c> bounds at a 64-bit C long here, not Windows' <c>2^31 -
        ///     1</c>.
        /// </summary>
        [TestMethod]
        public void Legacy_Binomial_NPast2Pow31_IsLP64NumPy()
        {
            // Windows NumPy: "n is out of bounds for long, ..." (LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG at 2^31 - 1).
            var rs = np.random.RandomState(42);
            using var n = np.array(new long[] { 3_000_000_000, 10 });
            using var p = NDArray.Scalar(0.5);
            using var r = rs.binomial(n, p);
            AssertInt64(r, new long[] { 2 }, 1499972652, 6);
            Assert.AreEqual(0x3FE32835D6632CB0UL, NextSample(rs));
        }

        /// <summary>
        ///     A legacy Poisson mean past <c>2^31</c> is legal and draws what LP64 NumPy draws:
        ///     <c>LEGACY_POISSON_LAM_MAX</c> derives from a 64-bit long, and the count itself exceeds int32.
        /// </summary>
        [TestMethod]
        public void Legacy_Poisson_LamPast2Pow31_IsLP64NumPy()
        {
            // Windows NumPy: "lam value too large" (LEGACY_POISSON_LAM_MAX from a 32-bit long).
            var rs = np.random.RandomState(42);
            using var lam = np.array(new[] { 1e10, 5.0 });
            using var r = rs.poisson(lam);
            AssertInt64(r, new long[] { 2 }, 9999964050, 4);
            Assert.AreEqual(0x3FEBB7B70955B7B5UL, NextSample(rs));
        }

        /// <summary>
        ///     The legacy negative binomial's Poisson of an extreme mean returns C's <c>(long)</c> cast of the double
        ///     as LP64 NumPy does: <c>p == 0</c> yields long's minimum (the indefinite integer), a tiny <c>p</c> or a
        ///     huge <c>n</c> a count past <c>2^31</c> — values only a 64-bit long carries.
        /// </summary>
        [TestMethod]
        public void Legacy_NegativeBinomial_ExtremeMeans_AreLP64IntegerCasts()
        {
            // p == 0: the Poisson mean is infinite and C's integer cast of it reads as long's minimum (int32's on
            // Windows); a tiny p / a huge n give a count past 2^31 that only a 64-bit long carries.
            {
                var rs = np.random.RandomState(42);
                using var n = np.array(new[] { 5.0, 3.0 });
                using var p = np.array(new[] { 0.0, 0.5 });
                using var r = rs.negative_binomial(n, p);
                AssertInt64(r, new long[] { 2 }, long.MinValue, 0);
                Assert.AreEqual(0x3FEBB7B70955B7B5UL, NextSample(rs));
            }
            {
                var rs = np.random.RandomState(42);
                using var n = np.array(new[] { 5.0, 0.5 });
                using var p = np.array(new[] { 0.5, 1e-18 });
                using var r = rs.negative_binomial(n, p);
                AssertInt64(r, new long[] { 2 }, 3, 1343377992127962112);
                Assert.AreEqual(0x3FEF098062E353A5UL, NextSample(rs));
            }
            {
                var rs = np.random.RandomState(42);
                using var n = np.array(new[] { 5.0, 1e18 });
                using var p = NDArray.Scalar(0.5);
                using var r = rs.negative_binomial(n, p);
                AssertInt64(r, new long[] { 2 }, 3, 1000000000149405440);
                Assert.AreEqual(0x3F95141A07387E80UL, NextSample(rs));
            }
        }

        /// <summary>
        ///     Legacy zipf rejects candidates above a 64-bit <c>LONG_MAX</c>: a candidate past <c>2^31</c> is accepted
        ///     and the stream continues where LP64 NumPy's does (Windows NumPy rejects it and continues on a different
        ///     stream).
        /// </summary>
        [TestMethod]
        public void Legacy_Zipf_SmallExponent_KeepsCandidatesBelowLongMax_LP64()
        {
            // legacy_random_zipf rejects every candidate above LONG_MAX: with a 64-bit long 3286471418949803 is accepted,
            // where Windows NumPy rejects it and continues on a different stream.
            var rs = np.random.RandomState(42);
            using var a = np.array(new[] { 3.0, 1.5, 1.05 });
            using var r = rs.zipf(a, new Shape(4, 3));
            AssertInt64(r, new long[] { 4, 3 }, 1, 13, 29, 1, 1, 3286471418949803, 1, 2, 81694, 1, 1, 86);
            Assert.AreEqual(0x3FE2F50F65DDF717UL, NextSample(rs));
        }

        // ---------------------------------------------------------------- RandomState's cached Gaussian

        /// <summary>
        ///     An odd number of legacy normals leaves the last polar pair's second half in RandomState's Gaussian
        ///     cache, and the NEXT call returns it without drawing — a broadcast fill that dropped or reused the cache
        ///     would shift every later normal.
        /// </summary>
        [TestMethod]
        public void Legacy_Normal_OddCount_LeavesTheCachedHalfForTheNextCall()
        {
            // Three polar normals = two pairs: the second pair's other half stays cached, and the next standard_normal
            // returns it without drawing.
            var rs = np.random.RandomState(42);
            using var loc = np.array(new[] { 0.0, 1.0, 2.0 });
            using var scale = NDArray.Scalar(1.0);
            using var r = rs.normal(loc, scale);
            AssertBits(r, new long[] { 3 }, 0x3FDFCA2A28A9307CUL, 0x3FEB9356BE887EB6UL, 0x40052E775409179AUL);
            Assert.AreEqual(0x3FF85E548E01AA2BUL, (ulong)BitConverter.DoubleToInt64Bits(rs.standard_normal().GetAtIndex<double>(0)));
        }

        /// <summary>
        ///     A legacy <c>standard_t</c> position whose df halves to 0 takes its normal from the cache and draws no
        ///     gamma: the read-ahead must stop at such a position, or the following draws land on the wrong words.
        /// </summary>
        [TestMethod]
        public void Legacy_StandardT_DfHalvingToZero_UsesTheCacheAndDrawsNothing()
        {
            // df = 5e-324 halves to 0: that position's normal comes from the cache and its gamma returns 0 without
            // drawing — a position that draws NOTHING, which the read-ahead must not run past.
            var rs = np.random.RandomState(42);
            using var df = np.array(new[] { 3.0, 5e-324 });
            using var r = rs.standard_t(df);
            AssertBits(r, new long[] { 2 }, 0x3FE33DC9C1F9191EUL, 0xFFF8000000000000UL);
            Assert.AreEqual(0x3FD468E4B5774960UL, (ulong)BitConverter.DoubleToInt64Bits(rs.standard_normal().GetAtIndex<double>(0)));
        }

        // ---------------------------------------------------------------- read-ahead: positions that draw nothing

        /// <summary>
        ///     Draw-free positions at the END of a legacy <c>chisquare</c> / <c>standard_t</c> array leave the stream
        ///     exactly where NumPy leaves it — a read-ahead sized by "one draw per position still owed" would consume
        ///     words NumPy never reads.
        /// </summary>
        [TestMethod]
        public void Legacy_ChisquareAndStandardT_TrailingDrawFreePositions_KeepTheStream()
        {
            {
                var rs = np.random.RandomState(42);
                using var df = np.array(new[] { 2.0, 5e-324, 5e-324 });
                using var r = rs.chisquare(df);
                AssertBits(r, new long[] { 3 }, 0x3FEE087D06E0729CUL, 0UL, 0UL);
                Assert.AreEqual(0x3FEE6C4068BBD654UL, NextSample(rs));
            }
            {
                var rs = np.random.RandomState(42);
                using var df = np.array(new[] { 3.0, 5e-324, 5e-324, 2.0 });
                using var r = rs.standard_t(df);
                AssertBits(r, new long[] { 4 }, 0x3FE33DC9C1F9191EUL, 0xFFF8000000000000UL, 0xFFF8000000000000UL, 0x3FD0182BDF940F61UL);
                Assert.AreEqual(0x3F95141A07387E80UL, NextSample(rs));
            }
        }

        /// <summary>
        ///     Draw-free positions at the END of a Generator <c>chisquare</c> / <c>f</c> array leave the PCG64 stream
        ///     exactly where NumPy leaves it (the next <c>random()</c> is NumPy's).
        /// </summary>
        [TestMethod]
        public void Generator_ChisquareAndF_TrailingDrawFreePositions_KeepTheStream()
        {
            // chisquare(2.0) is ONE exponential draw; the two 5e-324 positions draw nothing. A read-ahead sized by "one per
            // position still owed" would take 3 words where NumPy takes 1, and the next draw would be off by two.
            {
                var g = np.random.default_rng(42);
                using var df = np.array(new[] { 2.0, 5e-324, 5e-324 });
                using var r = g.chisquare(df);
                AssertBits(r, new long[] { 3 }, 0x40133BD1B8765DB9UL, 0UL, 0UL);
                Assert.AreEqual(0x3FDC16959869E47EUL, NextRandom(g));
            }
            {
                var g = np.random.default_rng(42);
                using var dfnum = np.array(new[] { 5e-324, 5e-324 });
                using var dfden = np.array(new[] { 2.0, 5e-324 });
                using var r = g.f(dfnum, dfden);
                AssertBits(r, new long[] { 2 }, 0UL, 0xFFF8000000000000UL);
                Assert.AreEqual(0x3FDC16959869E47EUL, NextRandom(g));
            }
        }

        // ---------------------------------------------------------------- size=() and the scalar path

        /// <summary>
        ///     <c>Shape.Scalar</c> is NumPy's GIVEN size <c>()</c>, not <c>None</c>: with non-0-d parameters both APIs
        ///     raise <c>validate_output_shape</c>'s text instead of broadcasting the parameters.
        /// </summary>
        [TestMethod]
        public void SizeEmptyTuple_WithArrayParameters_RaisesNumPysOutputSizeError()
        {
            using var loc = np.array(new[] { 0.0, 1.0, -2.0 });
            using var scale = NDArray.Scalar(2.0);
            var legacy = Assert.ThrowsException<ValueError>(() => np.random.RandomState(42).normal(loc, scale, Shape.Scalar));
            Assert.AreEqual("Output size () is not compatible with broadcast dimensions of inputs (3,).", legacy.Message);
            var gen = Assert.ThrowsException<ValueError>(() => np.random.default_rng(42).normal(loc, scale, Shape.Scalar));
            Assert.AreEqual("Output size () is not compatible with broadcast dimensions of inputs (3,).", gen.Message);
        }

        /// <summary>
        ///     With every parameter 0-d, <c>size=()</c> takes the scalar path and returns ONE 0-d draw, consuming
        ///     exactly the words NumPy's scalar call consumes (pinned by the next draw).
        /// </summary>
        [TestMethod]
        public void SizeEmptyTuple_WithZeroDParameters_ReturnsOneZeroDDraw()
        {
            using var loc = NDArray.Scalar(0.0);
            using var scale = NDArray.Scalar(2.0);
            var rs = np.random.RandomState(42);
            using (var r = rs.normal(loc, scale, Shape.Scalar))
                AssertBits(r, Array.Empty<long>(), 0x3FEFCA2A28A9307CUL);
            Assert.AreEqual(0x3FE76C7E8F1E6751UL, NextSample(rs));
            var g = np.random.default_rng(42);
            using (var r = g.normal(loc, scale, Shape.Scalar))
                AssertBits(r, Array.Empty<long>(), 0x3FE3807C1104FC6BUL);
            Assert.AreEqual(0x3FDC16959869E47EUL, NextRandom(g));
        }

        // ---------------------------------------------------------------- output allocation: np.empty(size) comes first

        /// <summary>
        ///     A given size is allocated BEFORE the parameters are broadcast against it, as NumPy's
        ///     <c>cont_broadcast_N</c> / <c>discrete_broadcast_*</c> do: a size that is both unallocatable and
        ///     incompatible reports <c>array is too big</c> (checking the shape first reported the mismatch), the
        ///     dimensions are scanned left to right, and an allocatable size that does not fit keeps NumPy's mismatch
        ///     text.
        /// </summary>
        [TestMethod]
        public void Broadcast_UnallocatableSize_ReportsTheAllocationBeforeTheShapeMismatch()
        {
            const string tooBig = "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.";
            const string mismatch = "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (4, 3) and arg 1 with shape (2,).";
            using var two = np.array(new[] { 0.0, 1.0 });
            using var one = NDArray.Scalar(1.0);
            using var counts = np.array(new long[] { 5, 6 });
            using var half = NDArray.Scalar(0.5);
            // (2^62, 3) is both unallocatable and incompatible with a (2,) parameter: NumPy's np.empty(size) fails first.
            var huge = new Shape(1L << 62, 3);
            (string expected, Func<NDArray> call)[] cases =
            {
                (tooBig, () => np.random.default_rng(42).normal(two, one, huge)),
                (tooBig, () => np.random.RandomState(42).normal(two, one, huge)),
                (tooBig, () => np.random.default_rng(42).binomial(counts, half, huge)),
                (tooBig, () => np.random.RandomState(42).binomial(counts, half, huge)),
                // The dimensions are scanned left to right, as np.empty does: the overflow is met before the -1 here...
                (tooBig, () => np.random.default_rng(42).normal(two, one, new Shape(1L << 62, -1))),
                (tooBig, () => np.random.RandomState(42).normal(two, one, new Shape(1L << 62, -1))),
                // ...and the -1 first here.
                ("negative dimensions are not allowed", () => np.random.default_rng(42).normal(two, one, new Shape(-1, 1L << 62))),
                ("negative dimensions are not allowed", () => np.random.RandomState(42).normal(two, one, new Shape(-1, 1L << 62))),
                // An allocatable size that does not fit still reports MultiIterNew's text, the output as arg 0.
                (mismatch, () => np.random.default_rng(42).normal(two, one, new Shape(4, 3))),
                (mismatch, () => np.random.RandomState(42).normal(two, one, new Shape(4, 3))),
            };
            foreach (var (expected, call) in cases)
                Assert.AreEqual(expected, Assert.ThrowsException<ValueError>(() => call()).Message);
        }

        // ---------------------------------------------------------------- uniform: the NumPy-shaped array overload

        /// <summary>
        ///     Array <c>uniform</c> bounds run NumPy's <c>np.subtract(high, low)</c> once, then <c>low + range *
        ///     next_double()</c> per broadcast position, on both APIs — NumPy's values and stream position.
        /// </summary>
        [TestMethod]
        public void Uniform_ArrayBounds_BroadcastLowAndRange_BothApis()
        {
            // NumPy computes np.subtract(high, low) whole, then low + range * next_double() per position. (The old
            // non-NumPy uniform(NDArray, NDArray, DType) required equal shapes and is gone.)
            using var low = np.array(new[] { 0.0, 1.0, -1.0 });
            using var high = NDArray.Scalar(5.0);
            var rs = np.random.RandomState(42);
            using (var r = rs.uniform(low, high))
                AssertBits(r, new long[] { 3 }, 0x3FFDF694E5F72667UL, 0x40133620345DEB2AUL, 0x400B22BDD6AD9AFAUL);
            Assert.AreEqual(0x3FE32835D6632CB0UL, NextSample(rs));
            var g = np.random.default_rng(42);
            using (var r = g.uniform(low, high))
                AssertBits(r, new long[] { 3 }, 0x400EF54F580B91EDUL, 0x40060B4ACC34F23FUL, 0x40109B39C23A6472UL);
            Assert.AreEqual(0x3FE650D6C1C2C011UL, NextRandom(g));
        }

        /// <summary>
        ///     The APIs' <c>uniform</c> checks differ: mtrand refuses only a non-finite range, so reversed bounds draw;
        ///     the Generator also requires <c>high - low &gt;= 0</c> and raises NumPy's text.
        /// </summary>
        [TestMethod]
        public void Uniform_Legacy_ReversedBoundsAreLegal_GeneratorRefusesThem()
        {
            // mtrand's uniform only rejects a non-finite range; the Generator also requires high - low >= 0.
            using var low = np.array(new[] { 1.0 });
            using var high = np.array(new[] { 0.0 });
            using (var r = np.random.RandomState(42).uniform(low, high))
                Assert.IsTrue(r.GetAtIndex<double>(0) is <= 1.0 and > 0.0);
            var ex = Assert.ThrowsException<ValueError>(() => np.random.default_rng(42).uniform(low, high));
            Assert.AreEqual("high - low < 0", ex.Message);
        }

        // ---------------------------------------------------------------- overload resolution

        /// <summary>
        ///     The NDArray <c>hypergeometric</c> overload must not capture integer calls: <c>(long, long, long,
        ///     Shape)</c> binds the scalar sampler, and the all-0-d NDArray form takes NumPy's scalar path with the
        ///     same draws.
        /// </summary>
        [TestMethod]
        public void Hypergeometric_IntegersAndShape_BindTheScalarSampler()
        {
            // With the NDArray overload beside hypergeometric(long, long, long, Shape?), a call passing integers and a Shape
            // was ambiguous; the exact (long, long, long, Shape) overload resolves it to the scalar sampler.
            using var viaShape = np.random.RandomState(42).hypergeometric(10L, 5L, 3L, new Shape(4));
            using var viaNullable = np.random.RandomState(42).hypergeometric(10L, 5L, 3L, (Shape?)new Shape(4));
            AssertInt64(viaShape, new long[] { 4 }, viaNullable.ToArray<long>());
            // The all-0-d NDArray form takes NumPy's scalar path: the same draws.
            using var g = NDArray.Scalar(10L);
            using var b = NDArray.Scalar(5L);
            using var s = NDArray.Scalar(3L);
            using var viaArrays = np.random.RandomState(42).hypergeometric(g, b, s, new Shape(4));
            AssertInt64(viaArrays, new long[] { 4 }, viaNullable.ToArray<long>());
        }

        /// <summary>
        ///     The Generator's <c>10^9</c> caps on <c>ngood</c>/<c>nbad</c> do not exist in mtrand: the legacy array
        ///     path draws where the Generator raises NumPy's text.
        /// </summary>
        [TestMethod]
        public void Legacy_HypergeometricArrays_HaveNoGeneratorCaps()
        {
            // The Generator refuses ngood/nbad >= 10^9; mtrand's legacy sampler has no such cap (only C long bounds).
            using var good = np.array(new long[] { 1_000_000_000 });
            using var bad = NDArray.Scalar(5L);
            using var sample = NDArray.Scalar(20L);
            using (var r = np.random.RandomState(42).hypergeometric(good, bad, sample))
                Assert.AreEqual(20L, r.GetAtIndex<long>(0));
            var ex = Assert.ThrowsException<ValueError>(() => np.random.default_rng(42).hypergeometric(good, bad, sample));
            Assert.AreEqual("both ngood and nbad must be less than 1000000000", ex.Message);
        }

        // ---------------------------------------------------------------- the scans: exact closed ranges, vector body and tail

        /// <summary>An 11-element float64 array of 0.5 with <paramref name="v"/> planted at <paramref name="at"/>.</summary>
        /// <param name="v">The probe value.</param>
        /// <param name="at">Where to plant it: index 1 lands in the float64 scans' vector body, index 10 in their scalar tail.</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray Planted(double v, int at)
        {
            var values = new double[11];
            Array.Fill(values, 0.5);
            values[at] = v;
            return np.array(values);
        }

        /// <summary>
        ///     The float64 scans behind the constraint checks and the read-ahead decisions answer exactly at every edge
        ///     — NaN of either sign, <c>±0</c>, the smallest subnormal and its halving, <c>±inf</c>, the neighbouring
        ///     doubles of strict bounds — both in the vectorized body and in the scalar tail.
        /// </summary>
        [TestMethod]
        public void Scans_ClosedRanges_AreExact_InTheVectorBodyAndTheTail()
        {
            // The constraint checks and read-ahead decisions scan with closed ranges whose strict bounds sit at the
            // neighbouring double; each probe goes through both the vectorized body and the scalar tail.
            var negativeNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000001UL));
            foreach (int at in new[] { 1, 10 })
            {
                using (var a = Planted(double.NaN, at))
                {
                    Assert.IsTrue(RandomBroadcast.AnyNaN(a));
                    Assert.IsFalse(RandomBroadcast.AllInRange(a, 0.0, 1.0), "a NaN fails every range (np.all)");
                    Assert.IsFalse(RandomBroadcast.AnyInRange(a, 0.6, double.PositiveInfinity), "a NaN is in no range (np.any)");
                }
                using (var a = Planted(-0.0, at))
                {
                    Assert.IsTrue(RandomBroadcast.AnyNegativeSign(a), "NumPy's signbit rejects -0.0 for CONS_NON_NEGATIVE");
                    Assert.IsTrue(RandomBroadcast.AnyZero(a));
                    Assert.IsTrue(RandomBroadcast.AllInRange(a, 0.0, 1.0), "-0.0 >= 0 holds, as np.greater_equal says");
                }
                using (var a = Planted(negativeNaN, at))
                    Assert.IsFalse(RandomBroadcast.AnyNegativeSign(a), "a NaN never counts, whatever its sign bit");
                using (var a = Planted(double.Epsilon, at))
                {
                    Assert.IsTrue(RandomBroadcast.AnyHalvesToZero(a), "5e-324 / 2 rounds to even: 0");
                    Assert.IsTrue(RandomBroadcast.AllInRange(a, double.Epsilon, 1.0), "v > 0 is v >= 5e-324");
                }
                using (var a = Planted(2 * double.Epsilon, at))
                    Assert.IsFalse(RandomBroadcast.AnyHalvesToZero(a), "1e-323 / 2 is exactly 5e-324");
                using (var a = Planted(0.0, at))
                {
                    Assert.IsFalse(RandomBroadcast.AllInRange(a, double.Epsilon, 1.0), "0 fails v > 0");
                    Assert.IsFalse(RandomBroadcast.AnyNegativeSign(a));
                }
                using (var a = Planted(double.PositiveInfinity, at))
                {
                    Assert.IsTrue(RandomBroadcast.AnyInRange(a, 1025.0, double.PositiveInfinity));
                    Assert.IsFalse(RandomBroadcast.AllInRange(a, -double.MaxValue, double.MaxValue), "inf is not finite");
                }
            }
            using var clean = Planted(0.5, 0);
            Assert.IsFalse(RandomBroadcast.AnyNaN(clean) || RandomBroadcast.AnyNegativeSign(clean) || RandomBroadcast.AnyZero(clean)
                           || RandomBroadcast.AnyHalvesToZero(clean) || RandomBroadcast.AnyInRange(clean, 0.6, 1.0));
            Assert.IsTrue(RandomBroadcast.AllInRange(clean, 0.5, 0.5));
        }

        /// <summary>
        ///     int64 and float32 parameters compare exactly against the double bounds: <c>long.MaxValue</c> against
        ///     <c>10^9</c> and the legacy long bound, float32 <c>-0</c> and NaN with their sign and NaN answers.
        /// </summary>
        [TestMethod]
        public void Scans_Int64AndFloat32Parameters_WidenExactly()
        {
            using var ints = np.array(new long[] { 3, 0, long.MaxValue, 12 });
            Assert.IsTrue(RandomBroadcast.AnyZero(ints));
            Assert.IsFalse(RandomBroadcast.AnyNegativeSign(ints));
            Assert.IsFalse(RandomBroadcast.AnyNaN(ints));
            Assert.IsTrue(RandomBroadcast.AnyInRange(ints, 1_000_000_000, double.PositiveInfinity), "long.MaxValue >= 10^9");
            Assert.IsTrue(RandomBroadcast.AllInRange(ints, double.NegativeInfinity, RandomConstraints.LegacyLongMax),
                "every int64 widens to at most 2^63");
            using var negative = np.array(new long[] { 5, -1 });
            Assert.IsTrue(RandomBroadcast.AnyNegativeSign(negative));
            Assert.IsTrue(RandomBroadcast.AnyInRange(negative, double.NegativeInfinity, -double.Epsilon), "v < 0 for an int64");
            using var floats = np.array(new float[] { 1f, -0f, float.NaN });
            Assert.IsTrue(RandomBroadcast.AnyNegativeSign(floats));
            Assert.IsTrue(RandomBroadcast.AnyNaN(floats));
            Assert.IsFalse(RandomBroadcast.AllInRange(floats, -1.0, 2.0), "the NaN fails the range");
        }

        /// <summary>
        ///     A parameter of <c>2 * lanes + 3</c> elements (so both the vector body and the scalar tail run whatever the
        ///     host's vector width) filled with <paramref name="fill"/>, with <paramref name="v"/> planted at
        ///     <paramref name="at"/>.
        /// </summary>
        /// <typeparam name="T">The element type (int64 or float32 here).</typeparam>
        /// <param name="fill">The background value, which no probe answers true for.</param>
        /// <param name="v">The probe value.</param>
        /// <param name="at">Where to plant it: a non-negative index counts from the start (1 lands in the vector body), a
        ///     negative one from the end (-1 is the last element, always in the scalar tail).</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray PlantedLanes<T>(T fill, T v, int at) where T : unmanaged
        {
            var values = new T[2 * System.Numerics.Vector<T>.Count + 3];
            Array.Fill(values, fill);
            values[at >= 0 ? at : values.Length + at] = v;
            return np.array(values);
        }

        /// <summary>
        ///     The vectorized sign scans over int64 and float32 parameters (the float32 one serves
        ///     <c>standard_gamma</c>'s float32 path) answer like NumPy's <c>signbit</c> on every edge — <c>-0</c>, a
        ///     NaN of either sign, the smallest negative subnormal, the extremes — whether the element sits in the
        ///     vector body or the scalar tail.
        /// </summary>
        [TestMethod]
        public void Scans_SignScans_VectorBodies_OverInt64AndFloat32()
        {
            // .NET's float.NaN is the NEGATIVE quiet NaN (0xffc00000); both NaN signs must answer false.
            float positiveNaN = BitConverter.Int32BitsToSingle(0x7FC00000), negativeNaN = BitConverter.Int32BitsToSingle(unchecked((int)0xFFC00001));
            (float v, bool expected)[] floats =
            {
                (-0f, true), (-1f, true), (-float.Epsilon, true), (float.MinValue, true), (float.NegativeInfinity, true),
                (positiveNaN, false), (negativeNaN, false), (0f, false), (float.Epsilon, false), (float.PositiveInfinity, false),
            };
            (long v, bool expected)[] longs = { (-1, true), (long.MinValue, true), (0, false), (long.MaxValue, false) };
            foreach (int at in new[] { 1, -1 })
            {
                foreach (var (v, expected) in floats)
                    using (var a = PlantedLanes(0.5f, v, at))
                        Assert.AreEqual(expected, RandomBroadcast.AnyNegativeSign(a), $"float32 0x{BitConverter.SingleToInt32Bits(v):X8} at {at}");
                foreach (var (v, expected) in longs)
                    using (var a = PlantedLanes(5L, v, at))
                        Assert.AreEqual(expected, RandomBroadcast.AnyNegativeSign(a), $"int64 {v} at {at}");
            }
        }

        // ---------------------------------------------------------------- constraint bounds (NumPy 2.4.2 values)

        /// <summary>
        ///     Each strict Generator bound accepts its neighbouring double and draws NumPy's values from it (geometric
        ///     <c>p = 5e-324</c>, logseries <c>p = 1 - ulp</c>, zipf <c>a = 1 + ulp</c>, a NaN exponential scale, a
        ///     zero gamma shape) — and rejects the bound itself with NumPy's array text.
        /// </summary>
        /// <remarks>
        ///     The <c>[Timeout]</c>: zipf at <c>a = 1</c> is rejected here, and its rejection loop never returns on that value
        ///     once a check lets it through — so a regression would HANG this test instead of failing it (see
        ///     <see cref="Constraints_LoopGuardingBounds_AreRejectedOnEveryPath"/>; logseries at <c>p = 1</c> returns 2 here
        ///     instead, and fails the assertion).
        /// </remarks>
        [TestMethod]
        [Timeout(60_000)]
        public void Constraints_StrictBounds_AcceptTheNeighbouringDouble_Generator()
        {
            // The smallest valid element next to the rejected bound — NumPy's draws and stream position (win-amd64).
            var g = np.random.default_rng(42);
            using (var p = np.array(new[] { double.Epsilon, 1.0 }))
            using (var r = g.geometric(p))
                AssertInt64(r, new long[] { 2 }, long.MaxValue, 1);
            Assert.AreEqual(0x3FEB79A2584DDB42UL, NextRandom(g));
            g = np.random.default_rng(42);
            using (var p = np.array(new[] { System.Math.BitDecrement(1.0), 0.5 }))
            using (var r = g.logseries(p))
                AssertInt64(r, new long[] { 2 }, 2574974, 1);
            Assert.AreEqual(0x3FE650D6C1C2C011UL, NextRandom(g));
            g = np.random.default_rng(42);
            using (var a = np.array(new[] { System.Math.BitIncrement(1.0), 2.0 }))
            using (var r = g.zipf(a))
                AssertInt64(r, new long[] { 2 }, 353887435612303, 1);
            Assert.AreEqual(0x3FE85B41A5F78B48UL, NextRandom(g));
            g = np.random.default_rng(42);
            using (var scale = np.array(new[] { BitConverter.Int64BitsToDouble(0x7FF8000000000000), 1.0 }))
            using (var r = g.exponential(scale))
                AssertBits(r, new long[] { 2 }, 0x7FF8000000000000UL, 0x4002B08433C827FCUL);
            Assert.AreEqual(0x3FEB79A2584DDB42UL, NextRandom(g));
            g = np.random.default_rng(42);
            using (var shape = np.array(new[] { 0.0, 1.0 }))
            using (var r = g.gamma(shape))
                AssertBits(r, new long[] { 2 }, 0x0UL, 0x40033BD1B8765DB9UL);
            Assert.AreEqual(0x3FDC16959869E47EUL, NextRandom(g));

            (string expected, Func<NDArray> call)[] rejected =
            {
                ("p <= 0, p > 1 or p contains NaNs", () => np.random.default_rng(42).geometric(np.array(new[] { 0.0, 0.5 }))),
                ("p < 0, p >= 1 or p contains NaNs", () => np.random.default_rng(42).logseries(np.array(new[] { 1.0, 0.5 }))),
                ("a <= 1 or a contains NaNs", () => np.random.default_rng(42).zipf(np.array(new[] { 1.0, 2.0 }))),
                ("scale < 0", () => np.random.default_rng(42).exponential(np.array(new[] { -0.0, 1.0 }))),
                ("lam value too large", () => np.random.default_rng(42).poisson(np.array(new[] { double.NaN, 1.0 }))),
                ("shape < 0", () => np.random.default_rng(42).gamma(np.array(new[] { -0.0, 1.0 }))),
                ("a < 0", () => np.random.default_rng(42).weibull(np.array(new[] { -0.0, 1.0 }))),
            };
            foreach (var (expected, call) in rejected)
                Assert.AreEqual(expected, Assert.ThrowsException<ValueError>(() => call()).Message);
        }

        /// <summary>
        ///     The legacy strict bounds at their neighbouring doubles, against LP64 NumPy: the geometric inversion
        ///     overflows to C's indefinite long, logseries near <c>p = 1</c> keeps its 64-bit count, and the bounds
        ///     themselves are rejected with NumPy's array texts.
        /// </summary>
        [TestMethod]
        public void Constraints_StrictBounds_AcceptTheNeighbouringDouble_LegacyLP64()
        {
            // LP64 NumPy (NumSharp's legacy long): the geometric inversion overflows to C's indefinite long, and the
            // logseries near p = 1 keeps its 64-bit count.
            var rs = np.random.RandomState(42);
            using (var p = np.array(new[] { double.Epsilon, 1.0 }))
            using (var r = rs.geometric(p))
                AssertInt64(r, new long[] { 2 }, long.MinValue, 1);
            Assert.AreEqual(0x3FE76C7E8F1E6751UL, NextSample(rs));
            rs = np.random.RandomState(42);
            using (var p = np.array(new[] { System.Math.BitDecrement(1.0), 0.5 }))
            using (var r = rs.logseries(p))
                AssertInt64(r, new long[] { 2 }, 1474262878646771, 1);
            Assert.AreEqual(0x3FE32835D6632CB0UL, NextSample(rs));

            (string expected, Func<NDArray> call)[] rejected =
            {
                ("lam value too large", () => np.random.RandomState(42).poisson(np.array(new[] { double.NaN, 1.0 }))),
                ("n < 0", () => np.random.RandomState(42).binomial(np.array(new long[] { -1, 3 }), NDArray.Scalar(0.5))),
                ("a < 0", () => np.random.RandomState(42).weibull(np.array(new[] { -0.0, 1.0 }))),
            };
            foreach (var (expected, call) in rejected)
                Assert.AreEqual(expected, Assert.ThrowsException<ValueError>(() => call()).Message);
        }

        /// <summary>
        ///     The parameters at which a sampler's rejection loop can never accept a draw are REJECTED on every path —
        ///     RandomState and Generator; the scalar path (a double, a 0-d array) and the array path with the bad value in
        ///     the range scan's vector body, in its scalar tail, and in F-ordered (used in place), strided and broadcast
        ///     (copied) parameters — with NumPy 2.4.2's texts (324 + 12 calls probed: every one raises there).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The legacy logseries at <c>p = 1</c>: <c>r = log(1 - p) = -inf</c> makes <c>q = 1 - exp(r*U) = 1</c>, so every
        ///     candidate count <c>floor(1 + log(V)/log(q))</c> is <c>-inf</c> and is rejected. zipf (both APIs) at
        ///     <c>a &lt;= 1</c> or NaN: every candidate <c>X = floor(U^(-1/(a-1)))</c> is <c>+inf</c>, 0 or NaN and is
        ///     rejected. NumPy's C samplers loop forever there too (its Generator logseries as well) — its parameter checks
        ///     keep these inputs out — so the checks are ALL that separates them from a hang, and a check that regresses
        ///     makes the call hang rather than fail: an in-progress build whose array check read <c>p &lt;= 1</c>
        ///     (<c>AllInRange(arr, 0.0, 1.0)</c> where <c>p &lt; 1</c> is <c>AllInRange(arr, 0.0, BitDecrement(1.0))</c>) spun
        ///     an oracle test host for 84 CPU-hours (2026-09-26 → 2026-10-01) on the corpus case
        ///     <c>rnd/logseries/bcast:viol:p&gt;=1/857</c>, <c>p = [0.5, 1.0]</c>. (NumSharp's Generator logseries returns 2
        ///     at an unchecked <c>p = 1</c> instead — its <c>log1p(-1)</c> is NaN, see <c>Generator.Log1p</c> — a wrong answer
        ///     this test reports as "returned instead of raising".)
        ///     </para>
        ///     <para>
        ///     So the calls run on a background thread under a deadline, and the test FAILS — naming the call — when one does
        ///     not return. Nothing can stop the abandoned thread (.NET has no thread abort); it spins on until the test host
        ///     exits, which a background thread does not delay.
        ///     </para>
        /// </remarks>
        [TestMethod]
        public void Constraints_LoopGuardingBounds_AreRejectedOnEveryPath()
        {
            double posNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000), negNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL));
            // NumPy's scalar path (a Python float or a 0-d array) says "is NaN", its array path "contains NaNs".
            var samplers = new (string dist, double fill, string scalarText, string arrayText, double[] bad)[]
            {
                ("logseries", 0.5, "p < 0, p >= 1 or p is NaN", "p < 0, p >= 1 or p contains NaNs",
                    new[] { 1.0, System.Math.BitIncrement(1.0), 2.0, double.PositiveInfinity, posNaN, negNaN, -double.Epsilon, -1.0, double.NegativeInfinity }),
                ("zipf", 2.0, "a <= 1 or a is NaN", "a <= 1 or a contains NaNs",
                    new[] { 1.0, System.Math.BitDecrement(1.0), 0.5, 0.0, -0.0, -1.0, double.NegativeInfinity, posNaN, negNaN }),
            };
            // (length, position): a one-element tail, positions inside the first vector of a 256-bit scan (n >= 4), its
            // scalar tail (n = 5, 9) and a later vector body (n = 17) — the AVX-512 tail too (n = 9, position 8).
            var positions = new (int n, int at)[] { (1, 0), (2, 1), (4, 0), (4, 3), (5, 4), (9, 8), (17, 5) };

            // The whole matrix runs on ONE background thread that publishes the call it is in; the test thread waits.
            var failures = new System.Collections.Generic.List<string>();
            string current = "(not started)";
            var worker = Task.Factory.StartNew(() =>
            {
                foreach (var (dist, fill, scalarText, arrayText, bad) in samplers)
                    foreach (string api in new[] { "RandomState", "Generator" })
                        foreach (double v in bad)
                        {
                            string value = $"{v:R} (0x{BitConverter.DoubleToInt64Bits(v):X16})";
                            Expect($"{api}.{dist}({value})", scalarText, () => DrawScalar(api, dist, v));
                            Expect($"{api}.{dist}(0-d {value})", scalarText, () =>
                            {
                                using var a = NDArray.Scalar(v);
                                return DrawArray(api, dist, a);
                            });
                            foreach (var (n, at) in positions)
                                Expect($"{api}.{dist}(array n={n}, {value} at {at})", arrayText, () =>
                                {
                                    using var a = np.full(new Shape(n), fill);
                                    a.SetDouble(v, at);
                                    return DrawArray(api, dist, a);
                                });
                        }

                // The parameter conversion's other branches, with the boundary value itself: an F-ordered parameter is
                // scanned in place, a strided or broadcast one is copied dense first.
                foreach (var (dist, fill, _, arrayText, bad) in samplers)
                    foreach (string api in new[] { "RandomState", "Generator" })
                    {
                        double v = bad[0];
                        Expect($"{api}.{dist}(F-ordered, {v:R} at [1, 2])", arrayText, () =>
                        {
                            using var a = np.full(new Shape(3, 3), fill);
                            a.SetDouble(v, 2, 1);
                            using var t = a.T;
                            return DrawArray(api, dist, t);
                        });
                        Expect($"{api}.{dist}(strided, {v:R} at 3)", arrayText, () =>
                        {
                            using var a = np.full(new Shape(10), fill);
                            a.SetDouble(v, 6);
                            using var s = a["::2"];
                            return DrawArray(api, dist, s);
                        });
                        Expect($"{api}.{dist}(broadcast {v:R} to (4,))", arrayText, () =>
                        {
                            using var one = np.full(new Shape(1), v);
                            using var b = np.broadcast_to(one, new Shape(4));
                            return DrawArray(api, dist, b);
                        });
                    }
            }, System.Threading.CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

            // ~350 calls take milliseconds; 60 s (the [Timeout] of the sibling tests) only ever runs out on a call that never returns.
            if (!worker.Wait(TimeSpan.FromSeconds(60)))
                Assert.Fail($"{System.Threading.Volatile.Read(ref current)} did not return within 60 s: the constraint check let a value " +
                            "through that the sampler's rejection loop never accepts (the thread is abandoned, spinning).");
            worker.GetAwaiter().GetResult();   // a harness exception, if any, surfaces with its own stack
            Assert.AreEqual(0, failures.Count, "\n" + string.Join("\n", failures));

            // One call: it must raise NumPy's ValueError with NumPy's text. Runs on the worker thread only.
            void Expect(string label, string expected, Func<NDArray> call)
            {
                System.Threading.Volatile.Write(ref current, label);
                try
                {
                    using var r = call();
                    failures.Add($"{label}: returned instead of raising ValueError('{expected}')");
                }
                catch (ValueError e) when (e.Message == expected)
                {
                }
                catch (Exception e)
                {
                    failures.Add($"{label}: {e.GetType().Name}('{e.Message}') instead of ValueError('{expected}')");
                }
            }
        }

        /// <summary>One draw through the scalar (<c>double</c>) overload of logseries or zipf, on a fresh seed-42 receiver.</summary>
        /// <param name="api"><c>"RandomState"</c> (legacy) or <c>"Generator"</c> (PCG64).</param>
        /// <param name="dist"><c>"logseries"</c> or <c>"zipf"</c>.</param>
        /// <param name="v">The parameter (<c>p</c> or <c>a</c>).</param>
        /// <returns>The draw (a 0-d int64 array).</returns>
        /// <exception cref="ValueError">The parameter violates the sampler's constraint.</exception>
        private static NDArray DrawScalar(string api, string dist, double v) => (api, dist) switch
        {
            ("RandomState", "logseries") => np.random.RandomState(42).logseries(v),
            ("RandomState", _) => np.random.RandomState(42).zipf(v),
            (_, "logseries") => np.random.default_rng(42).logseries(v),
            _ => np.random.default_rng(42).zipf(v),
        };

        /// <summary>One draw through the array-parameter (<c>NDArray</c>) overload of logseries or zipf, on a fresh seed-42 receiver.</summary>
        /// <param name="api"><c>"RandomState"</c> (legacy) or <c>"Generator"</c> (PCG64).</param>
        /// <param name="dist"><c>"logseries"</c> or <c>"zipf"</c>.</param>
        /// <param name="p">The parameter array (a 0-d array takes NumPy's scalar path).</param>
        /// <returns>The draws (int64, the parameter's shape).</returns>
        /// <exception cref="ValueError">An element violates the sampler's constraint.</exception>
        private static NDArray DrawArray(string api, string dist, NDArray p) => (api, dist) switch
        {
            ("RandomState", "logseries") => np.random.RandomState(42).logseries(p),
            ("RandomState", _) => np.random.RandomState(42).zipf(p),
            (_, "logseries") => np.random.default_rng(42).logseries(p),
            _ => np.random.default_rng(42).zipf(p),
        };

        // ---------------------------------------------------------------- draw-free thresholds NumPy never returns from

        /// <summary>
        ///     The legacy <c>vonmises</c> draw-free threshold is exactly <c>2^511</c>, where <c>4 * kappa^2</c>
        ///     overflows and NumPy never returns: the fill must equal the scalar calls there and must not read ahead
        ///     past those positions.
        /// </summary>
        [TestMethod]
        public void Legacy_VonmisesOverflowKappa_IsTwoPow511_AndDrawsNothingFromThere()
        {
            // 4 * kappa * kappa overflows exactly from 2^511: there the legacy envelope is NaN and NumPy loops forever, so the
            // expectation is self-consistency — the broadcast fill must equal the scalar calls in sequence, and a drawing
            // position followed by draw-free ones must not read ahead of the stream.
            double t = NumPyRandom.LegacyVonmisesOverflowKappa, below = System.Math.BitDecrement(t);
            Assert.AreEqual(0x5FE0000000000000L, BitConverter.DoubleToInt64Bits(t));
            Assert.IsTrue(double.IsPositiveInfinity(4 * t * t));
            Assert.IsTrue(double.IsFinite(4 * below * below));

            double[] mu = { 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
            double[] kappa = { 1.0, t, t, double.PositiveInfinity, t, t, t, t };
            var viaArrays = np.random.RandomState(42);
            var viaScalars = np.random.RandomState(42);
            using var m = np.array(mu);
            using var k = np.array(kappa);
            using var got = viaArrays.vonmises(m, k);
            for (int i = 0; i < mu.Length; i++)
                using (var one = viaScalars.vonmises(mu[i], kappa[i]))
                    Assert.AreEqual(BitConverter.DoubleToInt64Bits(one.GetAtIndex<double>(0)), BitConverter.DoubleToInt64Bits(got.GetAtIndex<double>(i)), $"position {i}");
            Assert.AreEqual(NextSample(viaScalars), NextSample(viaArrays));
        }

        /// <summary>
        ///     The legacy <c>zipf</c> draw-free threshold is exactly <c>a = 1025</c>, where <c>pow(2, a - 1)</c>
        ///     overflows and NumPy never returns: the fill must equal the scalar calls there and must not read ahead
        ///     past those positions.
        /// </summary>
        [TestMethod]
        public void Legacy_ZipfDrawFreeThreshold_Is1025_AndDrawsNothingFromThere()
        {
            // pow(2, a - 1) overflows exactly from a = 1025 (the largest double below 1024 still gives a finite power), where
            // NumPy's legacy zipf never returns: the fill must equal the scalar calls and must not read ahead.
            Assert.IsTrue(double.IsFinite(System.Math.Pow(2.0, System.Math.BitDecrement(1025.0) - 1.0)));
            Assert.IsTrue(double.IsPositiveInfinity(System.Math.Pow(2.0, 1025.0 - 1.0)));

            double[] a = { 2.0, 1025.0, 1025.0, 3.5, 1025.0, 1e300, 1025.0, 1025.0, 1025.0 };
            var viaArrays = np.random.RandomState(42);
            var viaScalars = np.random.RandomState(42);
            using var arr = np.array(a);
            using var got = viaArrays.zipf(arr);
            for (int i = 0; i < a.Length; i++)
                using (var one = viaScalars.zipf(a[i]))
                    Assert.AreEqual(one.GetAtIndex<long>(0), got.GetAtIndex<long>(i), $"position {i}");
            Assert.AreEqual(NextSample(viaScalars), NextSample(viaArrays));
        }

        // ---------------------------------------------------------------- runs: shared setups are bit-neutral

        /// <summary><paramref name="count"/> copies of <paramref name="v"/>.</summary>
        /// <param name="v">The value.</param>
        /// <param name="count">How many.</param>
        /// <returns>The run.</returns>
        private static double[] Run(double v, int count)
        {
            var r = new double[count];
            Array.Fill(r, v);
            return r;
        }

        /// <summary>The concatenation of <paramref name="parts"/>.</summary>
        /// <param name="parts">The pieces, in order.</param>
        /// <returns>One array.</returns>
        private static double[] Cat(params double[][] parts) => parts.SelectMany(p => p).ToArray();

        /// <summary>
        ///     Setups shared across a run of equal parameters — and the memos built for runs of 16 or more — are
        ///     bit-neutral: every position of long runs, short runs and single values equals the scalar call with its
        ///     parameters, and the stream ends where the scalar calls leave it.
        /// </summary>
        [TestMethod]
        public void Broadcast_RunsShareSetups_YetEqualTheScalarCallsInSequence()
        {
            // Runs long enough for the shared setups and memos (>= 16: PoissonSetup with the hoisted rejection log, zipf's
            // power memo, HRUA's memo, the urn's ratio table), runs of a few (PoissonSetup), and single values (NumPy's
            // per-call statements in locals): every position must still be exactly the scalar call with its parameters.
            double[] lam = Cat(Run(25.0, 20), Run(5.0, 3), new[] { 0.5, 60.0, 60.0, 60.0, 60.0, 12.0 }, Run(33.0, 17));
            double[] lamZero = Cat(lam, new[] { 0.0, 7.0 });
            double[] zipfA = Cat(Run(2.0, 20), new[] { 3.0 }, Run(2.5, 2), Run(2.0, 17));
            double[] geomP = Cat(Run(0.2, 20), Run(0.5, 20), new[] { 0.1, 0.9 });
            double[] nbN = Cat(Run(3.0, 20), new[] { 0.5 }, Run(7.5, 3));
            long[] good = { 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60, 50, 50, 50, 80, 80 };
            long[] urnGood = { 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40, 40 };

            void Check(string what, Func<NDArray> broadcast, Func<int, NDArray> scalar, Func<ulong> next, Func<ulong> nextScalar, int n, bool isInt)
            {
                using var got = broadcast();
                for (int i = 0; i < n; i++)
                    using (var one = scalar(i))
                        if (isInt)
                            Assert.AreEqual(one.GetAtIndex<long>(0), got.GetAtIndex<long>(i), $"{what} position {i}");
                        else
                            Assert.AreEqual(BitConverter.DoubleToInt64Bits(one.GetAtIndex<double>(0)), BitConverter.DoubleToInt64Bits(got.GetAtIndex<double>(i)), $"{what} position {i}");
                Assert.AreEqual(nextScalar(), next(), $"{what}: stream position");
            }

            foreach (var values in new[] { lam, lamZero })
            {
                var ga = np.random.default_rng(7); var gb = np.random.default_rng(7);
                using var arr = np.array(values);
                Check("gen poisson", () => ga.poisson(arr), i => gb.poisson(values[i]), () => NextRandom(ga), () => NextRandom(gb), values.Length, true);
                var ra = np.random.RandomState(7); var rb = np.random.RandomState(7);
                Check("legacy poisson", () => ra.poisson(arr), i => rb.poisson(values[i]), () => NextSample(ra), () => NextSample(rb), values.Length, true);
            }
            {
                var ga = np.random.default_rng(7); var gb = np.random.default_rng(7);
                using var arr = np.array(zipfA);
                Check("gen zipf", () => ga.zipf(arr), i => gb.zipf(zipfA[i]), () => NextRandom(ga), () => NextRandom(gb), zipfA.Length, true);
                var ra = np.random.RandomState(7); var rb = np.random.RandomState(7);
                Check("legacy zipf", () => ra.zipf(arr), i => rb.zipf(zipfA[i]), () => NextSample(ra), () => NextSample(rb), zipfA.Length, true);
            }
            {
                var ga = np.random.default_rng(7); var gb = np.random.default_rng(7);
                using var arr = np.array(geomP);
                Check("gen geometric", () => ga.geometric(arr), i => gb.geometric(geomP[i]), () => NextRandom(ga), () => NextRandom(gb), geomP.Length, true);
                var ra = np.random.RandomState(7); var rb = np.random.RandomState(7);
                Check("legacy geometric", () => ra.geometric(arr), i => rb.geometric(geomP[i]), () => NextSample(ra), () => NextSample(rb), geomP.Length, true);
            }
            {
                var ga = np.random.default_rng(7); var gb = np.random.default_rng(7);
                using var arr = np.array(nbN);
                using var p = NDArray.Scalar(0.4);
                Check("gen negative_binomial", () => ga.negative_binomial(arr, p), i => gb.negative_binomial(nbN[i], 0.4), () => NextRandom(ga), () => NextRandom(gb), nbN.Length, true);
                var ra = np.random.RandomState(7); var rb = np.random.RandomState(7);
                Check("legacy negative_binomial", () => ra.negative_binomial(arr, p), i => rb.negative_binomial(nbN[i], 0.4), () => NextSample(ra), () => NextSample(rb), nbN.Length, true);
            }
            foreach (var (goods, bad, sample) in new[] { (good, 100L, 20L), (urnGood, 25L, 7L) })
            {
                using var g = np.array(goods);
                using var b = NDArray.Scalar(bad);
                using var s = NDArray.Scalar(sample);
                var ga = np.random.default_rng(7); var gb = np.random.default_rng(7);
                Check($"gen hypergeometric nsample={sample}", () => ga.hypergeometric(g, b, s), i => gb.hypergeometric(goods[i], bad, sample), () => NextRandom(ga), () => NextRandom(gb), goods.Length, true);
                var ra = np.random.RandomState(7); var rb = np.random.RandomState(7);
                Check($"legacy hypergeometric nsample={sample}", () => ra.hypergeometric(g, b, s), i => rb.hypergeometric(goods[i], bad, sample), () => NextSample(ra), () => NextSample(rb), goods.Length, true);
            }
        }

        /// <summary>
        ///     Runs compare parameters BITWISE (<c>-0.0</c> and <c>+0.0</c> split a run, only the same NaN pattern
        ///     continues one), a stride-0 parameter is constant over its chunk, and a multi-parameter run ends where
        ///     ANY parameter changes.
        /// </summary>
        [TestMethod]
        public unsafe void RunEnd_ComparesBitwise_AndAStrideZeroParameterIsOneRun()
        {
            // -0.0 and +0.0 end a run (a setup may depend on the sign); the same NaN pattern continues one, another ends it.
            var otherNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000001);
            double[] v = { 1.0, 1.0, -0.0, 0.0, 0.0, double.NaN, double.NaN, otherNaN, 2.0 };
            fixed (double* p = v)
            {
                Assert.AreEqual(2, RandomBroadcast.RunEnd(0, v.Length, p, 1));
                Assert.AreEqual(3, RandomBroadcast.RunEnd(2, v.Length, p, 1));
                Assert.AreEqual(5, RandomBroadcast.RunEnd(3, v.Length, p, 1));
                Assert.AreEqual(7, RandomBroadcast.RunEnd(5, v.Length, p, 1));
                Assert.AreEqual(8, RandomBroadcast.RunEnd(7, v.Length, p, 1));
                Assert.AreEqual(9, RandomBroadcast.RunEnd(8, v.Length, p, 1));
                // A parameter broadcast along the chunk (stride 0) is one run over the whole chunk.
                Assert.AreEqual(9, RandomBroadcast.RunEnd(0, 9, p, 0));
                // Two parameters: a run ends where EITHER changes, and a stride-0 partner never ends one.
                double[] w = { 5.0, 5.0, 5.0, 6.0 };
                fixed (double* q = w)
                {
                    Assert.AreEqual(2, RandomBroadcast.RunEnd(0, 4, q, 1, p, 1));
                    Assert.AreEqual(3, RandomBroadcast.RunEnd(0, 4, q, 1, p, 0));
                    Assert.AreEqual(4, RandomBroadcast.RunEnd(0, 4, q, 0, p, 0));
                }
            }
            long[] g = { 7, 7, 7, 7 }, b = { 3, 3, 3, 3 }, s = { 2, 2, 9, 9 };
            fixed (long* pg = g, pb = b, ps = s)
            {
                Assert.AreEqual(2, RandomBroadcast.RunEnd(0, 4, pg, 1, pb, 1, ps, 1));
                Assert.AreEqual(4, RandomBroadcast.RunEnd(2, 4, pg, 1, pb, 1, ps, 1));
                Assert.AreEqual(4, RandomBroadcast.RunEnd(0, 4, pg, 0, pb, 0, ps, 0));
            }
        }

        // ---------------------------------------------------------------- conversion: which layouts are used in place

        /// <summary>
        ///     A parameter that already has the dtype and is C- or F-contiguous is used IN PLACE (NumPy's
        ///     <c>PyArray_FROM_OTF</c> copies nothing there); any other layout is copied densely. Either way the draws
        ///     read the parameter in the output's C order: an F-ordered or offset parameter draws exactly what its
        ///     C-ordered copy draws, and the whole-array scans see exactly the view's elements.
        /// </summary>
        [TestMethod]
        public void Convert_ContiguousParameters_AreUsedInPlace_AndDrawLikeTheirCopies()
        {
            using var c = np.arange(24.0).reshape(4, 6) + 1.0;          // C-contiguous, values 1..24 (valid means/shapes)
            using var f = c.T;                                            // (6, 4) F-contiguous view of it
            Assert.IsTrue(f.Shape.IsFContiguous && !f.Shape.IsContiguous, "precondition: an F-only layout");
            using (var p = RandomParam.Float64(c))
                Assert.IsTrue(ReferenceEquals(p.Array, c), "a C-contiguous float64 parameter is used in place");
            using (var p = RandomParam.Float64(f))
                Assert.IsTrue(ReferenceEquals(p.Array, f), "an F-contiguous float64 parameter is used in place");
            using (var strided = c[":, ::2"])
            using (var p = RandomParam.Float64(strided))
            {
                Assert.IsFalse(ReferenceEquals(p.Array, strided), "a strided view is copied");
                Assert.IsTrue(p.Array.Shape.IsContiguous || p.Array.Shape.IsFContiguous, "the copy is dense");
            }
            using (var ints = np.arange(24).reshape(4, 6).T)
            using (var p = RandomParam.Float64(ints))
                Assert.IsTrue(p.Array.Shape.IsFContiguous && !p.Array.Shape.IsContiguous,
                    "a converting copy keeps the input's F order (astype's K order): converted parameters are dense, not C-ordered");

            // The draws read the parameter in the OUTPUT's C order through its strides: an F-ordered parameter used in
            // place draws exactly what its C-ordered copy draws, run-structured samplers included, on both APIs.
            using var fAsC = np.ascontiguousarray(f);
            using var two = NDArray.Scalar(2.0);
            {
                var ga = np.random.default_rng(3); var gb = np.random.default_rng(3);
                using var viaF = ga.poisson(f);
                using var viaC = gb.poisson(fAsC);
                CollectionAssert.AreEqual(viaC.ToArray<long>(), viaF.ToArray<long>(), "gen poisson");
                Assert.AreEqual(NextRandom(gb), NextRandom(ga), "gen poisson: stream position");
            }
            {
                var ra = np.random.RandomState(3); var rb = np.random.RandomState(3);
                using var viaF = ra.gamma(f, two);
                using var viaC = rb.gamma(fAsC, two);
                CollectionAssert.AreEqual(Bits(viaC), Bits(viaF), "legacy gamma");
                Assert.AreEqual(NextSample(rb), NextSample(ra), "legacy gamma: stream position");
            }

            // An F-contiguous view at an OFFSET is used in place too; the whole-array checks must see exactly its
            // elements — a negative value just outside the view passes, one inside is rejected.
            using var outside = np.arange(24.0).reshape(4, 6) + 1.0;
            outside[0, 5] = -1.0;                                          // row 0: outside the view below
            using var fOutside = outside.T[":, 1:3"];                      // rows 1..2 of the base, F-contiguous, offset 6
            Assert.IsTrue(fOutside.Shape.IsFContiguous && fOutside.Shape.offset != 0, "precondition: an offset F-contiguous view");
            using (var p = RandomParam.Float64(fOutside))
                Assert.IsTrue(ReferenceEquals(p.Array, fOutside), "an offset F-contiguous parameter is used in place");
            using (var r = np.random.default_rng(3).exponential(fOutside))
                Assert.AreEqual(12, r.size);
            using var inside = np.arange(24.0).reshape(4, 6) + 1.0;
            inside[2, 0] = -1.0;                                           // row 2: inside the view below
            using var fInside = inside.T[":, 1:3"];
            Assert.AreEqual("scale < 0", Assert.ThrowsException<ValueError>(() => np.random.default_rng(3).exponential(fInside)).Message);
        }
    }
}
