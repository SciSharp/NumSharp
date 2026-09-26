using System;
using System.Linq;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     The array-valued (broadcast) distribution parameters of both random APIs — the parts the oracle corpus
    ///     (random_parity / generator_parity, <c>params["bargs"]</c>) cannot pin: legacy outcomes that depend on the width of
    ///     C <c>long</c>, RandomState's cached Gaussian carried into the next call, the read-ahead edge where a position draws
    ///     nothing, the <c>size=()</c> contract, the overloads C# must resolve — plus the exact closed-range scans behind the
    ///     constraint checks, the draw-free thresholds NumPy never returns from (legacy <c>vonmises</c> from <c>2^511</c>,
    ///     legacy <c>zipf</c> from 1025), and the bit-neutrality of the setups a run of equal parameters shares.
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

        // ---------------------------------------------------------------- uniform: the NumPy-shaped array overload

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

        // ---------------------------------------------------------------- constraint bounds (NumPy 2.4.2 values)

        [TestMethod]
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

        // ---------------------------------------------------------------- draw-free thresholds NumPy never returns from

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
    }
}
