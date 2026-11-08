using System;
using System.Linq;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Reproductions (then regression pins) for the legacy <c>RandomState</c> (<see cref="NumPyRandom"/>)
    ///     divergences surfaced while turning <see cref="MT19937"/> into a real bit generator: unseeded
    ///     seeding, seed validation, <c>randint</c>'s per-dtype stream consumption and messages, and the
    ///     legacy <c>shuffle</c>/<c>permutation</c>/<c>choice</c> contracts. Every expected value and message
    ///     comes from running the same call on NumPy 2.4.2. The legacy APIs' default integer is the C <c>long</c>, which
    ///     NumSharp models LP64 (int64, Linux/macOS NumPy): the values below are identical on win-amd64, whose 32-bit
    ///     <c>long</c> only changes the dtype and rejects bounds past <c>2**31</c>.
    /// </summary>
    [TestClass]
    public class RandomStateAuditFixesTest
    {
        /// <summary>A legacy generator seeded exactly like <c>np.random.RandomState(42)</c>.</summary>
        /// <returns>A freshly seeded legacy generator.</returns>
        private static NumPyRandom RS() => np.random.RandomState(42);

        /// <summary>Reads an integer/bool array of any rank in logical C order as <see cref="long"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values (bool as 0/1).</returns>
        private static long[] L(NDArray a)
        {
            var flat = a.flat;
            var r = new long[flat.size];
            for (long i = 0; i < flat.size; i++)
            {
                object o = flat.GetAtIndex(i);
                r[i] = o is bool b ? (b ? 1 : 0) : Convert.ToInt64(o);
            }
            return r;
        }

        // =====================================================================================
        //  Unseeded RandomState: OS entropy through SeedSequence, NumPy's MT19937() layout
        // =====================================================================================

        /// <summary>
        ///     NumPy seeds an unseeded RandomState from 128 bits of OS entropy via <c>SeedSequence</c>; NumSharp
        ///     seeded from <c>Environment.TickCount</c>, so instances created in the same millisecond produced
        ///     IDENTICAL streams (20/20 back-to-back pairs in the audit).
        /// </summary>
        [TestMethod]
        public void Unseeded_BackToBackInstances_DoNotCollide()
        {
            for (int t = 0; t < 20; t++)
            {
                var a = L(np.random.RandomState().randint(0, 1 << 30, new Shape(4)));
                var b = L(np.random.RandomState().randint(0, 1 << 30, new Shape(4)));
                a.Should().NotEqual(b);
            }
        }

        /// <summary>
        ///     An unseeded RandomState's state has NumPy's <c>MT19937()</c> layout: <c>key[0] = 0x80000000</c>
        ///     (the non-zero guarantee) and <c>pos = 623</c> (the last index of the SeedSequence fill loop).
        /// </summary>
        [TestMethod]
        public void Unseeded_StateHasNumPyMT19937Layout()
        {
            var st = np.random.RandomState().get_state();
            st.Key[0].Should().Be(0x80000000u);
            st.Pos.Should().Be(623);
        }

        // =====================================================================================
        //  seed() validation (NumPy MT19937._legacy_seeding)
        // =====================================================================================

        /// <summary>An empty seed array is NumPy's <c>ValueError: Seed must be non-empty</c>; NumSharp silently seeded 0.</summary>
        [TestMethod]
        public void Seed_EmptyArray_Throws()
        {
            Action act = () => np.random.RandomState().seed(new uint[0]);
            act.Should().Throw<ValueError>().WithMessage("Seed must be non-empty");
        }

        // =====================================================================================
        //  randint: NumPy's masked bounded fills, per dtype width
        // =====================================================================================

        /// <summary>
        ///     <c>randint(0, 2, dtype=bool)</c> crashed with an <see cref="ArgumentException"/> (a generic
        ///     constraint in the fill dispatch rejected bool). NumPy draws one bit per value from a buffered word.
        /// </summary>
        [TestMethod]
        public void Randint_Bool_ByteExact()
        {
            L(RS().randint(0, 2, new Shape(8), np.bool_)).Should().Equal(0, 1, 1, 0, 0, 1, 1, 0);
        }

        /// <summary>
        ///     8- and 16-bit dtypes split one 32-bit word into 4 bytes / 2 halves (NumPy's
        ///     <c>buffered_bounded_masked_uint8/16</c>); NumSharp drew a whole word per value, so the streams
        ///     diverged from the second value on.
        /// </summary>
        [TestMethod]
        public void Randint_SubWordDtypes_ByteExact()
        {
            L(RS().randint(-100, 100, new Shape(6), np.int8)).Should().Equal(2, -5, 79, -39, -8, -97);
            L(RS().randint(0, 10, new Shape(6), np.uint8)).Should().Equal(6, 1, 3, 3, 2, 3);
            L(RS().randint(0, 1000, new Shape(6), np.int16)).Should().Equal(102, 993, 435, 860, 866, 270);
            L(RS().randint(0, 60000, new Shape(6), np.uint16)).Should().Equal(56422, 24545, 15795, 52202, 860, 38158);
            L(RS().randint(-128, 128, new Shape(6), np.int8)).Should().Equal(-26, 92, 97, -33, 51, -67);
            L(RS().randint(0, 256, new Shape(6), np.uint8)).Should().Equal(102, 220, 225, 95, 179, 61);
            L(RS().randint(0, 65536, new Shape(6), np.uint16)).Should().Equal(56422, 24545, 15795, 52202, 860, 62306);
        }

        /// <summary>
        ///     A range that fits 32 bits is drawn from 32-bit words even when the BOUNDS need 64 bits (NumPy's
        ///     <c>random_bounded_uint64_fill</c> tests the RANGE); NumSharp switched to two-word draws whenever
        ///     <c>high &gt; int.MaxValue</c>.
        /// </summary>
        [TestMethod]
        public void Randint_32BitRange_64BitBounds_ByteExact()
        {
            L(RS().randint(2147483648L, 2147483658L, new Shape(6), np.int64))
                .Should().Equal(2147483654, 2147483651, 2147483655, 2147483652, 2147483654, 2147483657);
            L(RS().randint(0, 4294967296L, new Shape(6), np.uint64))
                .Should().Equal(1608637542, 3421126067, 4083286876, 787846414, 3143890026, 3348747335);
            L(RS().randint(0, 3000000000L, new Shape(6), np.uint32))
                .Should().Equal(1608637542, 787846414, 2571218620, 2563451924, 670094950, 1914837113);
            L(RS().randint(0, 4294967296L, new Shape(6), np.uint32))
                .Should().Equal(1608637542, 3421126067, 4083286876, 787846414, 3143890026, 3348747335);
        }

        /// <summary>The paths that already matched (int32/int64 small ranges, a 40-bit range) stay byte-exact.</summary>
        [TestMethod]
        public void Randint_ExistingPaths_StayByteExact()
        {
            L(RS().randint(0, 100, new Shape(6), np.int32)).Should().Equal(51, 92, 14, 71, 60, 20);
            L(RS().randint(0, 100, new Shape(6), np.int64)).Should().Equal(51, 92, 14, 71, 60, 20);
            L(RS().randint(0, 100, new Shape(5))).Should().Equal(51, 92, 14, 71, 60);
            RS().randint(0, 100, new Shape(5)).dtype.Should().Be(np.int64);
            L(RS().randint(0, 1099511627776L, new Shape(6), np.int64))
                .Should().Equal(441507790259, 395924837646, 458615280711, 810017303572, 440001501305, 902372521174);
            foreach (var dt in new[] { np.int32, np.uint32, np.int64, np.uint64 })
                L(RS().randint(0, 2, new Shape(8), dt)).Should().Equal(0, 1, 0, 0, 0, 1, 0, 0);
            foreach (var dt in new[] { np.int8, np.uint8 })
                L(RS().randint(0, 2, new Shape(8), dt)).Should().Equal(0, 0, 1, 1, 1, 1, 0, 1);
            foreach (var dt in new[] { np.int16, np.uint16 })
                L(RS().randint(0, 2, new Shape(8), dt)).Should().Equal(0, 1, 1, 0, 0, 0, 0, 1);
        }

        /// <summary>
        ///     NumPy's <c>format_bounds_error</c>: the default single-argument case (low == 0) reads
        ///     <c>high &lt;= 0</c>; a bound one past the dtype is <c>high is out of bounds</c>; a float dtype is a
        ///     TypeError naming randint.
        /// </summary>
        [TestMethod]
        public void Randint_ErrorMessages_MatchNumPy()
        {
            ((Action)(() => RS().randint(0))).Should().Throw<ValueError>().WithMessage("high <= 0");
            ((Action)(() => RS().randint(-5))).Should().Throw<ValueError>().WithMessage("high <= 0");
            ((Action)(() => RS().randint(0, 257, new Shape(3), np.uint8))).Should().Throw<ValueError>().WithMessage("high is out of bounds for uint8");
            ((Action)(() => RS().randint(0, 2147483649L, new Shape(3), np.int32))).Should().Throw<ValueError>().WithMessage("high is out of bounds for int32");
            ((Action)(() => RS().randint(0, 5, new Shape(3), np.float64))).Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('float64') for randint");
        }

        /// <summary>
        ///     <c>high=-1</c> is a real bound, not "high omitted": <c>randint(-10, -1, 3)</c> draws from
        ///     <c>[-10, -1)</c> and <c>randint(5, -1)</c> is <c>low &gt;= high</c> (NumSharp used -1 as its
        ///     "None" sentinel and treated the latter as <c>randint(5)</c>).
        /// </summary>
        [TestMethod]
        public void Randint_MinusOneHigh_IsARealBound()
        {
            L(RS().randint(-10, -1, new Shape(3))).Should().Equal(-4, -7, -3);
            ((Action)(() => RS().randint(5, -1))).Should().Throw<ValueError>().WithMessage("low >= high");
        }

        /// <summary>A zero-size request returns an empty array before the bounds are checked (NumPy's <c>_rand_*</c>).</summary>
        [TestMethod]
        public void Randint_ZeroSize_EmptyBeforeBoundsCheck()
        {
            var e = RS().randint(10, 5, new Shape(0));
            e.shape.Should().Equal(0L);
            e.dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     Legacy <c>randint</c> only WARNS about a non-native byte order and then draws the native dtype
        ///     (Generator.integers raises instead).
        /// </summary>
        [TestMethod]
        public void Randint_NonNativeDtype_DrawsNative()
        {
            var r = RS().randint(0, 100, new Shape(3), np.dtype(">i4"));
            L(r).Should().Equal(51, 92, 14);
            r.dtype.isnative.Should().BeTrue();
        }

        /// <summary>
        ///     <c>random_integers</c> is <c>randint(low, high + 1, dtype='l')</c> and inherits every fix above. With the LP64
        ///     <c>long</c> the whole int64 range is drawable (Linux NumPy 2.4.2 values; Windows NumPy rejects the last three as
        ///     <c>high is out of bounds for int32</c> / <c>low is out of bounds for int32</c>).
        /// </summary>
        [TestMethod]
        public void RandomIntegers_ByteExact_AndBounds()
        {
            L(RS().random_integers(5, 10, new Shape(6))).Should().Equal(8, 9, 7, 9, 9, 6);
            Convert.ToInt64(RS().random_integers(5).GetAtIndex(0)).Should().Be(4);
            L(RS().random_integers(0, 2147483647L, new Shape(3))).Should().Equal(1608637542, 1273642419, 1935803228);
            // [0, 2**31] still takes one 32-bit masked word per draw (mask 2**32 - 1, rejecting words above 2**31).
            L(RS().random_integers(0, 2147483648L, new Shape(3))).Should().Equal(1608637542, 787846414, 670094950);
            // high + 1 == 2**63 is exact (NumPy's Python int): the int64 maximum is reachable, never wrapped.
            L(RS().random_integers(0, long.MaxValue, new Shape(3))).Should().Equal(6909045637428952499, 8314211556539077902, 4279532810384561223);
            L(RS().random_integers(long.MinValue, long.MinValue + 5, new Shape(3))).Should().Equal(-9223372036854775805, -9223372036854775804, -9223372036854775806);
            RS().random_integers(0, 2147483648L, new Shape(3)).dtype.Should().Be(np.int64);
        }

        // =====================================================================================
        //  Legacy shuffle / permutation
        // =====================================================================================

        /// <summary>NumPy evaluates <c>len(x)</c> first, so shuffling a 0-d array is a TypeError (not ArgumentException).</summary>
        [TestMethod]
        public void Shuffle_ZeroDim_IsTypeError()
        {
            ((Action)(() => RS().shuffle(NDArray.Scalar(5L)))).Should().Throw<TypeError>().WithMessage("len() of unsized object");
        }

        /// <summary>2-D rows and a strided 1-D view shuffle byte-identically to NumPy.</summary>
        [TestMethod]
        public void Shuffle_2D_And_Strided_ByteExact()
        {
            var x = np.arange(12).reshape(4, 3);
            RS().shuffle(x);
            L(x).Should().Equal(3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8);

            var baseArr = np.arange(10);
            RS().shuffle(baseArr["::2"]);
            L(baseArr).Should().Equal(2, 1, 8, 3, 4, 5, 0, 7, 6, 9);
        }

        /// <summary>
        ///     <c>permutation(int)</c> is <c>arange(x, dtype=result_type(x, np.long))</c> — int64 under the LP64 <c>long</c>
        ///     (int32 on win-amd64, same values) — and a 0-d array is <c>IndexError: x must be an integer or at least
        ///     1-dimensional</c>.
        /// </summary>
        [TestMethod]
        public void Permutation_DtypeValuesAndErrors()
        {
            var p = RS().permutation(10);
            p.dtype.Should().Be(np.int64);
            L(p).Should().Equal(8, 1, 5, 0, 7, 2, 9, 4, 3, 6);

            L(RS().permutation(np.array(new long[] { 10, 20, 30, 40, 50 }))).Should().Equal(20, 50, 30, 10, 40);
            L(RS().permutation(np.arange(12).reshape(4, 3))).Should().Equal(3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8);

            ((Action)(() => RS().permutation(NDArray.Scalar(5L)))).Should().Throw<IndexError>()
                .WithMessage("x must be an integer or at least 1-dimensional");
        }

        // =====================================================================================
        //  Legacy choice (mtrand.pyx RandomState.choice)
        // =====================================================================================

        /// <summary>
        ///     The legacy sampling paths, byte-exact: uniform (<c>randint(0, pop)</c>, the LP64 long: int64), without
        ///     replacement (<c>permutation(pop)[:size]</c> — NumSharp ignored <c>replace=False</c>), weighted
        ///     (<c>searchsorted(side='right')</c> — NumSharp used 'left') and weighted without replacement.
        /// </summary>
        [TestMethod]
        public void Choice_SamplingPaths_ByteExact()
        {
            var u = RS().choice(5, new Shape(3));
            u.dtype.Should().Be(np.int64);
            L(u).Should().Equal(3, 4, 2);
            Convert.ToInt64(RS().choice(5).GetAtIndex(0)).Should().Be(3);

            L(RS().choice(5, new Shape(3), replace: false)).Should().Equal(1, 4, 2);
            var nr = RS().choice(5, new Shape(2, 2), replace: false);
            nr.shape.Should().Equal(2L, 2L);
            L(nr).Should().Equal(1, 4, 2, 0);

            var p = new[] { 0.1, 0.2, 0.3, 0.2, 0.2 };
            var w = RS().choice(5, new Shape(3), p: p);
            w.dtype.Should().Be(np.int64);
            L(w).Should().Equal(2, 4, 3);
            L(RS().choice(5, new Shape(3), replace: false, p: p)).Should().Equal(2, 4, 3);
        }

        /// <summary>
        ///     Sampling from an array: <c>size=None</c> is one element (0-d), <c>size=()</c> a 0-d array, a sized
        ///     request the gathered elements; a 2-D <c>a</c> is rejected (the legacy API is 1-D only).
        /// </summary>
        [TestMethod]
        public void Choice_FromArray_ShapesAndValues()
        {
            var a = np.array(new long[] { 10, 20, 30 });
            var one = RS().choice(a);
            one.ndim.Should().Be(0);
            Convert.ToInt64(one.GetAtIndex(0)).Should().Be(30);

            var zeroD = RS().choice(a, Shape.Scalar);
            zeroD.ndim.Should().Be(0);
            Convert.ToInt64(zeroD.GetAtIndex(0)).Should().Be(30);

            L(RS().choice(a, new Shape(4))).Should().Equal(30, 10, 30, 30);

            ((Action)(() => RS().choice(np.arange(6).reshape(3, 2)))).Should().Throw<ValueError>().WithMessage("a must be 1-dimensional");
        }

        /// <summary>The legacy validation messages (they differ in wording from Generator.choice's).</summary>
        [TestMethod]
        public void Choice_ValidationMessages_MatchNumPy()
        {
            ((Action)(() => RS().choice(0))).Should().Throw<ValueError>().WithMessage("a must be greater than 0 unless no samples are taken");
            ((Action)(() => RS().choice(NDArray.Scalar(3.5)))).Should().Throw<ValueError>().WithMessage("a must be 1-dimensional or an integer");
            ((Action)(() => RS().choice(np.array(new long[0])))).Should().Throw<ValueError>().WithMessage("'a' cannot be empty unless no samples are taken");
            ((Action)(() => RS().choice(4, p: new[] { 0.5, 0.5 }))).Should().Throw<ValueError>().WithMessage("'a' and 'p' must have same size");
            ((Action)(() => RS().choice(2, p: new[] { double.NaN, 0.5 }))).Should().Throw<ValueError>().WithMessage("probabilities contain NaN");
            ((Action)(() => RS().choice(2, p: new[] { -0.5, 1.5 }))).Should().Throw<ValueError>().WithMessage("probabilities are not non-negative");
            ((Action)(() => RS().choice(2, p: new[] { 0.4, 0.5 }))).Should().Throw<ValueError>().WithMessage("probabilities do not sum to 1");
            ((Action)(() => RS().choice(3, new Shape(5), replace: false))).Should().Throw<ValueError>()
                .WithMessage("Cannot take a larger sample than population when 'replace=False'");
            ((Action)(() => RS().choice(3, new Shape(3), replace: false, p: new[] { 0.5, 0.5, 0.0 }))).Should().Throw<ValueError>()
                .WithMessage("Fewer non-zero entries in p than size");
        }
    }
}
