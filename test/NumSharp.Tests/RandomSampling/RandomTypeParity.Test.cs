using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Pins the random suite's integer TYPES and API surface to NumPy 2.4.2's: every integer the suite accepts, stores
    ///     or returns uses NumPy's type (the C <c>long</c> modelled LP64 as int64, <c>npy_intp</c>/<c>Py_ssize_t</c> as
    ///     <see cref="long"/>, <c>uint32_t</c> as <see cref="uint"/>, a Python int as <see cref="BigInteger"/>), and the
    ///     members NumPy exposes exist with its names and positions.
    /// </summary>
    /// <remarks>
    ///     Every expected value and message was produced by NumPy 2.4.2 on Linux x86-64 — the LP64 build, whose C
    ///     <c>long</c> is 64-bit — and checked against the Windows build, which differs only in the dtype of the legacy
    ///     integers (int32) and in rejecting their bounds past <c>2**31</c>. Where NumPy's Windows and Linux messages differ,
    ///     the Linux one is pinned.
    /// </remarks>
    [TestClass]
    public class RandomTypeParityTests
    {
        /// <summary>A legacy generator seeded exactly like <c>np.random.RandomState(42)</c>.</summary>
        /// <returns>A freshly seeded legacy generator.</returns>
        private static NumPyRandom RS() => np.random.RandomState(42);

        /// <summary>Reads an integer array of any rank in logical C order as <see cref="long"/> values.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static long[] L(NDArray a)
        {
            var flat = a.flat;
            var r = new long[flat.size];
            for (long i = 0; i < flat.size; i++)
                r[i] = Convert.ToInt64(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads an unsigned array in logical C order as <see cref="ulong"/> values.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static ulong[] U(NDArray a)
        {
            var flat = a.flat;
            var r = new ulong[flat.size];
            for (long i = 0; i < flat.size; i++)
                r[i] = Convert.ToUInt64(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads a float array in logical C order as <see cref="double"/> values.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static double[] D(NDArray a)
        {
            var flat = a.flat;
            var r = new double[flat.size];
            for (long i = 0; i < flat.size; i++)
                r[i] = Convert.ToDouble(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Asserts that <paramref name="act"/> throws exactly <typeparamref name="TEx"/> with NumPy's message.</summary>
        /// <typeparam name="TEx">The expected exception type.</typeparam>
        /// <param name="act">The call.</param>
        /// <param name="message">The exact message.</param>
        private static void Throws<TEx>(Action act, string message) where TEx : Exception
            => act.Should().Throw<TEx>().Which.Message.Should().Be(message);

        // =====================================================================================
        //  The legacy C long: int64 (LP64) for every legacy integer
        // =====================================================================================

        /// <summary>
        ///     Every legacy integer mtrand types as <c>long</c> is int64: randint's default dtype, random_integers, the
        ///     permutation range, choice's indices, multinomial's counts and the discrete samplers — with NumPy's values.
        /// </summary>
        [TestMethod]
        public void LegacyIntegerSamplers_ReturnTheLp64Long()
        {
            void Pin(NDArray a, params long[] values)
            {
                a.dtype.Should().Be(np.int64);
                L(a).Should().Equal(values);
            }

            Pin(RS().randint(0, 10, new Shape(5)), 6, 3, 7, 4, 6);
            Pin(RS().randint(10, 5, new Shape(0)));
            Pin(RS().random_integers(5, 10, new Shape(6)), 8, 9, 7, 9, 9, 6);
            Pin(RS().permutation(10), 8, 1, 5, 0, 7, 2, 9, 4, 3, 6);
            Pin(RS().permutation(10L), 8, 1, 5, 0, 7, 2, 9, 4, 3, 6);
            Pin(RS().choice(5, new Shape(3)), 3, 4, 2);
            Pin(RS().choice(5, new Shape(3), replace: false), 1, 4, 2);
            Pin(RS().choice(5, new Shape(2, 2), replace: false), 1, 4, 2, 0);
            var p = new[] { 0.1, 0.2, 0.3, 0.2, 0.2 };
            Pin(RS().choice(5, new Shape(3), p: p), 2, 4, 3);
            Pin(RS().choice(5, new Shape(3), replace: false, p: p), 2, 4, 3);
            Pin(RS().multinomial(20, new[] { 0.2, 0.3, 0.5 }, 4L), 3, 10, 7, 5, 6, 9, 2, 5, 13, 1, 9, 10);
            Pin(RS().binomial(10, 0.5, new Shape(4)), 4, 8, 6, 5);
            Pin(RS().poisson(3, new Shape(4)), 4, 1, 3, 3);
            Pin(RS().geometric(0.3, new Shape(4)), 2, 9, 4, 3);
            Pin(RS().zipf(2.0, new Shape(4)), 1, 3, 1, 1);
            Pin(RS().hypergeometric(5, 5, 3, 4L), 2, 1, 2, 2);
            Pin(RS().logseries(0.5, 4L), 2, 1, 1, 1);
            Pin(RS().negative_binomial(3, 0.5, new Shape(4)), 2, 3, 1, 2);
        }

        /// <summary>
        ///     randint's default long accepts the whole int64 range: bounds past <c>2**31</c> (which Windows NumPy rejects as
        ///     <c>high is out of bounds for int32</c>) draw Linux NumPy's values, and only a bound past int64 is refused.
        /// </summary>
        [TestMethod]
        public void Randint_Lp64Bounds_MatchLinuxNumPy()
        {
            L(RS().randint(0, 2147483649L, new Shape(3))).Should().Equal(1608637542, 787846414, 670094950);
            L(RS().randint(0, 4294967296L, new Shape(3))).Should().Equal(1608637542, 3421126067, 4083286876);
            L(RS().randint(0, 1099511627776L, new Shape(3))).Should().Equal(441507790259, 395924837646, 458615280711);
            var top = RS().randint(0UL, 9223372036854775808UL, new Shape(3));
            top.dtype.Should().Be(np.int64);
            L(top).Should().Equal(6909045637428952499, 8314211556539077902, 4279532810384561223);
            Throws<ValueError>(() => RS().randint(0UL, 9223372036854775809UL, new Shape(3)), "high is out of bounds for int64");
            L(RS().randint(long.MinValue, 0, new Shape(3))).Should().Equal(-2314326399425823309, -909160480315697906, -4943839226470214585);
            L(RS().randint(-2147483649L, 0, new Shape(3))).Should().Equal(-538846107, -1359637235, -1477388699);
        }

        /// <summary>
        ///     choice over an integer population past int32: with replacement it is <c>randint(0, pop)</c> in the long dtype
        ///     (so a uint64 population above <c>2**63</c> is NumPy's int64 bounds error); without, it is
        ///     <c>permutation(pop)[:size]</c>, whose arange length follows NumPy's double computation at the int64 edge.
        /// </summary>
        [TestMethod]
        public void Choice_IntegerPopulationsPastInt32_MatchLinuxNumPy()
        {
            L(RS().choice(1099511627776L, new Shape(3))).Should().Equal(441507790259, 395924837646, 458615280711);
            L(RS().choice(long.MaxValue, new Shape(3))).Should().Equal(6909045637428952499, 8314211556539077902, 4279532810384561223);
            L(RS().choice(NDArray.Scalar(9223372036854775808UL), new Shape(3))).Should().Equal(6909045637428952499, 8314211556539077902, 4279532810384561223);
            Throws<ValueError>(() => RS().choice(NDArray.Scalar(9223372036854775813UL), new Shape(3)), "high is out of bounds for int64");
            Throws<ValueError>(() => RS().choice(NDArray.Scalar(ulong.MaxValue), new Shape(3)), "high is out of bounds for int64");
            // (double)(2**63 + 5) rounds to exactly 2**63, whose npy_intp cast is INT64_MIN on x86: an empty range, so the
            // shape assignment fails; 2**63 + 2048 rounds past it and the length computation itself raises.
            Throws<ValueError>(() => RS().choice(NDArray.Scalar(9223372036854775813UL), new Shape(3), replace: false),
                "cannot reshape array of size 0 into shape (3,)");
            Throws<ValueError>(() => RS().choice(NDArray.Scalar(9223372036854777856UL), new Shape(3), replace: false),
                "Maximum allowed size exceeded");
            Throws<ValueError>(() => RS().choice(long.MaxValue, new Shape(3), replace: false), "cannot reshape array of size 0 into shape (3,)");
            Throws<ValueError>(() => RS().choice(long.MaxValue, new Shape(3, 1), replace: false), "cannot reshape array of size 0 into shape (3,1)");
            var none = RS().choice(long.MaxValue, new Shape(0), replace: false);
            none.dtype.Should().Be(np.int64);
            none.shape.Should().Equal(0L);
        }

        /// <summary>
        ///     <c>permutation(n)</c> is <c>np.arange(n, dtype=np.long)</c>: non-positive lengths are empty, the band whose
        ///     double rounds to <c>2**63</c> is empty too (NumPy's x86 cast), and an addressable-but-huge one is the
        ///     allocator's error.
        /// </summary>
        [TestMethod]
        public void Permutation_ArangeLengthEdges_MatchNumPy()
        {
            foreach (long n in new[] { 0L, -3L, long.MaxValue, long.MaxValue - 511 })
            {
                var r = RS().permutation(n);
                r.dtype.Should().Be(np.int64);
                r.shape.Should().Equal(0L);
            }
            Throws<ValueError>(() => RS().permutation(4611686018427387904L),
                "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
        }

        /// <summary>
        ///     multinomial's <c>n</c> is a C <c>long</c>: an int64 count past <c>2**31</c> draws (Windows NumPy raises
        ///     <c>Python int too large to convert to C long</c>), and a negative one is <c>n &lt; 0</c>.
        /// </summary>
        [TestMethod]
        public void Multinomial_CountsPastInt32_MatchLinuxNumPy()
        {
            L(RS().multinomial(1099511627776L, new[] { 0.5, 0.5 })).Should().Equal(549755290311, 549756337465);
            L(RS().multinomial(2147483648L, new[] { 0.5, 0.5 })).Should().Equal(1073718686, 1073764962);
            Throws<ValueError>(() => RS().multinomial(-1, new[] { 0.5, 0.5 }), "n < 0");
        }

        // =====================================================================================
        //  Seed bookkeeping, sizes, random_sample(Shape)
        // =====================================================================================

        /// <summary>
        ///     <see cref="NumPyRandom.Seed"/> is a uint32 like NumPy's legacy seed, so a seed in <c>[2**31, 2**32 - 1]</c>
        ///     reads back as itself (it used to be an <c>int</c> and turned negative); the stream is NumPy's.
        /// </summary>
        [TestMethod]
        public void Seed_IsUInt32_AcrossTheFullLegacyRange()
        {
            var rs = np.random.RandomState(3000000000L);
            rs.Seed.Should().Be(3000000000u);
            D(rs.rand(2)).Should().Equal(0.5374315923900946, 0.3726939610994088);

            var top = np.random.RandomState((long)uint.MaxValue);
            top.Seed.Should().Be(uint.MaxValue);
            L(top.randint(0, 10, new Shape(3))).Should().Equal(3, 2, 7);

            var re = np.random.RandomState(1);
            re.seed(3000000000u);
            re.Seed.Should().Be(3000000000u);
            re.seed(new[] { 3000000000u, 5u });
            re.Seed.Should().Be(3000000000u);
            Throws<ValueError>(() => re.seed(4294967296L), "Seed must be between 0 and 2**32 - 1");
            Throws<ValueError>(() => re.seed(-1), "Seed must be between 0 and 2**32 - 1");
        }

        /// <summary>
        ///     NumPy's <c>random_sample(size)</c> family takes ONE size argument: <c>(2, 3)</c> is a shape, <c>()</c> a 0-d
        ///     array, <c>None</c> a single value — the <see cref="Shape"/> overloads, beside the loose-dimension ones.
        /// </summary>
        [TestMethod]
        public void RandomSample_SizeAsOneShape_MatchesNumPy()
        {
            var grid = RS().random_sample(new Shape(2, 3));
            grid.shape.Should().Equal(2L, 3L);
            D(grid).Should().Equal(0.3745401188473625, 0.9507143064099162, 0.7319939418114051,
                                   0.5986584841970366, 0.15601864044243652, 0.15599452033620265);
            var zeroD = RS().random_sample(Shape.Scalar);
            zeroD.ndim.Should().Be(0);
            D(zeroD).Should().Equal(0.3745401188473625);
            var none = RS().random_sample(default(Shape));
            none.ndim.Should().Be(0);
            D(none).Should().Equal(0.3745401188473625);
            foreach (var draw in new Func<NumPyRandom, NDArray>[] { r => r.random(new Shape(2)), r => r.ranf(new Shape(2)), r => r.sample(new Shape(2)) })
                D(draw(RS())).Should().Equal(0.3745401188473625, 0.9507143064099162);
        }

        /// <summary>
        ///     A NumPy <c>size</c> given as one integer is one <c>npy_intp</c> dimension: the single-integer size overloads
        ///     take a <see cref="long"/> (they took <c>int</c>) and produce NumPy's draws.
        /// </summary>
        [TestMethod]
        public void SizeOverloads_TakeOneInt64Dimension()
        {
            var dir = RS().dirichlet(new[] { 1.0, 2.0, 3.0 }, 4L);
            dir.shape.Should().Equal(4L, 3L);
            D(dir).Take(3).Should().Equal(0.10925336826910685, 0.14059620682266438, 0.7501504249082288);
            D(RS().pareto(3.0, 4L)).Should().Equal(0.16932036645405568, 1.7274682836709054, 0.5510238033601347, 0.35569492764272526);
            D(RS().power(3.0, 4L)).Should().Equal(0.7208298808378721, 0.983293895741171, 0.9012303919814004, 0.8428035972051934);
            var mvn = RS().multivariate_normal(new[] { 0.0, 0.0 }, new double[,] { { 1, 0 }, { 0, 1 } }, 3L);
            mvn.shape.Should().Equal(3L, 2L);
            D(mvn).Should().Equal(0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
                                  1.5230298564080254, -0.23415337472333597, -0.23413695694918055);
        }

        // =====================================================================================
        //  SeedSequence: NumPy's member types and coercion
        // =====================================================================================

        /// <summary>
        ///     SeedSequence's members carry NumPy's types: <c>pool_size</c> a <c>Py_ssize_t</c> (<see cref="long"/>),
        ///     <c>n_children_spawned</c> a <c>uint32_t</c>, <c>spawn_key</c> a tuple of Python ints (<see cref="BigInteger"/>
        ///     — a key element of any size is legal), <c>pool</c> and <c>generate_state</c> uint32 ndarrays.
        /// </summary>
        [TestMethod]
        public void SeedSequence_MemberTypes_AreNumPys()
        {
            var ss = new SeedSequence(42, new long[] { 1 });
            ((object)ss.pool_size).Should().BeOfType<long>().Which.Should().Be(4L);
            ((object)ss.n_children_spawned).Should().BeOfType<uint>().Which.Should().Be(0u);
            ss.spawn_key.Should().Equal(new BigInteger(1));
            ss.pool.dtype.Should().Be(np.uint32);
            U(ss.pool).Should().Equal(4069746273UL, 2430969354UL, 1995477525UL, 4234690265UL);
            ss.generate_state(2).dtype.Should().Be(np.uint32);

            var big = new SeedSequence(1, new[] { BigInteger.Pow(2, 70) });
            U(big.generate_state(4)).Should().Equal(48591740UL, 4075972060UL, 1127569541UL, 3504582927UL);
            big.ToString().Should().Be("SeedSequence(\n    entropy=1,\n    spawn_key=(1180591620717411303424,),\n)");
            big.spawn(1)[0].spawn_key.Should().Equal(BigInteger.Pow(2, 70), BigInteger.Zero);

            var nearFull = new SeedSequence(1, null, 4, uint.MaxValue - 1);
            nearFull.spawn(1)[0].spawn_key.Should().Equal(new BigInteger(uint.MaxValue - 1));
            nearFull.n_children_spawned.Should().Be(uint.MaxValue);
        }

        /// <summary>
        ///     <c>tuple(spawn_key)</c> as NumPy's coercion reads it: any iterable (a string is one of one-character strings),
        ///     nested sequences and integer arrays flattening to the same pool, strings by NumPy's decimal/hex rule, bools as
        ///     ints — and the coercion's refusals verbatim.
        /// </summary>
        [TestMethod]
        public void SeedSequence_SpawnKeyCoercion_MatchesNumPy()
        {
            var oneTwo = new[] { 1158480471UL, 3105224776UL };
            U(new SeedSequence(1, new long[] { 1, 2 }).generate_state(2)).Should().Equal(oneTwo);
            var fromString = new SeedSequence(1, "12"); // tuple("12") == ('1', '2')
            fromString.spawn_key.Should().Equal(new BigInteger(1), new BigInteger(2));
            U(fromString.generate_state(2)).Should().Equal(oneTwo);
            var nested = new SeedSequence(1, new object[] { new long[] { 1, 2 } }); // ((1, 2),) coerces like (1, 2)
            nested.spawn_key.Should().Equal(new BigInteger(1), new BigInteger(2));
            U(nested.generate_state(2)).Should().Equal(oneTwo);
            U(new SeedSequence(1, np.array(new long[,] { { 1, 2 }, { 3, 4 } })).generate_state(2)).Should().Equal(1734719507UL, 602822885UL);
            U(new SeedSequence(1, new long[] { 1, 2, 3, 4 }).generate_state(2)).Should().Equal(1734719507UL, 602822885UL);
            U(new SeedSequence(1, new object[] { true }).generate_state(2)).Should().Equal(1454127163UL, 941260221UL);
            new SeedSequence(1, np.array(new uint[] { 1, 2 })).spawn_key.Should().Equal(new BigInteger(1), new BigInteger(2));
            new SeedSequence(1, new List<int> { 3, 4 }).spawn_key.Should().Equal(new BigInteger(3), new BigInteger(4));
            // "0x..." is hex; a leading digit is DECIMAL - "012" is 12, not octal 10.
            var sixteen = new[] { 790193525UL, 2960481179UL };
            U(new SeedSequence(1, new object[] { "0x10" }).generate_state(2)).Should().Equal(sixteen);
            U(new SeedSequence(1, new long[] { 16 }).generate_state(2)).Should().Equal(sixteen);
            var twelve = new[] { 2540533722UL, 4073289697UL };
            U(new SeedSequence(1, new object[] { "12" }).generate_state(2)).Should().Equal(twelve);
            U(new SeedSequence(1, new object[] { "012" }).generate_state(2)).Should().Equal(twelve);
            U(new SeedSequence(1, new long[] { 10 }).generate_state(2)).Should().Equal(109501320UL, 445540211UL);

            Throws<TypeError>(() => new SeedSequence(1, 5), "'int' object is not iterable");
            Throws<ValueError>(() => new SeedSequence(1, "ab"), "unrecognized seed string");
            Throws<ValueError>(() => new SeedSequence(1, new object[] { "a" }), "unrecognized seed string");
            Throws<ValueError>(() => new SeedSequence(1, new object[] { "" }), "unrecognized seed string");
            Throws<ValueError>(() => new SeedSequence(1, new object[] { "0b11" }), "invalid literal for int() with base 10: '0b11'");
            Throws<ValueError>(() => new SeedSequence(1, new object[] { "0x" }), "invalid literal for int() with base 16: '0x'");
            Throws<TypeError>(() => new SeedSequence(1, new object[] { 1.5 }), "seed must be integer");
            Throws<TypeError>(() => new SeedSequence(1, new object[] { null }), "object of type 'NoneType' has no len()");
            Throws<ValueError>(() => new SeedSequence(1, new long[] { -1 }), "expected non-negative integer");
            Throws<TypeError>(() => new SeedSequence(1, np.array(new[] { true })), "object of type 'numpy.bool' has no len()");
            Throws<TypeError>(() => new SeedSequence(1, NDArray.Scalar(5L)), "iteration over a 0-d array");
            Throws<TypeError>(() => new SeedSequence(1, new object[] { NDArray.Scalar(5L) }), "len() of unsized object");
        }

        /// <summary>
        ///     String ENTROPY elements follow the same rule — the octal reading NumSharp used to apply changed every stream
        ///     seeded from a leading-zero string — while a bare string entropy is refused, as in NumPy.
        /// </summary>
        [TestMethod]
        public void SeedSequence_EntropyStrings_UseNumPysDecimalRule()
        {
            U(new SeedSequence((object)new object[] { "012" }).generate_state(2)).Should().Equal(2246587717UL, 2325260784UL);
            U(new SeedSequence(new long[] { 12 }).generate_state(2)).Should().Equal(2246587717UL, 2325260784UL);
            U(new SeedSequence((object)new object[] { "1_000 " }).generate_state(2)).Should().Equal(2989689681UL, 4191542321UL);
            U(new SeedSequence(new long[] { 1000 }).generate_state(2)).Should().Equal(2989689681UL, 4191542321UL);
            U(new SeedSequence((object)new object[] { "0x1f" }).generate_state(2)).Should().Equal(2523303487UL, 2774764755UL);
            U(new SeedSequence(new long[] { 31 }).generate_state(2)).Should().Equal(2523303487UL, 2774764755UL);
            Throws<ValueError>(() => new SeedSequence((object)new object[] { "a" }), "unrecognized seed string");
            Throws<ValueError>(() => new SeedSequence((object)new object[] { "-5" }), "unrecognized seed string");
            Throws<ValueError>(() => new SeedSequence((object)new object[] { "0X10" }), "invalid literal for int() with base 10: '0X10'");
            Throws<TypeError>(() => new SeedSequence((object)"0x10"), "SeedSequence expects int or sequence of ints for entropy not 0x10");
            Throws<ValueError>(() => new SeedSequence(-5), "expected non-negative integer");
        }

        /// <summary>
        ///     <c>generate_state</c> takes a 64-bit count and returns an ndarray; a uint64 request is DOUBLED before NumPy's
        ///     <c>np.zeros</c>, so its dimension error appears at half the single-width bound.
        /// </summary>
        [TestMethod]
        public void SeedSequence_GenerateState_LongCounts_MatchNumPysErrors()
        {
            var ss = new SeedSequence(1);
            var empty = ss.generate_state(0);
            empty.dtype.Should().Be(np.uint32);
            empty.shape.Should().Equal(0L);
            var wide = ss.generate_state(3, np.uint64);
            wide.dtype.Should().Be(np.uint64);
            U(wide).Should().Equal(7434755675892716031UL, 10007452063617845036UL, 5266248848659119178UL);
            const string tooBig = "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.";
            Throws<ValueError>(() => ss.generate_state(-1), "negative dimensions are not allowed");
            Throws<ValueError>(() => ss.generate_state(-1, np.uint64), "negative dimensions are not allowed");
            Throws<ValueError>(() => ss.generate_state(4611686018427387904L), tooBig);
            Throws<ValueError>(() => ss.generate_state(long.MaxValue), tooBig);
            Throws<ValueError>(() => ss.generate_state(2305843009213693952L, np.uint64), tooBig);
            Throws<ValueError>(() => ss.generate_state(4611686018427387903L, np.uint64), tooBig);
            Throws<ValueError>(() => ss.generate_state(4611686018427387904L, np.uint64), "Maximum allowed dimension exceeded");
            Throws<ValueError>(() => ss.generate_state(-4611686018427387904L, np.uint64), "negative dimensions are not allowed");
            Throws<ValueError>(() => ss.generate_state(long.MinValue, np.uint64), "Maximum allowed dimension exceeded");
            Throws<ValueError>(() => ss.generate_state(3, np.uint8), "only support uint32 or uint64");
        }

        /// <summary>
        ///     A custom <see cref="ISeedSequence"/> answers <c>generate_state</c> with an NDArray, as NumPy's protocol does;
        ///     the bit generators read its words (a wrong dtype or a short array is refused instead of read past).
        /// </summary>
        [TestMethod]
        public void CustomSeedSequence_WordsAreReadFromItsNDArray()
        {
            U(new PCG64(new Forwarding(new SeedSequence(5))).random_raw(new Shape(2))).Should().Equal(14849682912918955432UL, 14903876974979881461UL);
            U(new MT19937(new Forwarding(new SeedSequence(5))).random_raw(new Shape(2))).Should().Equal(1463718072UL, 2137291433UL);
            Throws<TypeError>(() => new PCG64(new Forwarding(new SeedSequence(5), wrongDtype: true)),
                "generate_state(n_words, np.uint64) must return a uint64 array of n_words words");
            Throws<TypeError>(() => new MT19937(new Forwarding(new SeedSequence(5), shortBy: 1)),
                "generate_state(n_words, np.uint32) must return a uint32 array of n_words words");
        }

        /// <summary>A user-written seed sequence that forwards to a real one, optionally returning a bad array.</summary>
        private sealed class Forwarding : ISeedSequence
        {
            private readonly SeedSequence _inner;
            private readonly bool _wrongDtype;
            private readonly long _shortBy;

            /// <summary>Wraps <paramref name="inner"/>.</summary>
            /// <param name="inner">The sequence whose words are returned.</param>
            /// <param name="wrongDtype">Return int64 words instead of the requested dtype.</param>
            /// <param name="shortBy">Return this many words fewer than requested.</param>
            public Forwarding(SeedSequence inner, bool wrongDtype = false, long shortBy = 0)
            {
                _inner = inner;
                _wrongDtype = wrongDtype;
                _shortBy = shortBy;
            }

            /// <inheritdoc/>
            public NDArray generate_state(long n_words, DType dtype = null)
            {
                NDArray words = _inner.generate_state(n_words - _shortBy, dtype);
                return _wrongDtype ? words.astype(np.int64) : words;
            }
        }

        // =====================================================================================
        //  Generator / bit generator surface
        // =====================================================================================

        /// <summary>
        ///     <c>Generator(bit_generator)</c>: NumPy's parameter name, its <c>AttributeError</c> for None (the constructor
        ///     reads <c>bit_generator.capsule</c>), and the public <c>_bit_generator</c> / <c>_poisson_lam_max</c>.
        /// </summary>
        [TestMethod]
        public void Generator_ConstructorAndAttributes_MatchNumPy()
        {
            Throws<AttributeError>(() => new Generator(null), "'NoneType' object has no attribute 'capsule'");
            var g = new Generator(bit_generator: new PCG64(1));
            ((double)g.random()).Should().Be(0.5118216247002567);
            Generator._poisson_lam_max.Should().Be(9.223372006484771e+18);
            NumPyRandom._poisson_lam_max.Should().Be(Generator._poisson_lam_max);
            ReferenceEquals(g._bit_generator, g.bit_generator).Should().BeTrue();
        }

        /// <summary>
        ///     NumPy's <c>permutation(x, axis=0)</c> and <c>choice(a, size, replace, p, axis, shuffle)</c>: for an integer
        ///     <c>x</c>/<c>a</c> the axis is accepted and ignored, and the fifth positional argument is <c>axis</c>, so a
        ///     ported positional call lands <c>shuffle</c> in the sixth slot.
        /// </summary>
        [TestMethod]
        public void Generator_PermutationAndChoice_AxisSlotIsNumPys()
        {
            L(new Generator(new PCG64(42)).permutation(10, axis: 5)).Should().Equal(5, 6, 0, 7, 3, 2, 4, 9, 1, 8);
            L(new Generator(new PCG64(42)).permutation(10)).Should().Equal(5, 6, 0, 7, 3, 2, 4, 9, 1, 8);
            L(new Generator(new PCG64(42)).choice(5, new Shape(3), true, null, 7)).Should().Equal(0, 3, 3);
            L(new Generator(new PCG64(42)).choice(5, new Shape(3), false, null, 7, false)).Should().Equal(0, 3, 4);
            L(new Generator(new PCG64(42)).choice(5, new Shape(3), false, null, 0, false)).Should().Equal(0, 3, 4);
            L(new Generator(new PCG64(42)).choice(5, new Shape(3), false)).Should().Equal(4, 0, 3);
        }

        /// <summary>
        ///     Every bit generator's seed parameter is named <c>seed</c> (NumPy's), a seed sequence is used as-is, and a null
        ///     one is NumPy's <c>seed=None</c> — fresh OS entropy — rather than an error.
        /// </summary>
        [TestMethod]
        public void BitGenerators_SeedParameter_IsNumPys()
        {
            U(new PCG64(seed: 5).random_raw(new Shape(2))).Should().Equal(14849682912918955432UL, 14903876974979881461UL);
            U(new PCG64(seed: new SeedSequence(5)).random_raw(new Shape(2))).Should().Equal(14849682912918955432UL, 14903876974979881461UL);
            U(new SFC64(seed: 5).random_raw(new Shape(2))).Should().Equal(12614612106463595544UL, 746723346124915927UL);
            U(new MT19937(seed: 5).random_raw(new Shape(2))).Should().Equal(1463718072UL, 2137291433UL);
            U(new MT19937(seed: new SeedSequence(5)).random_raw(new Shape(2))).Should().Equal(1463718072UL, 2137291433UL);
            U(new PCG64DXSM(seed: 5).random_raw(new Shape(2))).Should().Equal(18372055152263355045UL, 8489134709661829815UL);
            U(new Philox(seed: 5).random_raw(new Shape(2))).Should().Equal(15819541710926433573UL, 17064937530545973451UL);
            U(new Philox(seed: new SeedSequence(5)).random_raw(new Shape(2))).Should().Equal(15819541710926433573UL, 17064937530545973451UL);
            foreach (BitGenerator bg in new BitGenerator[]
            {
                new PCG64((ISeedSequence)null), new PCG64DXSM((ISeedSequence)null), new SFC64((ISeedSequence)null),
                new MT19937((ISeedSequence)null), new Philox((ISeedSequence)null),
            })
                bg.seed_seq.Should().BeOfType<SeedSequence>();
        }
    }

    /// <summary>
    ///     NumPy's module functions <c>get_bit_generator</c> / <c>set_bit_generator</c> and the module-level <c>seed</c>
    ///     over a hot-swapped singleton engine. These mutate the process-wide <c>np.random</c>, so the class is not
    ///     parallelized and every test restores the singleton's engine and state.
    /// </summary>
    [TestClass]
    [DoNotParallelize]
    public class RandomSingletonBitGeneratorTests
    {
        /// <summary>Reads an unsigned array in logical C order as <see cref="ulong"/> values.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static ulong[] U(NDArray a)
        {
            var flat = a.flat;
            var r = new ulong[flat.size];
            for (long i = 0; i < flat.size; i++)
                r[i] = Convert.ToUInt64(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Asserts that <paramref name="act"/> throws exactly <typeparamref name="TEx"/> with NumPy's message.</summary>
        /// <typeparam name="TEx">The expected exception type.</typeparam>
        /// <param name="act">The call.</param>
        /// <param name="message">The exact message.</param>
        private static void Throws<TEx>(Action act, string message) where TEx : Exception
            => act.Should().Throw<TEx>().Which.Message.Should().Be(message);

        /// <summary>
        ///     Runs <paramref name="body"/> against the singleton and puts its engine, state and seed bookkeeping back
        ///     afterwards, whatever happened: <see cref="NumPyRandom.set_bit_generator"/> restores the engine (and clears the
        ///     Gaussian cache), then the dict-form state — which works for any engine — restores the key/position and the
        ///     cached Gaussian, and <see cref="NumPyRandom.Seed"/> gets its recorded value back.
        /// </summary>
        /// <param name="body">The test body.</param>
        private static void WithSingleton(Action body)
        {
            BitGenerator engine = np.random.get_bit_generator();
            object state = np.random.get_state(legacy: false);
            uint seed = np.random.Seed;
            try
            {
                body();
            }
            finally
            {
                np.random.set_bit_generator(engine);
                np.random.set_state(state);
                np.random.Seed = seed;
            }
        }

        /// <summary>
        ///     <c>get_bit_generator</c> returns the singleton's engine itself; <c>set_bit_generator</c> swaps it (the module
        ///     functions then draw from the new engine) and discards the cached Gaussian, as NumPy's
        ///     <c>_initialize_bit_generator</c> does.
        /// </summary>
        [TestMethod]
        public void GetSetBitGenerator_SwapTheSingletonEngine()
        {
            WithSingleton(() =>
            {
                ReferenceEquals(np.random.get_bit_generator(), np.random._bit_generator).Should().BeTrue();
                var pcg = new PCG64(42);
                np.random.set_bit_generator(pcg);
                ReferenceEquals(np.random.get_bit_generator(), pcg).Should().BeTrue();
                np.random.random_sample(3).ToArray<double>().Should().Equal(0.7739560485559633, 0.4388784397520523, 0.8585979199113825);
                var ri = np.random.randint(0, 10, new Shape(5));
                ri.dtype.Should().Be(np.int64);
                ri.ToArray<long>().Should().Equal(5L, 1L, 0L, 3L, 3L);

                // The cached Gaussian belongs to the old engine: NumPy resets it on the swap.
                var mt = new MT19937(0);
                mt._legacy_seeding(42);
                np.random.set_bit_generator(mt);
                np.random.standard_normal(); // draws a pair, caches the second
                np.random.set_bit_generator(new PCG64(5));
                ((double)np.random.standard_normal()).Should().Be(0.537153394137919);
                ((double)np.random.RandomState(new PCG64(5)).standard_normal()).Should().Be(0.537153394137919);
            });
        }

        /// <summary>
        ///     <c>set_bit_generator(None)</c> is NumPy's <c>AttributeError</c>. NumPy assigns the attribute before it fails,
        ///     leaving its singleton holding <c>None</c>; here the engine is left as it was.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void SetBitGenerator_Null_IsAttributeError_AndChangesNothing()
        {
            WithSingleton(() =>
            {
                var pcg = new PCG64(1);
                np.random.set_bit_generator(pcg);
                Throws<AttributeError>(() => np.random.set_bit_generator(null), "'NoneType' object has no attribute 'capsule'");
                ReferenceEquals(np.random.get_bit_generator(), pcg).Should().BeTrue();
            });
        }

        /// <summary>
        ///     NumPy's MODULE <c>seed</c> on a hot-swapped singleton re-seeds the engine in place as
        ///     <c>type(engine)(seed).state</c> — for every engine type — with the SeedSequence's rules (no legacy
        ///     <c>2**32</c> cap, empty arrays legal, <c>expected non-negative integer</c> for a negative seed).
        /// </summary>
        [TestMethod]
        public void ModuleSeed_ReseedsASwappedSingletonEngine()
        {
            WithSingleton(() =>
            {
                var expected = new Dictionary<string, ulong[]>
                {
                    ["PCG64"] = new[] { 14849682912918955432UL, 14903876974979881461UL },
                    ["PCG64DXSM"] = new[] { 18372055152263355045UL, 8489134709661829815UL },
                    ["Philox"] = new[] { 15819541710926433573UL, 17064937530545973451UL },
                    ["SFC64"] = new[] { 12614612106463595544UL, 746723346124915927UL },
                };
                foreach (BitGenerator bg in new BitGenerator[] { new PCG64(1), new PCG64DXSM(1), new Philox(1), new SFC64(1) })
                {
                    np.random.set_bit_generator(bg);
                    np.random.seed(5);
                    ReferenceEquals(np.random.get_bit_generator(), bg).Should().BeTrue("the engine is re-seeded in place");
                    U(bg.random_raw(new Shape(2))).Should().Equal(expected[bg.GetType().Name]);
                }

                var pcg = new PCG64(1);
                np.random.set_bit_generator(pcg);
                np.random.seed(new[] { 1, 2, 3 });
                U(pcg.random_raw(new Shape(2))).Should().Equal(12368030237656201616UL, 2047858591075935483UL);
                np.random.seed(1099511627776L); // past the legacy 2**32 range: a SeedSequence seed
                U(pcg.random_raw(new Shape(1))).Should().Equal(6655761860222389633UL);
                np.random.seed(new int[0]);
                U(pcg.random_raw(new Shape(1))).Should().Equal(11749869230777074271UL);
                Throws<ValueError>(() => np.random.seed(-1), "expected non-negative integer");
                Throws<ValueError>(() => np.random.seed(new[] { -1 }), "expected non-negative integer");
                np.random.seed(); // fresh entropy, no error
            });
        }

        /// <summary>
        ///     The module <c>seed</c>'s engine branch leaves the cached Gaussian alone (only the engine state is replaced), so
        ///     the next normal is the cached second half of the previous pair — NumPy's observable behaviour.
        /// </summary>
        [TestMethod]
        public void ModuleSeed_OnASwappedEngine_KeepsTheCachedGaussian()
        {
            WithSingleton(() =>
            {
                np.random.set_bit_generator(new PCG64(3));
                ((double)np.random.standard_normal()).Should().Be(-0.14555610235430483);
                np.random.seed(9);
                ((double)np.random.standard_normal()).Should().Be(-0.22915539468468876);
                np.random.RandomState(new PCG64(3)).standard_normal(new Shape(2)).ToArray<double>()
                    .Should().Equal(-0.14555610235430483, -0.22915539468468876);
            });
        }

        /// <summary>
        ///     The engine branch is the MODULE function's only: any other RandomState over a non-MT19937 engine refuses to
        ///     re-seed (NumPy's method), and the MT19937 singleton keeps the legacy validation.
        /// </summary>
        [TestMethod]
        public void OnlyTheSingletonReseedsAnEngine_AndMT19937KeepsLegacySeeding()
        {
            WithSingleton(() =>
            {
                Throws<TypeError>(() => np.random.RandomState(new PCG64(1)).seed(5), "can only re-seed a MT19937 BitGenerator");
                Throws<TypeError>(() => np.random.RandomState(new PCG64(1)).seed(), "can only re-seed a MT19937 BitGenerator");
                np.random.set_bit_generator(new MT19937(0));
                Throws<ValueError>(() => np.random.seed(-1), "Seed must be between 0 and 2**32 - 1");
                Throws<ValueError>(() => np.random.seed(new int[0]), "Seed must be non-empty");
                np.random.seed(42);
                np.random.rand(1).ToArray<double>().Should().Equal(0.3745401188473625);
            });
        }
    }
}
