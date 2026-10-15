using System;
using System.Numerics;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Pins for the random-API wholeness pass (docs/plans/random-oracle-coverage.md, "pass 2"): NumPy's call SHAPES that
    ///     did not compile or bound the wrong overload, NumPy's array-bounds <c>integers</c>/<c>randint</c>, and the seed that
    ///     silently was not random.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The oracle invokes every overload by reflection, so it proves what each overload DOES but not that a NumPy call
    ///     spelled verbatim in C# BINDS it. Every test here is first a compile-time proof — the spelling is the NumPy one — and
    ///     then a value check against NumPy 2.4.2 (the legacy integer cases against the LP64 Linux build, NumSharp's model).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class RandomApiWholenessTests
    {
        /// <summary>
        ///     NumPy's <c>normal(loc=0.0, scale=1.0, size=None)</c> has no required parameter, so <c>np.random.normal(size=3)</c>
        ///     is a common spelling — which did not compile for the nine legacy samplers of that shape (the size overload had no
        ///     defaults, the defaulted overload no size). NumPy's <c>RandomState(42).X(size=3)</c> values.
        /// </summary>
        [TestMethod]
        public void LegacyAllDefaultSamplers_TakeASizeOnly()
        {
            np.random.RandomState(42).normal(size: 3).ToArray<double>().Should().Equal(0.4967141530112327, -0.13826430117118466, 0.6476885381006925);
            np.random.RandomState(42).uniform(size: 3).ToArray<double>().Should().Equal(0.3745401188473625, 0.9507143064099162, 0.7319939418114051);
            np.random.RandomState(42).exponential(size: 3).ToArray<double>().Should().Equal(0.4692680899768591, 3.010121430917521, 1.3167456935454493);
            np.random.RandomState(42).poisson(size: 3).ToArray<long>().Should().Equal(1L, 2L, 0L);
            np.random.RandomState(42).gumbel(size: 3).ToArray<double>().Should().Equal(0.7565810534288353, -1.101980420444563, -0.27516330885803314);
            np.random.RandomState(42).laplace(size: 3).ToArray<double>().Should().Equal(-0.28890917477433936, 2.316974250357576, 0.6235985129855038);
            np.random.RandomState(42).logistic(size: 3).ToArray<double>().Should().Equal(-0.5127882653574256, 2.9595797554777894, 1.0047626522764483);
            np.random.RandomState(42).lognormal(size: 3).ToArray<double>().Should().Equal(1.6433127155860012, 0.8708684897640193, 1.9111182426600926);
            np.random.RandomState(42).rayleigh(size: 3).ToArray<double>().Should().Equal(0.968780769810032, 2.453618320325116, 1.622803557763816);
        }

        /// <summary>
        ///     A keyword subset that skips a leading parameter (<c>uniform(high=5.0, size=2)</c>) binds the size overload
        ///     with NumPy's default for the skipped one; the all-arguments call still binds the one-draw overload.
        /// </summary>
        [TestMethod]
        public void LegacyAllDefaultSamplers_TakeAnyKeywordSubset()
        {
            np.random.RandomState(42).uniform(high: 5.0, size: 2).ToArray<double>().Should().Equal(1.8727005942368125, 4.75357153204958);
            np.random.RandomState(42).normal(scale: 2.0, size: 2).ToArray<double>().Should().Equal(0.9934283060224653, -0.2765286023423693);
            // No size at all: one draw, 0-d, from either overload (NumPy's size=None and size=() draw the same value).
            using var one = np.random.RandomState(42).normal();
            one.ndim.Should().Be(0);
            one.GetDouble().Should().Be(0.4967141530112327);
            using var both = np.random.RandomState(42).normal(0.0, 1.0);
            both.GetDouble().Should().Be(0.4967141530112327);
        }

        /// <summary>
        ///     numpy.random's classes through the module — <c>np.random.PCG64(42)</c>,
        ///     <c>np.random.Generator(np.random.PCG64(seed))</c>, <c>np.random.SeedSequence(42)</c> — the idiomatic NumPy
        ///     spellings, which had no C# counterpart (only <c>new PCG64(42)</c>). Each factory builds exactly what the
        ///     constructor builds; the values are NumPy's.
        /// </summary>
        [TestMethod]
        public void ModuleClasses_AreFactoriesOfTheConstructors()
        {
            np.random.MT19937(42).random_raw(3).ToArray<ulong>().Should().Equal(2327846034UL, 3904886566UL, 2661450408UL);
            np.random.PCG64(42).random_raw(3).ToArray<ulong>().Should().Equal(14276969152011380360UL, 8095878257575067585UL, 15838336090824644132UL);
            np.random.PCG64DXSM(42).random_raw(3).ToArray<ulong>().Should().Equal(12329818062196000797UL, 125530269004142706UL, 12137922674892001441UL);
            np.random.Philox(42).random_raw(3).ToArray<ulong>().Should().Equal(1587852024645073290UL, 2611271723512893552UL, 4982337093617253890UL);
            np.random.SFC64(42).random_raw(3).ToArray<ulong>().Should().Equal(9775594601838723485UL, 6977463094773878866UL, 17439770048677797496UL);
            np.random.Philox(key: 7).random_raw(2).ToArray<ulong>().Should().Equal(16086915834549238692UL, 5448529601018347655UL);
            np.random.SeedSequence(42).generate_state(4).ToArray<uint>().Should().Equal(3444837047u, 2669555309u, 2046530742u, 3581440988u);
            np.random.Generator(np.random.PCG64(42)).random(3).ToArray<double>().Should().Equal(0.7739560485559633, 0.4388784397520523, 0.8585979199113825);
            np.random.Generator(bit_generator: np.random.PCG64(seed: 42)).random(1).GetDouble().Should().Be(0.7739560485559633);
            np.random.SeedSequence(entropy: 42, spawn_key: new[] { 1, 2 }, pool_size: 8, n_children_spawned: 3).n_children_spawned.Should().Be(3u);

            // The factories return the constructor's object, not a wrapper.
            np.random.PCG64(1).Should().BeOfType<PCG64>();
            np.random.SeedSequence(1).Should().BeOfType<SeedSequence>();
            ((Action)(() => np.random.Generator(null))).Should().Throw<AttributeError>();
        }

        /// <summary>
        ///     A bare <c>null</c> seed — NumPy's <c>None</c> — was ambiguous between the <c>int[]</c>/<c>long[]</c>/<c>uint[]</c>
        ///     overloads (no array converts to another); each entry point now names the overload that takes it, and every one
        ///     draws fresh OS entropy.
        /// </summary>
        [TestMethod]
        public void NullLiteralSeeds_BindAndDrawFreshEntropy()
        {
            AssertFresh(() => new PCG64(null).random_raw(4));
            AssertFresh(() => new MT19937(null).random_raw(4));
            AssertFresh(() => new PCG64DXSM(seed: null).random_raw(4));
            AssertFresh(() => new Philox(null).random_raw(4));
            AssertFresh(() => new SFC64(null).random_raw(4));
            AssertFresh(() => new SeedSequence(null).generate_state(4));
            AssertFresh(() => new SeedSequence(entropy: null).generate_state(4));
            AssertFresh(() => np.random.default_rng(null).random(4));
            AssertFresh(() => np.random.default_rng(seed: null).random(4));
            AssertFresh(() => np.random.RandomState(null).random_sample(4));
            AssertFresh(() => np.random.PCG64(null).random_raw(4));
            AssertFresh(() => np.random.SeedSequence(null).generate_state(4));
            AssertFresh(() =>
            {
                var rs = np.random.RandomState(1);
                rs.seed(null);
                return rs.random_sample(4);
            });
            AssertFresh(() =>
            {
                var mt = new MT19937(1);
                mt._legacy_seeding(null);
                return mt.random_raw(4);
            });
        }

        /// <summary>
        ///     A null seed ARRAY was the empty list: <c>new SeedSequence((int[])null)</c> — and through it every engine's and
        ///     <c>default_rng</c>'s array overload — seeded NumPy's <c>SeedSequence([])</c>, the SAME stream every run, where a
        ///     null means Python's <c>None</c> (fresh OS entropy) everywhere else. The oracle's entropy cases mask the state,
        ///     so they could not see it.
        /// </summary>
        [TestMethod]
        public void TypedNullSeedArrays_DrawFreshEntropy_NotTheEmptyList()
        {
            AssertFresh(() => new SeedSequence((int[])null).generate_state(2));
            AssertFresh(() => new SeedSequence((long[])null).generate_state(2));
            AssertFresh(() => new SeedSequence((uint[])null).generate_state(2));
            AssertFresh(() => new PCG64((int[])null).random_raw(2));
            AssertFresh(() => new MT19937((long[])null).random_raw(2));
            AssertFresh(() => new SFC64((uint[])null).random_raw(2));
            AssertFresh(() => np.random.default_rng((int[])null).random(2));

            // NumPy's SeedSequence([]).generate_state(2): the stream a null array used to produce.
            new SeedSequence(new int[0]).generate_state(2).ToArray<uint>().Should().Equal(2968811710u, 3677149159u);
            new SeedSequence((int[])null).generate_state(2).ToArray<uint>().Should().NotEqual(new[] { 2968811710u, 3677149159u });
        }

        /// <summary>
        ///     <c>size: default</c> — the C# spelling of NumPy's explicit <c>size=None</c> (a <see cref="Shape"/> is a struct, so
        ///     <c>null</c> cannot convert) — was ambiguous for the legacy samplers with <c>int[]</c>/<c>long[]</c>/<c>long</c>
        ///     size shims, and for <c>multivariate_normal</c> it bound the <c>long</c> shim: a ZERO-length size, an empty
        ///     <c>(0, 2)</c> result where NumPy draws one vector.
        /// </summary>
        [TestMethod]
        public void SizeDefault_BindsTheNumPyShapedOverload()
        {
            var mean = new[] { 0.5, -1.0 };
            var cov = new double[,] { { 2.0, 0.3 }, { 0.3, 1.0 } };
            np.random.RandomState(42).multivariate_normal(mean, cov, size: default, check_valid: "warn", tol: 1e-8).shape
                .Should().Equal(2L);
            np.random.RandomState(42).pareto(3.0, size: default).ndim.Should().Be(0);
            np.random.RandomState(42).power(2.5, size: default).ndim.Should().Be(0);
            np.random.RandomState(42).logseries(0.6, size: default).ndim.Should().Be(0);
            np.random.RandomState(42).hypergeometric(10, 7, 8, size: default).ndim.Should().Be(0);
            np.random.RandomState(42).dirichlet(new[] { 1.0, 2.0, 3.0 }, size: default).shape.Should().Equal(3L);
            np.random.RandomState(42).multinomial(10, new[] { 0.2, 0.3, 0.5 }, size: default).shape.Should().Equal(3L);
            // The shims still bind for callers that pass them, with the same meaning as the Shape overload.
            np.random.RandomState(42).pareto(3.0, new[] { 2, 2 }).shape.Should().Equal(2L, 2L);
            np.random.RandomState(42).pareto(3.0, 3L).shape.Should().Equal(3L);
        }

        /// <summary>
        ///     <c>Generator.integers</c> with ARRAY bounds (NumPy's <c>_rand_&lt;dtype&gt;_broadcast</c>), which NumSharp lacked:
        ///     per-position bounds, every dtype's width, the closed interval, and float bounds truncated. NumPy's
        ///     <c>default_rng(42)</c> values.
        /// </summary>
        [TestMethod]
        public void Integers_ArrayBounds_MatchNumPy()
        {
            Generator G() => np.random.default_rng(42);
            G().integers(np.array(new long[] { 0, 10 }), np.array(new long[] { 5, 20 })).ToArray<long>().Should().Equal(0L, 17L);
            G().integers(np.arange(0, 20), 200, dtype: np.uint8).ToArray<byte>().Should().Equal(
                new byte[] { 106, 30, 169, 19, 160, 155, 152, 199, 116, 133, 81, 74, 38, 93, 154, 116, 113, 32, 27, 147 });
            G().integers(np.arange(-5, 5), np.arange(0, 10), dtype: np.int16).ToArray<short>().Should().Equal(
                new short[] { -5, -4, 1, 1, 3, 3, 2, 4, 3, 6 });
            using (var bits = G().integers(0, np.ones(40, np.int64) * 2, dtype: np.bool_))
            {
                bits.ToArray<bool>().Should().Equal(new[]
                {
                    false, false, false, true, false, false, false, true, false, true, true, false, false, true, false, false,
                    true, false, false, true, true, false, true, true, false, true, true, false, true, false, false, false,
                    true, false, true, true, false, false, true, true,
                });
            }
            G().integers(np.array(new long[] { 5, 0 }), np.array(new long[] { 5, 10 }), endpoint: true).ToArray<long>().Should().Equal(5L, 0L);
            G().integers(np.array(new[] { 0.5, 1.7 }), np.array(new[] { 5.9, 10.2 }), dtype: np.int32).ToArray<int>().Should().Equal(0, 7);
        }

        /// <summary>
        ///     NumPy's two broadcast quirks, reproduced: a size SMALLER than the bounds' broadcast takes the first positions
        ///     (no error), and 64-bit float bounds in a non-C layout are scrambled by NumPy's element-by-element conversion
        ///     into a layout-keeping <c>empty_like</c> — so F-ordered float bounds draw different numbers than F-ordered int
        ///     bounds with the same values.
        /// </summary>
        [TestMethod]
        public void Integers_ArrayBounds_ReproduceNumPysBroadcastQuirks()
        {
            Generator G() => np.random.default_rng(42);
            G().integers(np.zeros(new Shape(2, 3), np.int64), 10, size: 3).ToArray<long>().Should().Equal(0L, 7L, 6L);
            G().integers(np.asfortranarray(np.array(new double[,] { { 0, 1, 2 }, { 3, 4, 5 } })), 100).ToArray<long>()
                .Should().Equal(8L, 77L, 66L, 44L, 45L, 86L);
            G().integers(np.asfortranarray(np.array(new long[,] { { 0, 1, 2 }, { 3, 4, 5 } })), 100).ToArray<long>()
                .Should().Equal(8L, 77L, 66L, 45L, 45L, 86L);
        }

        /// <summary>
        ///     The legacy <c>randint</c> with array bounds: NumPy's masked sampler per position, in NumSharp's LP64 C
        ///     <c>long</c> (NumPy 2.4.2 on Linux: <c>RandomState(42).randint([0, 10], [5, 20])</c> is <c>[3, 17]</c>, int64 —
        ///     and the float F-order scramble shows through the same way).
        /// </summary>
        [TestMethod]
        public void Randint_ArrayBounds_MatchLinuxNumPy()
        {
            using var r = np.random.RandomState(42).randint(np.array(new long[] { 0, 10 }), np.array(new long[] { 5, 20 }));
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(3L, 17L);
            np.random.RandomState(42).randint(np.asfortranarray(np.array(new double[,] { { 0, 1, 2 }, { 3, 4, 5 } })), 100)
                .ToArray<long>().Should().Equal(51L, 94L, 18L, 72L, 63L, 25L);
        }

        /// <summary>
        ///     The array path's rejections, with NumPy's texts and check order: an empty interval anywhere, the one-argument
        ///     form's <c>high &lt;= 0</c>, bounds outside the dtype, NaN / infinity through Python's <c>int()</c>, and a size the
        ///     bounds do not broadcast with (the multi-iterator's mismatch, arg 2 being the output).
        /// </summary>
        [TestMethod]
        public void Integers_ArrayBounds_RaiseNumPysErrors()
        {
            var g = np.random.default_rng(1);
            ((Action)(() => g.integers(np.array(new long[] { 5, 0 }), np.array(new long[] { 5, 10 })))).Should().Throw<ValueError>().WithMessage("low >= high");
            ((Action)(() => g.integers(np.array(new long[] { 0, 10 })))).Should().Throw<ValueError>().WithMessage("high <= 0");
            ((Action)(() => g.integers(0, np.array(new long[] { 256, 10 }), dtype: np.uint8, endpoint: true))).Should().Throw<ValueError>().WithMessage("high is out of bounds for uint8");
            ((Action)(() => g.integers(np.array(new long[] { -1, 0 }), 5, dtype: np.uint64))).Should().Throw<ValueError>().WithMessage("low is out of bounds for uint64");
            ((Action)(() => g.integers(np.array(new[] { double.NaN, 1.0 }), 5))).Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            ((Action)(() => g.integers(0, np.array(double.PositiveInfinity)))).Should().Throw<OverflowException>().WithMessage("cannot convert float infinity to integer");
            ((Action)(() => g.integers(np.zeros(3, np.int64), 10, size: 2))).Should().Throw<ValueError>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 2 with shape (2,).");
            ((Action)(() => g.integers((NDArray)null))).Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            // A zero-size request returns before the bounds are read — even bad ones.
            g.integers(np.array(new long[] { 5 }), np.array(new long[] { 1 }), size: 0).size.Should().Be(0);
        }

        /// <summary>
        ///     Bounds past the <c>long</c>/<c>ulong</c> range, NumPy's Python ints: the full-range idiom
        ///     <c>integers(0, 2**64, dtype=np.uint64)</c> (an EXCLUSIVE <c>2**64</c>) and NumPy's out-of-bounds texts for
        ///     bigger values.
        /// </summary>
        [TestMethod]
        public void Integers_BigIntegerBounds_MatchNumPy()
        {
            var two64 = BigInteger.Pow(2, 64);
            np.random.default_rng(42).integers(0, two64, 3, dtype: np.uint64).ToArray<ulong>()
                .Should().Equal(14276969152011380360UL, 8095878257575067585UL, 15838336090824644132UL);
            np.random.default_rng(42).integers(0UL, ulong.MaxValue, 3, dtype: np.uint64, endpoint: true).ToArray<ulong>()
                .Should().Equal(14276969152011380360UL, 8095878257575067585UL, 15838336090824644132UL);
            ((Action)(() => np.random.default_rng(1).integers(0, BigInteger.Pow(2, 100)))).Should().Throw<ValueError>().WithMessage("high is out of bounds for int64");
            ((Action)(() => np.random.default_rng(1).integers(-BigInteger.Pow(2, 200), BigInteger.Pow(2, 200)))).Should().Throw<ValueError>().WithMessage("low is out of bounds for int64");
            // The legacy twin (LP64 Linux NumPy: RandomState(42).randint(0, 2**64, size=3, dtype=np.uint64)).
            np.random.RandomState(42).randint(0, two64, 3, dtype: np.uint64).ToArray<ulong>()
                .Should().Equal(6909045637428952499UL, 17537583593393853710UL, 13502904847239337031UL);
        }

        /// <summary>
        ///     Two independent evaluations of an entropy-seeded draw differ (a collision of 2+ 32/64-bit words is not a real
        ///     outcome) — the observable meaning of "fresh OS entropy".
        /// </summary>
        /// <param name="draw">Builds a freshly seeded object and draws from it.</param>
        private static void AssertFresh(Func<NDArray> draw)
        {
            using var a = draw();
            using var b = draw();
            a.ToString().Should().NotBe(b.ToString(), "an entropy-seeded draw must not repeat across constructions");
        }
    }
}
