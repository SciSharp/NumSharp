using System;
using System.Numerics;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact pins for <see cref="MT19937"/> as a NumPy 2.4.2 bit generator: SeedSequence seeding, the legacy
    ///     seeding, <c>random_raw</c> (including its output=False quirk), <c>jumped</c>, the <c>state</c> contract and a
    ///     <see cref="Generator"/> driven by it. Every expected value was produced by the same call on NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class MT19937ParityTest
    {
        /// <summary>Reads an integer array of any rank in logical C order as <see cref="ulong"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values.</returns>
        private static ulong[] U(NDArray a)
        {
            var flat = a.flat;
            var r = new ulong[flat.size];
            for (long i = 0; i < flat.size; i++) r[i] = Convert.ToUInt64(flat.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads a numeric array of any rank in logical C order as <see cref="double"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values.</returns>
        private static double[] D(NDArray a)
        {
            var flat = a.flat;
            var r = new double[flat.size];
            for (long i = 0; i < flat.size; i++) r[i] = Convert.ToDouble(flat.GetAtIndex(i));
            return r;
        }

        // ---- NumPy 2.x seeding: SeedSequence -> key[0]=0x80000000, key[1:]=generate_state(624)[1:], pos=623 ----

        /// <summary><c>MT19937(42)</c> is SeedSequence-seeded (NOT the legacy integer seeding) and matches NumPy's raw stream.</summary>
        [TestMethod]
        public void Seed42_SeedSequenceSeeding_RawStream()
        {
            U(new MT19937(42).random_raw(new Shape(5))).Should().Equal(2327846034UL, 3904886566UL, 2661450408UL, 1733955692UL, 246401338UL);
        }

        /// <summary>The seeded key starts with the non-zero guard word and the position is 623 (NumPy's fill loop leaves it there).</summary>
        [TestMethod]
        public void Seed42_StateLayout()
        {
            var st = new MT19937(42).state;
            st.bit_generator.Should().Be("MT19937");
            st.pos.Should().Be(623);
            st.key.Length.Should().Be(624);
            st.key[0].Should().Be(2147483648u);
            st.key[1].Should().Be(2669555309u);
            st.key[2].Should().Be(2046530742u);
            st.key[3].Should().Be(3581440988u);
            st.key[620].Should().Be(3956913338u);
            st.key[623].Should().Be(96769712u);
        }

        /// <summary>Every seed form NumPy accepts routes through SeedSequence identically.</summary>
        [TestMethod]
        public void SeedForms_MatchNumPy()
        {
            U(new MT19937(0).random_raw(new Shape(3))).Should().Equal(2058676884UL, 2606108953UL, 1230491694UL);
            U(new MT19937(new[] { 1, 2, 3 }).random_raw(new Shape(3))).Should().Equal(3088909719UL, 2329501708UL, 4078501282UL);
            U(new MT19937(BigInteger.Pow(2, 64) + 5).random_raw(new Shape(3))).Should().Equal(1657405558UL, 3561744789UL, 947526056UL);
            U(new MT19937(new SeedSequence(7)).random_raw(new Shape(3))).Should().Equal(1315568259UL, 1873995810UL, 2718702504UL);
        }

        /// <summary>An unseeded MT19937 has NumPy's seeded layout (OS entropy through SeedSequence).</summary>
        [TestMethod]
        public void Unseeded_HasSeedSequenceLayout()
        {
            var st = new MT19937().state;
            st.pos.Should().Be(623);
            st.key[0].Should().Be(0x80000000u);
            new MT19937().seed_seq.Should().NotBeNull();
        }

        // ---- random_raw ----

        /// <summary>random_raw returns uint64: a 0-d array for size=None, the requested shape otherwise.</summary>
        [TestMethod]
        public void RandomRaw_Shapes()
        {
            var s = new MT19937(42).random_raw();
            s.ndim.Should().Be(0);
            s.dtype.Should().Be(np.uint64);
            Convert.ToUInt64(s.GetAtIndex(0)).Should().Be(2327846034UL);

            var m = new MT19937(42).random_raw(new Shape(2, 2));
            m.shape.Should().Equal(2L, 2L);
            U(m).Should().Equal(2327846034UL, 3904886566UL, 2661450408UL, 1733955692UL);

            var zeroD = new MT19937(42).random_raw(Shape.Scalar);
            zeroD.ndim.Should().Be(0);
            Convert.ToUInt64(zeroD.GetAtIndex(0)).Should().Be(2327846034UL);
        }

        /// <summary>
        ///     With output=False NumPy draws the SUM of the size's dimensions (a quirk of <c>np.asarray(size).sum()</c>),
        ///     so <c>(3, 4)</c> advances the stream by 7, exactly like <c>random_raw(7)</c>; size=None advances by one.
        /// </summary>
        [TestMethod]
        public void RandomRaw_OutputFalse_AdvancesBySumOfDims()
        {
            var a = new MT19937(42);
            a.random_raw(output: false).Should().BeNull();
            U(a.random_raw(new Shape(2))).Should().Equal(3904886566UL, 2661450408UL);

            var b = new MT19937(42);
            b.random_raw(new Shape(3, 4), output: false);
            U(b.random_raw(new Shape(2))).Should().Equal(1498159427UL, 3694075699UL);

            var p = new PCG64(42);
            p.random_raw(new Shape(3, 4), output: false);
            U(p.random_raw(new Shape(2))).Should().Equal(14500327064922265408UL, 2363279394319028499UL);

            ((Action)(() => new MT19937(1).random_raw(Shape.Scalar, output: false))).Should().Throw<TypeError>()
                .WithMessage("'numpy.float64' object cannot be interpreted as an integer");
        }

        // ---- Generator over MT19937 ----

        /// <summary>A Generator driven by MT19937 uses its next_uint64 (two words, high first) and next_double (27+26 bits).</summary>
        [TestMethod]
        public void GeneratorOverMT19937_ByteExact()
        {
            Generator G() => new Generator(new MT19937(42));
            D(G().random(new Shape(3))).Should().Equal(0.5419938930062744, 0.6196672126927824, 0.05736978170666862);
            D(G().random(new Shape(3), np.float32)).Should().Equal(0.5419938564300537, 0.9091772437095642, 0.6196671724319458);
            U(G().integers(0, 100, new Shape(5))).Should().Equal(54UL, 90UL, 61UL, 40UL, 5UL);
            U(G().integers(0, 256, new Shape(6), np.uint8)).Should().Equal(146UL, 28UL, 192UL, 138UL, 38UL, 215UL);
            U(G().integers(0, 1099511627776L, new Shape(3))).Should().Equal(595928584936UL, 681331304551UL, 63078742721UL);
            D(G().standard_normal(new Shape(3))).Should().Equal(-0.2965179310498389, 1.3413208289199272, -1.367054573795688);
            D(G().standard_exponential(new Shape(3))).Should().Equal(2.0483417889412423, 0.2111184384375238, 0.44158195359784275);
            U(G().permutation(10)).Should().Equal(1UL, 8UL, 5UL, 3UL, 7UL, 9UL, 4UL, 0UL, 6UL, 2UL);
            U(G().bytes(6)).Should().Equal(146UL, 28UL, 192UL, 138UL, 38UL, 215UL);
            G().ToString().Should().Be("Generator(MT19937)");
        }

        // ---- jumped ----

        /// <summary>jumped() advances a copy by 2**128 draws with NumPy's precomputed polynomial (the source is unchanged).</summary>
        [TestMethod]
        public void Jumped_MatchesNumPy()
        {
            var src = new MT19937(42);
            U(src.jumped().random_raw(new Shape(3))).Should().Equal(3877897575UL, 3377564461UL, 113905337UL);
            U(src.random_raw(new Shape(3))).Should().Equal(2327846034UL, 3904886566UL, 2661450408UL); // untouched

            U(new MT19937(42).jumped(2).random_raw(new Shape(3))).Should().Equal(3245795878UL, 3614751274UL, 811645896UL);
            new MT19937(42).jumped().state.pos.Should().Be(588);

            var advanced = new MT19937(42);
            advanced.random_raw(new Shape(5));
            U(advanced.jumped().random_raw(new Shape(3))).Should().Equal(3754380164UL, 483884169UL, 1788337741UL);

            var legacy = new MT19937();
            legacy._legacy_seeding(42);
            U(legacy.jumped().random_raw(new Shape(3))).Should().Equal(876706316UL, 669659531UL, 952691915UL);
        }

        /// <summary>A zero or negative jump count copies the state unchanged (NumPy's <c>range(jumps)</c> is empty).</summary>
        [TestMethod]
        public void Jumped_ZeroOrNegative_IsAPlainCopy()
        {
            U(new MT19937(42).jumped(0).random_raw(new Shape(3))).Should().Equal(2327846034UL, 3904886566UL, 2661450408UL);
            U(new MT19937(42).jumped(-1).random_raw(new Shape(3))).Should().Equal(2327846034UL, 3904886566UL, 2661450408UL);
        }

        /// <summary>The jumped copy carries a FRESH seed sequence (NumPy builds it with <c>self.__class__()</c>), not the parent's.</summary>
        [TestMethod]
        public void Jumped_CarriesAFreshSeedSequence()
        {
            var src = new MT19937(42);
            var j = src.jumped();
            j.seed_seq.Should().NotBeNull();
            ReferenceEquals(j.seed_seq, src.seed_seq).Should().BeFalse();
            j.seed_seq.entropy.Should().NotBe(42L);
        }

        // ---- _legacy_seeding ----

        /// <summary>Legacy integer/array seeding reproduces RandomState's stream, leaves pos at 624 and drops the seed sequence.</summary>
        [TestMethod]
        public void LegacySeeding_MatchesNumPy()
        {
            var a = new MT19937(1);
            a._legacy_seeding(42);
            a.seed_seq.Should().BeNull();
            a.state.pos.Should().Be(624);
            U(a.random_raw(new Shape(3))).Should().Equal(1608637542UL, 3421126067UL, 4083286876UL);

            var b = new MT19937(1);
            b._legacy_seeding(new[] { 1, 2, 3 });
            b.state.pos.Should().Be(624);
            U(b.random_raw(new Shape(3))).Should().Equal(2619334238UL, 1552691353UL, 3808334787UL);

            var c = new MT19937(1);
            c._legacy_seeding(new uint[] { 1, 2, 3 });
            U(c.random_raw(new Shape(2))).Should().Equal(2619334238UL, 1552691353UL);
        }

        /// <summary>The legacy seed validation and messages.</summary>
        [TestMethod]
        public void LegacySeeding_Validation()
        {
            ((Action)(() => new MT19937(1)._legacy_seeding(-1))).Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
            ((Action)(() => new MT19937(1)._legacy_seeding(4294967296L))).Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
            ((Action)(() => new MT19937(1)._legacy_seeding(new long[0]))).Should().Throw<ValueError>().WithMessage("Seed must be non-empty");
            ((Action)(() => new MT19937(1)._legacy_seeding(new long[] { 4294967296L }))).Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
            ((Action)(() => new MT19937(1)._legacy_seeding(new[] { -1 }))).Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
        }

        /// <summary>
        ///     <c>_legacy_seeding(None)</c> re-keys from OS entropy but LEAVES the position where it was (NumPy quirk: after
        ///     five draws from pos 623 the position is 4, and stays 4).
        /// </summary>
        [TestMethod]
        public void LegacySeedingNone_KeepsPosition()
        {
            var b = new MT19937(42);
            b.random_raw(new Shape(5));
            b.state.pos.Should().Be(4);
            b._legacy_seeding();
            b.state.pos.Should().Be(4);
            b.state.key[0].Should().Be(0x80000000u);
            b.seed_seq.Should().BeNull();
        }

        // ---- state contract ----

        /// <summary>A snapshot restores exactly, through the typed and the polymorphic property alike.</summary>
        [TestMethod]
        public void State_RoundTrips()
        {
            var a = new MT19937(42);
            a.random_raw(new Shape(700)); // crosses a twist
            var snap = a.state;
            var expected = U(a.random_raw(new Shape(4)));

            var b = new MT19937(1);
            b.state = snap;
            U(b.random_raw(new Shape(4))).Should().Equal(expected);

            BitGenerator poly = new MT19937(2);
            poly.state = snap;
            U(poly.random_raw(new Shape(4))).Should().Equal(expected);
        }

        /// <summary>The legacy RandomState state tuple is accepted (NumPy's setter takes <c>('MT19937', key, pos[, ...])</c>).</summary>
        [TestMethod]
        public void State_AcceptsTheLegacyTuple()
        {
            var b = new MT19937(1);
            b.state = np.random.RandomState(42).get_state();
            U(b.random_raw(new Shape(2))).Should().Equal(1608637542UL, 3421126067UL);

            Action bad = () => b.state = new NativeRandomState { Algorithm = "PCG64", Key = new uint[624], Pos = 0 };
            bad.Should().Throw<ValueError>().WithMessage("state is not a legacy MT19937 state");
        }

        /// <summary>The setter's errors: another generator's state, null, a short key.</summary>
        [TestMethod]
        public void State_SetterErrors()
        {
            BitGenerator b = new MT19937(1);
            ((Action)(() => b.state = new PCG64(1).state)).Should().Throw<ValueError>().WithMessage("state must be for a MT19937 PRNG");
            ((Action)(() => b.state = null)).Should().Throw<TypeError>().WithMessage("state must be a dict");
            ((Action)(() => ((MT19937)b).state = new MT19937.State(new uint[3], 0))).Should().Throw<IndexError>()
                .WithMessage("index 3 is out of bounds for axis 0 with size 3");
        }

        /// <summary>
        ///     NumPy stores any position (and then reads past the key — undefined behaviour in C); NumSharp rejects a
        ///     position outside [0, 624] at the assignment instead of failing on a later draw.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void State_OutOfRangePosition_IsRejected()
        {
            ((Action)(() => new MT19937(1).state = new MT19937.State(new uint[624], 700))).Should().Throw<ValueError>();
        }
    }
}
