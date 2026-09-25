using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact pins for NumPy 2.4.2's bit-generator family: <see cref="SeedSequence"/> spawning / pool size / state /
    ///     repr / entropy coercion, <see cref="BitGenerator.spawn"/> and <see cref="Generator.spawn"/>, <see cref="PCG64"/>
    ///     <c>advance</c>/<c>jumped</c>, and the three generators NumSharp lacked — <see cref="PCG64DXSM"/>,
    ///     <see cref="Philox"/> and <see cref="SFC64"/> — plus the <c>default_rng</c> forms. Every expected value and error
    ///     text was produced by the same call on NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class BitGeneratorFamilyTest
    {
        private const ulong M = ulong.MaxValue;

        /// <summary>Reads an integer array of any rank in logical C order as <see cref="ulong"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values.</returns>
        private static ulong[] U(NDArray a)
        {
            var r = new ulong[a.size];
            for (long i = 0; i < a.size; i++) r[i] = Convert.ToUInt64(a.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads a numeric array of any rank in logical C order as <see cref="double"/> values.</summary>
        /// <param name="a">The array to read.</param>
        /// <returns>The values.</returns>
        private static double[] D(NDArray a)
        {
            var r = new double[a.size];
            for (long i = 0; i < a.size; i++) r[i] = Convert.ToDouble(a.GetAtIndex(i));
            return r;
        }

        /// <summary>Reads the <c>uint[]</c>/<c>ulong[]</c> that <see cref="SeedSequence.generate_state"/> returns as <see cref="ulong"/> values.</summary>
        /// <param name="words">The generated words.</param>
        /// <returns>The values.</returns>
        private static ulong[] W(Array words) => words.Cast<object>().Select(Convert.ToUInt64).ToArray();

        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="TEx"/> with EXACTLY <paramref name="message"/>.</summary>
        /// <typeparam name="TEx">The expected exception type (NumPy's exception mapped to NumSharp's).</typeparam>
        /// <param name="act">The call.</param>
        /// <param name="message">NumPy's message, verbatim (compared literally — no wildcards, since the texts contain <c>**</c>).</param>
        private static void Throws<TEx>(Action act, string message) where TEx : Exception
            => act.Should().Throw<TEx>().Which.Message.Should().Be(message);

        // =====================================================================================
        //  SeedSequence: spawn / spawn_key / pool_size / n_children_spawned / state / repr
        // =====================================================================================

        /// <summary>Spawned children extend the spawn key and reproduce NumPy's child pools.</summary>
        [TestMethod]
        public void SeedSequence_Spawn_ChildrenMatchNumPy()
        {
            W(new SeedSequence(42).generate_state(4)).Should().Equal(3444837047UL, 2669555309UL, 2046530742UL, 3581440988UL);
            var kids = new SeedSequence(42).spawn(2);
            kids.Length.Should().Be(2);
            W(kids[0].generate_state(4)).Should().Equal(2684470948UL, 3757501821UL, 1691896351UL, 1126406280UL);
            W(kids[1].generate_state(4)).Should().Equal(4091952314UL, 31242083UL, 366899054UL, 1794014678UL);
            kids[1].spawn_key.Should().Equal(1L);
            kids[1].entropy.Should().Be(42L);
        }

        /// <summary>Repeated spawns continue the numbering; the parent counts every child it ever spawned.</summary>
        [TestMethod]
        public void SeedSequence_SpawnTwice_ContinuesNumbering()
        {
            var s = new SeedSequence(42);
            s.spawn(2);
            var c = s.spawn(1)[0];
            c.spawn_key.Should().Equal(2L);
            s.n_children_spawned.Should().Be(3);
            new SeedSequence(1).spawn(0).Length.Should().Be(0);
        }

        /// <summary>A grandchild's key is the full path, and its pool matches NumPy's.</summary>
        [TestMethod]
        public void SeedSequence_Grandchild_KeyPathAndPool()
        {
            var g = new SeedSequence(42).spawn(1)[0].spawn(2)[1];
            g.spawn_key.Should().Equal(0L, 1L);
            W(g.generate_state(4)).Should().Equal(3908812709UL, 3341582407UL, 3454793571UL, 679120907UL);
        }

        /// <summary>
        ///     An explicit spawn key pads the run entropy to the pool size before appending the key (the gh-16539 fix)
        ///     — for short and long entropy alike.
        /// </summary>
        [TestMethod]
        public void SeedSequence_ExplicitSpawnKey_PadsRunEntropy()
        {
            W(new SeedSequence(42, new long[] { 1, 2 }).generate_state(4)).Should().Equal(4007147622UL, 3772432517UL, 3162052391UL, 3078089812UL);
            W(new SeedSequence(new[] { 1, 2 }, new long[] { 3 }).generate_state(4)).Should().Equal(1207666766UL, 1361338344UL, 2530703459UL, 2759864538UL);
            W(new SeedSequence(BigInteger.Pow(2, 70), new long[] { 0 }).generate_state(4)).Should().Equal(3389591344UL, 426551014UL, 1847785401UL, 1094270253UL);
            W(new SeedSequence(new[] { 1, 2, 3, 4, 5 }, new long[] { 0 }).generate_state(4)).Should().Equal(2678533964UL, 4152742634UL, 1996422145UL, 1946681658UL);
            W(new SeedSequence(1, new long[] { 1L << 40 }).generate_state(2)).Should().Equal(1955033281UL, 1495014721UL);
            W(new SeedSequence(new int[0], new long[] { 1 }).generate_state(2)).Should().Equal(673228719UL, 1136656250UL);
        }

        /// <summary>A larger pool mixes more words, is exposed as <c>pool</c>, and is inherited by children.</summary>
        [TestMethod]
        public void SeedSequence_PoolSize8()
        {
            var s = new SeedSequence(42, null, 8);
            s.pool_size.Should().Be(8);
            s.pool.Should().Equal(2438911046u, 2499837615u, 2402278905u, 1814575481u, 575835601u, 2584694812u, 3897248574u, 124566269u);
            W(s.generate_state(4)).Should().Equal(3148411915UL, 2066703841UL, 814058201UL, 172636690UL);
            var child = s.spawn(1)[0];
            child.pool_size.Should().Be(8);
            W(child.generate_state(4)).Should().Equal(3475118689UL, 157057634UL, 2577142221UL, 3008509395UL);
        }

        /// <summary><c>n_children_spawned</c> seeds the numbering of the next spawn (rebuilding a serialized sequence).</summary>
        [TestMethod]
        public void SeedSequence_NChildrenSpawned_ContinuesNumbering()
        {
            new SeedSequence(42, null, 4, 5).spawn(1)[0].spawn_key.Should().Equal(5L);
        }

        /// <summary>An entropy-less sequence draws a 128-bit Python int (kept for reproducibility) — two in a row differ.</summary>
        [TestMethod]
        public void SeedSequence_NoEntropy_Draws128BitInt()
        {
            var a = new SeedSequence();
            a.entropy.Should().BeOfType<BigInteger>();
            ((BigInteger)a.entropy).Should().BeLessThan(BigInteger.Pow(2, 128));
            ((BigInteger)a.entropy).Sign.Should().BeGreaterThanOrEqualTo(0);
            new SeedSequence().entropy.Should().NotBe(a.entropy);
            // Logging the entropy reproduces the sequence.
            W(new SeedSequence(a.entropy).generate_state(4)).Should().Equal(W(a.generate_state(4)));
        }

        /// <summary>NumPy's constructor validation texts: pool size, entropy type, negative values, child-count range.</summary>
        [TestMethod]
        public void SeedSequence_Validation_MatchesNumPy()
        {
            Throws<ValueError>(() => new SeedSequence(42, null, 3), "The size of the entropy pool should be at least 4");
            Throws<TypeError>(() => new SeedSequence((object)3.5), "SeedSequence expects int or sequence of ints for entropy not 3.5");
            Throws<TypeError>(() => new SeedSequence((object)"5"), "SeedSequence expects int or sequence of ints for entropy not 5");
            Throws<ValueError>(() => new SeedSequence(-1), "expected non-negative integer");
            Throws<ValueError>(() => new SeedSequence(0, new long[] { -1 }), "expected non-negative integer");
            Throws<OverflowException>(() => new SeedSequence(1, null, 4, -1), "can't convert negative value to uint32_t");
            Throws<OverflowException>(() => new SeedSequence(1, null, 4, 1L << 32), "Python int too large to convert to C unsigned long");
            Throws<OverflowException>(() => new SeedSequence(1).spawn(-1), "can't convert negative value to uint32_t");
        }

        /// <summary>
        ///     Spawning past the uint32 child count raises before creating any child. NumPy itself hangs there (its
        ///     <c>cdef uint32_t</c> loop variable wraps and never reaches the bound) — a deliberate divergence.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void SeedSequence_SpawnPastUInt32_RaisesInsteadOfHanging()
        {
            var s = new SeedSequence(1, null, 4, uint.MaxValue);
            Throws<OverflowException>(() => s.spawn(2), "Python int too large to convert to C unsigned long");
            s.n_children_spawned.Should().Be(uint.MaxValue);
        }

        /// <summary>
        ///     A signed <c>int[]</c> is a Python list, NOT the uint32 pass-through — the CLR lets an <c>int[]</c> pass an
        ///     <c>is uint[]</c> test, which used to reinterpret <c>-1</c> as <c>0xFFFFFFFF</c> instead of rejecting it.
        /// </summary>
        [TestMethod]
        public void SeedSequence_SignedIntArray_IsAListNotAUInt32PassThrough()
        {
            Throws<ValueError>(() => new SeedSequence(new[] { -1 }), "expected non-negative integer");
            Throws<ValueError>(() => new SeedSequence(new long[] { -1 }), "expected non-negative integer");
            new SeedSequence(new[] { 1, 2, 3 }).ToString().Should().Be("SeedSequence(\n    entropy=[1, 2, 3],\n)");
            new SeedSequence(new uint[] { 1, 2, 3 }).ToString().Should().Be("SeedSequence(\n    entropy=array([1, 2, 3], dtype=uint32),\n)");
        }

        /// <summary>NumPy's repr: the entropy, then only the non-default fields, one per line.</summary>
        [TestMethod]
        public void SeedSequence_Repr_MatchesNumPy()
        {
            new SeedSequence(42).ToString().Should().Be("SeedSequence(\n    entropy=42,\n)");
            new SeedSequence(42, new long[] { 1, 2 }).ToString().Should().Be("SeedSequence(\n    entropy=42,\n    spawn_key=(1, 2),\n)");
            new SeedSequence(42, new long[] { 1 }).ToString().Should().Be("SeedSequence(\n    entropy=42,\n    spawn_key=(1,),\n)");
            new SeedSequence(42, null, 8).ToString().Should().Be("SeedSequence(\n    entropy=42,\n    pool_size=8,\n)");
            new SeedSequence(42, null, 4, 3).ToString().Should().Be("SeedSequence(\n    entropy=42,\n    n_children_spawned=3,\n)");
            var s = new SeedSequence(42);
            s.spawn(2);
            s.ToString().Should().Be("SeedSequence(\n    entropy=42,\n    n_children_spawned=2,\n)");
            new SeedSequence((object)np.array(new uint[] { 1, 2, 3 })).ToString().Should().Be("SeedSequence(\n    entropy=array([1, 2, 3], dtype=uint32),\n)");
            new SeedSequence((object)np.array(new long[] { 1, 2, 3 })).ToString().Should().Be("SeedSequence(\n    entropy=array([1, 2, 3]),\n)");
            new SeedSequence((object)new object[] { "0x10", 5 }).ToString().Should().Be("SeedSequence(\n    entropy=['0x10', 5],\n)");
        }

        /// <summary><c>state</c> is NumPy's dict of the four constructor arguments, in NumPy's key order.</summary>
        [TestMethod]
        public void SeedSequence_State_MatchesNumPy()
        {
            var st = new SeedSequence(42, new long[] { 1 }).state;
            st.Keys.Should().Equal("entropy", "spawn_key", "pool_size", "n_children_spawned");
            st["entropy"].Should().Be(42L);
            ((long[])st["spawn_key"]).Should().Equal(1L);
            st["pool_size"].Should().Be(4);
            st["n_children_spawned"].Should().Be(0L);
        }

        /// <summary>A value of 2**40 spans two words whether it arrives as int64, uint64 or a C# long array.</summary>
        [TestMethod]
        public void SeedSequence_WideValues_SpanWords()
        {
            var expected = new[] { 2437431422UL, 4286480934UL };
            W(new SeedSequence((object)np.array(new long[] { 1L << 40, 7 })).generate_state(2)).Should().Equal(expected);
            W(new SeedSequence((object)np.array(new ulong[] { 1UL << 40, 7 })).generate_state(2)).Should().Equal(expected);
            W(new SeedSequence(new long[] { 1L << 40, 7 }).generate_state(2)).Should().Equal(expected);
        }

        /// <summary>
        ///     NumPy's <c>_coerce_to_uint32_array</c> corners: nested sequences flatten, string elements parse (hex/octal/
        ///     decimal by prefix), a 2-D int array flattens in C order, a uint32 array element passes through, empties are
        ///     empty, bools count as ints, and a C# list stands for a Python list.
        /// </summary>
        [TestMethod]
        public void SeedSequence_EntropyCoercion_Corners()
        {
            var one23 = new[] { 3822189696UL, 3026158655UL };
            W(new SeedSequence((object)new object[] { new[] { 1, 2 }, new[] { 3 } }).generate_state(2)).Should().Equal(one23);
            W(new SeedSequence((object)new object[] { np.array(new uint[] { 1, 2 }), 3 }).generate_state(2)).Should().Equal(one23);
            W(new SeedSequence((object)new object[] { "0x10", "5" }).generate_state(2)).Should().Equal(242428781UL, 3874957636UL);
            var one234 = new[] { 4251844491UL, 3567780332UL };
            W(new SeedSequence((object)np.array(new long[,] { { 1, 2 }, { 3, 4 } })).generate_state(2)).Should().Equal(one234);
            W(new SeedSequence(new[] { 1, 2, 3, 4 }).generate_state(2)).Should().Equal(one234);
            var empty = new[] { 2968811710UL, 3677149159UL };
            W(new SeedSequence(new int[0]).generate_state(2)).Should().Equal(empty);
            W(new SeedSequence((object)np.array(new long[0])).generate_state(2)).Should().Equal(empty);
            var oneTwo = new[] { 1596810411UL, 3615836353UL };
            W(new SeedSequence((object)new object[] { true, 2 }).generate_state(2)).Should().Equal(oneTwo);
            W(new SeedSequence((object)new List<int> { 1, 2 }).generate_state(2)).Should().Equal(oneTwo);
        }

        /// <summary>The coercion's error texts, including NumPy's <c>np.concatenate</c> errors for non-1-D uint32 arrays.</summary>
        [TestMethod]
        public void SeedSequence_EntropyCoercion_Errors()
        {
            Throws<TypeError>(() => new SeedSequence((object)new object[] { 1.5 }), "seed must be integer");
            Throws<TypeError>(() => new SeedSequence((object)new object[] { null }), "object of type 'NoneType' has no len()");
            Throws<TypeError>(() => new SeedSequence((object)np.array(new[] { 1.0, 2.0 })), "seed must be integer");
            Throws<TypeError>(() => new SeedSequence((object)np.array(new[] { true })), "object of type 'numpy.bool' has no len()");
            Throws<ValueError>(() => new SeedSequence((object)np.array(new uint[,] { { 1, 2 }, { 3, 4 } })),
                "all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)");
            Throws<ValueError>(() => new SeedSequence((object)NDArray.Scalar(5u)), "zero-dimensional arrays cannot be concatenated");
            Throws<TypeError>(() => new SeedSequence(NDArray.Scalar(5u), new long[] { 1 }), "len() of unsized object");
            Throws<ValueError>(() => new SeedSequence((object)new object[] { "abc" }), "invalid literal for int() with base 10: 'abc'");
            Throws<ValueError>(() => new SeedSequence((object)new object[] { "08" }), "invalid literal for int() with base 8: '08'");
        }

        /// <summary><c>generate_state</c>: uint64 words, the empty request, and NumPy's dtype check (native uint32/uint64 only).</summary>
        [TestMethod]
        public void SeedSequence_GenerateState_Dtypes()
        {
            W(new SeedSequence(42).generate_state(3, np.uint64)).Should().Equal(11465652750463011511UL, 15382171918060459190UL, 9018504550953525431UL);
            W(new SeedSequence(42).generate_state(2, "<u8")).Should().Equal(11465652750463011511UL, 15382171918060459190UL);
            new SeedSequence(42).generate_state(0).Length.Should().Be(0);
            Throws<ValueError>(() => new SeedSequence(42).generate_state(2, np.int32), "only support uint32 or uint64");
            Throws<ValueError>(() => new SeedSequence(42).generate_state(2, ">u4"), "only support uint32 or uint64");
        }

        /// <summary><see cref="SeedlessSeedSequence"/> cannot seed anything and spawns itself.</summary>
        [TestMethod]
        public void SeedlessSeedSequence_MatchesNumPy()
        {
            var s = new SeedlessSeedSequence();
            Throws<NotImplementedException>(() => s.generate_state(2), "seedless SeedSequences cannot generate state");
            var kids = s.spawn(3);
            kids.Length.Should().Be(3);
            kids.All(k => ReferenceEquals(k, s)).Should().BeTrue();
            Throws<NotImplementedException>(() => new PCG64(s), "seedless SeedSequences cannot generate state");
        }

        // =====================================================================================
        //  BitGenerator.spawn / Generator.spawn
        // =====================================================================================

        /// <summary>A spawned child is a bit generator of the same type seeded from the matching seed-sequence child.</summary>
        [TestMethod]
        public void BitGenerator_Spawn_MatchesNumPy()
        {
            var kids = new PCG64(42).spawn(2);
            kids[0].Should().BeOfType<PCG64>();
            U(kids[0].random_raw(new Shape(3))).Should().Equal(16910944855483863638UL, 16804737912411866312UL, 16170277589469884630UL);
            U(kids[1].random_raw(new Shape(3))).Should().Equal(8623682774590505111UL, 856830905295172750UL, 10985220740352260511UL);
            var mt = new MT19937(42).spawn(1)[0];
            mt.Should().BeOfType<MT19937>();
            U(mt.random_raw(new Shape(3))).Should().Equal(1824649662UL, 3368690883UL, 1689735191UL);
            U(new PCG64DXSM(42).spawn(1)[0].random_raw(new Shape(3))).Should().Equal(13719008326363809935UL, 11367353641529148353UL, 4416631002723781746UL);
            U(new SFC64(42).spawn(1)[0].random_raw(new Shape(3))).Should().Equal(607877528894555031UL, 6338024872074868750UL, 5663039522575169809UL);
            new PCG64(1).spawn(0).Length.Should().Be(0);
        }

        /// <summary>Spawning advances the parent's seed sequence, so repeated spawns never repeat a child.</summary>
        [TestMethod]
        public void BitGenerator_Spawn_AdvancesTheSeedSequence()
        {
            var b = new PCG64(42);
            b.spawn(3);
            ((SeedSequence)b.seed_seq).n_children_spawned.Should().Be(3);
        }

        /// <summary>Generators without a spawnable seed sequence raise NumPy's TypeError.</summary>
        [TestMethod]
        public void BitGenerator_Spawn_WithoutSeedSequence_IsTypeError()
        {
            var legacy = new MT19937(1);
            legacy._legacy_seeding(5);
            Throws<TypeError>(() => legacy.spawn(1), "The underlying SeedSequence does not implement spawning.");
            Throws<TypeError>(() => new Philox(key: 5).spawn(1), "The underlying SeedSequence does not implement spawning.");
            Throws<TypeError>(() => new Generator(new Philox(key: 5)).spawn(1), "The underlying SeedSequence does not implement spawning.");
        }

        /// <summary><c>Generator.spawn</c> wraps the spawned bit generators; negative counts are NumPy's OverflowError.</summary>
        [TestMethod]
        public void Generator_Spawn_MatchesNumPy()
        {
            var kids = np.random.default_rng(42).spawn(2);
            kids.Length.Should().Be(2);
            kids[0].bit_generator.Should().BeOfType<PCG64>();
            D(kids[1].random(new Shape(3))).Should().Equal(0.4674907799518424, 0.04644889644868733, 0.5955100095961371);
            Throws<OverflowException>(() => np.random.default_rng(1).spawn(-1), "can't convert negative value to uint32_t");
        }

        // =====================================================================================
        //  PCG64.advance / jumped
        // =====================================================================================

        /// <summary><c>advance</c> skips raw draws in O(log n), wrapping mod 2**128 (negative steps back).</summary>
        [TestMethod]
        public void PCG64_Advance_MatchesNumPy()
        {
            U(new PCG64(42).advance(5).random_raw(new Shape(2))).Should().Equal(17997055833233904524UL, 14040549286955598961UL);
            U(new PCG64(42).advance(0).random_raw(new Shape(2))).Should().Equal(14276969152011380360UL, 8095878257575067585UL);
            U(new PCG64(42).advance(-1).random_raw(new Shape(2))).Should().Equal(468196377545690179UL, 14276969152011380360UL);
            U(new PCG64(42).advance(BigInteger.Pow(2, 128) + 3).random_raw(new Shape(2))).Should().Equal(12864169557245331597UL, 1737265434024182251UL);
            U(new PCG64(42).advance(BigInteger.Pow(2, 100)).random_raw(new Shape(2))).Should().Equal(11466488495143500310UL, 15182494446638615426UL);
        }

        /// <summary><c>advance</c> discards the buffered 32-bit half (NumPy "to ensure exact reproducibility").</summary>
        [TestMethod]
        public void PCG64_Advance_ResetsBufferedHalf()
        {
            var b = new PCG64(42);
            new Generator(b).random(new Shape(1), np.float32);
            b.state.has_uint32.Should().Be(1);
            b.advance(0);
            b.state.has_uint32.Should().Be(0);
            b.state.uinteger.Should().Be(0u);
        }

        /// <summary><c>jumped</c> copies the state and advances it by <c>jumps * (phi - 1) * 2**128</c>; the source is untouched.</summary>
        [TestMethod]
        public void PCG64_Jumped_MatchesNumPy()
        {
            var src = new PCG64(42);
            U(src.jumped().random_raw(new Shape(3))).Should().Equal(13948710574210763863UL, 11637761307587064314UL, 9384314469793298068UL);
            U(new PCG64(42).jumped(2).random_raw(new Shape(3))).Should().Equal(8679114779050513949UL, 9457325719678509308UL, 13496067618142150237UL);
            U(new PCG64(42).jumped(0).random_raw(new Shape(3))).Should().Equal(14276969152011380360UL, 8095878257575067585UL, 15838336090824644132UL);
            U(new PCG64(42).jumped(-1).random_raw(new Shape(3))).Should().Equal(2701073675766915402UL, 12111095884381728047UL, 13444761135894037977UL);
            var st = new PCG64(42).jumped().state;
            st.state.Should().Be(UInt128.Parse("246721301968239085263295379140720340427"));
            st.inc.Should().Be(UInt128.Parse("332724090758049132448979897138935081983"));
            src.state.state.Should().Be(UInt128.Parse("274674114334540486603088602300644985544"));
            ReferenceEquals(src.jumped().seed_seq, src.seed_seq).Should().BeFalse();
        }

        // =====================================================================================
        //  PCG64DXSM
        // =====================================================================================

        /// <summary>Same seeding as PCG64 (identical state), DXSM output of the pre-iterated state, cheap-multiplier step.</summary>
        [TestMethod]
        public void PCG64DXSM_RawStreamAndState_MatchNumPy()
        {
            U(new PCG64DXSM(42).random_raw(new Shape(5))).Should().Equal(12329818062196000797UL, 125530269004142706UL,
                12137922674892001441UL, 6848431486601849532UL, 3812337789277959813UL);
            var st = new PCG64DXSM(42).state;
            st.bit_generator.Should().Be("PCG64DXSM");
            st.state.Should().Be(new PCG64(42).state.state);
            st.inc.Should().Be(new PCG64(42).state.inc);
            U(new PCG64DXSM(0).random_raw(new Shape(3))).Should().Equal(15672045205194312304UL, 10230625629676741203UL, 1393141542142426128UL);
        }

        /// <summary><c>advance</c> composes the cheap-multiplier step; <c>jumped</c> uses PCG64's jump distance.</summary>
        [TestMethod]
        public void PCG64DXSM_AdvanceAndJumped_MatchNumPy()
        {
            U(new PCG64DXSM(42).advance(5).random_raw(new Shape(2))).Should().Equal(3575850668235994163UL, 14001613239349552815UL);
            U(new PCG64DXSM(42).advance(-1).random_raw(new Shape(2))).Should().Equal(999837659158574823UL, 12329818062196000797UL);
            U(new PCG64DXSM(42).jumped().random_raw(new Shape(3))).Should().Equal(12255520594600849659UL, 14432627000476523311UL, 5025260346042806266UL);
            U(new PCG64DXSM(42).jumped(3).random_raw(new Shape(3))).Should().Equal(6779012181160038566UL, 9298966129022454859UL, 17833755464604659735UL);
        }

        /// <summary>A Generator over PCG64DXSM reproduces NumPy's doubles, floats, integers and normals.</summary>
        [TestMethod]
        public void PCG64DXSM_Generator_MatchesNumPy()
        {
            D(new Generator(new PCG64DXSM(42)).random(new Shape(3))).Should().Equal(0.6684007764691958, 0.006805009518349059, 0.6579981066789486);
            D(new Generator(new PCG64DXSM(42)).random(new Shape(3), np.float32)).Should().Equal(
                (double)0.5562024116516113f, (double)0.668400764465332f, (double)0.3302779197692871f);
            U(new Generator(new PCG64DXSM(42)).integers(0, 100, new Shape(5))).Should().Equal(55UL, 66UL, 33UL, 0UL, 1UL);
            D(new Generator(new PCG64DXSM(42)).standard_normal(new Shape(3))).Should().Equal(0.27546266544254505, 0.07845904998353895, 0.46697136428073827);
            new Generator(new PCG64DXSM(42)).ToString().Should().Be("Generator(PCG64DXSM)");
        }

        /// <summary>A PCG64 state is not a PCG64DXSM state, although the fields are identical.</summary>
        [TestMethod]
        public void PCG64DXSM_ForeignState_IsValueError()
        {
            var d = new PCG64DXSM(1);
            Throws<ValueError>(() => ((BitGenerator)d).state = new PCG64(1).state, "state must be for a PCG64DXSM RNG");
        }

        // =====================================================================================
        //  Philox
        // =====================================================================================

        /// <summary>The raw stream crosses a block boundary exactly as NumPy's buffered 4x64-10 does.</summary>
        [TestMethod]
        public void Philox_RawStream_MatchesNumPy()
        {
            U(new Philox(42).random_raw(new Shape(6))).Should().Equal(1587852024645073290UL, 2611271723512893552UL, 4982337093617253890UL,
                16123152800351476682UL, 3138981475030020977UL, 9349039075912046124UL);
            U(new Philox(0).random_raw(new Shape(3))).Should().Equal(259491006799949737UL, 4754966410622352325UL, 8698845897610382596UL);
        }

        /// <summary>State layout before and after one draw: the counter increments BEFORE encryption and the block is buffered.</summary>
        [TestMethod]
        public void Philox_State_MatchesNumPy()
        {
            var st = new Philox(42).state;
            st.bit_generator.Should().Be("Philox");
            st.counter.Should().Equal(0UL, 0UL, 0UL, 0UL);
            st.key.Should().Equal(11465652750463011511UL, 15382171918060459190UL);
            st.buffer.Should().Equal(0UL, 0UL, 0UL, 0UL);
            st.buffer_pos.Should().Be(4);
            st.has_uint32.Should().Be(0);
            st.uinteger.Should().Be(0u);

            var b = new Philox(42);
            b.random_raw(new Shape(1));
            var after = b.state;
            after.counter.Should().Equal(1UL, 0UL, 0UL, 0UL);
            after.buffer.Should().Equal(1587852024645073290UL, 2611271723512893552UL, 4982337093617253890UL, 16123152800351476682UL);
            after.buffer_pos.Should().Be(1);
        }

        /// <summary>Explicit keys and counters in every NumPy form; a keyed Philox has no seed sequence.</summary>
        [TestMethod]
        public void Philox_KeyAndCounter_MatchNumPy()
        {
            var keyed = new Philox(key: 5);
            keyed.seed_seq.Should().BeNull();
            U(keyed.random_raw(new Shape(3))).Should().Equal(13535223855206698129UL, 10893183200674769480UL, 3833398344621921443UL);
            U(new Philox(key: BigInteger.Pow(2, 127) + 9).random_raw(new Shape(3))).Should().Equal(7589142747770494645UL, 4547647955350555698UL, 14389942624022327308UL);
            U(new Philox(key: new long[] { 5, 6 }).random_raw(new Shape(3))).Should().Equal(2872608186901354967UL, 17625563014048335106UL, 10329226678761915209UL);
            U(new Philox(42, counter: 7).random_raw(new Shape(3))).Should().Equal(5018231186413187675UL, 3307371308634238189UL, 8304758364223038998UL);
            U(new Philox(counter: new long[] { 1, 2, 3, 4 }, key: new long[] { 5, 6 }).random_raw(new Shape(3))).Should().Equal(
                10568657560886415971UL, 15636345889197363040UL, 4975934743979906624UL);
            U(new Philox(counter: BigInteger.Pow(2, 255), key: 1).random_raw(new Shape(3))).Should().Equal(3667462195032723939UL, 13650499583026342101UL, 17010584963620489924UL);
            U(new Philox(counter: BigInteger.Pow(2, 256) - 1, key: 1).random_raw(new Shape(3))).Should().Equal(14663341350739098444UL, 11767532808736069200UL, 16779231742903463967UL);
            U(new Philox(key: new long[] { 5, 6 }, counter: new[] { M, M, M, M }).random_raw(new Shape(3))).Should().Equal(
                932400633431777695UL, 12070657395441989121UL, 9452969912890316051UL);
            new Philox(new SeedSequence(7)).state.key.Should().Equal(16920295385781661272UL, 610735763742393210UL);
        }

        /// <summary>
        ///     NumPy's <c>int_to_array</c> scalar coercion: Python <c>int()</c> of floats (truncation), bools, 0-d arrays and
        ///     strings, then the range check.
        /// </summary>
        [TestMethod]
        public void Philox_ScalarKeyCoercion_MatchesNumPy()
        {
            new Philox(key: 3.7).state.key.Should().Equal(3UL, 0UL);
            new Philox(key: -0.5).state.key.Should().Equal(0UL, 0UL);
            new Philox(key: System.Math.Pow(2, 64)).state.key.Should().Equal(0UL, 1UL);
            new Philox(key: true).state.key.Should().Equal(1UL, 0UL);
            new Philox(key: NDArray.Scalar(7L)).state.key.Should().Equal(7UL, 0UL);
            new Philox(key: NDArray.Scalar(7.9)).state.key.Should().Equal(7UL, 0UL);
            new Philox(key: "5").state.key.Should().Equal(5UL, 0UL);
            new Philox(key: " 1_000 ").state.key.Should().Equal(1000UL, 0UL);
            new Philox(key: "+7").state.key.Should().Equal(7UL, 0UL);
            new Philox(1, counter: 3.5).state.counter.Should().Equal(3UL, 0UL, 0UL, 0UL);
        }

        /// <summary>
        ///     The array form is NumPy's unsafe <c>astype(uint64)</c> (negatives wrap, floats truncate, out-of-range/NaN
        ///     floats become 2**63 as MSVC converts them, complex drops the imaginary part), then an exact-size check.
        /// </summary>
        [TestMethod]
        public void Philox_ArrayKeyCoercion_MatchesNumPy()
        {
            new Philox(key: np.array(new long[] { -1, 5 })).state.key.Should().Equal(M, 5UL);
            new Philox(key: new[] { 1.9, 2.1 }).state.key.Should().Equal(1UL, 2UL);
            new Philox(key: new[] { true, false }).state.key.Should().Equal(1UL, 0UL);
            new Philox(key: np.array(new sbyte[] { -1, 2 })).state.key.Should().Equal(M, 2UL);
            new Philox(key: np.array(new[] { new Complex(1, 2), new Complex(3, 0) })).state.key.Should().Equal(1UL, 3UL);
            new Philox(key: np.array(new[] { -1.5, 3.5 })).state.key.Should().Equal(M, 3UL);
            new Philox(key: np.array(new[] { double.NaN, 1.0 })).state.key.Should().Equal(9223372036854775808UL, 1UL);
            new Philox(key: np.array(new[] { double.PositiveInfinity, 1.0 })).state.key.Should().Equal(9223372036854775808UL, 1UL);
            new Philox(key: np.array(new[] { 1e30, 1.0 })).state.key.Should().Equal(9223372036854775808UL, 1UL);
            new Philox(key: new[] { System.Math.Pow(2, 64), 1.0 }).state.key.Should().Equal(9223372036854775808UL, 1UL);
        }

        /// <summary>
        ///     A Python-list-like key follows NumPy's list inference: int64 + uint64 ints promote to float64 (correctly
        ///     rounded — .NET's own BigInteger→double truncates), and an int beyond uint64 makes an object array that raises.
        /// </summary>
        [TestMethod]
        public void Philox_ListKeyInference_MatchesNumPy()
        {
            BigInteger p63 = BigInteger.Pow(2, 63), p64 = BigInteger.Pow(2, 64);
            new Philox(key: new object[] { -1, p63 }).state.key.Should().Equal(M, 9223372036854775808UL);
            new Philox(key: new object[] { p63 + 1, 5 }).state.key.Should().Equal(9223372036854775808UL, 5UL);
            new Philox(key: new object[] { p63 + 1, p63 + 3 }).state.key.Should().Equal(9223372036854775809UL, 9223372036854775811UL);
            new Philox(key: new object[] { -5, p63 + 1025 }).state.key.Should().Equal(18446744073709551611UL, 9223372036854777856UL);
            Throws<OverflowException>(() => new Philox(key: new object[] { p64, 1 }), "int too big to convert");
            Throws<OverflowException>(() => new Philox(key: new object[] { p64, -1 }), "int too big to convert");
            Throws<OverflowException>(() => new Philox(key: new object[] { -1, p64 }), "Python integer -1 out of bounds for uint64");
            Throws<OverflowException>(() => new Philox(key: new object[] { -p63 - 1, 5 }), "int too big to convert");
        }

        /// <summary>Philox's constructor errors, in NumPy's order (seed/key conflict, seed, key, counter).</summary>
        [TestMethod]
        public void Philox_Validation_MatchesNumPy()
        {
            Throws<ValueError>(() => new Philox(1, key: 2), "seed and key cannot be both used");
            Throws<ValueError>(() => new Philox(key: -1), "key must be positive and less than 2**128.");
            Throws<ValueError>(() => new Philox(key: BigInteger.Pow(2, 128)), "key must be positive and less than 2**128.");
            Throws<ValueError>(() => new Philox(key: "-7"), "key must be positive and less than 2**128.");
            Throws<ValueError>(() => new Philox(counter: -1), "counter must be positive and less than 2**256.");
            Throws<ValueError>(() => new Philox(counter: BigInteger.Pow(2, 256)), "counter must be positive and less than 2**256.");
            Throws<ValueError>(() => new Philox(counter: new long[] { 1, 2 }), "counter must have 4 elements when using array form");
            Throws<ValueError>(() => new Philox(counter: new long[] { 1, 2, 3, 4, 5 }), "counter must have 4 elements when using array form");
            Throws<ValueError>(() => new Philox(key: new long[] { 1, 2, 3 }), "key must have 2 elements when using array form");
            Throws<ValueError>(() => new Philox(key: new long[0]), "key must have 2 elements when using array form");
            Throws<ValueError>(() => new Philox(key: new long[,] { { 1, 2 } }), "key must have 2 elements when using array form");
            Throws<ValueError>(() => new Philox(key: new object[] { new[] { 1 }, new[] { 2 } }), "key must have 2 elements when using array form");
            Throws<ValueError>(() => new Philox(key: double.NaN), "cannot convert float NaN to integer");
            Throws<OverflowException>(() => new Philox(key: double.PositiveInfinity), "cannot convert float infinity to integer");
            Throws<TypeError>(() => new Philox(key: NDArray.Scalar(new Complex(1, 2))), "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            Throws<ValueError>(() => new Philox(key: "abc"), "invalid literal for int() with base 10: 'abc'");
            Throws<ValueError>(() => new Philox(key: "1__0"), "invalid literal for int() with base 10: '1__0'");
            Throws<ValueError>(() => new Philox(key: "_10"), "invalid literal for int() with base 10: '_10'");
            Throws<ValueError>(() => new Philox(key: ""), "invalid literal for int() with base 10: ''");
            // Order: key before counter; the seed (via SeedSequence) before the counter.
            Throws<ValueError>(() => new Philox(counter: -1, key: -1), "key must be positive and less than 2**128.");
            Throws<ValueError>(() => new Philox(-1, counter: -1), "expected non-negative integer");
            Throws<TypeError>(() => new Philox(3.5), "SeedSequence expects int or sequence of ints for entropy not 3.5");
        }

        /// <summary><c>advance</c> adds to the counter mod 2**256 and discards the buffered block.</summary>
        [TestMethod]
        public void Philox_Advance_MatchesNumPy()
        {
            U(new Philox(42).advance(5).random_raw(new Shape(3))).Should().Equal(7884165492903707825UL, 4266091862756903345UL, 8963462020418319140UL);
            U(new Philox(42).advance(-1).random_raw(new Shape(3))).Should().Equal(8325030314314475915UL, 9740503905916979327UL, 14661952272905906144UL);
            U(new Philox(42).advance(BigInteger.Pow(2, 64)).random_raw(new Shape(3))).Should().Equal(8071730641624022976UL, 6747693920690648496UL, 616662189251774573UL);
            var st = new Philox(42).advance(5).state;
            st.counter.Should().Equal(5UL, 0UL, 0UL, 0UL);
            st.buffer.Should().Equal(0UL, 0UL, 0UL, 0UL);
            st.buffer_pos.Should().Be(4);
            new Philox(42).advance(BigInteger.Pow(2, 256) + 5).state.counter.Should().Equal(5UL, 0UL, 0UL, 0UL);
            new Philox(42).advance(-BigInteger.Pow(2, 64)).state.counter.Should().Equal(0UL, M, M, M);

            var mid = new Philox(42);
            mid.random_raw(new Shape(2));
            mid.advance(1);
            U(mid.random_raw(new Shape(3))).Should().Equal(8833152655518396711UL, 14612918126939214882UL, 4148435705760359294UL);
        }

        /// <summary>NumPy's <c>philox_advance</c> carry chain, transcribed literally, and the per-block counter increment's carry.</summary>
        [TestMethod]
        public void Philox_CounterCarries_MatchNumPy()
        {
            new Philox(1, counter: (BigInteger)M).advance(1).state.counter.Should().Equal(0UL, 1UL, 0UL, 0UL);
            new Philox(1, counter: new[] { M, M, 0UL, 0UL }).advance(1).state.counter.Should().Equal(0UL, 0UL, 1UL, 0UL);
            new Philox(1, counter: new[] { M, M, 0UL, 0UL }).advance(BigInteger.Pow(2, 64) + 1).state.counter.Should().Equal(0UL, 1UL, 1UL, 0UL);
            new Philox(1, counter: new[] { 5UL, M, 0UL, 0UL }).advance((BigInteger)M + BigInteger.Pow(2, 64)).state.counter.Should().Equal(4UL, 1UL, 1UL, 0UL);
            new Philox(1, counter: new[] { M, M, M, 0UL }).advance(BigInteger.Pow(2, 128) - 1).state.counter.Should().Equal(M - 1, M, 0UL, 1UL);
            new Philox(1, counter: new[] { 1UL, M, 0UL, 0UL }).advance((BigInteger)M + ((BigInteger)M << 64)).state.counter.Should().Equal(0UL, M, 1UL, 0UL);

            var b = new Philox(1, counter: new[] { M, M, M, 0UL });
            b.random_raw(new Shape(1));
            b.state.counter.Should().Equal(0UL, 0UL, 0UL, 1UL);
            var w = new Philox(1, counter: BigInteger.Pow(2, 256) - 1);
            w.random_raw(new Shape(1));
            w.state.counter.Should().Equal(0UL, 0UL, 0UL, 0UL);
        }

        /// <summary><c>jumped</c> advances the counter's upper half by <c>jumps</c> (2**128 draws each).</summary>
        [TestMethod]
        public void Philox_Jumped_MatchesNumPy()
        {
            U(new Philox(42).jumped().random_raw(new Shape(3))).Should().Equal(5874559101895519209UL, 7332921304122003723UL, 5542365866194672767UL);
            var st = new Philox(42).jumped(2).state;
            st.counter.Should().Equal(0UL, 0UL, 2UL, 0UL);
            st.key.Should().Equal(11465652750463011511UL, 15382171918060459190UL);
            new Philox(42).jumped(0).state.counter.Should().Equal(0UL, 0UL, 0UL, 0UL);
            new Philox(42).jumped(-1).state.counter.Should().Equal(0UL, 0UL, M, M);
        }

        /// <summary>A Generator over Philox reproduces NumPy's doubles, floats (split words), integers and normals.</summary>
        [TestMethod]
        public void Philox_Generator_MatchesNumPy()
        {
            D(new Generator(new Philox(42)).random(new Shape(3))).Should().Equal(0.08607763073528474, 0.14155732377913233, 0.27009303504774695);
            D(new Generator(new Philox(42)).random(new Shape(3), np.float32)).Should().Equal(
                (double)0.9252124428749084f, (double)0.08607763051986694f, (double)0.1406564712524414f);
            U(new Generator(new Philox(42)).integers(0, 100, new Shape(5))).Should().Equal(92UL, 8UL, 14UL, 14UL, 40UL);
            D(new Generator(new Philox(42)).standard_normal(new Shape(3))).Should().Equal(-1.1043995228921153, 0.1891281100736375, 0.04600092882122236);

            var b = new Philox(42);
            new Generator(b).random(new Shape(1), np.float32);
            b.state.has_uint32.Should().Be(1);
            b.state.uinteger.Should().Be(369700608u);
            b.state.buffer_pos.Should().Be(1);
        }

        /// <summary>
        ///     Restoring a state restores the block buffer and its position (a position of 4 or more encrypts a new block
        ///     first); a foreign state is NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void Philox_StateSetter_MatchesNumPy()
        {
            var b = new Philox(42);
            var st = b.state;
            st.buffer = new ulong[] { 11, 22, 33, 44 };
            st.buffer_pos = 2;
            b.state = st;
            U(b.random_raw(new Shape(3))).Should().Equal(33UL, 44UL, 1587852024645073290UL);

            var c = new Philox(42);
            var st10 = c.state;
            st10.buffer_pos = 10;
            c.state = st10;
            U(c.random_raw(new Shape(1))).Should().Equal(1587852024645073290UL);
            c.state.buffer_pos.Should().Be(1);

            Throws<ValueError>(() => ((BitGenerator)new Philox(1)).state = new PCG64(1).state, "state must be for a Philox PRNG");
        }

        /// <summary>
        ///     Malformed Philox states fail in NumPy's read order with its IndexError text, and — unlike NumPy, which has
        ///     already written the words it read — leave the generator untouched. A negative <c>buffer_pos</c> (undefined
        ///     behavior in NumPy's C) is refused.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Philox_StateSetter_MalformedStates()
        {
            var b = new Philox(42);
            var before = b.state;
            Throws<IndexError>(() => b.state = new Philox.State(new ulong[] { 1 }, new ulong[] { 1, 2 }, new ulong[4], 4), "index 1 is out of bounds for axis 0 with size 1");
            Throws<IndexError>(() => b.state = new Philox.State(new ulong[4], new ulong[] { 1 }, new ulong[4], 4), "index 1 is out of bounds for axis 0 with size 1");
            Throws<IndexError>(() => b.state = new Philox.State(new ulong[4], new ulong[2], new ulong[3], 4), "index 3 is out of bounds for axis 0 with size 3");
            Throws<ValueError>(() => b.state = new Philox.State(new ulong[4], new ulong[2], new ulong[4], -1), "state['buffer_pos'] must be non-negative, got -1");
            b.state.counter.Should().Equal(before.counter);
            b.state.key.Should().Equal(before.key);
        }

        // =====================================================================================
        //  SFC64
        // =====================================================================================

        /// <summary>Three seeded words, counter 1 and twelve warm-up draws reproduce NumPy's stream and state.</summary>
        [TestMethod]
        public void SFC64_RawStreamAndState_MatchNumPy()
        {
            U(new SFC64(42).random_raw(new Shape(5))).Should().Equal(9775594601838723485UL, 6977463094773878866UL, 17439770048677797496UL,
                7768405669198076140UL, 11828679036797625575UL);
            var st = new SFC64(42).state;
            st.bit_generator.Should().Be("SFC64");
            st.state.Should().Equal(9143715722600539226UL, 631878879238184246UL, 4804307118799948943UL, 13UL);
            st.has_uint32.Should().Be(0);
            U(new SFC64(0).random_raw(new Shape(3))).Should().Equal(10490465040999277362UL, 4331856608414834465UL, 7312684695965765022UL);
            U(new SFC64(new[] { 1, 2, 3 }).random_raw(new Shape(3))).Should().Equal(638631490462651809UL, 14672256278229396831UL, 3576229714778753261UL);
            U(new SFC64(BigInteger.Pow(2, 100)).random_raw(new Shape(3))).Should().Equal(12072699153697711433UL, 13676488678798140390UL, 6198400221021126201UL);
        }

        /// <summary>A Generator over SFC64 reproduces NumPy's doubles, floats, integers and normals.</summary>
        [TestMethod]
        public void SFC64_Generator_MatchesNumPy()
        {
            D(new Generator(new SFC64(42)).random(new Shape(3))).Should().Equal(0.5299360452325557, 0.3782490322895635, 0.9454118287212049);
            D(new Generator(new SFC64(42)).random(new Shape(3), np.float32)).Should().Equal(
                (double)0.24540334939956665f, (double)0.5299360156059265f, (double)0.4273233413696289f);
            U(new Generator(new SFC64(42)).integers(0, 100, new Shape(5))).Should().Equal(24UL, 52UL, 42UL, 37UL, 60UL);
            D(new Generator(new SFC64(42)).standard_normal(new Shape(3))).Should().Equal(-0.41650720505797584, 0.03184219797455374, 0.8344613070953106);
        }

        /// <summary>
        ///     The state setter is NumPy's broadcast assignment (one word fills all four, four copy, anything else is the
        ///     broadcast error), and a foreign state reports NumPy's literal, un-interpolated message.
        /// </summary>
        [TestMethod]
        public void SFC64_StateSetter_MatchesNumPy()
        {
            var b = new SFC64(1);
            var st = b.state;
            st.state = new ulong[] { 5 };
            b.state = st;
            b.state.state.Should().Equal(5UL, 5UL, 5UL, 5UL);

            var c = new SFC64(1);
            c.state = new SFC64.State(new ulong[] { 1, 2, 3, 4 });
            U(c.random_raw(new Shape(2))).Should().Equal(7UL, 34UL);

            var d = new SFC64(1);
            var bad = d.state;
            bad.state = new ulong[] { 1, 2, 3 };
            Throws<ValueError>(() => d.state = bad, "could not broadcast input array from shape (3,) into shape (4,)");
            Throws<ValueError>(() => ((BitGenerator)new SFC64(1)).state = new PCG64(1).state, "state must be for a {self.__class__.__name__} RNG");
        }

        // =====================================================================================
        //  default_rng forms
        // =====================================================================================

        /// <summary><c>default_rng</c> accepts the full uint64 range, an integer array, a RandomState and null, like NumPy's.</summary>
        [TestMethod]
        public void DefaultRng_Forms_MatchNumPy()
        {
            D(np.random.default_rng(ulong.MaxValue).random(new Shape(2))).Should().Equal(0.6800266789616931, 0.8453117585624743);
            D(np.random.default_rng(np.array(new long[] { 1, 2, 3 })).random(new Shape(2))).Should().Equal(0.6704722626516632, 0.1110146366693816);
            D(np.random.default_rng(BigInteger.Pow(2, 64) - 1).random(new Shape(2))).Should().Equal(0.6800266789616931, 0.8453117585624743);
            D(np.random.default_rng((object)new object[] { 1, 2, 3 }).random(new Shape(2))).Should().Equal(0.6704722626516632, 0.1110146366693816);
            np.random.default_rng((object)null).Should().BeOfType<Generator>();
        }

        /// <summary><c>default_rng(RandomState)</c> wraps the legacy engine itself (NumPy's <c>seed._bit_generator</c>).</summary>
        [TestMethod]
        public void DefaultRng_RandomState_SharesTheEngine()
        {
            var rs = np.random.RandomState(42);
            var g = np.random.default_rng(rs);
            g.bit_generator.Should().BeOfType<MT19937>();
            D(g.random(new Shape(2))).Should().Equal(0.3745401188473625, 0.9507143064099162);
            ReferenceEquals(np.random.default_rng((object)rs).bit_generator, g.bit_generator).Should().BeTrue();
        }

        /// <summary>The object form dispatches like NumPy's: bit generators wrap, generators pass through, seed sequences seed PCG64.</summary>
        [TestMethod]
        public void DefaultRng_ObjectDispatch()
        {
            var bg = new Philox(42);
            np.random.default_rng((object)bg).bit_generator.Should().BeSameAs(bg);
            var g = np.random.default_rng(1);
            np.random.default_rng((object)g).Should().BeSameAs(g);
            D(np.random.default_rng((object)new SeedSequence(42)).random(new Shape(2))).Should().Equal(D(np.random.default_rng(42).random(new Shape(2))));
            Throws<TypeError>(() => np.random.default_rng((object)1.5), "SeedSequence expects int or sequence of ints for entropy not 1.5");
        }

        // =====================================================================================
        //  Bulk fills: identical streams and states to the per-draw primitives
        // =====================================================================================

        /// <summary>The five engines, freshly seeded, for the bulk-fill equivalence checks.</summary>
        /// <returns>Factories producing an identically seeded engine on every call.</returns>
        private static IEnumerable<Func<BitGenerator>> AllEngines() => new Func<BitGenerator>[]
        {
            () => new PCG64(123), () => new PCG64DXSM(123), () => new Philox(123), () => new SFC64(123), () => new MT19937(123),
        };

        /// <summary>Snapshots a bit generator's state as a comparable string (every member, arrays expanded).</summary>
        /// <param name="bg">The bit generator.</param>
        /// <returns>The state's members, in declaration order.</returns>
        private static string StateText(BitGenerator bg)
        {
            var st = bg.state;
            return string.Join("|", st.GetType().GetProperties().Select(p =>
            {
                object v = p.GetValue(st);
                return p.Name + "=" + (v is Array a ? string.Join(",", a.Cast<object>()) : Convert.ToString(v));
            }));
        }

        /// <summary>
        ///     Every bulk fill (the double / float / raw paths behind <c>random</c>, <c>random(float32)</c> and
        ///     <c>random_raw</c>) must produce EXACTLY the per-draw stream and leave EXACTLY the per-draw state, for every
        ///     engine, across chunk/block/twist boundaries and with a pending 32-bit half or a part-consumed Philox block
        ///     going in — the fills are interleaved freely with single draws in NumPy's streams.
        /// </summary>
        [TestMethod]
        public void BulkFills_MatchThePerDrawStreamAndState()
        {
            long[] sizes = { 1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 511, 512, 513, 623, 624, 625, 1249, 5000 };
            foreach (var make in AllEngines())
            {
                foreach (long n in sizes)
                {
                    foreach (int lead in new[] { 0, 1, 2, 3 })
                    {
                        // lead float draws leave a pending half (odd) and part-consume Philox's block.
                        var bulk = make();
                        var single = make();
                        var gb = new Generator(bulk);
                        var gs = new Generator(single);
                        for (int k = 0; k < lead; k++)
                        {
                            gb.random(dtype: np.float32);
                            gs.random(dtype: np.float32);
                        }

                        double[] d = D(gb.random(new Shape(n)));
                        var ds = new double[n];
                        for (long k = 0; k < n; k++) ds[k] = Convert.ToDouble(gs.random().GetAtIndex(0));
                        d.Should().Equal(ds, $"{bulk.GetType().Name} doubles n={n} lead={lead}");
                        StateText(bulk).Should().Be(StateText(single), $"{bulk.GetType().Name} state after doubles n={n} lead={lead}");

                        double[] f = D(gb.random(new Shape(n), np.float32));
                        var fs = new double[n];
                        for (long k = 0; k < n; k++) fs[k] = Convert.ToDouble(gs.random(dtype: np.float32).GetAtIndex(0));
                        f.Should().Equal(fs, $"{bulk.GetType().Name} floats n={n} lead={lead}");
                        StateText(bulk).Should().Be(StateText(single), $"{bulk.GetType().Name} state after floats n={n} lead={lead}");

                        ulong[] r = U(bulk.random_raw(new Shape(n)));
                        var rs = new ulong[n];
                        for (long k = 0; k < n; k++) rs[k] = Convert.ToUInt64(single.random_raw().GetAtIndex(0));
                        r.Should().Equal(rs, $"{bulk.GetType().Name} raw n={n} lead={lead}");
                        StateText(bulk).Should().Be(StateText(single), $"{bulk.GetType().Name} state after raw n={n} lead={lead}");
                    }
                }
            }
        }

        /// <summary>
        ///     The chunked bounded-integer fills (words pulled a chunk at a time, never past what the per-draw loop would
        ///     consume) produce exactly NumPy's per-draw Lemire sequence — <c>buffered_bounded_lemire_uint32</c> for ranges
        ///     that fit 32 bits, <c>bounded_lemire_uint64</c> above, the raw word at the full ranges — and leave the engine in
        ///     the per-draw state, for every engine, across chunk boundaries and with a pending 32-bit half going in.
        /// </summary>
        [TestMethod]
        public void ChunkedIntegers_MatchThePerDrawLemireStreamAndState()
        {
            ulong[] ranges = { 1, 5, 1000, 1UL << 31, 0xFFFFFFFEUL, 0xFFFFFFFFUL, 1UL << 32, (1UL << 40) + 3, 1UL << 63, ulong.MaxValue - 1, ulong.MaxValue };
            long[] sizes = { 1, 7, 511, 512, 513, 1500 };
            foreach (var make in AllEngines())
            {
                foreach (ulong rng in ranges)
                {
                    foreach (long n in sizes)
                    {
                        foreach (int lead in new[] { 0, 1 })
                        {
                            var bulk = make();
                            var single = make();
                            for (int k = 0; k < lead; k++)
                            {
                                new Generator(bulk).random(dtype: np.float32);
                                new Generator(single).random(dtype: np.float32);
                            }
                            ulong[] got = U(new Generator(bulk).integers(0UL, rng, new Shape(n), np.uint64, endpoint: true));
                            var expected = new ulong[n];
                            for (long k = 0; k < n; k++)
                            {
                                expected[k] = rng <= 0xFFFFFFFFUL
                                    ? (rng == 0xFFFFFFFFUL ? single.NextUInt32() : BoundedIntegers.LemireUInt32(single, (uint)rng))
                                    : (rng == ulong.MaxValue ? single.NextUInt64() : BoundedIntegers.LemireUInt64(single, rng));
                            }
                            got.Should().Equal(expected, $"{bulk.GetType().Name} rng={rng} n={n} lead={lead}");
                            StateText(bulk).Should().Be(StateText(single), $"{bulk.GetType().Name} state rng={rng} n={n} lead={lead}");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The ziggurat fills (normal / exponential, float64 and float32, plus <c>method='inv'</c>, <c>normal(loc, scale)</c>,
        ///     <c>exponential(scale)</c>, <c>uniform</c>) read their draws ahead through a buffer; over 20,000 outputs — hundreds
        ///     of wedge/tail rejections included — every engine must give exactly the per-draw values and end in the per-draw
        ///     state (a read-ahead that drew one word too many would shift every later value).
        /// </summary>
        [TestMethod]
        public void ZigguratFills_MatchThePerDrawStreamAndState()
        {
            const int n = 20_000;
            var cases = new (string Name, Func<Generator, NDArray> Bulk, Func<Generator, double> Single)[]
            {
                ("standard_normal", g => g.standard_normal(new Shape(n)), g => Convert.ToDouble(g.standard_normal().GetAtIndex(0))),
                ("standard_normal f32", g => g.standard_normal(new Shape(n), np.float32), g => Convert.ToDouble(g.standard_normal(dtype: np.float32).GetAtIndex(0))),
                ("standard_exponential", g => g.standard_exponential(new Shape(n)), g => Convert.ToDouble(g.standard_exponential().GetAtIndex(0))),
                ("standard_exponential f32", g => g.standard_exponential(new Shape(n), np.float32), g => Convert.ToDouble(g.standard_exponential(dtype: np.float32).GetAtIndex(0))),
                ("standard_exponential inv", g => g.standard_exponential(new Shape(n), method: "inv"), g => Convert.ToDouble(g.standard_exponential(method: "inv").GetAtIndex(0))),
                ("standard_exponential inv f32", g => g.standard_exponential(new Shape(n), np.float32, "inv"), g => Convert.ToDouble(g.standard_exponential(dtype: np.float32, method: "inv").GetAtIndex(0))),
                ("normal(3, 2)", g => g.normal(3.0, 2.0, new Shape(n)), g => Convert.ToDouble(g.normal(3.0, 2.0).GetAtIndex(0))),
                ("exponential(2.5)", g => g.exponential(2.5, new Shape(n)), g => Convert.ToDouble(g.exponential(2.5).GetAtIndex(0))),
                ("uniform(-1, 4)", g => g.uniform(-1.0, 4.0, new Shape(n)), g => Convert.ToDouble(g.uniform(-1.0, 4.0).GetAtIndex(0))),
            };
            foreach (var make in AllEngines())
            {
                foreach (var (name, bulkCall, singleCall) in cases)
                {
                    var bulk = make();
                    var single = make();
                    // A pending 32-bit half going in exercises the float32 paths' buffered-half hand-off.
                    new Generator(bulk).random(dtype: np.float32);
                    new Generator(single).random(dtype: np.float32);
                    double[] got = D(bulkCall(new Generator(bulk)));
                    var gs = new Generator(single);
                    var expected = new double[n];
                    for (int k = 0; k < n; k++) expected[k] = singleCall(gs);
                    got.Should().Equal(expected, $"{bulk.GetType().Name} {name}");
                    StateText(bulk).Should().Be(StateText(single), $"{bulk.GetType().Name} {name} state");
                }
            }
        }

        /// <summary>The legacy <c>rand</c> bulk fill keeps RandomState's stream (it now fills in one locked pass).</summary>
        [TestMethod]
        public void LegacyRand_BulkFill_MatchesNumPy()
        {
            var rs = np.random.RandomState(42);
            D(rs.rand(5)).Should().Equal(0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652);
            var single = np.random.RandomState(42);
            var bulk = np.random.RandomState(42);
            double[] b = D(bulk.rand(new Shape(700)));
            var s = new double[700];
            for (int k = 0; k < 700; k++) s[k] = Convert.ToDouble(single.rand().GetAtIndex(0));
            b.Should().Equal(s);
        }

        // =====================================================================================
        //  Thread safety: the new generators serialize on BitGenerator.lock like the others
        // =====================================================================================

        /// <summary>
        ///     Threads sharing one Philox / SFC64 / PCG64DXSM Generator jointly consume exactly the sequential stream
        ///     (Philox's block buffer and every engine's buffered 32-bit half would tear without the lock).
        /// </summary>
        [TestMethod]
        public void SharedNewGenerators_ConcurrentDraws_ConsumeTheSequentialStream()
        {
            foreach (Func<BitGenerator> make in new Func<BitGenerator>[] { () => new Philox(7), () => new SFC64(7), () => new PCG64DXSM(7) })
            {
                const int threads = 4, per = 25_000;
                var reference = D(new Generator(make()).random(new Shape(threads * per), np.float32));
                Array.Sort(reference);

                var shared = new Generator(make());
                var parts = new double[threads][];
                Parallel.For(0, threads, t => parts[t] = D(shared.random(new Shape(per), np.float32)));
                var union = parts.SelectMany(p => p).ToArray();
                Array.Sort(union);
                union.Should().Equal(reference);
            }
        }
    }
}
