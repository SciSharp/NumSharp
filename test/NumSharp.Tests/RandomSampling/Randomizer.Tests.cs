using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Behavioural tests for the MT19937 Mersenne Twister engine through NumPy's surface (the legacy seeding,
    ///     <c>state</c>, <c>random_raw</c>, and a <see cref="Generator"/> over it). The byte-exact NumPy pins live in
    ///     <see cref="MT19937ParityTest"/>; these keep the original intent of this suite — save/restore, reseeding,
    ///     copies, bounds — on the NumPy-shaped API that replaced the old ad-hoc members.
    /// </summary>
    [TestClass]
    public class MT19937Tests
    {
        /// <summary>A RandomState's state restored into another continues with the identical next draw.</summary>
        [TestMethod]
        public void SaveAndRestore()
        {
            var original = np.random.RandomState(42);
            original.randomizer.NextDouble();
            original.randomizer.NextDouble();
            var copy = np.random.RandomState();
            copy.set_state(original.get_state());
            var expectedNext = original.randomizer.NextDouble();
            copy.randomizer.NextDouble().Should().Be(expectedNext);
        }

        /// <summary>Legacy re-seeding with the same integer restarts the same stream (NumPy's <c>_legacy_seeding</c>).</summary>
        [TestMethod]
        public void LegacySeed42_ProducesConsistentSequence()
        {
            var mt = new MT19937();
            mt._legacy_seeding(42);
            var first = mt.random_raw(new Shape(3));

            mt._legacy_seeding(42);
            mt.random_raw(new Shape(3)).array_equal(first).Should().BeTrue();
        }

        /// <summary>Copying the state into another generator makes the two produce identical sequences from there on.</summary>
        [TestMethod]
        public void StateCopy_ProducesIdenticalSequence()
        {
            var mt1 = new MT19937(42);
            mt1.random_raw(new Shape(2));

            var mt2 = new MT19937(7);
            mt2.state = mt1.state;

            mt1.random_raw(new Shape(5)).array_equal(mt2.random_raw(new Shape(5))).Should().BeTrue();
        }

        /// <summary>The internal key/position restore used by RandomState.set_state resumes exactly where the source was.</summary>
        [TestMethod]
        public void SetState_RestoresExactState()
        {
            var mt1 = new MT19937(42);

            // Advance the state
            for (int i = 0; i < 100; i++)
                mt1.NextDouble();

            // Save state
            var key = mt1.Key;
            var pos = mt1.Pos;

            // Get next value
            var expected = mt1.NextDouble();

            // Create new generator and restore state
            var mt2 = new MT19937(0);
            mt2.SetState(key, pos);

            mt2.NextDouble().Should().Be(expected);
        }

        /// <summary>random_raw widens MT19937's 32-bit words to uint64 and never exceeds 2**32 - 1.</summary>
        [TestMethod]
        public void RandomRaw_Is32BitWordsWidened()
        {
            var raw = new MT19937(42).random_raw(new Shape(100));
            raw.dtype.Should().Be(np.uint64);
            for (long i = 0; i < raw.size; i++)
                ((ulong)raw.GetAtIndex(i)).Should().BeLessThanOrEqualTo(uint.MaxValue);
        }

        /// <summary>Bounded integers drawn through a Generator over MT19937 stay inside the requested range.</summary>
        [TestMethod]
        public void GeneratorIntegers_WithRange_StaysInBounds()
        {
            var g = new Generator(new MT19937(42));
            var small = g.integers(0, 10, new Shape(1000));
            var large = g.integers(0, 1000000000L, new Shape(1000));
            for (long i = 0; i < 1000; i++)
            {
                Convert.ToInt64(small.GetAtIndex(i)).Should().BeInRange(0, 9);
                Convert.ToInt64(large.GetAtIndex(i)).Should().BeInRange(0, 999999999L);
            }
        }

        /// <summary>The engine's doubles lie in [0, 1).</summary>
        [TestMethod]
        public void NextDouble_IsInRange()
        {
            var mt = new MT19937(42);

            for (int i = 0; i < 1000; i++)
            {
                var val = mt.NextDouble();
                val.Should().BeGreaterThanOrEqualTo(0.0);
                val.Should().BeLessThan(1.0);
            }
        }

        /// <summary>Array (init_by_array) legacy seeding is deterministic.</summary>
        [TestMethod]
        public void LegacyArraySeed_ProducesConsistentSequence()
        {
            var mt1 = new MT19937();
            var mt2 = new MT19937();

            var initKey = new uint[] { 1, 2, 3, 4 };
            mt1._legacy_seeding(initKey);
            mt2._legacy_seeding(initKey);

            // Both should produce identical sequences
            mt1.random_raw(new Shape(100)).array_equal(mt2.random_raw(new Shape(100))).Should().BeTrue();
        }
    }
}
