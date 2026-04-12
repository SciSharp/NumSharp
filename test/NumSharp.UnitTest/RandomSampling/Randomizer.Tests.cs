using System;
using AwesomeAssertions;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    ///     Tests for the MT19937 Mersenne Twister random number generator.
    ///     These tests verify NumPy compatibility.
    /// </summary>
    public class MT19937Tests
    {
        [Test]
        public void SaveAndRestore()
        {
            var original = np.random.RandomState(42);
            var a = original.randomizer.NextDouble();
            var b = original.randomizer.NextDouble();
            var copy = np.random.RandomState();
            copy.set_state(original.get_state());
            var expectedNext = original.randomizer.NextDouble();
            copy.randomizer.NextDouble().Should().Be(expectedNext);
        }

        [Test]
        public void Seed42_ProducesConsistentSequence()
        {
            var mt = new MT19937(42);

            // First few uint32 values should be deterministic
            var v1 = mt.NextUInt32();
            var v2 = mt.NextUInt32();
            var v3 = mt.NextUInt32();

            // Reseed and verify same sequence
            mt.Seed(42);
            mt.NextUInt32().Should().Be(v1);
            mt.NextUInt32().Should().Be(v2);
            mt.NextUInt32().Should().Be(v3);
        }

        [Test]
        public void Clone_ProducesIdenticalSequence()
        {
            var mt1 = new MT19937(42);
            mt1.NextDouble();
            mt1.NextDouble();

            var mt2 = mt1.Clone();

            // Both should produce identical sequences from this point
            mt1.NextDouble().Should().Be(mt2.NextDouble());
            mt1.NextDouble().Should().Be(mt2.NextDouble());
            mt1.NextUInt32().Should().Be(mt2.NextUInt32());
        }

        [Test]
        public void SetState_RestoresExactState()
        {
            var mt1 = new MT19937(42);

            // Advance the state
            for (int i = 0; i < 100; i++)
                mt1.NextDouble();

            // Save state
            var key = (uint[])mt1.Key.Clone();
            var pos = mt1.Pos;

            // Get next value
            var expected = mt1.NextDouble();

            // Create new generator and restore state
            var mt2 = new MT19937(0);
            mt2.SetState(key, pos);

            mt2.NextDouble().Should().Be(expected);
        }

        [Test]
        public void NextBytes_FillsBuffer()
        {
            var mt = new MT19937(42);
            var buffer = new byte[100];

            mt.NextBytes(buffer);

            // Should have some non-zero values
            var hasNonZero = false;
            foreach (var b in buffer)
                if (b != 0) hasNonZero = true;

            hasNonZero.Should().BeTrue();
        }

        [Test]
        public void Next_WithRange_StaysInBounds()
        {
            var mt = new MT19937(42);

            for (int i = 0; i < 1000; i++)
            {
                var val = mt.Next(10);
                val.Should().BeGreaterThanOrEqualTo(0);
                val.Should().BeLessThan(10);
            }
        }

        [Test]
        public void NextLong_WithRange_StaysInBounds()
        {
            var mt = new MT19937(42);

            for (int i = 0; i < 1000; i++)
            {
                var val = mt.NextLong(1000000000L);
                val.Should().BeGreaterThanOrEqualTo(0);
                val.Should().BeLessThan(1000000000L);
            }
        }

        [Test]
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

        [Test]
        public void SeedByArray_ProducesConsistentSequence()
        {
            var mt1 = new MT19937();
            var mt2 = new MT19937();

            var initKey = new uint[] { 1, 2, 3, 4 };
            mt1.SeedByArray(initKey);
            mt2.SeedByArray(initKey);

            // Both should produce identical sequences
            for (int i = 0; i < 100; i++)
            {
                mt1.NextUInt32().Should().Be(mt2.NextUInt32());
            }
        }
    }
}
