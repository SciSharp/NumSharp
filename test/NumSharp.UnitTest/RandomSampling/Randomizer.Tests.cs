using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    [TestClass]
    public class RandomizerTests
    {
        [TestMethod]
        public void SaveAndRestore()
        {
            var original = np.random.RandomState(42);
            var a = original.randomizer.Next();
            var b = original.randomizer.Next();
            var copy = np.random.RandomState();
            copy.set_state(original.get_state());
            var expectedNext = original.randomizer.Next();
            copy.randomizer.Next().Should().Be(expectedNext);
        }

        [TestMethod]
        public void CompareRandomizerToRandom()
        {
            var rnd = new System.Random(42);
            var rndizer = new Randomizer(42);

            rnd.Next().Should().Be(rndizer.Next());
            rnd.Next().Should().Be(rndizer.Next());

            var bytes_a = new byte[50];
            var bytes_b = new byte[50];
            rnd.NextBytes(bytes_a);
            rndizer.NextBytes(bytes_b);
            bytes_a.Should().BeEquivalentTo(bytes_b);

            rnd.NextBytes(bytes_a);
            rndizer.NextBytes(bytes_b);
            bytes_a.Should().BeEquivalentTo(bytes_b);

            rnd.NextDouble().Should().Be(rndizer.NextDouble());
            rnd.NextDouble().Should().Be(rndizer.NextDouble());
        }
    }
}
