using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity tests for <see cref="NumPyRandom.default_rng"/> and the PCG64-backed
    ///     <see cref="Generator"/> against NumPy 2.4.2's <c>np.random.default_rng(42)</c>.
    /// </summary>
    [TestClass]
    public class NpRandomDefaultRngTest
    {
        private static void AssertDoubles(NDArray a, params double[] expected)
        {
            a.size.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
                Convert.ToDouble(a.GetAtIndex(i)).Should().Be(expected[i]);
        }

        private static void AssertInts(NDArray a, params long[] expected)
        {
            a.size.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
                Convert.ToInt64(a.GetAtIndex(i)).Should().Be(expected[i]);
        }

        [TestMethod]
        public void DefaultRng_Repr_And_BitGenerator()
        {
            var rng = np.random.default_rng(42);
            rng.ToString().Should().Be("Generator(PCG64)");
            rng.bit_generator.Name.Should().Be("PCG64");
            rng.bit_generator.Should().BeOfType<PCG64>();
        }

        [TestMethod]
        public void Random_ByteExact()
        {
            var rng = np.random.default_rng(42);
            AssertDoubles(rng.random(5),
                0.7739560485559633, 0.4388784397520523, 0.8585979199113825,
                0.6973680290593639, 0.09417734788764953);
        }

        [TestMethod]
        public void Integers_ByteExact()
        {
            var rng = np.random.default_rng(42);
            var a = rng.integers(0, 10, 5);
            a.dtype.Should().Be(typeof(long)); // default int64
            AssertInts(a, 0, 7, 6, 4, 4);
        }

        [TestMethod]
        public void StandardNormal_ByteExact_Ziggurat()
        {
            var rng = np.random.default_rng(42);
            AssertDoubles(rng.standard_normal(5),
                0.30471707975443135, -1.0399841062404955, 0.7504511958064572,
                0.9405647163912139, -1.9510351886538364);
        }

        [TestMethod]
        public void Exponential_And_Uniform_ByteExact()
        {
            AssertDoubles(np.random.default_rng(42).exponential(2.5, 4),
                6.0105215099149865, 5.840474139561134, 5.961902499685637, 0.6994857246289581);
            AssertDoubles(np.random.default_rng(42).uniform(-1, 1, 4),
                0.5479120971119267, -0.12224312049589536, 0.7171958398227649, 0.3947360581187278);
        }

        [TestMethod]
        public void Bytes_ByteExact()
        {
            np.random.default_rng(42).bytes(8).Should().Equal((byte)136, 38, 217, 22, 205, 251, 33, 198);
        }

        [TestMethod]
        public void Permutation_And_Choice_ByteExact()
        {
            AssertInts(np.random.default_rng(42).permutation(10), 5, 6, 0, 7, 3, 2, 4, 9, 1, 8);
            AssertInts(np.random.default_rng(42).choice(10, new Shape(5), replace: false), 5, 3, 7, 0, 4);
            AssertInts(np.random.default_rng(42).choice(6, new Shape(8),
                    p: np.array(new double[] { .4, .3, .1, .1, .05, .05 })),
                2, 1, 3, 1, 0, 5, 2, 2);
        }

        [TestMethod]
        public void Shuffle_InPlace_ByteExact()
        {
            var x = np.arange(10);
            np.random.default_rng(42).shuffle(x);
            AssertInts(x, 5, 6, 0, 7, 3, 2, 4, 9, 1, 8); // same permutation as permutation(10)
        }

        [TestMethod]
        public void SameSeed_IsDeterministic()
        {
            var a = np.random.default_rng(42).random(20);
            var b = np.random.default_rng(42).random(20);
            for (long i = 0; i < 20; i++)
                Convert.ToDouble(a.GetAtIndex(i)).Should().Be(Convert.ToDouble(b.GetAtIndex(i)));
        }

        [TestMethod]
        public void Float32_Random_ProducesFloatArray()
        {
            var a = np.random.default_rng(42).random(4, np.float32);
            a.dtype.Should().Be(typeof(float));
            a.size.Should().Be(4);
        }

        [TestMethod]
        public void NoSeed_RunsAndShapes()
        {
            var rng = np.random.default_rng();
            rng.random(new Shape(3, 4)).shape.Should().Equal(3, 4);
        }

        [TestMethod]
        public void Integers_OutOfBounds_Throws()
        {
            var rng = np.random.default_rng(0);
            Action act = () => rng.integers(0, 300, new Shape(5), np.uint8);
            act.Should().Throw<ValueError>().WithMessage("*high is out of bounds for uint8*");
        }

        [TestMethod]
        public void Integers_BadDtype_ThrowsNumpyMessage()
        {
            var rng = np.random.default_rng(0);
            Action act = () => rng.integers(0, 10, new Shape(5), np.float64);
            act.Should().Throw<TypeError>().WithMessage("Unsupported dtype dtype('float64') for integers");
        }

        [TestMethod]
        public void Uniform_HighLessThanLow_Throws()
        {
            // NumPy raises ValueError("high - low < 0") (CONS_BOUNDED_0 on the range).
            var rng = np.random.default_rng(42);
            Action act = () => rng.uniform(5, 1, 3);
            act.Should().Throw<ValueError>().WithMessage("high - low < 0");
        }

        [TestMethod]
        public void StandardGamma_Float32_IsSupported()
        {
            var a = np.random.default_rng(42).standard_gamma(2.0, new Shape(4), np.float32);
            a.dtype.Should().Be(typeof(float));
            a.size.Should().Be(4);
        }

        [TestMethod]
        public void BitGenerator_State_RoundTrips()
        {
            var rng = np.random.default_rng(99);
            rng.random(3); // advance
            var pcg = (PCG64)rng.bit_generator;
            var state = pcg.GetState();
            var a = rng.random(5);
            pcg.SetState(state);
            var b = rng.random(5);
            for (long i = 0; i < 5; i++)
                Convert.ToDouble(a.GetAtIndex(i)).Should().Be(Convert.ToDouble(b.GetAtIndex(i)));
        }

        [TestMethod]
        public void Out_Random_ReturnsSameInstance_ByteExact()
        {
            var rng = np.random.default_rng(42);
            var outp = np.empty(new Shape(5), np.float64);
            var r = rng.random(new Shape(5), @out: outp);
            ReferenceEquals(r, outp).Should().BeTrue();
            AssertDoubles(outp,
                0.7739560485559633, 0.4388784397520523, 0.8585979199113825,
                0.6973680290593639, 0.09417734788764953);
        }
    }
}
