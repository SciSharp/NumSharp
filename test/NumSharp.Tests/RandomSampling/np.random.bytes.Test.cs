using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity tests for <see cref="NumPyRandom.bytes"/> against NumPy 2.4.2's
    ///     <c>np.random.bytes</c> (MT19937, seed 42).
    /// </summary>
    [TestClass]
    public class NpRandomBytesTest
    {
        [TestMethod]
        public void Bytes_Length10_ByteExact()
        {
            np.random.seed(42);
            np.random.bytes(10).Should().Equal((byte)102, 220, 225, 95, 179, 61, 234, 203, 92, 3);
        }

        [TestMethod]
        public void Bytes_Length7_ByteExact()
        {
            np.random.seed(42);
            np.random.bytes(7).Should().Equal((byte)102, 220, 225, 95, 179, 61, 234);
        }

        [TestMethod]
        public void Bytes_Zero_IsEmpty_ButConsumesOneWord()
        {
            // NumPy: n_uint32 = ((0-1)//4 + 1) == 1 (C truncation) -> draws one uint32 then slices [:0].
            np.random.seed(42);
            var empty = np.random.bytes(0);
            empty.Should().BeEmpty();

            // Because bytes(0) consumed one 32-bit word, the next 4 bytes are the SECOND word.
            np.random.seed(42);
            var b8 = np.random.bytes(8); // words[0..1]
            var tail4 = new byte[4];
            Array.Copy(b8, 4, tail4, 0, 4);

            np.random.seed(42);
            np.random.bytes(0);
            np.random.bytes(4).Should().Equal(tail4); // == second word
        }

        [TestMethod]
        public void Bytes_NegativeLength_MatchesNumpyQuirk()
        {
            // NumPy bytes(-1): n_uint32 == 1 -> 4 bytes then Python slice [:-1] -> first 3 bytes.
            np.random.seed(42);
            np.random.bytes(-1).Should().Equal((byte)102, 220, 225);

            // bytes(-5): n_uint32 == 0 -> empty.
            np.random.seed(42);
            np.random.bytes(-5).Should().BeEmpty();
        }

        [TestMethod]
        public void Bytes_TooNegative_Throws()
        {
            // NumPy: bytes(-8) -> n_uint32 == -1 -> ValueError "negative dimensions are not allowed".
            np.random.seed(42);
            Action act = () => np.random.bytes(-8);
            act.Should().Throw<ValueError>().WithMessage("*negative dimensions are not allowed*");
        }
    }
}
