using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity tests for <see cref="NumPyRandom.bytes"/> against NumPy 2.4.2's
    ///     <c>np.random.bytes</c> (MT19937, seed 42). The result is a 1-D uint8 <see cref="NDArray"/>
    ///     (NumSharp's analogue of NumPy's <c>bytes</c> object) and — unlike a managed <c>byte[]</c> —
    ///     can exceed 2 GiB, matching NumPy's 64-bit <c>npy_intp</c> length.
    /// </summary>
    [TestClass]
    public class NpRandomBytesTest
    {
        // Mirrors AssertInts in np.random.default_rng.Test.cs: asserts a 1-D uint8 array's bytes.
        private static void AssertBytes(NDArray a, params byte[] expected)
        {
            a.typecode.Should().Be(NPTypeCode.Byte);
            a.ndim.Should().Be(1);
            a.size.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
                Convert.ToByte(a.GetAtIndex(i)).Should().Be(expected[i]);
        }

        [TestMethod]
        public void Bytes_Length10_ByteExact()
        {
            np.random.seed(42);
            AssertBytes(np.random.bytes(10), 102, 220, 225, 95, 179, 61, 234, 203, 92, 3);
        }

        [TestMethod]
        public void Bytes_Length7_ByteExact()
        {
            np.random.seed(42);
            AssertBytes(np.random.bytes(7), 102, 220, 225, 95, 179, 61, 234);
        }

        [TestMethod]
        public void Bytes_ReturnsOneDimensionalUInt8NDArray()
        {
            // NumPy returns a `bytes` object; NumSharp's analogue is a 1-D uint8 NDArray.
            // The explicit typed local is a compile-time assertion that bytes(long) returns NDArray<byte>.
            np.random.seed(42);
            NumSharp.Generic.NDArray<byte> typed = np.random.bytes(5);
            typed.typecode.Should().Be(NPTypeCode.Byte);
            typed.ndim.Should().Be(1);
            typed.shape.Should().Equal(5);
        }

        [TestMethod]
        public void Bytes_Zero_IsEmpty_ButConsumesOneWord()
        {
            // NumPy: n_uint32 = ((0-1)//4 + 1) == 1 (C truncation) -> draws one uint32 then slices [:0].
            np.random.seed(42);
            var empty = np.random.bytes(0);
            empty.size.Should().Be(0);
            empty.ndim.Should().Be(1);
            empty.typecode.Should().Be(NPTypeCode.Byte);

            // Because bytes(0) consumed one 32-bit word, the next 4 bytes are the SECOND word.
            np.random.seed(42);
            var b8 = np.random.bytes(8); // words[0..1]
            var tail4 = new byte[4];
            for (int i = 0; i < 4; i++)
                tail4[i] = Convert.ToByte(b8.GetAtIndex(4 + i));

            np.random.seed(42);
            np.random.bytes(0);
            AssertBytes(np.random.bytes(4), tail4); // == second word
        }

        [TestMethod]
        public void Bytes_NegativeLength_MatchesNumpyQuirk()
        {
            // NumPy bytes(-1): n_uint32 == 1 -> 4 bytes then Python slice [:-1] -> first 3 bytes.
            np.random.seed(42);
            AssertBytes(np.random.bytes(-1), 102, 220, 225);

            // bytes(-5): n_uint32 == 0 -> empty.
            np.random.seed(42);
            np.random.bytes(-5).size.Should().Be(0);
        }

        [TestMethod]
        public void Bytes_TooNegative_Throws()
        {
            // NumPy: bytes(-8) -> n_uint32 == -1 -> ValueError "negative dimensions are not allowed".
            np.random.seed(42);
            Action act = () => np.random.bytes(-8);
            act.Should().Throw<ValueError>().WithMessage("*negative dimensions are not allowed*");
        }

        [TestMethod]
        [HighMemory]
        [LongIndexing]
        public void Bytes_ExceedsIntMaxValue_Allocates()
        {
            // NumPy's bytes() length is a 64-bit npy_intp and its result can exceed 2 GiB. NumSharp
            // returns an unmanaged-backed NDArray<byte>, so a request larger than a .NET byte[]
            // (Array.MaxLength ~ 2 GiB) succeeds where it used to raise. Byte-exact with NumPy's
            // MT19937 stream in the region PAST int.MaxValue (verified: bytes[2147483690] == 114).
            const long length = 2_147_483_700L; // 53 bytes past int.MaxValue (2_147_483_647)
            np.random.seed(42);
            var big = np.random.bytes(length);
            try
            {
                big.size.Should().Be(length);
                big.size.Should().BeGreaterThan(int.MaxValue);
                big.ndim.Should().Be(1);
                big.typecode.Should().Be(NPTypeCode.Byte);

                // Same seed 42 -> identical MT19937 stream prefix as bytes(10).
                var prefix = new byte[] { 102, 220, 225, 95 };
                for (int i = 0; i < prefix.Length; i++)
                    Convert.ToByte(big.GetAtIndex(i)).Should().Be(prefix[i]);

                // A byte BEYOND int.MaxValue is addressable and byte-exact with NumPy's stream.
                Convert.ToByte(big.GetAtIndex(2_147_483_690L)).Should().Be(114);
            }
            finally
            {
                big.Dispose();
            }
        }
    }
}
