using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The two layout helpers of the sliding kernels behind np.correlate / np.convolve: <c>SlidingDot.RevertInPlace</c>
    ///     (NumPy's <c>_pyarray_revert</c>, which reverses a swapped product in its own buffer) and
    ///     <c>SlidingDot.MaterializeForSliding</c>'s one-dimensional copy (one strided move instead of the general copy).
    ///     Each must give the bits of the route it replaced - the reversed view's copy, and <see cref="NDArray.copy"/> of
    ///     the view - for every element width, with the bit patterns a value-level move could lose: NaN payloads of both
    ///     signs, signed zeros, a decimal's scale, complex parts.
    /// </summary>
    [TestClass]
    public class SlidingDotHelpersTests
    {
        /// <summary>The array's elements in C order as raw bytes.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>The bytes.</returns>
        private static byte[] Bytes(NDArray a) => np.ascontiguousarray(a).Unsafe.ReadOnlyBytes().ToArray();

        /// <summary>
        ///     One fresh array of <paramref name="n"/> elements per dtype (all 15): random bit patterns for the numeric
        ///     dtypes (so NaNs with payloads, infinities, subnormals and signed zeros occur), 0 / 1 for bool, valid decimals
        ///     with random scales, and fixed special values at the front where the length allows.
        /// </summary>
        /// <param name="n">The length.</param>
        /// <param name="seed">The random seed.</param>
        /// <returns>The arrays.</returns>
        private static NDArray[] Sources(int n, int seed = 7)
        {
            var rng = new System.Random(seed);
            NDArray Raw(NPTypeCode dtype, int itemSize)
            {
                if (n == 0)
                    return np.empty(new Shape(0L), dtype);
                var bytes = new byte[n * itemSize];
                rng.NextBytes(bytes);
                return np.frombuffer(bytes, dtype).copy();
            }

            var bools = new bool[n];
            for (int i = 0; i < n; i++)
                bools[i] = rng.Next(2) == 1;
            // Small mantissas with random scales: the scale is what a value-level move could lose, and the correlation test
            // multiplies and sums these, which full 96-bit mantissas would overflow.
            var decimals = new decimal[n];
            for (int i = 0; i < n; i++)
                decimals[i] = new decimal(rng.Next(100_000), 0, 0, rng.Next(2) == 1, (byte)rng.Next(29));
            var doubles = Raw(NPTypeCode.Double, 8);
            var halfs = Raw(NPTypeCode.Half, 2);
            var complexes = Raw(NPTypeCode.Complex, 16);
            if (n >= 3)
            {
                // A positive NaN with a payload, a negative zero and a negative NaN, at fixed places.
                doubles.SetDouble(BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0123), 0);
                doubles.SetDouble(-0.0, 1);
                doubles.SetDouble(BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8_0000_0000_0456)), 2);
                halfs.SetHalf(BitConverter.UInt16BitsToHalf(0x7E05), 0);
                halfs.SetHalf(BitConverter.UInt16BitsToHalf(0x8000), 1);
                complexes.SetComplex(new Complex(-0.0, BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0789)), 0);
            }

            return new[]
            {
                np.array(bools), Raw(NPTypeCode.Byte, 1), Raw(NPTypeCode.SByte, 1), Raw(NPTypeCode.Int16, 2),
                Raw(NPTypeCode.UInt16, 2), Raw(NPTypeCode.Char, 2), halfs, Raw(NPTypeCode.Int32, 4),
                Raw(NPTypeCode.UInt32, 4), Raw(NPTypeCode.Single, 4), Raw(NPTypeCode.Int64, 8), Raw(NPTypeCode.UInt64, 8),
                doubles, np.array(decimals), complexes,
            };
        }

        /// <summary>
        ///     <c>RevertInPlace</c> leaves a fresh array holding exactly the bytes of its reversed view's copy, for every
        ///     element width and for lengths around the vectorized reversal's widths (0, 1, odd, even, one past a vector).
        /// </summary>
        [TestMethod]
        public void RevertInPlace_EveryWidth_IsTheReversedViewsCopy()
        {
            foreach (int n in new[] { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 1000 })
            {
                foreach (var src in Sources(n, seed: n))
                {
                    using var scope = NDScope.Open();
                    byte[] expected = Bytes(src["::-1"].copy());
                    var ret = src.copy();
                    SlidingDot.RevertInPlace(ret);
                    Bytes(ret).Should().Equal(expected, "a {0} array of {1} elements reverses bit for bit", src.dtype, n);
                    ret.Shape.IsContiguous.Should().BeTrue();
                    ret.Shape.offset.Should().Be(0);
                }
            }
        }

        /// <summary>
        ///     <c>MaterializeForSliding</c> of a one-dimensional view of the compute dtype - reversed, stepped either way,
        ///     offset, or broadcast - is a contiguous, offset-0 array with the bytes of the view's general copy.
        /// </summary>
        [TestMethod]
        public void MaterializeForSliding_OneDimensionalViews_AreTheGeneralCopy()
        {
            foreach (var src in Sources(37))
            {
                foreach (string slice in new[] { "::-1", "::2", "::-3", "1:", "5:30", "-1::-2", "30:4:-5", "36:", "3:4" })
                {
                    using var scope = NDScope.Open();
                    var view = src[slice];
                    var got = SlidingDot.MaterializeForSliding(view, view.GetTypeCode);
                    Bytes(got).Should().Equal(Bytes(view.copy()), "the {0} view [{1}] copies bit for bit", src.dtype, slice);
                    got.Shape.IsContiguous.Should().BeTrue();
                    got.Shape.IsSliced.Should().BeFalse("the sliding kernels read the buffer from its start");
                }

                using (NDScope.Open())
                {
                    var broadcast = np.broadcast_to(src["3:4"], new Shape(9L));
                    var got = SlidingDot.MaterializeForSliding(broadcast, broadcast.GetTypeCode);
                    Bytes(got).Should().Equal(Bytes(broadcast.copy()), "a broadcast {0} view repeats its element", src.dtype);
                    got.Shape.IsBroadcasted.Should().BeFalse();
                }
            }
        }

        /// <summary>
        ///     <c>MaterializeForSliding</c> hands back the operand itself when it already is a contiguous, offset-0 array of
        ///     the compute dtype (no copy), and a cast for another dtype.
        /// </summary>
        [TestMethod]
        public void MaterializeForSliding_ContiguousOperand_IsNotCopied()
        {
            var a = np.arange(10.0);
            ReferenceEquals(SlidingDot.MaterializeForSliding(a, NPTypeCode.Double), a).Should().BeTrue();
            using var scope = NDScope.Open();
            var cast = SlidingDot.MaterializeForSliding(np.arange(10), NPTypeCode.Double);
            cast.typecode.Should().Be(NPTypeCode.Double);
            cast.GetDouble(9).Should().Be(9.0);
        }

        /// <summary>
        ///     For a real dtype NumPy's <c>PyArray_Correlate2</c> computes the same products in the same order whichever
        ///     operand is first (the longer one is the data), so <c>np.correlate(v, a)</c> is <c>np.correlate(a, v)</c>
        ///     reversed, bit for bit - NaN payloads included - in every mode: the swapped product's in-place reversal at work.
        /// </summary>
        [TestMethod]
        public void Correlate_SwappedRealOperands_AreTheReversedProductBitForBit()
        {
            var longs = Sources(37, seed: 11);
            var shorts = Sources(5, seed: 12);
            for (int i = 0; i < longs.Length; i++)
            {
                if (longs[i].typecode == NPTypeCode.Complex)
                    continue; // conj(v) is taken before the swap, so the swapped complex product is another value
                foreach (string mode in new[] { "valid", "same", "full" })
                {
                    using var scope = NDScope.Open();
                    var forward = np.correlate(longs[i], shorts[i], mode);
                    var swapped = np.correlate(shorts[i], longs[i], mode);
                    Bytes(swapped).Should().Equal(Bytes(forward["::-1"]), "{0} in mode {1}", longs[i].dtype, mode);
                }
            }
        }
    }
}
