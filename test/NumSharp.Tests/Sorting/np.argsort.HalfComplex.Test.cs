using System;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    ///     np.argsort of float16 and complex128 is NumPy's stable order (<c>kind='stable'</c>: timsort with
    ///     <c>npy::half_tag::less</c> / <c>CDOUBLE_LT</c>) for every kind, by radix: float16 over a 16-bit key with both
    ///     zeros folded and the NaNs last in input order; complex128 partitioned by NaN class (no NaN part, <c>R+nanj</c>,
    ///     <c>nan+Rj</c>, <c>nan+nanj</c>) and radix-sorted per class. The comparison argsort these kernels replaced gave the
    ///     same indices at about half NumPy's speed. Expected values are NumPy 2.4.2's <c>np.argsort(a, kind='stable')</c>
    ///     over the same formula-built lines (the lengths 200 and 300 are past the radix's insertion thresholds, 80 and 120
    ///     keys, so the scatter passes run); NumPy's default kind orders some of these ties its own way.
    /// </summary>
    [TestClass]
    public class np_argsort_half_complex_tests
    {
        private static readonly ushort[] HalfPool = { 0x7E00, 0xFE00, 0x7E01, 0x0000, 0x8000, 0x7C00, 0xFC00, 0x0001, 0x8001 };

        private static readonly ulong[] DoublePool =
        {
            0x7FF8000000000000, 0x7FF8000000000001, 0xFFF8000000000000, 0x0000000000000000, 0x8000000000000000,
            0x7FF0000000000000, 0xFFF0000000000000, 0x3FF0000000000000,
        };

        /// <summary>
        ///     The float16 line of the probe: every fifth element from <see cref="HalfPool"/> (NaNs of both signs and two
        ///     payloads, both zeros, both infinities, the smallest subnormals), the others repeated halves in [-2.5, 2.5].
        /// </summary>
        /// <param name="n">The length.</param>
        /// <returns>The float16 array.</returns>
        private static NDArray HalfLine(int n)
        {
            var v = new Half[n];
            for (int i = 0; i < n; i++)
                v[i] = i % 5 == 0 ? BitConverter.UInt16BitsToHalf(HalfPool[(i / 5) % HalfPool.Length]) : (Half)(((i * 37) % 11 - 5) * 0.5f);
            return np.array(v);
        }

        /// <summary>
        ///     The complex128 line of the probe: the real part from <see cref="DoublePool"/> at every fourth element, the
        ///     imaginary part at every third (so all four NaN classes, both zeros and both infinities meet), the others
        ///     repeated halves.
        /// </summary>
        /// <param name="n">The length.</param>
        /// <returns>The complex128 array.</returns>
        private static NDArray ComplexLine(int n)
        {
            var v = new Complex[n];
            for (int i = 0; i < n; i++)
            {
                double re = i % 4 == 0 ? BitConverter.UInt64BitsToDouble(DoublePool[(i * 3) % DoublePool.Length]) : ((i * 37) % 11 - 5) * 0.5;
                double im = i % 3 == 0 ? BitConverter.UInt64BitsToDouble(DoublePool[(i * 5) % DoublePool.Length]) : ((i * 13) % 7 - 3) * 0.5;
                v[i] = new Complex(re, im);
            }
            return np.array(v);
        }

        private static readonly string[] Kinds = { null, "quicksort", "stable", "mergesort", "heapsort" };

        [TestMethod]
        public void Float16_IsNumPysStableOrder_ForEveryKind()
        {
            var shortLine = HalfLine(12);
            var longLine = HalfLine(200);
            foreach (var kind in Kinds)
            {
                np.argsort(shortLine, kind: kind).ToArray<long>().Should().Equal(11L, 3, 6, 9, 1, 4, 7, 2, 8, 0, 5, 10);
                np.argsort(longLine, kind: kind).ToArray<long>().Should().Equal(
                    30L, 75, 120, 165, 11, 22, 33, 44, 66, 77, 88, 99, 121, 132, 143, 154, 176, 187, 198, 3, 14, 36, 47, 58, 69,
                    91, 102, 113, 124, 146, 157, 168, 179, 6, 17, 28, 39, 61, 72, 83, 94, 116, 127, 138, 149, 171, 182, 193, 9,
                    31, 42, 53, 64, 86, 97, 108, 119, 141, 152, 163, 174, 196, 1, 12, 23, 34, 56, 67, 78, 89, 111, 122, 133, 144,
                    166, 177, 188, 199, 40, 85, 130, 175, 4, 15, 20, 26, 37, 48, 59, 60, 65, 81, 92, 103, 105, 110, 114, 136, 147,
                    150, 155, 158, 169, 191, 195, 35, 80, 125, 170, 7, 18, 29, 51, 62, 73, 84, 106, 117, 128, 139, 161, 172, 183,
                    194, 21, 32, 43, 54, 76, 87, 98, 109, 131, 142, 153, 164, 186, 197, 2, 13, 24, 46, 57, 68, 79, 101, 112, 123,
                    134, 156, 167, 178, 189, 16, 27, 38, 49, 71, 82, 93, 104, 126, 137, 148, 159, 181, 192, 8, 19, 41, 52, 63, 74,
                    96, 107, 118, 129, 151, 162, 173, 184, 25, 70, 115, 160, 0, 5, 10, 45, 50, 55, 90, 95, 100, 135, 140, 145, 180,
                    185, 190);
            }
        }

        [TestMethod]
        public void Complex128_IsNumPysStableOrder_ForEveryKind()
        {
            var shortLine = ComplexLine(12);
            var longLine = ComplexLine(300);
            foreach (var kind in Kinds)
            {
                np.argsort(shortLine, kind: kind).ToArray<long>().Should().Equal(11L, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 0);
                np.argsort(longLine, kind: kind).ToArray<long>().Should().Equal(
                    198L, 77, 154, 55, 209, 286, 110, 187, 11, 231, 242, 143, 99, 121, 275, 22, 253, 33, 297, 102, 14, 91, 245,
                    146, 223, 47, 278, 25, 135, 179, 157, 3, 58, 267, 289, 113, 190, 201, 6, 270, 182, 259, 83, 61, 215, 39, 193,
                    17, 94, 149, 171, 226, 50, 127, 281, 105, 174, 119, 97, 251, 229, 53, 130, 207, 31, 185, 262, 75, 86, 163, 218,
                    295, 9, 273, 78, 133, 287, 34, 265, 89, 166, 67, 111, 221, 298, 122, 199, 23, 243, 254, 1, 155, 177, 246, 28,
                    70, 140, 196, 20, 76, 125, 188, 202, 244, 26, 68, 103, 124, 236, 257, 292, 4, 12, 15, 36, 60, 84, 108, 116,
                    132, 156, 158, 172, 180, 204, 228, 235, 252, 276, 279, 284, 52, 59, 164, 220, 290, 37, 44, 100, 147, 191, 212,
                    268, 92, 148, 169, 260, 81, 150, 7, 161, 238, 62, 139, 293, 194, 271, 95, 183, 73, 227, 51, 205, 29, 106, 249,
                    54, 98, 175, 230, 131, 87, 109, 263, 10, 241, 65, 142, 219, 43, 197, 274, 153, 222, 35, 266, 13, 167, 145, 299,
                    46, 255, 277, 101, 178, 2, 79, 123, 233, 134, 211, 57, 126, 49, 203, 181, 5, 82, 137, 159, 214, 38, 115, 269,
                    27, 170, 247, 291, 71, 225, 30, 294, 217, 41, 118, 19, 173, 250, 63, 74, 151, 206, 283, 107, 195, 85, 239, 129,
                    66, 165, 69, 234, 138, 237, 42, 141, 45, 210, 114, 213, 18, 117, 282, 21, 186, 285, 90, 189, 93, 258, 162, 261,
                    56, 112, 224, 280, 104, 160, 272, 40, 152, 208, 32, 88, 200, 256, 80, 136, 248, 16, 128, 184, 296, 8, 64, 176,
                    232, 0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288);
            }
        }

        /// <summary>
        ///     NumPy's complex NaN classes in their own order: <c>R+nanj</c> by R before <c>nan+Rj</c> by R, then
        ///     <c>nan+nanj</c>; both zeros equal in either part, so equal values keep their input order.
        /// </summary>
        [TestMethod]
        public void Complex128_NaNClassesAndSignedZeros()
        {
            double nan = BitConverter.UInt64BitsToDouble(0x7FF8000000000000);
            // np.argsort([nan+0j, 1+nanj, 2+0j], kind='stable') is [2, 1, 0]
            np.argsort(np.array(new[] { new Complex(nan, 0), new Complex(1, nan), new Complex(2, 0) })).ToArray<long>()
              .Should().Equal(2L, 1, 0);
            // np.argsort([nan+nanj, nan+1j, 0+nanj, 1+0j], kind='stable') is [3, 2, 1, 0]
            np.argsort(np.array(new[] { new Complex(nan, nan), new Complex(nan, 1), new Complex(0, nan), new Complex(1, 0) }))
              .ToArray<long>().Should().Equal(3L, 2, 1, 0);
            // np.argsort([0j, complex(-0.0, 0.0), complex(0.0, -0.0), -0j], kind='stable') is [0, 1, 2, 3]
            np.argsort(np.array(new[] { new Complex(0, 0), new Complex(-0.0, 0), new Complex(0, -0.0), new Complex(-0.0, -0.0) }))
              .ToArray<long>().Should().Equal(0L, 1, 2, 3);
        }

        /// <summary>Along axis 0 of a C array the lines are strided: the radix reads them through their stride.</summary>
        [TestMethod]
        public void Axis0_StridedLines_AreNumPysStableOrder()
        {
            np.argsort(HalfLine(24).reshape(6, 4), 0).ToArray<long>().Should().Equal(
                3L, 4, 5, 2, 1, 2, 3, 0, 5, 0, 1, 5, 4, 5, 4, 3, 2, 3, 0, 1, 0, 1, 2, 4);
            np.argsort(ComplexLine(24).reshape(6, 4), 0).ToArray<long>().Should().Equal(
                5L, 4, 5, 2, 1, 2, 3, 0, 3, 0, 1, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0, 5, 4, 4);
        }

        /// <summary>A line of one element is index 0, and an empty line gives no index (the radix never sees either).</summary>
        [TestMethod]
        public void OneElementAndEmpty()
        {
            np.argsort(np.array(new[] { BitConverter.UInt16BitsToHalf(0xFE00) })).ToArray<long>().Should().Equal(0L);
            np.argsort(np.array(new[] { new Complex(double.NaN, 1) })).ToArray<long>().Should().Equal(0L);
            np.argsort(np.array(Array.Empty<Half>())).size.Should().Be(0);
            np.argsort(np.array(Array.Empty<Complex>())).size.Should().Be(0);
        }
    }
}
