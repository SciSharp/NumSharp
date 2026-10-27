using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Statistics
{
    /// <summary>
    ///     np.average over an axis TUPLE: NumPy's <c>a.mean(axis=tuple)</c> and <c>wgt.sum(axis=tuple, dtype=...)</c> are ONE
    ///     add.reduce over the whole set, whose iterator decides the order of the additions, and the empty tuple still
    ///     adds every element to the identity. NumSharp averaged one axis at a time (a mean of means, rounded at each
    ///     step) and returned the operand itself for the empty tuple. Expected values are NumPy 2.4.2's bits.
    /// </summary>
    [TestClass]
    public class np_average_TupleAxes_Tests
    {
        /// <summary>The raw bits of every float64 element in C order.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>The bits.</returns>
        private static long[] Bits(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            return c.ToArray<double>().Select(BitConverter.DoubleToInt64Bits).ToArray();
        }

        /// <summary>The raw bits of every complex128 element's two parts in C order.</summary>
        /// <param name="a">A complex128 array.</param>
        /// <returns>The bits, real then imaginary per element.</returns>
        private static long[] ComplexBits(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            return c.ToArray<Complex>().SelectMany(z => new[] { BitConverter.DoubleToInt64Bits(z.Real), BitConverter.DoubleToInt64Bits(z.Imaginary) }).ToArray();
        }

        [TestMethod]
        public void Mean_OverTwoLeadingAxes_IsOneReductionNotAMeanOfMeans()
        {
            // np.average(a, axis=(0, 1)) of a C-ordered (2, 3, 4) array: the innermost axis is kept, so each output adds
            // its six elements one after the other and divides once. The mean of means rounds after each axis: it is one
            // ULP off in all four outputs (0x...A26, 0x...5D7, 0x...396, 0x...27A).
            var a = np.array(new[]
            {
                -801.931, -1324.359, -248.362, 420.445, 1136.047, 109.706, -552.647, -784.78, 748.746, 1634.783, 272.769,
                -1233.329, -958.265, 1600.019, 202.882, -1732.135, -83.696, -1163.226, -629.288, -488.006, -713.313, 553.378,
                -63.086, -589.431,
            }).reshape(2, 3, 4);
            using var m = np.average(a, new[] { 0, 1 });
            Bits(m).Should().Equal(unchecked((long)0xC05C046508DFEA27), 0x406D619AF72015D8, unchecked((long)0xC06533E76C8B4395),
                unchecked((long)0xC086F4508DFEA279));
            using var swapped = np.average(a, new[] { 1, 0 });
            Bits(swapped).Should().Equal(Bits(m), "the order of the tuple does not change the reduction");
        }

        [TestMethod]
        public void Mean_OverAxesAroundAKeptOne_IsNumPysSchedule()
        {
            // (0, 2) of a C-ordered (3, 2, 9) array: the innermost axis is reduced (one pairwise sum of 9 per row and
            // call), the outer one adds the rows' sums in order.
            var a = np.array(new[]
            {
                409.638, 829.855, -1643.023, -256.73, -980.747, -173.155, -1289.419, 20.69, -37.886, -304.338, -1047.927,
                -396.19, -1091.329, -1355.209, 224.786, -1109.35, 1170.296, 716.588, -1997.817, 272.129, -1101.717, 33.057,
                43.632, -1988.43, -233.423, -255.79, 962.001, -1181.447, 738.042, -1098.973, -331.291, -840.473, 1448.731,
                568.213, 2431.733, 641.916, 844.993, 840.683, -606.612, -70.028, 1350.389, -396.551, 188.8, -21.223, 609.216,
                -364.909, -152.362, 242.381, 103.023, -864.973, 895.783, -1298.481, -1201.116, -1282.492,
            }).reshape(3, 2, 9);
            using var m = np.average(a, new[] { 0, 2 });
            Bits(m).Should().Equal(unchecked((long)0xC065841C378607BE), unchecked((long)0xC065F1075B8D9D8B));
        }

        [TestMethod]
        public void Mean_OverTheEmptyTuple_AddsTheIdentityAndDividesByOne()
        {
            // a.mean(axis=()) is umr_sum over no axis (0 + x for each element: -0.0 becomes +0.0) divided by a count of
            // 1, in _mean's loop: an int32 operand gives float64, float16 sums in float32 and comes back float16, and a
            // complex element is divided by 1+0j (Smith's algorithm turns an infinite part's partner into NaN).
            using (var ints = np.average(np.array(new[,] { { 1, -2 }, { 3, 4 } }), Array.Empty<int>()))
            {
                ints.dtype.Should().Be(np.float64);
                ints.shape.Should().Equal(2, 2);
                ints.flags.owndata.Should().BeTrue("the mean over no axis is a fresh array, not the operand");
            }

            using (var f = np.average(np.array(new[] { -0.0, 1.5, -2.25 }), Array.Empty<int>()))
                Bits(f).Should().Equal(0L, 0x3FF8000000000000, unchecked((long)0xC002000000000000));

            var c = np.array(new[] { new Complex(double.PositiveInfinity, 1.0), new Complex(1.0, double.PositiveInfinity), new Complex(-0.0, 2.0) });
            using (var z = np.average(c, Array.Empty<int>()))
                ComplexBits(z).Should().Equal(0x7FF0000000000000, unchecked((long)0xFFF8000000000000), unchecked((long)0xFFF8000000000000),
                    0x7FF0000000000000, 0L, 0x4000000000000000);

            using (var h = np.average(np.array(new[] { (Half)(-0.0), (Half)1.5, Half.MaxValue }), Array.Empty<int>()))
            {
                h.dtype.Should().Be(np.float16);
                h.ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).Should().Equal((ushort)0x0000, (ushort)0x3E00, (ushort)0x7BFF);
            }
        }

        [TestMethod]
        public void Returned_OverTheEmptyTuple_SumOfWeightsIsTheFloat64Count()
        {
            // scl = avg.dtype.type(a.size / avg.size): the mean is float64, so the count is float64 1.0, broadcast to the
            // mean's shape (it was the operand's int32 dtype while the operand itself came back).
            var (avg, scl) = np.average_returned(np.array(new[,] { { 1, -2 }, { 3, 4 } }), Array.Empty<int>());
            using (avg)
            using (scl)
            {
                scl.dtype.Should().Be(np.float64);
                scl.ToArray<double>().Should().Equal(1.0, 1.0, 1.0, 1.0);
            }
        }

        [TestMethod]
        public void Mean_OverATuple_OwnsItsDataInNumPysLayout()
        {
            // NumPy's iterator allocates the result in the operand's axis order: an F-ordered (2, 3, 4, 5) operand averaged
            // over (1, 3) gives an F-ordered (2, 4) result (strides (8, 16)); with keepdims the reduced axes stay as
            // length-1 dimensions of an array that owns its data (a reshape view did not).
            using var a4 = np.asfortranarray(np.arange(120.0).reshape(2, 3, 4, 5));
            using (var m = np.average(a4, new[] { 1, 3 }))
            {
                m.shape.Should().Equal(2, 4);
                m.flags.f_contiguous.Should().BeTrue();
                m.flags.c_contiguous.Should().BeFalse();
                m.flags.owndata.Should().BeTrue();
                m.ToArray<double>().Should().Equal(22.0, 27.0, 32.0, 37.0, 82.0, 87.0, 92.0, 97.0);
            }

            using var af = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4));
            using (var kd = np.average(af, new[] { 0, 2 }, keepdims: true))
            {
                kd.shape.Should().Equal(1, 3, 1);
                kd.flags.owndata.Should().BeTrue();
                kd.ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            }
        }

        [TestMethod]
        public void Mean_OverTheOnlyAxisAsATuple_KeepsTheAxisRoute()
        {
            // (0,) of a 1-D operand is one axis AND every axis. NumPy reduces it like axis=0 and axis=None (one call of
            // 2317 stride-0 elements); NumSharp's flat mean still declines a broadcast operand (its own fold, another
            // order), so the one-axis route must win: it is NumPy's schedule here. Found by the random differential.
            var z = new Complex(BitConverter.Int64BitsToDouble(0x3F719DEA7CEC23CD), BitConverter.Int64BitsToDouble(unchecked((long)0xC053252E6FCAEE21)));
            using var b = np.broadcast_to(np.array(new[] { z }), new Shape(2317));
            using var m = np.average(b, new[] { 0 }, keepdims: true);
            m.shape.Should().Equal(1);
            ComplexBits(m).Should().Equal(0x3F719DEA7CEC23CC, unchecked((long)0xC053252E6FCAEE21));
        }

        [TestMethod]
        public void WeightedComplex_OverATuple_SumsEachOnceAndOwnsTheResults()
        {
            // A complex operand has no fused weighted-sum kernel: its weight sum and weighted sum are add.reduce over the
            // tuple, NumPy's wgt.sum(axis=(0, 1), dtype=complex128) and np.multiply(a, wgt).sum(axis=(0, 1)).
            using var z = np.arange(24.0).reshape(2, 3, 4) * new Complex(1.0, 0.5);
            using var w = np.ones(new Shape(2, 3, 4)) * np.array(new[] { 0.5, 2.0, 1.0, 3.0 });
            var (avg, scl) = np.average_returned(z, new[] { 0, 1 }, w);
            using (avg)
            using (scl)
            {
                ComplexBits(avg).Should().Equal(0x4024000000000000, 0x4014000000000000, 0x4026000000000000, 0x4016000000000000,
                    0x4028000000000000, 0x4018000000000000, 0x402A000000000000, 0x401A000000000000);
                ComplexBits(scl).Should().Equal(0x4008000000000000, 0L, 0x4028000000000000, 0L, 0x4018000000000000, 0L,
                    0x4032000000000000, 0L);
                avg.flags.owndata.Should().BeTrue();
                scl.flags.owndata.Should().BeTrue();
            }
        }
    }
}
