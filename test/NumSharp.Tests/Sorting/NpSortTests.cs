using System;
using System.Numerics;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    /// np.sort / np.argsort parity with NumPy 2.4.2 (values produced by running NumPy).
    /// Implementation: NDIter IterAllButAxis drive + native-width LSD radix line kernel
    /// (scalar introsort with NumPy comparators for Half/Complex/Decimal).
    /// </summary>
    [TestClass]
    public class NpSortTests
    {
        // -------------------- 1-D --------------------
        [TestMethod]
        public void Sort_1D_Int32()
        {
            var s = np.sort(np.array(new[] { 3, 1, 2, 5, 4 }));
            s.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 2, 3, 4, 5 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Argsort_1D_Int32()
        {
            // np.argsort([3,1,2,5,4]) == [1,2,0,4,3]
            var g = np.argsort(np.array(new[] { 3, 1, 2, 5, 4 }));
            g.dtype.Should().Be(typeof(long));
            g.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 2, 0, 4, 3 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Argsort_Stable_Ties()
        {
            // np.argsort([3,1,3,1,3,1], kind='stable') == [1,3,5,0,2,4]
            var g = np.argsort(np.array(new[] { 3, 1, 3, 1, 3, 1 }));
            g.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 3, 5, 0, 2, 4 }, o => o.WithStrictOrdering());
        }

        // -------------------- 2-D axis --------------------
        [TestMethod]
        public void Sort_2D_Axis0()
        {
            // np.sort([[3,1,2],[0,5,4]], axis=0) == [[0,1,2],[3,5,4]]
            var s = np.sort(np.array(new[,] { { 3, 1, 2 }, { 0, 5, 4 } }), 0);
            s.GetInt32(0, 0).Should().Be(0); s.GetInt32(0, 1).Should().Be(1); s.GetInt32(0, 2).Should().Be(2);
            s.GetInt32(1, 0).Should().Be(3); s.GetInt32(1, 1).Should().Be(5); s.GetInt32(1, 2).Should().Be(4);
        }

        [TestMethod]
        public void Sort_2D_Axis1_Default()
        {
            // default axis = -1 (last). np.sort([[3,1,2],[0,5,4]]) == [[1,2,3],[0,4,5]]
            var s = np.sort(np.array(new[,] { { 3, 1, 2 }, { 0, 5, 4 } }));
            s.GetInt32(0, 0).Should().Be(1); s.GetInt32(0, 2).Should().Be(3);
            s.GetInt32(1, 0).Should().Be(0); s.GetInt32(1, 2).Should().Be(5);
        }

        [TestMethod]
        public void Argsort_2D_Axis1()
        {
            // np.argsort([[3,1,2],[0,5,4]], axis=1) == [[1,2,0],[0,2,1]]
            var g = np.argsort(np.array(new[,] { { 3, 1, 2 }, { 0, 5, 4 } }), 1);
            g.GetInt64(0, 0).Should().Be(1); g.GetInt64(0, 1).Should().Be(2); g.GetInt64(0, 2).Should().Be(0);
            g.GetInt64(1, 0).Should().Be(0); g.GetInt64(1, 1).Should().Be(2); g.GetInt64(1, 2).Should().Be(1);
        }

        // -------------------- axis=None / negative --------------------
        [TestMethod]
        public void Sort_AxisNone_Flattens()
        {
            // np.sort([[3,1],[2,0]], axis=None) == [0,1,2,3]
            var s = np.sort(np.array(new[,] { { 3, 1 }, { 2, 0 } }), (int?)null);
            s.ndim.Should().Be(1);
            s.ToArray<int>().Should().BeEquivalentTo(new[] { 0, 1, 2, 3 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Sort_NegativeAxis_EqualsLast()
        {
            var m = np.array(new[,] { { 3, 1, 2 }, { 0, 5, 4 } });
            var byNeg = np.sort(m, -1);
            var byPos = np.sort(m, 1);
            byNeg.ToArray<int>().Should().BeEquivalentTo(byPos.ToArray<int>(), o => o.WithStrictOrdering());
        }

        // -------------------- strided / transposed input (DOD) --------------------
        [TestMethod]
        public void Sort_TransposedView_MatchesNumPy()
        {
            // a=[[3,1,2],[0,5,4]]; aT (non-contig) = [[3,0],[1,5],[2,4]]
            // np.sort(aT, axis=0) == [[1,0],[2,4],[3,5]]
            var aT = np.array(new[,] { { 3, 1, 2 }, { 0, 5, 4 } }).T;
            aT.Shape.IsContiguous.Should().BeFalse();
            var s = np.sort(aT, 0);
            s.GetInt32(0, 0).Should().Be(1); s.GetInt32(0, 1).Should().Be(0);
            s.GetInt32(1, 0).Should().Be(2); s.GetInt32(1, 1).Should().Be(4);
            s.GetInt32(2, 0).Should().Be(3); s.GetInt32(2, 1).Should().Be(5);
        }

        // -------------------- floats: NaN sorts last --------------------
        [TestMethod]
        public void Sort_Float64_NaNLast()
        {
            // np.sort([3.0, nan, 1.0, nan, 2.0]) == [1,2,3,nan,nan]
            var s = np.sort(np.array(new[] { 3.0, double.NaN, 1.0, double.NaN, 2.0 }));
            ((double)s.GetAtIndex(0)).Should().Be(1.0);
            ((double)s.GetAtIndex(1)).Should().Be(2.0);
            ((double)s.GetAtIndex(2)).Should().Be(3.0);
            double.IsNaN((double)s.GetAtIndex(3)).Should().BeTrue();
            double.IsNaN((double)s.GetAtIndex(4)).Should().BeTrue();
        }

        /// <summary>
        ///     Pins the BITS of the NaNs a float32 / float64 sort leaves (the 2026-10-05 polynomial re-review found them
        ///     through <c>polyfromroots</c>). NumPy's default sort (x86-simd-sort) rewrites every NaN of a line longer
        ///     than one element — negative, signalling or payload-bearing — to the positive quiet NaN
        ///     (<c>0x7ff8000000000000</c> / <c>0x7fc00000</c>); NumSharp wrote .NET's negative <c>double.NaN</c> /
        ///     <c>float.NaN</c>. A one-element line is not sorted, so its NaN keeps its bits; float16 NaNs keep theirs
        ///     (NumPy's float16 sort is a comparison sort). <c>double.IsNaN</c> cannot see any of this: only bits can.
        /// </summary>
        [TestMethod]
        public void Sort_FloatNaN_IsNumPysCanonicalQuietNaN()
        {
            static double D(ulong bits) => BitConverter.UInt64BitsToDouble(bits);
            static ulong B(object v) => BitConverter.DoubleToUInt64Bits((double)v);

            // np.sort(f64 [nan(payload 5), 1.0, -nan, -1.0]) -> [-1.0, 1.0, 0x7ff8000000000000, 0x7ff8000000000000]
            var x = np.array(new[] { D(0x7ff8000000000005UL), 1.0, D(0xfff8000000000000UL), -1.0 });
            var s = np.sort(x);
            B(s.GetAtIndex(0)).Should().Be(0xbff0000000000000UL);
            B(s.GetAtIndex(1)).Should().Be(0x3ff0000000000000UL);
            B(s.GetAtIndex(2)).Should().Be(0x7ff8000000000000UL, "every NaN becomes NumPy's positive quiet NaN");
            B(s.GetAtIndex(3)).Should().Be(0x7ff8000000000000UL);

            // ndarray.sort in place: the same bits.
            var y = x.copy();
            y.sort();
            B(y.GetAtIndex(2)).Should().Be(0x7ff8000000000000UL);
            B(y.GetAtIndex(3)).Should().Be(0x7ff8000000000000UL);

            // An axis-0 line of a 2-D array (a strided line): np.sort([[nan5, 1], [-nan, -1]], axis=0)
            // -> [[0x7ff8.., -1], [0x7ff8.., 1]].
            var m = np.sort(x.reshape(2, 2), 0);
            B(m.GetAtIndex(0)).Should().Be(0x7ff8000000000000UL);
            B(m.GetAtIndex(1)).Should().Be(0xbff0000000000000UL);
            B(m.GetAtIndex(2)).Should().Be(0x7ff8000000000000UL);
            B(m.GetAtIndex(3)).Should().Be(0x3ff0000000000000UL);

            // A one-element line is left alone: np.sort(f64 [-nan(payload 5)]) keeps 0xfff8000000000005.
            B(np.sort(np.array(new[] { D(0xfff8000000000005UL) })).GetAtIndex(0)).Should().Be(0xfff8000000000005UL);

            // float32: np.sort([nan(payload 5), 1.0, -sNaN]) -> [1.0, 0x7fc00000, 0x7fc00000].
            var f = np.sort(np.array(new[]
            {
                BitConverter.UInt32BitsToSingle(0x7fc00005U), 1f, BitConverter.UInt32BitsToSingle(0xff800001U),
            }));
            BitConverter.SingleToUInt32Bits((float)f.GetAtIndex(0)).Should().Be(0x3f800000U);
            BitConverter.SingleToUInt32Bits((float)f.GetAtIndex(1)).Should().Be(0x7fc00000U);
            BitConverter.SingleToUInt32Bits((float)f.GetAtIndex(2)).Should().Be(0x7fc00000U);

            // float16 keeps every NaN's bits, in input order: np.sort(f16 [0x7e05, 1.0, 0xfe00]) -> [1.0, 0x7e05, 0xfe00].
            var h = np.sort(np.array(new[]
            {
                BitConverter.UInt16BitsToHalf(0x7e05), (Half)1, BitConverter.UInt16BitsToHalf(0xfe00),
            }));
            BitConverter.HalfToUInt16Bits((Half)h.GetAtIndex(1)).Should().Be((ushort)0x7e05);
            BitConverter.HalfToUInt16Bits((Half)h.GetAtIndex(2)).Should().Be((ushort)0xfe00);
        }

        [TestMethod]
        public void Argsort_Float64_NaNIndicesLast()
        {
            // np.argsort([3.0,nan,1.0,nan,2.0], kind='stable') == [2,4,0,1,3]
            var g = np.argsort(np.array(new[] { 3.0, double.NaN, 1.0, double.NaN, 2.0 }));
            g.ToArray<long>().Should().BeEquivalentTo(new long[] { 2, 4, 0, 1, 3 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Sort_Float_NegZero_Inf()
        {
            // np.sort([inf, -inf, 0.0, -1.0, 1.0]) == [-inf,-1,0,1,inf]
            var s = np.sort(np.array(new[] { float.PositiveInfinity, float.NegativeInfinity, 0f, -1f, 1f }));
            float.IsNegativeInfinity((float)s.GetAtIndex(0)).Should().BeTrue();
            ((float)s.GetAtIndex(1)).Should().Be(-1f);
            ((float)s.GetAtIndex(4)).Should().Be(float.PositiveInfinity, "+inf sorts before NaN but after finite");
        }

        // -------------------- Complex: lexicographic, NaN-part last --------------------
        [TestMethod]
        public void Sort_Complex_Lexicographic()
        {
            // np.sort([3+1j,1+2j,1+1j,nan+0j,2+0j]) == [1+1j,1+2j,2+0j,3+1j,nan+0j]
            var s = np.sort(np.array(new Complex[]
            {
                new(3, 1), new(1, 2), new(1, 1), new(double.NaN, 0), new(2, 0)
            }));
            ((Complex)s.GetAtIndex(0)).Should().Be(new Complex(1, 1));
            ((Complex)s.GetAtIndex(1)).Should().Be(new Complex(1, 2));
            ((Complex)s.GetAtIndex(2)).Should().Be(new Complex(2, 0));
            ((Complex)s.GetAtIndex(3)).Should().Be(new Complex(3, 1));
            double.IsNaN(((Complex)s.GetAtIndex(4)).Real).Should().BeTrue();
        }

        // -------------------- other dtypes --------------------
        [TestMethod]
        public void Sort_Half_NaNLast()
        {
            var s = np.sort(np.array(new[] { (Half)3, Half.NaN, (Half)1, (Half)2 }));
            ((double)(Half)s.GetAtIndex(0)).Should().Be(1.0);
            ((double)(Half)s.GetAtIndex(2)).Should().Be(3.0);
            Half.IsNaN((Half)s.GetAtIndex(3)).Should().BeTrue();
        }

        [TestMethod]
        public void Sort_Decimal()
        {
            var s = np.sort(np.array(new decimal[] { 3.5m, -1.2m, 0m, 2.1m }));
            s.ToArray<decimal>().Should().BeEquivalentTo(new decimal[] { -1.2m, 0m, 2.1m, 3.5m }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Sort_UInt8_And_Int16()
        {
            np.sort(np.array(new byte[] { 200, 5, 99, 0, 255 })).ToArray<byte>()
                .Should().BeEquivalentTo(new byte[] { 0, 5, 99, 200, 255 }, o => o.WithStrictOrdering());
            np.sort(np.array(new short[] { -300, 300, 0, -1, 1 })).ToArray<short>()
                .Should().BeEquivalentTo(new short[] { -300, -1, 0, 1, 300 }, o => o.WithStrictOrdering());
        }

        // -------------------- in-place ndarray.sort --------------------
        [TestMethod]
        public void Sort_InPlace_MutatesArray()
        {
            var a = np.array(new[] { 4, 2, 5, 1, 3 });
            a.sort();
            a.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 2, 3, 4, 5 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Sort_InPlace_Axis0()
        {
            var a = np.array(new[,] { { 3, 1 }, { 0, 5 }, { 2, 4 } });
            a.sort(0); // sort each column
            a.GetInt32(0, 0).Should().Be(0); a.GetInt32(2, 0).Should().Be(3);
            a.GetInt32(0, 1).Should().Be(1); a.GetInt32(2, 1).Should().Be(5);
        }

        // -------------------- edges --------------------
        [TestMethod]
        public void Sort_Empty_And_Single()
        {
            np.sort(np.array(new int[0])).size.Should().Be(0);
            np.sort(np.array(new[] { 42 })).GetInt32(0).Should().Be(42);
            np.argsort(np.array(new[] { 42 })).GetInt64(0).Should().Be(0);
        }

        [TestMethod]
        public void Argsort_ReconstructsSort_3D()
        {
            // take_along_axis(a, argsort(a, axis), axis) == sort(a, axis), for a 3-D array.
            var rng = new System.Random(99);
            var data = new int[2 * 3 * 4];
            for (int i = 0; i < data.Length; i++) data[i] = rng.Next(-50, 50);
            var a = np.array(data).reshape(2, 3, 4);
            for (int axis = 0; axis < 3; axis++)
            {
                var s = np.sort(a, axis);
                var g = np.argsort(a, axis);
                // manual take_along_axis: gather a along `axis` using g, must equal s
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 3; j++)
                        for (int k = 0; k < 4; k++)
                        {
                            long gi = g.GetInt64(i, j, k);
                            int taken = axis == 0 ? a.GetInt32((int)gi, j, k)
                                      : axis == 1 ? a.GetInt32(i, (int)gi, k)
                                                  : a.GetInt32(i, j, (int)gi);
                            taken.Should().Be(s.GetInt32(i, j, k),
                                $"take_along_axis(a, argsort, {axis}) at [{i},{j},{k}] must equal sort(a, {axis})");
                        }
            }
        }
    }
}
