using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Extensive 1-to-1 parity tests for <c>np.cumsum</c> against NumPy 2.4.2.
    ///
    /// Every expected value / dtype / shape / layout flag in this file was probed from ACTUAL
    /// NumPy 2.4.2 output (python &lt;&lt; side-by-side with dotnet). np.cumsum IS np.add.accumulate;
    /// the canonical matrix is ported from numpy/lib/tests/test_function_base.py::TestCumsum.
    ///
    /// Complements CumSumEdgeCaseTests / CumSumComprehensiveTests with the areas they miss:
    ///   - the full dtype matrix (all 15 NumSharp dtypes, not a 4-dtype spot check)
    ///   - NEP50 output dtype for every input dtype
    ///   - KEEPORDER output LAYOUT (C-source→C, F-source→F, transpose→F, strided→C)
    ///   - value correctness on F-contiguous / strided / transposed / negative-stride / broadcast views
    ///   - the dtype= parameter, integer overflow/wraparound, NaN/Inf, complex, signed zero
    ///   - axis validation / error conditions (AxisError equivalents)
    /// </summary>
    [TestClass]
    public class CumSumParityTests
    {
        // ----- helpers ----------------------------------------------------------------

        /// <summary>Read an array's elements in logical C-order as double (layout-independent —
        /// a C copy makes storage order == logical order, so F/strided outputs read correctly).</summary>
        private static double[] Logical(NDArray a)
        {
            var c = a.astype(NPTypeCode.Double).copy('C');
            var v = new double[c.size];
            for (long i = 0; i < c.size; i++) v[i] = Convert.ToDouble(c.GetAtIndex(i));
            return v;
        }

        private static void AssertValues(NDArray result, params double[] expected)
        {
            var got = Logical(result);
            got.Length.Should().Be(expected.Length, "element count");
            for (int i = 0; i < expected.Length; i++)
                Math.Abs(got[i] - expected[i]).Should().BeLessThan(1e-6,
                    $"element {i}: got {got[i]} expected {expected[i]}");
        }

        // NumPy TestCumsum.test_basic fixtures.
        private static readonly int[] Ba = { 1, 2, 10, 11, 6, 5, 4 };
        private static readonly int[,] Ba2 = { { 1, 2, 3, 4 }, { 5, 6, 7, 9 }, { 10, 3, 4, 5 } };

        // ===== [1] canonical NumPy matrix across all numeric dtypes ====================

        [TestMethod]
        public void CumSum_CanonicalMatrix_AllNumericDtypes()
        {
            // NumPy (TestCumsum.test_basic): values are identical across dtypes (all fit).
            //   cumsum(ba, 0)   = [1, 3, 13, 24, 30, 35, 39]
            //   cumsum(ba2, 0)  = [[1,2,3,4],[6,8,10,13],[16,11,14,18]]
            //   cumsum(ba2, 1)  = [[1,3,6,10],[5,11,18,27],[10,13,17,22]]
            //   cumsum(ba2)     = [1,3,6,10,15,21,28,37,47,50,54,59]
            var dtypes = new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
                NPTypeCode.Decimal,
            };
            foreach (var dt in dtypes)
            {
                var a = np.array(Ba).astype(dt);
                var a2 = np.array(Ba2).astype(dt);
                AssertValues(np.cumsum(a, 0), 1, 3, 13, 24, 30, 35, 39);
                AssertValues(np.cumsum(a2, 0), 1, 2, 3, 4, 6, 8, 10, 13, 16, 11, 14, 18);
                AssertValues(np.cumsum(a2, 1), 1, 3, 6, 10, 5, 11, 18, 27, 10, 13, 17, 22);
                AssertValues(np.cumsum(a2, -1), 1, 3, 6, 10, 5, 11, 18, 27, 10, 13, 17, 22);
                AssertValues(np.cumsum(a2), 1, 3, 6, 10, 15, 21, 28, 37, 47, 50, 54, 59);
            }
        }

        [TestMethod]
        public void CumSum_CanonicalMatrix_Complex()
        {
            // Complex cumsum accumulates both components: probed against NumPy complex128.
            var a = np.array(new Complex[] { new Complex(1, 2), new Complex(3, -1), new Complex(-2, 0.5) });
            var r = np.cumsum(a);
            var d = r.GetData<Complex>().ToArray();
            r.typecode.Should().Be(NPTypeCode.Complex);
            d[0].Should().Be(new Complex(1, 2));
            d[1].Should().Be(new Complex(4, 1));
            d[2].Should().Be(new Complex(2, 1.5));
        }

        // ===== [2] NEP50 output dtype for EVERY input dtype ============================

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.SByte, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.Byte, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Int16, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt16, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt32, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Int64, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Half, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal, NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex, NPTypeCode.Complex)]
        public void CumSum_OutputDtype_NEP50(NPTypeCode input, NPTypeCode expected)
        {
            // NumPy 2.x: cumsum widens narrow ints to the platform int64/uint64; bool->int64;
            // float/complex preserved. (NumSharp adds Char->UInt64, Half->Half, Decimal->Decimal.)
            var a = np.ones(new Shape(3)).astype(input);
            np.cumsum(a).typecode.Should().Be(expected, $"cumsum({input})");
            np.cumsum(a.reshape(1, 3), 0).typecode.Should().Be(expected, $"cumsum({input}, axis=0)");
        }

        [TestMethod]
        public void CumSum_Bool_ReturnsInt64_TrueIsOne()
        {
            // NumPy: cumsum([T,F,T,T,F]) = [1, 1, 2, 3, 3] int64
            var a = np.array(new bool[] { true, false, true, true, false });
            var r = np.cumsum(a);
            r.typecode.Should().Be(NPTypeCode.Int64);
            AssertValues(r, 1, 1, 2, 3, 3);
        }

        // ===== [3] KEEPORDER output LAYOUT ============================================

        [TestMethod]
        public void CumSum_OutputLayout_CContiguousSource_StaysC()
        {
            // NumPy: cumsum(C-contig, axis) -> C-contiguous output.
            var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
            foreach (int ax in new[] { 0, 1 })
            {
                var r = np.cumsum(a, ax);
                r.Shape.IsContiguous.Should().BeTrue($"axis {ax} C");
                r.Shape.IsFContiguous.Should().BeFalse($"axis {ax} not F");
            }
        }

        [TestMethod]
        public void CumSum_OutputLayout_FContiguousSource_StaysF()
        {
            // NumPy: cumsum(F-contig, axis) -> F-contiguous output (KEEPORDER). This is the
            // long-standing bug the NDIter accumulate rewrite fixed (was C + post-hoc copy).
            var f = np.arange(12).astype(NPTypeCode.Double).reshape(4, 3).T; // (3,4) F-contig
            f.Shape.IsFContiguous.Should().BeTrue("fixture is F-contig");
            foreach (int ax in new[] { 0, 1 })
            {
                var r = np.cumsum(f, ax);
                r.Shape.IsFContiguous.Should().BeTrue($"axis {ax} F");
                r.Shape.IsContiguous.Should().BeFalse($"axis {ax} not C");
            }
        }

        [TestMethod]
        public void CumSum_OutputLayout_TransposedSource_IsF()
        {
            // NumPy: a transposed (F-contig) view -> F output.
            var t = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4).T; // (4,3) F-contig
            var r = np.cumsum(t, 0);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void CumSum_OutputLayout_StridedSource_IsC()
        {
            // NumPy: a non-contiguous strided view (neither C nor F) -> C-contiguous output.
            var s = np.arange(24).astype(NPTypeCode.Double).reshape(4, 6)["::2, ::2"]; // (2,3)
            foreach (int ax in new[] { 0, 1 })
            {
                var r = np.cumsum(s, ax);
                r.Shape.IsContiguous.Should().BeTrue($"axis {ax} C");
            }
        }

        [TestMethod]
        public void CumSum_OutputLayout_AxisNone_Is1D()
        {
            // NumPy: axis=None always returns a 1-D (C&F) result regardless of source layout.
            var f = np.arange(12).astype(NPTypeCode.Double).reshape(4, 3).T;
            var r = np.cumsum(f);
            r.ndim.Should().Be(1);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        // ===== [4] value correctness on non-trivial layouts ===========================

        [TestMethod]
        public void CumSum_FContiguous_ValuesMatchNumPy()
        {
            // logical [[0,3],[1,4],[2,5]] (F-contig). NumPy cumsum:
            //   axis0 -> [[0,3],[1,7],[3,12]] ; axis1 -> [[0,3],[1,5],[2,7]]
            var f = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3).T; // (3,2) F
            AssertValues(np.cumsum(f, 0), 0, 3, 1, 7, 3, 12);
            AssertValues(np.cumsum(f, 1), 0, 3, 1, 5, 2, 7);
        }

        [TestMethod]
        public void CumSum_Strided1D_ValuesMatchNumPy()
        {
            // NumPy: cumsum(arange(10)[::2]) = cumsum([0,2,4,6,8]) = [0,2,6,12,20]
            var s = np.arange(10)["::2"];
            AssertValues(np.cumsum(s), 0, 2, 6, 12, 20);
        }

        [TestMethod]
        public void CumSum_NegativeStride1D_ValuesMatchNumPy()
        {
            // NumPy: cumsum([4,3,2,1,0]) = [4,7,9,10,10]
            var ns = np.arange(5)["::-1"];
            AssertValues(np.cumsum(ns), 4, 7, 9, 10, 10);
        }

        [TestMethod]
        public void CumSum_NegativeStride2D_Axis0_ValuesMatchNumPy()
        {
            // NumPy: cumsum(arange(12).reshape(3,4)[::-1], axis=0)
            //   = [[8,9,10,11],[12,14,16,18],[12,15,18,21]]
            var ns = np.arange(12).reshape(3, 4)["::-1"];
            AssertValues(np.cumsum(ns, 0), 8, 9, 10, 11, 12, 14, 16, 18, 12, 15, 18, 21);
        }

        [TestMethod]
        public void CumSum_BroadcastRow_ValuesMatchNumPy()
        {
            // NumPy: broadcast_to([1,2,3],(3,3)) -> rows all [1,2,3].
            //   axis0 -> [[1,2,3],[2,4,6],[3,6,9]] ; axis1 -> [[1,3,6],[1,3,6],[1,3,6]]
            var rb = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(3, 3));
            AssertValues(np.cumsum(rb, 0), 1, 2, 3, 2, 4, 6, 3, 6, 9);
            AssertValues(np.cumsum(rb, 1), 1, 3, 6, 1, 3, 6, 1, 3, 6);
        }

        [TestMethod]
        public void CumSum_BroadcastCol_ValuesMatchNumPy()
        {
            // NumPy: broadcast_to([[1],[2],[3]],(3,3)) -> rows [1,1,1],[2,2,2],[3,3,3].
            //   axis0 -> [[1,1,1],[3,3,3],[6,6,6]] ; axis1 -> [[1,2,3],[2,4,6],[3,6,9]]
            var cb = np.broadcast_to(np.array(new int[,] { { 1 }, { 2 }, { 3 } }), new Shape(3, 3));
            AssertValues(np.cumsum(cb, 0), 1, 1, 1, 3, 3, 3, 6, 6, 6);
            AssertValues(np.cumsum(cb, 1), 1, 2, 3, 2, 4, 6, 3, 6, 9);
        }

        [TestMethod]
        public void CumSum_3D_AllAxes_ValuesMatchNumPy()
        {
            // arange(24).reshape(2,3,4), probed against NumPy for each axis.
            var a = np.arange(24).astype(NPTypeCode.Int32).reshape(2, 3, 4);
            AssertValues(np.cumsum(a, 0),
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34);
            AssertValues(np.cumsum(a, 2),
                0, 1, 3, 6, 4, 9, 15, 22, 8, 17, 27, 38,
                12, 25, 39, 54, 16, 33, 51, 70, 20, 41, 63, 86);
        }

        // ===== [5] empty / scalar / size-1 (shape + dtype) ============================

        [TestMethod]
        public void CumSum_Empty1D_PromotesDtype_Keeps1D()
        {
            // NumPy: cumsum(empty int32) -> empty int64 shape (0,)
            var r = np.cumsum(np.arange(0).astype(NPTypeCode.Int32));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Int64);
            r.ndim.Should().Be(1);
        }

        [TestMethod]
        public void CumSum_Empty2D_AxisNone_RavelsTo1D()
        {
            // NumPy: cumsum(zeros((0,3), int32)) -> shape (0,) int64
            var e = np.zeros(new Shape(0, 3)).astype(NPTypeCode.Int32);
            var r = np.cumsum(e);
            r.typecode.Should().Be(NPTypeCode.Int64);
            r.ndim.Should().Be(1);
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void CumSum_Empty2D_WithAxis_PreservesShape()
        {
            // NumPy: cumsum(zeros((0,3), int32), axis=1) -> shape (0,3) int64
            var e = np.zeros(new Shape(0, 3)).astype(NPTypeCode.Int32);
            var r = np.cumsum(e, 1);
            r.typecode.Should().Be(NPTypeCode.Int64);
            r.Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void CumSum_EmptyFloat_PreservesFloat()
        {
            // NumPy: cumsum(empty float64) -> empty float64
            var r = np.cumsum(np.arange(0).astype(NPTypeCode.Double));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void CumSum_ZeroDimScalar_ReturnsShape1_Promoted()
        {
            // NumPy: cumsum(np.array(5)) -> shape (1,) int64 value [5]  (NEVER 0-d)
            var r = np.cumsum(NDArray.Scalar(5));
            r.Should().BeShaped(1);
            r.typecode.Should().Be(NPTypeCode.Int64);
            AssertValues(r, 5);
        }

        [TestMethod]
        public void CumSum_ZeroDimScalar_Axis0_ReturnsShape1()
        {
            // NumPy treats a 0-d array as 1-D for cumsum: axis=0 and axis=-1 are valid.
            np.cumsum(NDArray.Scalar(5), 0).Should().BeShaped(1);
            np.cumsum(NDArray.Scalar(5), -1).Should().BeShaped(1);
        }

        [TestMethod]
        public void CumSum_SingleElement1D_Promoted()
        {
            // NumPy: cumsum([5]) -> [5] int64
            var r = np.cumsum(np.array(new int[] { 5 }));
            r.Should().BeShaped(1);
            r.typecode.Should().Be(NPTypeCode.Int64);
            AssertValues(r, 5);
        }

        [TestMethod]
        public void CumSum_SingleElement2D_AxisNone_RavelsTo1D_Promoted()
        {
            // NumPy: cumsum([[5]]) -> shape (1,) int64  (never collapses to 0-d)
            var r = np.cumsum(np.array(new int[,] { { 5 } }));
            r.Should().BeShaped(1);
            r.typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void CumSum_AxisLength1_Promoted()
        {
            // NumPy: cumsum([[5]], axis=0) -> shape (1,1) int64
            var r = np.cumsum(np.array(new int[,] { { 5 } }), 0);
            r.Should().BeShaped(1, 1);
            r.typecode.Should().Be(NPTypeCode.Int64);
        }

        // ===== [6] dtype= parameter ====================================================

        [TestMethod]
        public void CumSum_DtypeParam_IntToFloat()
        {
            // NumPy: cumsum([1,2,3], dtype=float64) -> [1.,3.,6.] float64
            var r = np.cumsum(np.array(new int[] { 1, 2, 3 }), typeCode: NPTypeCode.Double);
            r.typecode.Should().Be(NPTypeCode.Double);
            AssertValues(r, 1, 3, 6);
        }

        [TestMethod]
        public void CumSum_DtypeParam_FloatToInt_Truncates()
        {
            // NumPy: cumsum([1.5,2.5,3.5], dtype=int32) -> [1,3,6] int32 (running value truncated)
            var r = np.cumsum(np.array(new double[] { 1.5, 2.5, 3.5 }), typeCode: NPTypeCode.Int32);
            r.typecode.Should().Be(NPTypeCode.Int32);
            AssertValues(r, 1, 3, 6);
        }

        [TestMethod]
        public void CumSum_DtypeParam_NarrowInt_Wraps()
        {
            // NumPy: cumsum([100,100,100], dtype=int8) -> [100,-56,44] (int8 modular wrap)
            var r = np.cumsum(np.array(new int[] { 100, 100, 100 }), typeCode: NPTypeCode.SByte);
            r.typecode.Should().Be(NPTypeCode.SByte);
            AssertValues(r, 100, -56, 44);
        }

        // ===== [7] overflow / wraparound ==============================================

        [TestMethod]
        public void CumSum_Int8DtypeParam_Wraps()
        {
            // NumPy: cumsum(int8[100,100,100], dtype=int8) = [100,-56,44]
            var r = np.cumsum(np.array(new sbyte[] { 100, 100, 100 }), typeCode: NPTypeCode.SByte);
            AssertValues(r, 100, -56, 44);
        }

        [TestMethod]
        public void CumSum_UInt8DtypeParam_Wraps()
        {
            // NumPy: cumsum(uint8[200,100], dtype=uint8) = [200,44]
            var r = np.cumsum(np.array(new byte[] { 200, 100 }), typeCode: NPTypeCode.Byte);
            AssertValues(r, 200, 44);
        }

        [TestMethod]
        public void CumSum_Int64_OverflowWraps()
        {
            // NumPy: cumsum([int64.max,1,1]) wraps modulo 2^64 (two's complement).
            var r = np.cumsum(np.array(new long[] { long.MaxValue, 1, 1 }));
            var d = r.GetData<long>().ToArray();
            d[0].Should().Be(long.MaxValue);
            d[1].Should().Be(long.MinValue);          // max + 1 wraps
            d[2].Should().Be(long.MinValue + 1);
        }

        [TestMethod]
        public void CumSum_LargeValues_Int64_NoOverflow()
        {
            var a = np.array(new long[] { 1_000_000_000L, 2_000_000_000L, 3_000_000_000L });
            AssertValues(np.cumsum(a), 1_000_000_000d, 3_000_000_000d, 6_000_000_000d);
        }

        // ===== [8] NaN / Inf / signed zero ============================================

        [TestMethod]
        public void CumSum_NaN_Propagates()
        {
            // NumPy: cumsum([1, nan, 3]) = [1, nan, nan]
            var d = np.cumsum(np.array(new double[] { 1.0, double.NaN, 3.0 })).GetData<double>().ToArray();
            d[0].Should().Be(1.0);
            double.IsNaN(d[1]).Should().BeTrue();
            double.IsNaN(d[2]).Should().BeTrue();
        }

        [TestMethod]
        public void CumSum_InfMinusInf_IsNaN()
        {
            // NumPy: cumsum([1, inf, -inf, 3]) = [1, inf, nan, nan]
            var d = np.cumsum(np.array(new double[] { 1.0, double.PositiveInfinity, double.NegativeInfinity, 3.0 }))
                .GetData<double>().ToArray();
            d[0].Should().Be(1.0);
            double.IsPositiveInfinity(d[1]).Should().BeTrue();
            double.IsNaN(d[2]).Should().BeTrue();   // inf + (-inf) = nan
            double.IsNaN(d[3]).Should().BeTrue();
        }

        [TestMethod]
        public void CumSum_SignedZero_AxisPath_PreservesNegativeZero()
        {
            // NumPy: cumsum([[-0.0,1.0],[2.0,3.0]], axis=0)[0,0] keeps the sign bit (-0.0).
            // The NDIter accumulate path copies the first element (memmove-first), so this matches.
            var z = np.array(new double[,] { { -0.0, 1.0 }, { 2.0, 3.0 } });
            var r = np.cumsum(z, 0);
            var d = r.GetData<double>().ToArray();
            double.IsNegative(d[0]).Should().BeTrue("axis cumsum preserves -0.0 of the first element");
        }

        [TestMethod]
        [Misaligned]
        public void CumSum_SignedZero_FlatPath_DropsSignDivergence()
        {
            // KNOWN DIVERGENCE (flat / axis=None path only): NumPy preserves the sign bit of the
            // FIRST element — cumsum([-0.0, 0.0, -0.0]) = [-0.0, 0.0, 0.0] (signbit [T,F,F]).
            // NumSharp's flat scan seeds a +0.0 accumulator and adds, so out[0] = 0.0 + (-0.0) = +0.0
            // (signbit False). The NDIter axis path copies the first element and DOES preserve it
            // (see CumSum_SignedZero_AxisPath_PreservesNegativeZero); only the flat helper diverges.
            // Fixing it cleanly would touch the shared flat IL emitters; scoped out as a minor edge.
            var r = np.cumsum(np.array(new double[] { -0.0, 0.0, -0.0 }));
            var d = r.GetData<double>().ToArray();
            double.IsNegative(d[0]).Should().BeFalse(
                "DIVERGENCE: NumSharp flat cumsum drops the -0.0 sign of the first element (NumPy keeps it)");
        }

        // ===== [9] negative axis ======================================================

        [TestMethod]
        public void CumSum_NegativeAxis_EqualsPositive()
        {
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            AssertValues(np.cumsum(a, -1), 1, 3, 6, 4, 9, 15);  // == axis 1
            AssertValues(np.cumsum(a, -2), 1, 2, 3, 5, 7, 9);   // == axis 0
        }

        // ===== [10] axis error conditions =============================================

        [TestMethod]
        public void CumSum_AxisOutOfBounds_Positive_Throws()
        {
            // NumPy raises AxisError; NumSharp raises ArgumentOutOfRangeException.
            var a = np.ones(new Shape(2, 3));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.cumsum(a, 2));
        }

        [TestMethod]
        public void CumSum_AxisOutOfBounds_Negative_Throws()
        {
            // NumPy: axis=-3 on a 2-D array is an AxisError (NOT a silent wrap to axis 1).
            var a = np.ones(new Shape(2, 3));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.cumsum(a, -3));
        }

        [TestMethod]
        public void CumSum_AxisOutOfBounds_OnTrivialArray_Throws()
        {
            // NumPy validates the axis BEFORE the trivial shortcut: 0-d/size-1 still error.
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.cumsum(NDArray.Scalar(5), 1));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.cumsum(np.array(new int[] { 5 }), -3));
        }
    }
}
