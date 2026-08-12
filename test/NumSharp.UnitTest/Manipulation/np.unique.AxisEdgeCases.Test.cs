using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// np.unique axis-path edge cases + full parameter parity with NumPy 2.4.2.
    /// The axis path compares each sub-array (slab) as a structured scalar: NaN slabs are ALWAYS
    /// distinct (nan != nan; equal_nan has no effect on axis), signed-zero slabs collapse
    /// (-0.0 == +0.0). Expected values were produced by running NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class UniqueAxisEdgeCasesTests
    {
        private static void AssertInt64(NDArray a, params long[] expected)
        {
            a.typecode.Should().Be(NPTypeCode.Int64);
            a.ToArray<long>().Should().Equal(expected);
        }

        // ---- NaN slabs are distinct (the fixed bug: NumSharp used to collapse them) -------
        [TestMethod]
        public void Axis_NaNRows_AreDistinct_Float64()
        {
            // >>> np.unique([[1,2],[nan,2],[1,2],[nan,2]], axis=0, return_*=True)
            // values [[1,2],[nan,2],[nan,2]] idx [0,1,3] inv [0,1,0,2] counts [2,1,1]
            var a = np.array(new[,] { { 1.0, 2.0 }, { double.NaN, 2.0 }, { 1.0, 2.0 }, { double.NaN, 2.0 } });
            var r = np.unique(a, true, true, true, axis: 0);
            r[0].shape.Should().Equal(3, 2);
            var v = r[0].ToArray<double>();          // C-order flat: [1,2, nan,2, nan,2]
            v[0].Should().Be(1.0); v[1].Should().Be(2.0);
            double.IsNaN(v[2]).Should().BeTrue(); v[3].Should().Be(2.0);
            double.IsNaN(v[4]).Should().BeTrue(); v[5].Should().Be(2.0);
            AssertInt64(r[1], 0, 1, 3);   // first-occurrence indices
            AssertInt64(r[2], 0, 1, 0, 2);
            AssertInt64(r[3], 2, 1, 1);
        }

        [TestMethod]
        public void Axis_EqualNan_HasNoEffect()
        {
            // NumPy: equal_nan is ignored for the axis path (structured comparison).
            var a = np.array(new[,] { { 1.0, 2.0 }, { double.NaN, 2.0 }, { double.NaN, 2.0 } });
            np.unique(a, false, false, false, axis: 0, equal_nan: true)[0].shape.Should().Equal(3, 2);
            np.unique(a, false, false, false, axis: 0, equal_nan: false)[0].shape.Should().Equal(3, 2);
        }

        [TestMethod]
        public void Axis_NaNRows_AreDistinct_Complex()
        {
            // >>> np.unique([[1+2j,3],[nan+1j,2],[1+2j,3],[nan+1j,2]], axis=0) -> 3 unique rows
            var a = np.array(new[,]
            {
                { new Complex(1, 2), new Complex(3, 0) },
                { new Complex(double.NaN, 1), new Complex(2, 2) },
                { new Complex(1, 2), new Complex(3, 0) },
                { new Complex(double.NaN, 1), new Complex(2, 2) },
            });
            np.unique(a, false, false, false, axis: 0)[0].shape.Should().Equal(3, 2);
        }

        // ---- signed-zero slabs collapse; representative follows input order ---------------
        [TestMethod]
        public void Axis_SignedZeroRows_Collapse_KeepInputRepresentative()
        {
            // >>> np.unique([[0.,5.],[-0.,5.],[1.,1.]], axis=0) -> [[0.,5.],[1.,1.]] (2 rows, +0.0)
            var a = np.array(new[,] { { 0.0, 5.0 }, { -0.0, 5.0 }, { 1.0, 1.0 } });
            var r = np.unique(a, false, false, true, axis: 0);   // [values, counts]
            r[0].shape.Should().Equal(2, 2);
            var v = r[0].ToArray<double>();
            v[0].Should().Be(0.0);
            double.IsNegative(v[0]).Should().BeFalse("representative is +0.0 (first in input order)");
            AssertInt64(r[1], 2, 1);
        }

        // ---- sorted= parameter (NumPy 2.3) -----------------------------------------------
        [TestMethod]
        public void Sorted_Parameter_IsAccepted_AlwaysSorted()
        {
            var x = np.array(new[] { 3, 1, 2, 1, 3, 2, 5, 4 });
            // sorted=false is accepted for API parity; NumSharp always returns sorted output.
            np.unique(x, false, false, false, null, true, sorted: false)[0]
                .array_equal(np.array(new[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
            np.unique(x, false, false, false, null, true, sorted: true)[0]
                .array_equal(np.array(new[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        // ---- axis out of bounds -> AxisError with NumPy-exact message ---------------------
        [TestMethod]
        public void Axis_OutOfBounds_ThrowsAxisError()
        {
            var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            Action pos = () => np.unique(a, false, false, false, axis: 2);
            pos.Should().Throw<AxisError>().WithMessage("axis 2 is out of bounds for array of dimension 2*");

            Action neg = () => np.unique(a, false, false, false, axis: -3);
            neg.Should().Throw<AxisError>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
        }

        // ---- empty-with-axis shapes ------------------------------------------------------
        [TestMethod]
        public void Axis_Empty_Shapes()
        {
            np.unique(np.zeros(new Shape(0, 3)), false, false, false, axis: 0)[0].shape.Should().Equal(0, 3);
            np.unique(np.zeros(new Shape(0, 3)), false, false, false, axis: 1)[0].shape.Should().Equal(0, 1);
            np.unique(np.zeros(new Shape(3, 0)), false, false, false, axis: 0)[0].shape.Should().Equal(1, 0);
            np.unique(np.zeros(new Shape(3, 0)), false, false, false, axis: 1)[0].shape.Should().Equal(3, 0);
        }

        // ---- Char / Decimal (no NumPy dtype): axis reconstruct invariant -----------------
        [TestMethod]
        public void Axis_CharDecimal_Reconstruct()
        {
            var c = np.array(new char[,] { { 'a', 'b' }, { 'a', 'b' }, { 'c', 'a' } });
            var rc = np.unique(c, false, true, false, axis: 0);
            np.take(rc[0], rc[1], axis: 0).array_equal(c).Should().BeTrue();

            var d = np.array(new decimal[,] { { 1.5m, 2m }, { 1.5m, 2m }, { 3m, 1m } });
            var rd = np.unique(d, false, true, false, axis: 0);
            np.take(rd[0], rd[1], axis: 0).array_equal(d).Should().BeTrue();
        }

        /// <summary>
        /// DELIBERATE DIVERGENCE ([Misaligned]): NumPy 2.4.2's structured/void sort mis-orders
        /// float16 NaN — it places NaN slabs FIRST, inconsistent with its own 1-D float16 sort and
        /// its float32/float64 structured sorts (all NaN-last). NumSharp keeps the consistent,
        /// correct NaN-to-end ordering across every float width. The unique-row SET and counts are
        /// identical; only the NaN-slab position differs, and only for float16 + axis + NaN.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Axis_Float16_NaN_ConsistentNaNToEnd_NotNumPyVoidSortBug()
        {
            var a = np.array(new Half[,]
            {
                { (Half)1, (Half)2 }, { Half.NaN, (Half)2 }, { (Half)1, (Half)2 }, { Half.NaN, (Half)2 },
            });
            var r = np.unique(a, true, false, true, axis: 0);
            r[0].shape.Should().Equal(3, 2);
            // NumSharp: NaN to end -> idx [0,1,3], counts [2,1,1]. (NumPy f16: NaN first -> [1,3,0], [1,1,2].)
            AssertInt64(r[1], 0, 1, 3);
            AssertInt64(r[2], 2, 1, 1);
        }
    }
}
