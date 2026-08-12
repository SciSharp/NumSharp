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

        // ---- Bare single-return overload: np.unique(ar, axis=/equal_nan=/sorted=) -> NDArray
        //      (NumPy returns a bare array when no return_* flag is set; this overload matches that,
        //      so `np.unique(m, axis=0)` ports verbatim instead of forcing `np.unique(m, false, axis:0)[0]`).
        [TestMethod]
        public void BareReturn_Axis0_ReturnsSingleNDArray()
        {
            var m = np.array(new[,] { { 1, 0, 1 }, { 2, 3, 2 }, { 1, 0, 1 }, { 0, 0, 0 } });
            // >>> np.unique(m, axis=0) -> [[0,0,0],[1,0,1],[2,3,2]]
            NDArray u = np.unique(m, axis: 0);                 // binds to the bare NDArray overload
            u.shape.Should().Equal(3, 3);
            u.array_equal(np.unique(m, false, false, false, axis: 0)[0]).Should().BeTrue();
            u.array_equal(np.array(new[,] { { 0, 0, 0 }, { 1, 0, 1 }, { 2, 3, 2 } })).Should().BeTrue();
        }

        [TestMethod]
        public void BareReturn_NegativeAxis()
        {
            var m = np.array(new[,] { { 1, 0, 1 }, { 2, 3, 2 }, { 5, 4, 5 } });
            // >>> np.unique(m, axis=-1) collapses the duplicate columns
            NDArray u = np.unique(m, axis: -1);
            u.array_equal(np.unique(m, false, false, false, axis: -1)[0]).Should().BeTrue();
        }

        [TestMethod]
        public void BareReturn_Flat_Plain_And_Keywords()
        {
            var x = np.array(new[] { 3, 1, 2, 1, 3, 2, 5, 4 });
            // plain bare-return still works and stays sorted
            np.unique(x).array_equal(np.array(new[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
            // sorted=false keyword reaches the bare overload (still sorted in NumSharp)
            np.unique(x, sorted: false).array_equal(np.array(new[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void BareReturn_EqualNanFalse_KeepsEveryNaN()
        {
            // >>> np.unique([1., nan, nan, 2.], equal_nan=False) -> [1., 2., nan, nan]  (bare array)
            NDArray u = np.unique(np.array(new[] { 1.0, double.NaN, double.NaN, 2.0 }), equal_nan: false);
            u.size.Should().Be(4);
            var v = u.ToArray<double>();
            v[0].Should().Be(1.0); v[1].Should().Be(2.0);
            double.IsNaN(v[2]).Should().BeTrue(); double.IsNaN(v[3]).Should().BeTrue();
            // equal_nan defaults to true -> a single NaN survives (still bare-return)
            np.unique(np.array(new[] { 1.0, double.NaN, double.NaN, 2.0 })).size.Should().Be(3);
        }

        // ---- UniqueResult: NumPy's bare-array-OR-tuple return as one type. The whole point of the
        //      redesign — np.unique(x, return_counts=True) / return_inverse=True port from Python VERBATIM
        //      (no forced `false` for return_index), plus Deconstruct / implicit NDArray[] / indexer / named fields.
        [TestMethod]
        public void UniqueResult_ReturnCounts_Verbatim_Deconstruct()
        {
            var x = np.array(new[] { 3, 1, 2, 1, 3, 3, 5 });
            // >>> values, counts = np.unique(x, return_counts=True)  — assume_unique/return_index NOT needed
            var (values, counts) = np.unique(x, return_counts: true);
            values.array_equal(np.array(new[] { 1, 2, 3, 5 })).Should().BeTrue();
            counts.array_equal(np.array(new long[] { 2, 1, 3, 1 })).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueResult_ReturnInverse_Verbatim_Reconstructs()
        {
            var x = np.array(new[] { 3, 1, 2, 1, 3, 3, 5 });
            // >>> values, inv = np.unique(x, return_inverse=True); np.take(values, inv) == x
            var (values, inv) = np.unique(x, return_inverse: true);
            inv.array_equal(np.array(new long[] { 2, 0, 1, 0, 2, 2, 3 })).Should().BeTrue();
            np.take(values, inv).array_equal(x).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueResult_NamedFields_And_Nulls()
        {
            var x = np.array(new[] { 5, 1, 5, 3 });
            var r = np.unique(x, return_counts: true);
            r.Values.array_equal(np.array(new[] { 1, 3, 5 })).Should().BeTrue();
            r.Counts.array_equal(np.array(new long[] { 1, 1, 2 })).Should().BeTrue();
            r.Index.Should().BeNull("return_index was False");
            r.Inverse.Should().BeNull("return_inverse was False");
            r.Length.Should().Be(2);       // [values, counts]
        }

        [TestMethod]
        public void UniqueResult_ImplicitConversions_And_Indexer()
        {
            var x = np.array(new[] { 5, 1, 5, 3 });
            // implicit NDArray (bare) yields Values
            NDArray bare = np.unique(x);
            bare.array_equal(np.array(new[] { 1, 3, 5 })).Should().BeTrue();
            // implicit NDArray[] (tuple) yields the present outputs, NumPy field order
            NDArray[] tup = np.unique(x, return_index: true, return_counts: true);
            tup.Length.Should().Be(3);                                          // [values, index, counts]
            tup[0].array_equal(np.array(new[] { 1, 3, 5 })).Should().BeTrue();
            tup[1].array_equal(np.array(new long[] { 1, 3, 0 })).Should().BeTrue();
            tup[2].array_equal(np.array(new long[] { 1, 1, 2 })).Should().BeTrue();
            // indexer selects the k-th present output (matches the old NDArray[] return)
            var r = np.unique(x, return_counts: true);
            r[0].array_equal(np.array(new[] { 1, 3, 5 })).Should().BeTrue();
            r[1].array_equal(np.array(new long[] { 1, 1, 2 })).Should().BeTrue();
        }
    }
}
