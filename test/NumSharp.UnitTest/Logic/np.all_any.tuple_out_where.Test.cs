using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Coverage for the NumPy 2.x additions to <c>np.all</c> / <c>np.any</c>:
    ///   - tuple-axis (axis=(int, int, ...))
    ///   - out= keyword
    ///   - where= keyword
    ///   - axis=() (empty tuple) — input cast to bool, no reduction
    ///   - keepdims=True with axis=None
    /// Expected values were generated against numpy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpAllAnyTupleOutWhereTest
    {
        // === Tuple axis ===

        [TestMethod]
        public void all_tuple_axis_basic()
        {
            // numpy: np.all(np.ones((2,3,4)), axis=(0,2)).shape == (3,) and all True
            var a = np.ones(new Shape(2, 3, 4));
            var r = np.all(a, new[] { 0, 2 });
            CollectionAssert.AreEqual(new long[] { 3 }, r.shape);
            Assert.IsTrue(r.GetValue<bool>(0) && r.GetValue<bool>(1) && r.GetValue<bool>(2));
        }

        [TestMethod]
        public void all_tuple_axis_all_axes_returns_scalar()
        {
            var a = np.ones(new Shape(2, 3, 4));
            var r = np.all(a, new[] { 0, 1, 2 });
            CollectionAssert.AreEqual(Array.Empty<long>(), r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void all_tuple_axis_empty_tuple_no_reduction()
        {
            // numpy: np.all(a, axis=()) returns input cast to bool, same shape.
            var a = np.array(new[,] { { 1, 0, 1 }, { 1, 1, 1 } });
            var r = np.all(a, Array.Empty<int>());
            CollectionAssert.AreEqual(new long[] { 2, 3 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0, 0));
            Assert.AreEqual(false, r.GetValue<bool>(0, 1));
            Assert.AreEqual(true, r.GetValue<bool>(0, 2));
            Assert.AreEqual(true, r.GetValue<bool>(1, 0));
        }

        [TestMethod]
        public void all_tuple_axis_keepdims_preserves_dim_positions()
        {
            // numpy: shape == (1, 3, 1)
            var a = np.ones(new Shape(2, 3, 4));
            var r = np.all(a, new[] { 0, 2 }, keepdims: true);
            CollectionAssert.AreEqual(new long[] { 1, 3, 1 }, r.shape);
        }

        [TestMethod]
        public void all_tuple_axis_negative_axis_resolves()
        {
            // axis=(-1, 0) equivalent to (0, 2) for 3-D input.
            var a = np.ones(new Shape(2, 3, 4));
            var r = np.all(a, new[] { -1, 0 });
            CollectionAssert.AreEqual(new long[] { 3 }, r.shape);
        }

        [TestMethod]
        public void all_tuple_axis_duplicate_throws()
        {
            // numpy raises ValueError("duplicate value in 'axis'")
            var a = np.ones(new Shape(2, 3, 4));
            Assert.ThrowsException<ArgumentException>(() => np.all(a, new[] { 0, 0 }));
        }

        [TestMethod]
        public void all_tuple_axis_out_of_bounds_throws()
        {
            var a = np.ones(new Shape(2, 3, 4));
            Assert.ThrowsException<AxisError>(() => np.all(a, new[] { 0, 3 }));
        }

        [TestMethod]
        public void all_tuple_axis_with_falsy_values()
        {
            // numpy: 2x2x3 array, all(axis=(0,2)) should report per-column truthiness.
            var a = np.array(new[, ,]
            {
                { { 1, 1, 1 }, { 1, 1, 1 } },
                { { 1, 0, 1 }, { 1, 1, 1 } }
            });
            var r = np.all(a, new[] { 0, 2 });
            CollectionAssert.AreEqual(new long[] { 2 }, r.shape);
            Assert.AreEqual(false, r.GetValue<bool>(0));  // 2nd plane row 0 has 0
            Assert.AreEqual(true, r.GetValue<bool>(1));
        }

        [TestMethod]
        public void any_tuple_axis_basic()
        {
            var a = np.zeros(new Shape(2, 3, 4));
            a[1, 1, 2] = 1;  // single truthy element
            var r = np.any(a, new[] { 0, 2 });
            CollectionAssert.AreEqual(new long[] { 3 }, r.shape);
            Assert.AreEqual(false, r.GetValue<bool>(0));
            Assert.AreEqual(true, r.GetValue<bool>(1));
            Assert.AreEqual(false, r.GetValue<bool>(2));
        }

        [TestMethod]
        public void any_tuple_axis_empty_tuple_no_reduction()
        {
            var a = np.array(new[,] { { 1, 0 }, { 0, 1 } });
            var r = np.any(a, Array.Empty<int>());
            CollectionAssert.AreEqual(new long[] { 2, 2 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0, 0));
            Assert.AreEqual(false, r.GetValue<bool>(0, 1));
        }

        // === keepdims with axis=None ===

        [TestMethod]
        public void all_axis_none_keepdims_returns_1x1x1()
        {
            var a = np.ones(new Shape(2, 3, 4));
            var r = np.all(a, keepdims: true);
            CollectionAssert.AreEqual(new long[] { 1, 1, 1 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0, 0, 0));
        }

        [TestMethod]
        public void any_axis_none_keepdims_returns_1x1x1()
        {
            var a = np.zeros(new Shape(2, 3, 4));
            a[0, 0, 0] = 1;
            var r = np.any(a, keepdims: true);
            CollectionAssert.AreEqual(new long[] { 1, 1, 1 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0, 0, 0));
        }

        [TestMethod]
        public void all_axis_none_keepdims_false_returns_0d()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var r = np.all(a, keepdims: false);
            CollectionAssert.AreEqual(Array.Empty<long>(), r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        // === where= keyword ===

        [TestMethod]
        public void all_where_drops_falsy_from_consideration()
        {
            // numpy: np.all([[1,1],[1,0]], where=[[True,True],[False,False]]) == True
            var a = np.array(new[,] { { 1, 1 }, { 1, 0 } });
            var w = np.array(new[,] { { true, true }, { false, false } });
            var r = np.all(a, @where: w);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void all_where_with_axis_per_row()
        {
            // numpy:
            //   a = [[1, 0, 1], [1, 1, 1]]
            //   w = [[T, T, F], [T, F, T]]
            //   axis=1 → row 0: check a[0,0]=1, a[0,1]=0 → False
            //            row 1: check a[1,0]=1, a[1,2]=1 → True
            var a = np.array(new[,] { { 1, 0, 1 }, { 1, 1, 1 } });
            var w = np.array(new[,] { { true, true, false }, { true, false, true } });
            var r = np.all(a, axis: 1, @where: w);
            CollectionAssert.AreEqual(new long[] { 2 }, r.shape);
            Assert.AreEqual(false, r.GetValue<bool>(0));
            Assert.AreEqual(true, r.GetValue<bool>(1));
        }

        [TestMethod]
        public void any_where_drops_truthy_from_consideration()
        {
            // numpy: np.any([0,0,0,1], where=[T,T,T,F]) → False (the 1 is masked out)
            var a = np.array(new[] { 0, 0, 0, 1 });
            var w = np.array(new[] { true, true, true, false });
            var r = np.any(a, @where: w);
            Assert.AreEqual(false, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void any_where_scalar_false_is_vacuous_false()
        {
            // numpy: np.any(anything, where=False) is False (vacuous false)
            var a = np.ones(new Shape(3, 4));
            var w = np.array(false);
            var r = np.any(a, @where: w);
            Assert.AreEqual(false, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void all_where_scalar_false_is_vacuous_true()
        {
            // numpy: np.all(anything, where=False) is True (vacuous truth)
            var a = np.zeros(new Shape(3, 4));
            var w = np.array(false);
            var r = np.all(a, @where: w);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void all_where_broadcasts_against_input()
        {
            // numpy:
            //   a = [[1, 0, 1], [1, 1, 1]]
            //   w = [True, False, True]  # broadcasts along axis 0
            //   axis=0 → col 0: a[:,0]=[1,1] all True; col 2: a[:,2]=[1,1] all True
            //            col 1: masked out → True (vacuous)
            //   so result = [True, True, True]
            var a = np.array(new[,] { { 1, 0, 1 }, { 1, 1, 1 } });
            var w = np.array(new[] { true, false, true });
            var r = np.all(a, axis: 0, @where: w);
            CollectionAssert.AreEqual(new long[] { 3 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0));
            Assert.AreEqual(true, r.GetValue<bool>(1));
            Assert.AreEqual(true, r.GetValue<bool>(2));
        }

        [TestMethod]
        public void all_where_with_tuple_axis()
        {
            // 3-D input with tuple-axis reduction under a 3-D mask.
            // numpy:
            //   a = ones((2,3,4)); a[0,1,2] = 0
            //   w = ones((2,3,4), bool); w[0,1,2] = False
            //   axis=(0,2): row 1 has the zero masked → all True for j=1
            //   result shape (3,); all values True.
            var a = np.ones(new Shape(2, 3, 4));
            a[0, 1, 2] = 0;
            var w = np.ones(new Shape(2, 3, 4), NPTypeCode.Boolean);
            w[0, 1, 2] = false;
            var r = np.all(a, new[] { 0, 2 }, @out: null, @where: w);
            CollectionAssert.AreEqual(new long[] { 3 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0));
            Assert.AreEqual(true, r.GetValue<bool>(1));
            Assert.AreEqual(true, r.GetValue<bool>(2));
        }

        [TestMethod]
        public void all_non_bool_where_treated_as_truthy()
        {
            // numpy: where=int → non-zero treated as True (truthy)
            var a = np.array(new[] { 1, 1, 0 });
            var w = np.array(new[] { 1, 1, 0 });  // last is "false" → ignored
            var r = np.all(a, @where: w);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        // === out= keyword ===

        [TestMethod]
        public void all_out_returns_same_instance_and_writes_into_it()
        {
            var a = np.ones(new Shape(2, 3));
            var outArr = np.zeros(new Shape(3), NPTypeCode.Boolean);
            var r = np.all(a, axis: 0, @out: outArr);
            Assert.IsTrue(ReferenceEquals(r, outArr));
            Assert.AreEqual(true, outArr.GetValue<bool>(0));
            Assert.AreEqual(true, outArr.GetValue<bool>(1));
            Assert.AreEqual(true, outArr.GetValue<bool>(2));
        }

        [TestMethod]
        public void all_out_with_int_dtype_stores_zero_one()
        {
            // numpy preserves out's dtype: int32 receives 1/0.
            var a = np.ones(new Shape(2, 3, 4));
            var outArr = np.empty(new Shape(3), NPTypeCode.Int32);
            var r = np.all(a, axis: new[] { 0, 2 }, @out: outArr);
            Assert.AreEqual(typeof(int), r.dtype);
            Assert.AreEqual(1, r.GetValue<int>(0));
            Assert.AreEqual(1, r.GetValue<int>(1));
            Assert.AreEqual(1, r.GetValue<int>(2));
        }

        [TestMethod]
        public void all_out_with_float_dtype_stores_zero_one_floats()
        {
            var a = np.array(new[,] { { 1, 1, 1 }, { 1, 0, 1 } });
            var outArr = np.empty(new Shape(3), NPTypeCode.Double);
            var r = np.all(a, axis: 0, @out: outArr);
            Assert.AreEqual(1.0, r.GetValue<double>(0));
            Assert.AreEqual(0.0, r.GetValue<double>(1));
            Assert.AreEqual(1.0, r.GetValue<double>(2));
        }

        [TestMethod]
        public void any_out_returns_same_instance()
        {
            var a = np.array(new[,] { { 0, 0, 1 }, { 0, 1, 0 } });
            var outArr = np.empty(new Shape(3), NPTypeCode.Boolean);
            var r = np.any(a, axis: 0, @out: outArr);
            Assert.IsTrue(ReferenceEquals(r, outArr));
            Assert.AreEqual(false, outArr.GetValue<bool>(0));
            Assert.AreEqual(true, outArr.GetValue<bool>(1));
            Assert.AreEqual(true, outArr.GetValue<bool>(2));
        }

        [TestMethod]
        public void all_out_shape_mismatch_throws()
        {
            var a = np.ones(new Shape(2, 3));
            var badOut = np.empty(new Shape(4), NPTypeCode.Boolean);
            Assert.ThrowsException<ArgumentException>(() => np.all(a, axis: 0, @out: badOut));
        }

        [TestMethod]
        public void all_out_with_where_and_axis_full_combo()
        {
            // Full combo: tuple-axis + where + out (non-bool dtype).
            var a = np.ones(new Shape(2, 3, 4));
            a[0, 0, 0] = 0;  // a single zero
            var w = np.ones(new Shape(2, 3, 4), NPTypeCode.Boolean);
            w[0, 0, 0] = false;  // mask out the zero
            var outArr = np.empty(new Shape(3), NPTypeCode.Int32);
            var r = np.all(a, axis: new[] { 0, 2 }, @out: outArr, @where: w);
            Assert.IsTrue(ReferenceEquals(r, outArr));
            Assert.AreEqual(1, outArr.GetValue<int>(0));
            Assert.AreEqual(1, outArr.GetValue<int>(1));
            Assert.AreEqual(1, outArr.GetValue<int>(2));
        }

        // === Empty / edge ===

        [TestMethod]
        public void all_empty_array_with_tuple_axis()
        {
            // numpy: np.all(empty((0,3,4)), axis=(1,)) shape (0,4)
            var a = np.empty(new Shape(0, 3, 4));
            var r = np.all(a, new[] { 1 });
            CollectionAssert.AreEqual(new long[] { 0, 4 }, r.shape);
        }

        [TestMethod]
        public void all_empty_array_reduce_to_scalar_via_all_axes()
        {
            // numpy: np.all(empty((0,3,4)), axis=(0,1,2)) → True (vacuous truth, all axes reduced)
            // and result has shape (4,) when we reduce (0,1) since axis 2 stays
            var a = np.empty(new Shape(0, 3, 4));
            var r = np.all(a, new[] { 0, 1 });
            CollectionAssert.AreEqual(new long[] { 4 }, r.shape);
            Assert.AreEqual(true, r.GetValue<bool>(0));
        }

        [TestMethod]
        public void all_NaN_is_truthy()
        {
            // numpy: np.all([1.0, np.nan]) == True (NaN is non-zero)
            var a = np.array(new[] { 1.0, double.NaN });
            Assert.AreEqual(true, np.all(a));
        }

        [TestMethod]
        public void any_NaN_is_truthy()
        {
            var a = np.array(new[] { 0.0, double.NaN });
            Assert.AreEqual(true, np.any(a));
        }
    }
}
