using System;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.Tests.Logic
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

        /// <summary>
        ///     A non-bool <c>where=</c> ARRAY raises: NumPy's converter casts the mask with the <c>safe</c> rule, and
        ///     only bool reaches bool (<c>np.all(np.array([1,1,0], np.int32), where=np.array([1,1,0], np.int32))</c> is a
        ///     <c>TypeError</c> in NumPy 2.4.2). Only a Python LIST of ints is read by truth value there, and a C# array
        ///     is an ndarray in NumSharp's mapping, so it raises like NumPy's array. The house type for this check is
        ///     <see cref="ArgumentException"/>, as on the elementwise ufuncs; the text is NumPy's.
        /// </summary>
        [TestMethod]
        public void all_non_bool_where_array_raises_numpys_cast_error()
        {
            var a = np.array(new[] { 1, 1, 0 });
            var w = np.array(new[] { 1, 1, 0 });
            var e = Assert.ThrowsException<ArgumentException>(() => np.all(a, @where: w));
            Assert.AreEqual("Cannot cast array data from dtype('int32') to dtype('bool') according to the rule 'safe'", e.Message);
        }

        /// <summary>
        ///     Every non-bool mask dtype, the method forms, the tuple-axis overload and a 0-d mask (NumPy words a 0-d
        ///     source as a "scalar"). NumPy 2.4.2 texts. Was the [OpenBugs] pin
        ///     <c>AllAny_Where_NonBoolMask_RaisesNumPysCastError</c> until the reduce prologue (ReductionArguments) landed.
        /// </summary>
        [TestMethod]
        public void where_non_bool_mask_raises_for_every_form()
        {
            var b = np.array(new[] { true, false, true, true, true, true }).reshape(2, 3);
            var int64Mask = np.array(new long[] { 1, 0, 1, 0, 1, 1 }).reshape(2, 3);
            var float64Mask = np.array(new[] { 1.0, 0.0, 1.0, 0.0, 1.0, 1.0 }).reshape(2, 3);
            var uint8Mask = np.array(new byte[] { 1, 0, 1, 0, 1, 1 }).reshape(2, 3);
            const string tail = " to dtype('bool') according to the rule 'safe'";
            (string Label, Func<object> Call, string Text)[] cases =
            {
                ("np.all int64", () => np.all(b, @where: int64Mask), "Cannot cast array data from dtype('int64')" + tail),
                ("np.any int64", () => np.any(b, @where: int64Mask), "Cannot cast array data from dtype('int64')" + tail),
                ("np.all float64", () => np.all(b, @where: float64Mask), "Cannot cast array data from dtype('float64')" + tail),
                ("np.any uint8", () => np.any(b, @where: uint8Mask), "Cannot cast array data from dtype('uint8')" + tail),
                ("b.all int64", () => b.all(@where: int64Mask), "Cannot cast array data from dtype('int64')" + tail),
                ("b.any float64", () => b.any(@where: float64Mask), "Cannot cast array data from dtype('float64')" + tail),
                ("np.all tuple axis int64", () => np.all(b, new[] { 0, 1 }, null, false, int64Mask), "Cannot cast array data from dtype('int64')" + tail),
                ("np.all 0-d int64", () => np.all(b, @where: np.array(1L)), "Cannot cast scalar from dtype('int64')" + tail),
            };
            foreach (var (label, call, text) in cases)
            {
                var e = Assert.ThrowsException<ArgumentException>(call, label);
                Assert.AreEqual(text, e.Message, label);
            }

            var boolMask = np.array(new[] { true, false, true, false, true, true }).reshape(2, 3);
            Assert.AreEqual(true, (bool)np.all(b, @where: boolMask));
        }

        /// <summary>
        ///     NumPy's reduce prologue reports in its argument-parsing order (probed 2.4.2): a read-only out before the
        ///     mask, the mask before the axis, the axis before any shape. Each case pairs two faults; the first one
        ///     must win.
        /// </summary>
        [TestMethod]
        public void reduce_prologue_reports_in_numpys_order()
        {
            var b = np.ones(new Shape(2, 3), NPTypeCode.Boolean);
            var readOnlyOut = np.zeros(new Shape(3), NPTypeCode.Boolean);
            readOnlyOut.flags.writeable = false;
            var intMask = np.ones(new Shape(2, 3), NPTypeCode.Int64);

            var e1 = Assert.ThrowsException<ValueError>(() => np.all(b, 0, readOnlyOut, false, intMask));
            Assert.AreEqual("output array is read-only", e1.Message);

            var e2 = Assert.ThrowsException<ArgumentException>(() => np.all(b, 7, null, false, intMask));
            Assert.AreEqual("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'", e2.Message);

            var e3 = Assert.ThrowsException<ArgumentException>(() => np.any(b, null, null, false, np.ones(new Shape(2), NPTypeCode.Int64)));
            StringAssert.StartsWith(e3.Message, "Cannot cast array data from dtype('int64')");

            Assert.ThrowsException<AxisError>(() => np.all(b, 7, np.zeros(new Shape(9, 9), NPTypeCode.Boolean), false, null));
        }

        /// <summary>
        ///     The shape checks of NumPy's reduction iterator, verbatim (probed 2.4.2): the mask broadcasts TO the
        ///     input (never the reverse, and never with more axes), out has the result's rank and extents.
        /// </summary>
        [TestMethod]
        public void reduce_shape_errors_use_numpys_iterator_texts()
        {
            var b = np.ones(new Shape(2, 3), NPTypeCode.Boolean);
            var row = np.ones(new Shape(1, 3), NPTypeCode.Boolean);

            var e1 = Assert.ThrowsException<IncorrectShapeException>(() => np.all(b, null, null, false, np.ones(new Shape(2), NPTypeCode.Boolean)));
            Assert.AreEqual("operands could not be broadcast together with remapped shapes [original->remapped]: (2,3) (2,) ", e1.Message);

            var e2 = Assert.ThrowsException<ValueError>(() => np.all(b, null, null, false, np.ones(new Shape(1, 2, 3), NPTypeCode.Boolean)));
            Assert.AreEqual("input operand has more dimensions than allowed by the axis remapping", e2.Message);

            var e3 = Assert.ThrowsException<ValueError>(() => np.all(row, 0, null, false, np.ones(new Shape(2, 3), NPTypeCode.Boolean)));
            Assert.AreEqual("non-broadcastable operand with shape (1,3) doesn't match the broadcast shape (2,3)", e3.Message);

            var e4 = Assert.ThrowsException<ValueError>(() => np.any(b, 1, np.zeros(new Shape(2, 1), NPTypeCode.Boolean)));
            Assert.AreEqual("output parameter for reduction operation logical_or has the wrong number of dimensions: Found 2 but expected 1", e4.Message);

            var e5 = Assert.ThrowsException<ValueError>(() => np.all(b, null, np.zeros(new Shape(1), NPTypeCode.Boolean), true));
            Assert.AreEqual("output parameter for reduction operation logical_and has the wrong number of dimensions: Found 1 but expected 2 (must match the operand's when keepdims=True)", e5.Message);

            var e6 = Assert.ThrowsException<ValueError>(() => np.all(b, 1, np.zeros(new Shape(1), NPTypeCode.Boolean)));
            Assert.AreEqual("output operand requires a reduction along dimension 0, but the reduction is not enabled. The dimension size of 1 does not match the expected output shape.", e6.Message);

            var e7 = Assert.ThrowsException<ValueError>(() => np.all(b, 1, np.zeros(new Shape(2, 3), NPTypeCode.Boolean), true));
            Assert.AreEqual("operand was set up as a reduction along axis 1, but the length of the axis is 3 (it has to be 1)", e7.Message);

            var e8 = Assert.ThrowsException<IncorrectShapeException>(() => np.all(b, 1, np.zeros(new Shape(3), NPTypeCode.Boolean), false, np.ones(new Shape(2), NPTypeCode.Boolean)));
            Assert.AreEqual("operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3,newaxis) (2,3) (2,) ", e8.Message);
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

        /// <summary>
        ///     An out of the right rank but the wrong extent fails NumPy's reduction-iterator broadcast, with its
        ///     remapped-shapes text (<c>np.all(np.ones((2,3)), axis=0, out=np.empty(4, bool))</c>, NumPy 2.4.2). The
        ///     dropped reduced axis is a leading newaxis, which NumPy's shape printer omits: <c>(4,)-&gt;(4)</c>.
        /// </summary>
        [TestMethod]
        public void all_out_shape_mismatch_throws()
        {
            var a = np.ones(new Shape(2, 3));
            var badOut = np.empty(new Shape(4), NPTypeCode.Boolean);
            var e = Assert.ThrowsException<IncorrectShapeException>(() => np.all(a, axis: 0, @out: badOut));
            Assert.AreEqual("operands could not be broadcast together with remapped shapes [original->remapped]: (4,)->(4) (2,3) ", e.Message);
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
