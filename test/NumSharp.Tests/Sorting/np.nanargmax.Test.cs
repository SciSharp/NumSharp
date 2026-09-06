using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    /// np.nanargmax / np.nanargmin parity with NumPy 2.4.2 (every expected value produced by
    /// running NumPy). Implementation: port of numpy/lib/_nanfunctions_impl.py — _replace_nan
    /// with ∓inf + the all-NaN-slice ValueError guard, then argmax/argmin.
    /// </summary>
    [TestClass]
    public class NpNanArgMaxTests
    {
        // -------------------- flat --------------------
        [TestMethod]
        public void NanArgMax_Flat()
        {
            // np.nanargmax([[nan, 4], [2, 3]]) == 1 ; np.argmax would give 0 (NaN propagates)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            np.nanargmax(a).Should().Be(1L);
            np.argmax(a).Should().Be(0L);
        }

        [TestMethod]
        public void NanArgMin_Flat()
        {
            // np.nanargmin([[nan, 4], [2, 3]]) == 2
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            np.nanargmin(a).Should().Be(2L);
        }

        // -------------------- axis --------------------
        [TestMethod]
        public void NanArgMax_Axis0()
        {
            // np.nanargmax([[nan, 4], [2, 3]], axis=0) == [1, 0], dtype int64
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var r = np.nanargmax(a, 0);
            r.dtype.Should().Be(typeof(long));
            r.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMin_Axis1()
        {
            // np.nanargmin([[nan, 4], [2, 3]], axis=1) == [1, 0]
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            np.nanargmin(a, 1).ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_NegativeAxis()
        {
            // np.nanargmax([[nan, 4], [2, 3]], axis=-1) == [1, 1]
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            np.nanargmax(a, -1).ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 1 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_3D_MiddleAxis()
        {
            // a = arange(24).reshape(2,3,4).astype(f8); a[0,1,2] = nan
            // np.nanargmax(a, axis=1) == [[2,2,2,2],[2,2,2,2]] shape (2,4)
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            a.SetValue(double.NaN, 0, 1, 2);
            var r = np.nanargmax(a, 1);
            r.Shape.Should().Be(new Shape(2, 4));
            r.ToArray<long>().Should().OnlyContain(x => x == 2);
        }

        // -------------------- keepdims --------------------
        [TestMethod]
        public void NanArgMax_Keepdims_Axis()
        {
            // np.nanargmax([[nan, 4], [2, 3]], axis=0, keepdims=True) == [[1, 0]] shape (1,2)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var r = np.nanargmax(a, 0, keepdims: true);
            r.Shape.Should().Be(new Shape(1, 2));
            r.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_Keepdims_FlatAxisNull()
        {
            // np.nanargmax([[nan, 4], [2, 3]], keepdims=True) == [[1]] shape (1,1)  (axis=None)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var r = np.nanargmax(a, null, keepdims: true);
            r.Shape.Should().Be(new Shape(1, 1));
            r.GetValue<long>(0, 0).Should().Be(1L);
        }

        [TestMethod]
        public void NanArgMax_FlatAxisNull_NoKeepdims_Is0d()
        {
            // np.nanargmax(a, axis=None) → np.int64 scalar
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var r = np.nanargmax(a, null);
            r.Shape.IsScalar.Should().BeTrue();
            ((long)r).Should().Be(1L);
        }

        // -------------------- all-NaN slices --------------------
        [TestMethod]
        public void NanArgMax_AllNaN_Flat_Throws()
        {
            // np.nanargmax([nan, nan]) → ValueError: All-NaN slice encountered
            var act = () => np.nanargmax(np.array(new[] { double.NaN, double.NaN }));
            act.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
        }

        [TestMethod]
        public void NanArgMin_AllNaN_Flat_Throws()
        {
            var act = () => np.nanargmin(np.array(new[] { double.NaN, double.NaN }));
            act.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
        }

        [TestMethod]
        public void NanArgMax_AllNaNRow_AxisThatHitsIt_Throws()
        {
            // a = [[nan, nan], [1, 2]]: axis=1 has an all-NaN slice → ValueError
            var a = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
            var act = () => np.nanargmax(a, 1);
            act.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
        }

        [TestMethod]
        public void NanArgMax_AllNaNRow_OtherAxis_Ok()
        {
            // same array, axis=0: no column is all-NaN → [1, 1]
            var a = np.array(new double[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
            np.nanargmax(a, 0).ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 1 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_0d_NaN_Throws()
        {
            // np.nanargmax(np.array(nan)) → ValueError
            var act = () => np.nanargmax(NDArray.Scalar(double.NaN));
            act.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
        }

        // -------------------- integer / bool passthrough --------------------
        [TestMethod]
        public void NanArgMax_Int_Passthrough()
        {
            // np.nanargmax([[3,1],[2,9]]) == 3 ; np.nanargmin == 1 (no NaN concept, no error)
            var i = np.array(new int[,] { { 3, 1 }, { 2, 9 } });
            np.nanargmax(i).Should().Be(3L);
            np.nanargmin(i).Should().Be(1L);
        }

        [TestMethod]
        public void NanArgMax_Bool()
        {
            // np.nanargmax([True, False, True]) == 0
            np.nanargmax(np.array(new[] { true, false, true })).Should().Be(0L);
        }

        [TestMethod]
        public void NanArgMax_Int_Keepdims_FlatAxisNull()
        {
            // np.nanargmax([[1,2],[3,4]], keepdims=True) == [[3]] shape (1,1)
            var i = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var r = np.nanargmax(i, null, keepdims: true);
            r.Shape.Should().Be(new Shape(1, 1));
            r.GetValue<long>(0, 0).Should().Be(3L);
        }

        // -------------------- Half / Complex --------------------
        [TestMethod]
        public void NanArgMax_Half()
        {
            // np.nanargmax(f16 [nan, 2, 1]) == 1 ; nanargmin == 2
            var h = np.array(new Half[] { Half.NaN, (Half)2.0, (Half)1.0 });
            np.nanargmax(h).Should().Be(1L);
            np.nanargmin(h).Should().Be(2L);
        }

        [TestMethod]
        public void NanArgMax_Complex()
        {
            // np.nanargmax([nan+1j, 2+3j, 1+nan*j]) == 1 (NaN in either component masks the value);
            // nanargmin == 1 as well — indices 0 and 2 are NaN, only one candidate survives
            var c = np.array(new[] { new Complex(double.NaN, 1), new Complex(2, 3), new Complex(1, double.NaN) });
            np.nanargmax(c).Should().Be(1L);
            np.nanargmin(c).Should().Be(1L);
        }

        [TestMethod]
        public void NanArgMax_Complex_AllNaN_Throws()
        {
            // both entries have a NaN component → all-NaN → ValueError
            var c = np.array(new[] { new Complex(double.NaN, 0), new Complex(0, double.NaN) });
            var act = () => np.nanargmax(c);
            act.Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
        }

        // -------------------- empty --------------------
        [TestMethod]
        public void NanArgMax_Empty_Flat_Throws()
        {
            // np.nanargmax([]) → ValueError: attempt to get argmax of an empty sequence
            // (house type: ArgumentException, NumPy's exact text — the engine's own error)
            var act = () => np.nanargmax(np.array(new double[0]));
            act.Should().Throw<ArgumentException>().WithMessage("attempt to get argmax of an empty sequence*");
        }

        [TestMethod]
        public void NanArgMin_Empty_ReduceAlongZeroAxis_Throws()
        {
            // np.nanargmin(np.zeros((0,3)), axis=0) → ValueError (reducing along zero-size axis)
            var act = () => np.nanargmin(np.zeros(new Shape(0, 3)), 0);
            act.Should().Throw<ArgumentException>().WithMessage("attempt to get argmin of an empty sequence*");
        }

        [TestMethod]
        public void NanArgMax_Empty_OtherAxis_EmptyResult()
        {
            // np.nanargmax(np.zeros((0,3)), axis=1) == array([], dtype=int64) shape (0,)
            var r = np.nanargmax(np.zeros(new Shape(0, 3)), 1);
            r.size.Should().Be(0);
            r.dtype.Should().Be(typeof(long));
        }

        // -------------------- 0-d --------------------
        [TestMethod]
        public void NanArgMax_0d_Value()
        {
            // np.nanargmax(np.array(5.0)) == 0; axis=0 and axis=-1 are legal on 0-d (NumPy quirk)
            np.nanargmax(NDArray.Scalar(5.0)).Should().Be(0L);
            ((long)np.nanargmax(NDArray.Scalar(5.0), 0)).Should().Be(0L);
            ((long)np.nanargmax(NDArray.Scalar(5.0), -1)).Should().Be(0L);
        }

        [TestMethod]
        public void NanArgMax_0d_AxisOut_Throws()
        {
            // np.nanargmax(np.array(5.0), axis=1) → AxisError: axis 1 is out of bounds for array of dimension 0
            var act = () => np.nanargmax(NDArray.Scalar(5.0), 1);
            act.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 0*");
        }

        [TestMethod]
        public void NanArgMax_0d_Keepdims_StaysScalar()
        {
            // np.nanargmax(np.array(5.0), keepdims=True) → np.int64(0), shape ()
            var r = np.nanargmax(NDArray.Scalar(5.0), null, keepdims: true);
            r.Shape.IsScalar.Should().BeTrue();
            ((long)r).Should().Be(0L);
        }

        // -------------------- axis validation (both dtype paths) --------------------
        [TestMethod]
        public void NanArgMax_AxisOutOfBounds_Positive()
        {
            // np.nanargmax(a, axis=5) → AxisError: axis 5 is out of bounds for array of dimension 2
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var act = () => np.nanargmax(a, 5);
            act.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
        }

        [TestMethod]
        public void NanArgMax_AxisOutOfBounds_Negative_BothDtypePaths()
        {
            // np.nanargmax(a, axis=-3) → AxisError reporting the ORIGINAL -3, for float AND int input
            // (the int path must not inherit the engine's silent negative-axis wrap-around)
            var f = np.array(new double[,] { { 1.0, 4.0 }, { 2.0, 3.0 } });
            var i = np.array(new int[,] { { 1, 4 }, { 2, 3 } });
            ((Action)(() => np.nanargmax(f, -3))).Should().Throw<AxisError>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
            ((Action)(() => np.nanargmax(i, -3))).Should().Throw<AxisError>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
        }

        // -------------------- NaN/Inf untrusted-tie caveat (falls out of the composition) --------------------
        [TestMethod]
        public void NanArgMin_NaNInfTies_MatchNumPy()
        {
            // np.nanargmin([-inf, nan, -inf]) == 0 (nan→+inf, first -inf wins)
            // np.nanargmax([-inf, nan, -inf]) == 0 (nan→-inf, all equal, first wins)
            // np.nanargmin([nan, inf]) == 0 (nan→+inf ties inf; FIRST position wins — the documented
            //   "cannot be trusted if a slice contains only NaNs and Infs" NumPy caveat)
            var tie = np.array(new[] { double.NegativeInfinity, double.NaN, double.NegativeInfinity });
            np.nanargmin(tie).Should().Be(0L);
            np.nanargmax(tie).Should().Be(0L);
            np.nanargmin(np.array(new[] { double.NaN, double.PositiveInfinity })).Should().Be(0L);
        }

        // -------------------- layouts --------------------
        [TestMethod]
        public void NanArgMax_StridedView()
        {
            // a = arange(12).reshape(3,4).astype(f8); a[1,2]=nan; v = a[:, ::2]
            // np.nanargmax(v, axis=0) == [2, 2]; np.nanargmax(v) == 5
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            a.SetValue(double.NaN, 1, 2);
            var v = a[":, ::2"];
            np.nanargmax(v, 0).ToArray<long>().Should().BeEquivalentTo(new long[] { 2, 2 }, o => o.WithStrictOrdering());
            np.nanargmax(v).Should().Be(5L);
        }

        [TestMethod]
        public void NanArgMax_Transposed()
        {
            // np.nanargmax(a.T, axis=1) == [2, 2, 2, 2]
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            a.SetValue(double.NaN, 1, 2);
            np.nanargmax(a.T, 1).ToArray<long>().Should().BeEquivalentTo(new long[] { 2, 2, 2, 2 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_DoesNotMutateInput()
        {
            // _replace_nan copies — the original keeps its NaN
            var a = np.array(new[] { double.NaN, 1.0 });
            np.nanargmax(a);
            double.IsNaN(a.GetValue<double>(0)).Should().BeTrue();
            a.GetValue<double>(1).Should().Be(1.0);
        }

        [TestMethod]
        public void NanArgMax_Float32()
        {
            // np.nanargmax(f4 [nan, 2, 7, nan, 3]) == 2; nanargmin == 1
            var a = np.array(new float[] { float.NaN, 2f, 7f, float.NaN, 3f });
            np.nanargmax(a).Should().Be(2L);
            np.nanargmin(a).Should().Be(1L);
        }

        [TestMethod]
        public void NanArgMax_DecimalAndChar_Passthrough()
        {
            // no NumPy analog dtypes take the integer passthrough route (mask=None): plain argmax/argmin
            var d = np.array(new decimal[] { 3, 9, 1, 5 });
            np.nanargmax(d).Should().Be(1L);
            np.nanargmin(d).Should().Be(2L);
            var c = np.array(new[] { 'c', 'z', 'a' });
            np.nanargmax(c).Should().Be(1L);
            np.nanargmin(c).Should().Be(2L);
        }

        // -------------------- out= (probed NumPy 2.4.2 semantics) --------------------
        [TestMethod]
        public void NanArgMax_Out_SameInstance_WrittenThrough()
        {
            // np.nanargmax(a, axis=0, out=o) writes o and returns o ITSELF (r is out → True)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var o = np.zeros(new Shape(2), NPTypeCode.Int64);
            var r = np.nanargmax(a, 0, o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, x => x.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_Out_IntegerDtypes_CastUnsafely()
        {
            // out may be ANY integer dtype safe-castable to intp; values cast UNSAFELY into it:
            // int32 out keeps int32 (probed), bool out reads index != 0, int8 out WRAPS (200 → -56)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var o32 = np.zeros(new Shape(2), NPTypeCode.Int32);
            np.nanargmax(a, 0, o32).typecode.Should().Be(NPTypeCode.Int32);
            o32.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 0 }, x => x.WithStrictOrdering());

            var ob = np.zeros(new Shape(2), NPTypeCode.Boolean);
            np.nanargmax(a, 0, ob);
            ob.GetValue<bool>(0).Should().BeTrue();
            ob.GetValue<bool>(1).Should().BeFalse();

            var big = np.zeros(new Shape(300));
            big.SetValue(5.0, 200);
            var o8 = np.zeros(new Shape(), NPTypeCode.SByte);
            np.nanargmax(big, null, o8);
            o8.GetValue<sbyte>(0).Should().Be(-56, "index 200 wraps into int8 exactly like NumPy");
        }

        [TestMethod]
        public void NanArgMax_Out_BadDtypes_VerbatimTypeError()
        {
            // uint64/floats/complex are NOT safe-castable to intp — NumPy's verbatim cast TypeError
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            ((Action)(() => np.nanargmax(a, 0, np.zeros(new Shape(2), NPTypeCode.UInt64))))
                .Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule 'safe'*");
            ((Action)(() => np.nanargmax(a, 0, np.zeros(new Shape(2), NPTypeCode.Half))))
                .Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('float16')*");
            ((Action)(() => np.nanargmax(a, 0, np.zeros(new Shape(2), NPTypeCode.Complex))))
                .Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('complex128')*");
        }

        [TestMethod]
        public void NanArgMax_Out_ShapeMismatch_LeaksArgmaxArgminNames()
        {
            // "output array does not match result of np.argmax." — nanargmax leaks argmax,
            // nanargmin leaks argmin (probed verbatim, trailing period included)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            ((Action)(() => np.nanargmax(a, 0, np.zeros(new Shape(3), NPTypeCode.Int64))))
                .Should().Throw<ValueError>().WithMessage("output array does not match result of np.argmax.*");
            ((Action)(() => np.nanargmin(a, 0, np.zeros(new Shape(3), NPTypeCode.Int64))))
                .Should().Throw<ValueError>().WithMessage("output array does not match result of np.argmin.*");
            // keepdims changes the required out shape: (2,) out with keepdims=True (needs (1,2)) rejects
            ((Action)(() => np.nanargmax(a, 0, np.zeros(new Shape(2), NPTypeCode.Int64), keepdims: true)))
                .Should().Throw<ValueError>().WithMessage("output array does not match result of np.argmax.*");
            var okd = np.zeros(new Shape(1, 2), NPTypeCode.Int64);
            np.nanargmax(a, 0, okd, keepdims: true);
            okd.GetValue<long>(0, 0).Should().Be(1L);
        }

        [TestMethod]
        public void NanArgMax_Out_StridedView_WritesThroughBase()
        {
            // a strided out view is legal and writes through to its base (probed: base [1 0 0 0])
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var baseArr = np.zeros(new Shape(4), NPTypeCode.Int64);
            var vo = baseArr["::2"];
            var r = np.nanargmax(a, 0, vo);
            ReferenceEquals(r, vo).Should().BeTrue();
            baseArr.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0, 0, 0 }, x => x.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_Out_AllNaN_RaisesBeforeWriting()
        {
            // the all-NaN guard fires BEFORE out is touched (probed: out keeps its prior contents)
            var allnan = np.array(new double[,] { { double.NaN, double.NaN }, { double.NaN, double.NaN } });
            var ox = np.full(new Shape(2), 77L);
            ((Action)(() => np.nanargmax(allnan, 0, ox)))
                .Should().Throw<ValueError>().WithMessage("All-NaN slice encountered*");
            ox.ToArray<long>().Should().BeEquivalentTo(new long[] { 77, 77 }, x => x.WithStrictOrdering());
        }

        [TestMethod]
        public void NanArgMax_Out_Flat0d()
        {
            // axis=None with a 0-d out: np.nanargmax(a, out=o0) == array(1)
            var a = np.array(new double[,] { { double.NaN, 4.0 }, { 2.0, 3.0 } });
            var o0 = np.zeros(new Shape(), NPTypeCode.Int64);
            var r = np.nanargmax(a, null, o0);
            ((long)r).Should().Be(1L);
        }
    }

    /// <summary>
    /// Pins the two engine bugs the nanargmax dtype sweep exposed in np.argmax/np.argmin itself:
    /// the FLAT Decimal path rode the IL arg kernel whose Bgt/Blt cannot compare a 16-byte struct
    /// (argmax([3,9,1,5]) silently returned 0 — the worst finding class), and the FLAT Char path
    /// threw NotSupportedException while the AXIS path supported char all along. Decimal now takes
    /// the same ExecuteReducing fallback family as Half/Complex; Char rides the zero-copy uint16
    /// reinterpret its amax/amin already use. The AXIS paths were always correct (plain C# generic
    /// compares in AxisArgReductionHelper) and stay on the typed kernels.
    /// </summary>
    [TestClass]
    public class ArgMaxArgMinDecimalCharFixTests
    {
        [TestMethod]
        public void ArgMax_Decimal_Flat()
        {
            np.argmax(np.array(new decimal[] { 3, 9, 1, 5 })).Should().Be(1L);
            np.argmin(np.array(new decimal[] { 3, 9, 1, 5 })).Should().Be(2L);
            // first-occurrence tiebreak (NumPy contract)
            np.argmax(np.array(new decimal[] { 5, 9, 9, 1 })).Should().Be(1L);
        }

        [TestMethod]
        public void ArgMax_Decimal_Axis_TypedKernelStaysCorrect()
        {
            var m = np.array(new decimal[,] { { 3, 9 }, { 7, 5 } });
            np.argmax(m, 0).ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, o => o.WithStrictOrdering());
            np.argmin(m, 1).ToArray<long>().Should().BeEquivalentTo(new long[] { 0, 1 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void ArgMax_Decimal_StridedView()
        {
            var v = np.array(new decimal[] { 9, 1, 8, 2, 7, 3 })["::2"];   // [9, 8, 7]
            np.argmax(v).Should().Be(0L);
            np.argmin(v).Should().Be(2L);
        }

        [TestMethod]
        public void ArgMax_Char_Flat_NoLongerThrows()
        {
            var c = np.array(new[] { 'c', 'z', 'a' });
            np.argmax(c).Should().Be(1L);
            np.argmin(c).Should().Be(2L);
        }

        [TestMethod]
        public void ArgMax_Char_Axis()
        {
            var m = np.array(new[,] { { 'c', 'z' }, { 'd', 'a' } });
            np.argmax(m, 0).ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void ArgMax_NonContiguous_FirstNaNWins()
        {
            // The strided/scalar IL step compared with Bgt_Un/Blt_Un — the UNORDERED branch also
            // fires when the ACCUM holds NaN, so every later finite value stole the index and
            // argmax(strided [nan,2,3,nan]) walked to 3 instead of NumPy's 0 (first NaN wins).
            // Contiguous inputs were immune (they ride the C# NaN helpers). Fixed to ordered
            // Bgt/Blt; caught by the reduce tier's new flat argmax cells over NaN pools (G13).
            var b = np.array(new double[,] { { double.NaN, 3 }, { 2, double.NaN } });
            np.argmax(b.T).Should().Be(0L);   // b.T C-order = [nan, 2, 3, nan]
            np.argmin(b.T).Should().Be(0L);
            var f = np.array(new float[,] { { float.NaN, 3f }, { 2f, float.NaN } });
            np.argmax(f.T).Should().Be(0L);
            // reversed stride: view [3, nan, 1, 7, nan, 5] → first NaN at 1
            var c = np.array(new double[] { 5, double.NaN, 7, 1, double.NaN, 3 });
            np.argmax(c["::-1"]).Should().Be(1L);
            // finite non-contiguous unchanged
            var a = np.array(new double[,] { { 1, 9 }, { 2, 4 } });
            np.argmax(a.T).Should().Be(2L);
            np.argmin(a.T).Should().Be(0L);
        }
    }
}
