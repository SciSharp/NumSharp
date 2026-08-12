using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Sorting
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
    }
}
