using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// NEP50 accumulator widening on the FLAT (axis=null) size-&#8804;1 reduction path
/// (<c>DefaultEngine.HandleScalarReduction</c>, reached from <c>ReduceAdd</c>/<c>ReduceProduct</c>
/// when the input is a 0-d scalar or a single-element 1-D array).
///
/// The regression this pins: that path used to <c>arr.Clone()</c> — preserving the INPUT dtype —
/// so <c>np.sum</c>/<c>np.prod</c> returned int32/uint8/bool instead of NumPy's widened
/// int64/uint64/int64. NumPy 2.4.2 widens the sum/prod accumulator to int64/uint64 for every
/// integer/bool dtype REGARDLESS of size (floats/complex preserved) — verified at n&#8805;2 and at
/// the degenerate n&#8804;1 boundary alike.
///
/// The helper is SHARED with amin/amax (which must PRESERVE dtype), so the widening lives at the
/// sum/prod call sites, not in the helper — the amin/amax guard tests below pin that split.
/// All expected values/dtypes are from NumPy 2.4.2.
/// </summary>
[TestClass]
public class ScalarReductionWideningTests
{
    #region np.sum — size-1 1-D widens (the bug)

    [TestMethod]
    public void Sum_Int32_SizeOne_WidensToInt64()
    {
        // NumPy: np.sum(np.array([5], dtype=np.int32)) -> np.int64(5), dtype int64, shape ()
        var s = np.sum(np.array(new[] { 5 }));

        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode, "dtype");
        Assert.AreEqual(0, s.ndim, "flat sum of a 1-D array is a 0-d scalar");
        Assert.AreEqual(5L, s.GetInt64(0), "value");
    }

    [TestMethod]
    public void Sum_Int16_SizeOne_WidensToInt64()
    {
        var s = np.sum(np.array(new short[] { 5 }));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(5L, s.GetInt64(0));
    }

    [TestMethod]
    public void Sum_UInt8_SizeOne_WidensToUInt64()
    {
        var s = np.sum(np.array(new byte[] { 5 }));
        Assert.AreEqual(NPTypeCode.UInt64, s.GetTypeCode);
        Assert.AreEqual(5UL, s.GetUInt64(0));
    }

    [TestMethod]
    public void Sum_Bool_SizeOne_WidensToInt64()
    {
        // NumPy: np.sum(np.array([True])) -> np.int64(1), dtype int64
        var s = np.sum(np.array(new[] { true }));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(1L, s.GetInt64(0));
    }

    [TestMethod]
    public void Sum_Float32_SizeOne_PreservesFloat32()
    {
        // Floats are their own accumulating type — preserved, not widened.
        var s = np.sum(np.array(new[] { 5f }));
        Assert.AreEqual(NPTypeCode.Single, s.GetTypeCode);
        Assert.AreEqual(5f, s.GetSingle(0));
    }

    [TestMethod]
    public void Sum_Float64_SizeOne_PreservesFloat64()
    {
        var s = np.sum(np.array(new[] { 5.0 }));
        Assert.AreEqual(NPTypeCode.Double, s.GetTypeCode);
        Assert.AreEqual(5.0, s.GetDouble(0));
    }

    #endregion

    #region np.sum — 0-d scalar widens (the bug)

    [TestMethod]
    public void Sum_Int32_Scalar_WidensToInt64()
    {
        // NumPy: np.sum(np.array(5, dtype=np.int32)) -> np.int64(5)
        var s = np.sum(NDArray.Scalar(5));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(0, s.ndim);
        Assert.AreEqual(5L, s.GetInt64(0));
    }

    [TestMethod]
    public void Sum_Bool_Scalar_WidensToInt64()
    {
        var s = np.sum(NDArray.Scalar(true));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(1L, s.GetInt64(0));
    }

    #endregion

    #region np.prod — size-1 / scalar widens (the bug), mirrors sum

    [TestMethod]
    public void Prod_Int32_SizeOne_WidensToInt64()
    {
        var p = np.prod(np.array(new[] { 5 }));
        Assert.AreEqual(NPTypeCode.Int64, p.GetTypeCode);
        Assert.AreEqual(5L, p.GetInt64(0));
    }

    [TestMethod]
    public void Prod_UInt16_SizeOne_WidensToUInt64()
    {
        var p = np.prod(np.array(new ushort[] { 7 }));
        Assert.AreEqual(NPTypeCode.UInt64, p.GetTypeCode);
        Assert.AreEqual(7UL, p.GetUInt64(0));
    }

    [TestMethod]
    public void Prod_Bool_Scalar_WidensToInt64()
    {
        var p = np.prod(NDArray.Scalar(true));
        Assert.AreEqual(NPTypeCode.Int64, p.GetTypeCode);
        Assert.AreEqual(1L, p.GetInt64(0));
    }

    [TestMethod]
    public void Prod_Float32_SizeOne_PreservesFloat32()
    {
        var p = np.prod(np.array(new[] { 5f }));
        Assert.AreEqual(NPTypeCode.Single, p.GetTypeCode);
        Assert.AreEqual(5f, p.GetSingle(0));
    }

    #endregion

    #region keepdims + explicit dtype= still honored

    [TestMethod]
    public void Sum_Int32_SizeOne_Keepdims_WidensAndKeepsShape()
    {
        // NumPy: np.sum(np.array([5], np.int32), keepdims=True) -> array([5]) int64, shape (1,)
        var s = np.sum(np.array(new[] { 5 }), keepdims: true);
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        s.Should().BeShaped(1);
        Assert.AreEqual(5L, s.GetInt64(0));
    }

    [TestMethod]
    public void Sum_Int32_SizeOne_ExplicitDtype_Honored()
    {
        // An explicit dtype= must still win over the default accumulator.
        // NumPy: np.sum(np.array([5], np.int32), dtype=np.int16) -> np.int16(5)
        var s = np.sum(np.array(new[] { 5 }), dtype: np.int16);
        Assert.AreEqual(NPTypeCode.Int16, s.GetTypeCode);
        // GetInt64 reinterprets raw bytes on a non-int64 dtype — read the boxed value and convert.
        Assert.AreEqual(5L, Convert.ToInt64(s.GetValue(0)));
    }

    #endregion

    #region guards — the boundary and the sibling reductions must be unaffected

    [TestMethod]
    public void Sum_Int32_SizeTwo_StillWidens()
    {
        // The n>=2 path (HandleElementWiseSum) already widened — pin it so a fix to the
        // size-1 path cannot regress it.
        var s = np.sum(np.array(new[] { 1, 2 }));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(3L, s.GetInt64(0));
    }

    [TestMethod]
    public void Sum_Int32_SizeOne_2D_StillWidens()
    {
        // A size-1 (1,1) array is 2-D, so it never hit the buggy scalar branch — pin the contrast.
        var s = np.sum(np.array(new[] { 5 }).reshape(1, 1));
        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        Assert.AreEqual(5L, s.GetInt64(0));
    }

    [TestMethod]
    public void Min_Int32_SizeOne_PreservesDtype()
    {
        // amin/amax share HandleScalarReduction but must PRESERVE dtype (no accumulator).
        // NumPy: np.min(np.array([5], np.int32)) -> np.int32(5)
        var m = np.amin(np.array(new[] { 5 }));
        Assert.AreEqual(NPTypeCode.Int32, m.GetTypeCode, "min must NOT widen");
        Assert.AreEqual(5, m.GetInt32(0));
    }

    [TestMethod]
    public void Max_UInt8_SizeOne_PreservesDtype()
    {
        var m = np.amax(np.array(new byte[] { 5 }));
        Assert.AreEqual(NPTypeCode.Byte, m.GetTypeCode, "max must NOT widen");
        Assert.AreEqual((byte)5, m.GetByte(0));
    }

    [TestMethod]
    public void Mean_Int32_SizeOne_IsFloat64()
    {
        // mean has its own path (always float64) — pin it as unaffected.
        var m = np.mean(np.array(new[] { 5 }));
        Assert.AreEqual(NPTypeCode.Double, m.GetTypeCode);
        Assert.AreEqual(5.0, m.GetDouble(0), 1e-12);
    }

    [TestMethod]
    public void CumSum_Int32_SizeOne_WidensToInt64()
    {
        // cumsum already widens at n=1 (ReduceCumAdd uses GetAccumulatingType) — guard.
        var c = np.cumsum(np.array(new[] { 5 }));
        Assert.AreEqual(NPTypeCode.Int64, c.GetTypeCode);
        Assert.AreEqual(5L, c.GetInt64(0));
    }

    [TestMethod]
    public void CumProd_Int32_SizeOne_WidensToInt64()
    {
        // cumprod's size-1 widening was fixed earlier (ReduceCumMul) — guard the sibling.
        var c = np.cumprod(np.array(new[] { 5 }));
        Assert.AreEqual(NPTypeCode.Int64, c.GetTypeCode);
        Assert.AreEqual(5L, c.GetInt64(0));
    }

    #endregion

    #region the exposing case — apply_along_axis(np.sum) over a length-1 slice

    [TestMethod]
    public void ApplyAlongAxis_Sum_LengthOneSlice_WidensToInt64()
    {
        // shape (2,1): each 1-D slice along axis=1 is length 1, so np.sum returns the widened
        // dtype and apply_along_axis propagates it. NumPy: dtype int64, values [3, 7].
        var a2d = np.array(new[] { 3, 7 }).reshape(2, 1);
        var aa = np.apply_along_axis(x => np.sum(x), 1, a2d);

        Assert.AreEqual(NPTypeCode.Int64, aa.GetTypeCode, "apply_along_axis must inherit the widened sum dtype");
        aa.Should().BeShaped(2);
        Assert.AreEqual(3L, aa.GetInt64(0));
        Assert.AreEqual(7L, aa.GetInt64(1));
    }

    #endregion
}
