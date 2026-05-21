using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation;

/// <summary>
/// Battle tests for np.asarray. Verified 1-to-1 against numpy 2.4.2 — covers the
/// tristate `copy=None/True/False` keyword, dtype-as-string, dtype-as-DType,
/// dtype-as-NPTypeCode, the `like` and `device` parameters, plus the `order=K/A/C/F`
/// no-copy contract.
/// </summary>
[TestClass]
public class np_asarray_BattleTests
{
    // ─── copy parameter (tri-state) ─────────────────────────────────────

    [TestMethod]
    public void Asarray_DefaultCopy_ReturnsSameStorage()
    {
        // NumPy: np.asarray(a) is a (no copy). copy=None (default) reuses storage.
        var a = np.arange(6);
        var b = np.asarray(a);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_CopyTrue_AllocatesNewStorage()
    {
        // NumPy: np.asarray(a, copy=True) — always copies.
        var a = np.arange(6);
        var b = np.asarray(a, copy: true);
        ReferenceEquals(a.Storage, b.Storage).Should().BeFalse();
        b.SetAtIndex<long>(999L, 0);
        a.GetAtIndex<long>(0).Should().Be(0L, "copy=True must not alias the source");
    }

    [TestMethod]
    public void Asarray_CopyFalse_NoCopyNeeded_ReturnsSameStorage()
    {
        // NumPy: np.asarray(a, copy=False) on already-OK input is a no-op.
        var a = np.arange(6);
        var b = np.asarray(a, copy: false);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_CopyFalse_DtypeMismatch_RaisesValueError()
    {
        // NumPy: np.asarray(int_arr, dtype=float64, copy=False) raises ValueError.
        var a = np.arange(6);
        Action act = () => np.asarray(a, dtype: typeof(double), copy: false);
        act.Should().Throw<ArgumentException>(
            "dtype change requires a copy — copy=False must throw");
    }

    [TestMethod]
    public void Asarray_CopyFalse_LayoutMismatch_RaisesValueError()
    {
        // NumPy: np.asarray(c_contig_2d, order='F', copy=False) raises ValueError.
        var a = np.arange(12).reshape(3, 4);
        Action act = () => np.asarray(a, order: 'F', copy: false);
        act.Should().Throw<ArgumentException>(
            "layout change requires a copy — copy=False must throw");
    }

    // ─── dtype overloads ────────────────────────────────────────────────

    [TestMethod]
    public void Asarray_DtypeString_Float32_CastsAndAllocates()
    {
        var a = np.arange(6);
        var b = np.asarray(a, dtype: "float32");
        b.dtype.Should().Be(typeof(float));
        ReferenceEquals(a.Storage, b.Storage).Should().BeFalse();
    }

    [TestMethod]
    public void Asarray_DtypeString_ByteOrderPrefix_StripsAndCasts()
    {
        // NumPy accepts '<i8' (little-endian int64). NumSharp is host-endian so
        // the prefix is stripped via np.dtype(string).
        var a = np.arange(6, dtype: NPTypeCode.Int32);
        var b = np.asarray(a, dtype: "<i8");
        b.dtype.Should().Be(typeof(long));
    }

    [TestMethod]
    public void Asarray_DtypeString_Invalid_ThrowsNotSupported()
    {
        var a = np.arange(6);
        Action act = () => np.asarray(a, dtype: "xyz");
        act.Should().Throw<NotSupportedException>();
    }

    [TestMethod]
    public void Asarray_NPTypeCode_Casts()
    {
        var a = np.arange(6);
        var b = np.asarray(a, dtype: NPTypeCode.Single);
        b.dtype.Should().Be(typeof(float));
    }

    [TestMethod]
    public void Asarray_DType_Casts()
    {
        var a = np.arange(6);
        var b = np.asarray(a, dtype: np.dtype("float64"));
        b.dtype.Should().Be(typeof(double));
    }

    [TestMethod]
    public void Asarray_NPTypeCode_Empty_ReturnsSame()
    {
        // NPTypeCode.Empty is the "no dtype" sentinel.
        var a = np.arange(6);
        var b = np.asarray(a, dtype: NPTypeCode.Empty);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_DType_Null_ReturnsSame()
    {
        var a = np.arange(6);
        var b = np.asarray(a, dtype: (DType)null);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_DtypeString_Null_ReturnsSame()
    {
        var a = np.arange(6);
        var b = np.asarray(a, dtype: (string)null);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    // ─── order parameter ────────────────────────────────────────────────

    [TestMethod]
    public void Asarray_OrderC_OnCContig_ReturnsSame()
    {
        var a = np.arange(12).reshape(3, 4);
        var b = np.asarray(a, order: 'C');
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_OrderF_OnCContig2D_AllocatesFContig()
    {
        var a = np.arange(12).reshape(3, 4);
        var b = np.asarray(a, order: 'F');
        b.Shape.IsFContiguous.Should().BeTrue();
        ReferenceEquals(a.Storage, b.Storage).Should().BeFalse();
    }

    [TestMethod]
    public void Asarray_OrderF_On1D_ReturnsSame()
    {
        // 1-D arrays are both C and F contiguous.
        var a = np.arange(6);
        var b = np.asarray(a, order: 'F');
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_OrderK_OnStridedView_ReturnsSame()
    {
        // 'K' (keep) imposes no layout constraint — no copy even for non-contig.
        var a = np.arange(12).reshape(3, 4);
        var view = a[":, ::2"];
        var b = np.asarray(view, order: 'K');
        ReferenceEquals(view.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_OrderA_OnStridedView_ReturnsSame()
    {
        // 'A' (any) imposes no layout constraint (NumPy STRIDING_OK semantics).
        var a = np.arange(12).reshape(3, 4);
        var view = a[":, ::2"];
        var b = np.asarray(view, order: 'A');
        ReferenceEquals(view.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_OrderF_OnFContig_ReturnsSame()
    {
        var a = np.asfortranarray(np.arange(12).reshape(3, 4));
        var b = np.asarray(a, order: 'F');
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_OrderInvalid_ThrowsArgumentException()
    {
        var a = np.arange(6);
        Action act = () => np.asarray(a, order: 'X');
        act.Should().Throw<ArgumentException>();
    }

    // ─── device parameter ───────────────────────────────────────────────

    [TestMethod]
    public void Asarray_DeviceCpu_Allowed()
    {
        var a = np.arange(6);
        var b = np.asarray(a, device: "cpu");
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_DeviceGpu_Throws()
    {
        var a = np.arange(6);
        Action act = () => np.asarray(a, device: "gpu");
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Asarray_DeviceNull_Allowed()
    {
        var a = np.arange(6);
        var b = np.asarray(a, device: null);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    // ─── like parameter ─────────────────────────────────────────────────

    [TestMethod]
    public void Asarray_LikeNDArray_IsNoOp()
    {
        // NumPy: like=array is for __array_function__ dispatch; for plain ndarrays
        // it has no observable effect. NumSharp accepts it for API parity.
        var a = np.arange(6);
        var like = np.arange(2);
        var b = np.asarray(a, like: like);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_LikeNull_Allowed()
    {
        var a = np.arange(6);
        var b = np.asarray(a, like: null);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    // ─── error paths ────────────────────────────────────────────────────

    [TestMethod]
    public void Asarray_NullInput_ThrowsArgumentNullException()
    {
        Action act = () => np.asarray((NDArray)null);
        act.Should().Throw<ArgumentNullException>();
    }

    // ─── edge: 0-D scalar ───────────────────────────────────────────────

    [TestMethod]
    public void Asarray_ZeroD_CopyFalse_ReturnsSame()
    {
        var s = NDArray.Scalar(5);
        var b = np.asarray(s, copy: false);
        ReferenceEquals(s.Storage, b.Storage).Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_ZeroD_CopyTrue_Allocates()
    {
        var s = NDArray.Scalar(5);
        var b = np.asarray(s, copy: true);
        ReferenceEquals(s.Storage, b.Storage).Should().BeFalse();
        ((int)b).Should().Be(5);
    }

    // ─── edge: empty array ──────────────────────────────────────────────

    [TestMethod]
    public void Asarray_Empty_DefaultCopy_ReturnsSame()
    {
        var a = np.zeros(new Shape(0, 3));
        var b = np.asarray(a);
        ReferenceEquals(a.Storage, b.Storage).Should().BeTrue();
    }

    // ─── compound: dtype + order + copy interactions ────────────────────

    [TestMethod]
    public void Asarray_DtypeAndOrder_Combined()
    {
        // Cast + F-layout in one call — both must apply.
        var a = np.arange(12).reshape(3, 4);
        var b = np.asarray(a, dtype: "float32", order: 'F');
        b.dtype.Should().Be(typeof(float));
        b.Shape.IsFContiguous.Should().BeTrue();
    }

    [TestMethod]
    public void Asarray_DtypeMatchOrderMatchCopyTrue_StillCopies()
    {
        // Even when nothing would be needed, copy=True forces it.
        var a = np.arange(6);
        var b = np.asarray(a, dtype: typeof(long), order: 'C', copy: true);
        ReferenceEquals(a.Storage, b.Storage).Should().BeFalse();
    }
}
