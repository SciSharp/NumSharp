using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Reflection;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// Audit V2 Tier 1 correctness bugs for NDArray core + Creation APIs.
/// Each test reproduces a verified divergence between NumSharp and NumPy 2.4.2
/// or an internal correctness defect. Tests are kept FAILING via
/// <see cref="OpenBugsAttribute"/> so they document the bug and start passing
/// once it is fixed.
///
/// FALSE POSITIVES / ALREADY FIXED entries appear as comment blocks (no test).
/// </summary>
[TestClass]
public class AuditV2_NDArrayCreation
{
    // -----------------------------------------------------------------------
    // T1.7 (FIXED) — np.array(NDArray) default flipped from copy=false to
    //   copy=true to match NumPy 2.x. Calling `np.array(a)` now returns an
    //   independent copy. Pair with the new np.asarray's tristate `copy`
    //   parameter for explicit "copy if needed" semantics.
    //   File: src/NumSharp.Core/Creation/np.array.cs:18-29
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_7_NpArray_NDArrayInput_DefaultAliases()
    {
        var a = np.arange(10); // int64 in NumSharp
        var b = np.array(a);
        b.SetAtIndex<long>(999L, 0);

        a.GetAtIndex<long>(0).Should().Be(0L,
            "NumPy default copy=True; np.array(a) must NOT alias the source");
    }

    // -----------------------------------------------------------------------
    // T1.8 (FIXED) — np.concatenate now routes through np.result_type for
    //   NEP50-compliant promotion. concatenate([float32, int64]) returns
    //   float64 matching NumPy 2.x.
    //   File: src/NumSharp.Core/Creation/np.concatenate.cs
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_8_Concatenate_F32_I64_PromotesToFloat32_Not_Float64()
    {
        var f32 = np.array(new float[] { 1f, 2f, 3f });
        var i64 = np.array(new long[] { 4L, 5L, 6L });
        var r = np.concatenate(new[] { f32, i64 });

        // NumPy: np.concatenate([f4, i8]).dtype == 'float64'.
        // Per np._FindCommonArrayType(Single, Int64) the NumSharp table
        // *itself* says Double — but np.concatenate doesn't call it.
        r.typecode.Should().Be(NPTypeCode.Double,
            "NEP50: float32 + int64 -> float64; concatenate must use _FindCommonType, " +
            "not NPTypeCode.CompareTo group/size ordering");
    }

    // -----------------------------------------------------------------------
    // T1.9 — np.concatenate crashes on mixed SByte/Half/Complex inputs.
    //   File: src/NumSharp.Core/Creation/np.concatenate.cs:108
    //          src/NumSharp.Core/Backends/Iterators/NDIterCasting.cs (root)
    //   When two arrays of *different* dtypes are concatenated and one is
    //   SByte / Half / Complex, NDIter.Copy routes through
    //   CopyStridedToStridedWithCast which throws
    //   NotSupportedException("Unsupported type: ..."). The branch added
    //   SByte/Half to NPTypeCode but the NDIter cast read/write paths
    //   never gained corresponding entries.
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_9_Concatenate_Mixed_SByte_Byte_ThrowsNotSupported()
    {
        var s8 = np.array(new sbyte[] { 1, 2 });
        var u8 = np.array(new byte[] { 3, 4 });
        Action act = () => np.concatenate(new[] { s8, u8 });
        act.Should().NotThrow<NotSupportedException>(
            "concatenating int8 + uint8 must not crash — NumPy promotes to int16");
    }

    [TestMethod]
    public void T1_9_Concatenate_Mixed_Half_Single_ThrowsNotSupported()
    {
        var h = np.array(new Half[] { (Half)1f, (Half)2f });
        var f = np.array(new float[] { 3f, 4f });
        Action act = () => np.concatenate(new[] { h, f });
        act.Should().NotThrow<NotSupportedException>(
            "concatenating float16 + float32 must not crash — NumPy promotes to float32");
    }

    [TestMethod]
    public void T1_9_Concatenate_Mixed_Complex_Double_ThrowsNotSupported()
    {
        var c = np.array(new System.Numerics.Complex[] { new(1, 0), new(2, 0) });
        var d = np.array(new double[] { 3.0, 4.0 });
        Action act = () => np.concatenate(new[] { c, d });
        act.Should().NotThrow<NotSupportedException>(
            "concatenating complex128 + float64 must not crash — NumPy promotes to complex128");
    }

    // -----------------------------------------------------------------------
    // T1.16 / T1.43 — DefaultEngine.Cast(nd, dtype, copy=false) mutates
    //   the caller's NDArray.Storage (and reassigns TensorEngine to itself).
    //   File: src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Cast.cs
    //         (lines 22-25, 36-37, 48-50, 72-74)
    //
    //   `copy=false` should mean "in-place if possible". Instead the engine
    //   creates a brand-new UnmanagedStorage and ASSIGNS it back over
    //   nd.Storage — every existing reference to that NDArray now observes
    //   a different dtype, with the old storage detached. This is the same
    //   issue reported as both T1.16 and T1.43; the latter additionally
    //   flagged the redundant `nd.TensorEngine = engine` reassignment which
    //   is a no-op aliasing (engine is captured from nd.TensorEngine at
    //   line 15).
    //
    //   NumPy: `arr.astype(dtype, copy=False)` returns a new view-or-array
    //   *without* mutating arr itself.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.16")]
    public void T1_16_Cast_CopyFalse_MutatesCallerStorage()
    {
        var orig = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
        var origStorage = orig.Storage;
        var origDtype = orig.dtype;

        var result = orig.TensorEngine.Cast(orig, NPTypeCode.Double, copy: false);

        // The bug: result is `orig` itself, and `orig.Storage` was replaced.
        ReferenceEquals(orig, result).Should().BeFalse(
            "Cast(copy=false) must NOT return the same NDArray reference (caller mutation)");
        ReferenceEquals(origStorage, orig.Storage).Should().BeTrue(
            "Cast(copy=false) must NOT replace caller's Storage field");
        orig.dtype.Should().Be(origDtype,
            "Cast(copy=false) must NOT change the dtype on the input NDArray");
    }

    // T1.43 is the same defect as T1.16 — Default.Cast.cs reassigns BOTH
    // nd.Storage AND nd.TensorEngine when copy=false. The engine reassignment
    // is a no-op aliasing today (engine = nd.TensorEngine captured at line 15,
    // then written back to nd.TensorEngine), but it documents the intent: the
    // cast path mutates the caller in-place rather than returning a clean
    // value. Tracking the bug fix through T1.16 — no separate test needed.
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.43")]
    public void T1_43_Cast_CopyFalse_EmptyArray_AlsoMutatesCaller()
    {
        // The empty-array branch (Default.Cast.cs:18-26) has the same defect
        // as the regular branch: when copy=false on an empty input, Storage
        // and TensorEngine are reassigned on the caller's NDArray. This is
        // the original line 23 the audit referenced.
        var orig = new NDArray(NPTypeCode.Int32);  // empty
        orig.Shape.IsEmpty.Should().BeTrue("sanity");

        var origStorage = orig.Storage;
        var result = orig.TensorEngine.Cast(orig, NPTypeCode.Double, copy: false);

        ReferenceEquals(orig, result).Should().BeFalse(
            "Cast(empty, copy=false) should not return the same instance with mutated storage");
        ReferenceEquals(origStorage, orig.Storage).Should().BeTrue(
            "Cast(empty, copy=false) must not replace caller's Storage even for empty arrays");
    }

    // -----------------------------------------------------------------------
    // T1.44 — `new NDArray(Array values, Shape shape, char order)` silently
    //   ignores the `order` parameter.
    //   File: src/NumSharp.Core/Backends/NDArray.cs:178-186, :197-205, :224
    //
    //   The constructor accepts an `order` parameter ('C' / 'F') but the
    //   body has a comment "F-order not supported, order parameter is
    //   accepted but ignored (C-order only)". There is no OrderResolver
    //   dispatch; the storage is laid out C-order regardless of what the
    //   caller requested. Users passing `order='F'` get C-order storage
    //   with no warning.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.44")]
    public void T1_44_NDArrayCtor_FOrderArg_SilentlyIgnored()
    {
        // Given 1..6 in row-major slot 0..5, C-order (2,3) yields
        //   [[1,2,3],
        //    [4,5,6]]
        // F-order (2,3) would yield (NumPy reference):
        //   [[1,3,5],
        //    [2,4,6]]
        var arr = new int[] { 1, 2, 3, 4, 5, 6 };
        var ndF = new NDArray(arr, new Shape(2, 3), 'F');

        ndF.Shape.IsFContiguous.Should().BeTrue(
            "ctor 'F' must produce an F-contiguous shape; today the order arg is dropped");
        ndF.GetInt32(0, 1).Should().Be(3,
            "F-order (2,3) of [1..6]: element [0,1] is 3, not 2 (C-order)");
    }

    // -----------------------------------------------------------------------
    // T1.45 — FALSE POSITIVE / matches NumPy.
    //
    //   OrderResolver.cs:54-66 returns 'C' when called with order='K' on a
    //   source that is NEITHER C- nor F-contiguous. The audit master
    //   document called this a bug ("should preserve order"). Verified
    //   against NumPy 2.4.2: `np.copy(a[::2,::2], order='K')` returns a
    //   C-contiguous result for any non-contiguous strided input. The
    //   "conservative fallback" comment in OrderResolver is correct.
    //
    //   No test added.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // T1.46 — find_common_type / arithmetic with uint8 array + scalar 1000
    //   produces silent overflow.
    //   File: src/NumSharp.Core/Logic/np.find_common_type.cs (table) plus
    //         arithmetic kernels.
    //
    //   NEP50 type promotion correctly says (uint8 array, Python int) ->
    //   uint8 — NumSharp matches that part. But the ACTUAL add operation
    //   in NumSharp wraps silently (e.g., 1 + 1000 -> 233 mod 256), where
    //   NumPy 2.4.2 raises `OverflowError: Python integer 1000 out of
    //   bounds for uint8` before the kernel even runs.
    //
    //   The audit grouped both observations under T1.46; the arithmetic
    //   path is the real defect.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.46")]
    public void T1_46_FindCommonType_UInt8Array_PlusLargeInt_SilentOverflow()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });

        // Type promotion *resolves* to uint8 (matches NumPy NEP50 type rule).
        np.find_common_type(new[] { typeof(byte) }, new[] { typeof(int) })
            .Should().Be(NPTypeCode.Byte,
                "NEP50: uint8 array + Python int -> uint8 dtype is correct");

        // But the actual `arr + 1000` must raise OverflowError, not wrap to 233.
        Action act = () => { var _ = arr + 1000; };
        act.Should().Throw<OverflowException>(
            "NumPy raises OverflowError when adding a Python int that doesn't fit in uint8; " +
            "NumSharp wraps silently to 233/234/235");
    }

    // -----------------------------------------------------------------------
    // T1.47 — `_can_coerce_all(NPTypeCode[] dtypelist, int start)` has a
    //   wrong-destIndex Array.Copy.
    //   File: src/NumSharp.Core/Logic/np.find_common_type.cs:1065
    //
    //       Array.Copy(dtypelist, start, sub, len, len);
    //                                      ^^^^^^ destIndex
    //
    //   This passes `len` as destIndex AND `len` as count. The destIndex
    //   must be 0 (writing the prefix of `sub`). When called with start>0
    //   this throws `ArgumentException: Destination array was not long
    //   enough.` Currently dead-code latent — only hit from
    //   _find_common_coerce which itself is gated on a kind-index mismatch
    //   that hasn't been exercised by callers yet.
    //
    //   The List<NPTypeCode> sibling at :1097-1098 has a different but
    //   equally-broken pattern (assigning into an uninitialized List slot).
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.47")]
    public void T1_47_CanCoerceAll_Array_StartIndex_ThrowsArgumentException()
    {
        var method = typeof(np).GetMethod(
            "_can_coerce_all",
            BindingFlags.NonPublic | BindingFlags.Static,
            null,
            new[] { typeof(NPTypeCode[]), typeof(int) },
            null);
        method.Should().NotBeNull("internal _can_coerce_all(NPTypeCode[], int) overload");

        // Calling with start>0 must NOT throw ArgumentException — the bug
        // produces "Destination array was not long enough" because
        // Array.Copy is called with destIndex=len instead of destIndex=0.
        // When fixed, the method returns the coerced common type.
        Action act = () => method!.Invoke(
            null,
            new object[]
            {
                new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double },
                1
            });

        act.Should().NotThrow(
            "_can_coerce_all(arr, start>0) must not throw ArgumentException via wrong destIndex; " +
            "np.find_common_type.cs:1065 has Array.Copy(src, start, dst, len, len) — dst offset should be 0");
    }

    // -----------------------------------------------------------------------
    // T1.48 — np.ascontiguousarray(0-D) / np.asfortranarray(0-D) returns
    //   ndim=0; NumPy promotes to ndim=1.
    //   Files: src/NumSharp.Core/Creation/np.ascontiguousarray.cs (new file)
    //          src/NumSharp.Core/Creation/np.asfortranarray.cs   (new file)
    //
    //   Both wrappers delegate to `np.asarray(a, dtype, 'C'/'F')` and
    //   asarray returns the input unchanged when dtype+layout match. For
    //   a 0-D scalar, that means ndim stays 0.
    //
    //   NumPy contract:
    //     >>> np.ascontiguousarray(np.array(42)).shape
    //     (1,)
    //
    //   The NumSharp docstring even claims "Return a contiguous array
    //   (ndim >= 1)", but the implementation does not promote.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.48")]
    public void T1_48_AsContiguousArray_Scalar_DoesNotPromoteTo_1D()
    {
        var scalar = NDArray.Scalar(42);
        scalar.ndim.Should().Be(0, "sanity check that input is 0-D");

        var r = np.ascontiguousarray(scalar);
        r.ndim.Should().Be(1,
            "NumPy: np.ascontiguousarray(0-D).ndim == 1; ndim>=1 is the function contract");
        r.shape.Should().Equal(new long[] { 1L });
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.48")]
    public void T1_48_AsFortranArray_Scalar_DoesNotPromoteTo_1D()
    {
        var scalar = NDArray.Scalar(42);
        var r = np.asfortranarray(scalar);
        r.ndim.Should().Be(1,
            "NumPy: np.asfortranarray(0-D).ndim == 1; ndim>=1 is the function contract");
        r.shape.Should().Equal(new long[] { 1L });
    }

    // -----------------------------------------------------------------------
    // T1.50 — np.arange(0, 5, 1, Boolean) returns alternating bool[];
    //   NumPy raises TypeError for length > 2.
    //   File: src/NumSharp.Core/Creation/np.arange.cs:81-90
    //
    //   NumPy raises:
    //     TypeError: arange() is only supported for booleans when the
    //     result has at most length 2.
    //
    //   NumSharp instead produces `[False, True, False, True, False]`
    //   by alternating between `start_t` and `next_t` — a NumSharp-specific
    //   extension that silently swallows an obvious user error.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.50")]
    public void T1_50_Arange_BoolDtype_LengthOver2_DoesNotRaise()
    {
        Action act = () => np.arange(0, 5, 1, NPTypeCode.Boolean);
        act.Should().Throw<Exception>(
            "NumPy: arange(..., dtype=bool) is only supported for length <= 2; raises TypeError");
    }

    [TestMethod]
    public void T1_50_Arange_BoolDtype_LengthAtMost2_Allowed()
    {
        // The audit explicitly notes len=2 should still work. Sanity guard
        // so the eventual fix does not over-restrict.
        Action act = () => np.arange(0, 2, 1, NPTypeCode.Boolean);
        act.Should().NotThrow("arange(0, 2, 1, bool) must keep working — length 2 is legal");
    }

    // -----------------------------------------------------------------------
    // T1.54 — np.frombuffer('F' / 'c8' / 'complex64') silently widens to
    //   complex128.
    //   File: src/NumSharp.Core/Creation/np.frombuffer.cs:720
    //         (ParseDtypeString)
    //
    //   The comment at line 717-719 acknowledges: "NumSharp only ships
    //   complex128. 'c8'/'F' (single-precision complex) map to complex128
    //   rather than throwing so the round-trip still works on the common
    //   path; the storage widens but values are exact."
    //
    //   This is inconsistent with np.dtype.cs which explicitly REJECTS
    //   'F' / 'c8' / 'complex64' via _unsupported_numpy_codes. The two
    //   surfaces disagree.
    //
    //   Worse: feeding genuine complex64 bytes (8 bytes per element) to
    //   frombuffer('c8') hits the "buffer size must be a multiple of
    //   element size" check (16 bytes), because the reader expects
    //   complex128.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.54")]
    public void T1_54_FromBuffer_c8_AlignsWith_npDtype_AndRejects()
    {
        // 16 bytes — accepted as Complex128 today. Should be rejected (c8 = complex64).
        var bytes = new byte[16];

        // np.dtype("c8") raises NotSupportedException — keep that contract.
        // np.frombuffer should not silently widen it to Complex128.
        Action c8 = () => np.frombuffer(bytes, "c8");
        c8.Should().Throw<NotSupportedException>(
            "'c8' (complex64) is not a NumSharp dtype; frombuffer must be consistent with np.dtype()");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.54")]
    public void T1_54_FromBuffer_F_AlignsWith_npDtype_AndRejects()
    {
        // 16 bytes — accepted as Complex128 today. Should be rejected ('F' = complex64).
        var bytes = new byte[16];

        Action f = () => np.frombuffer(bytes, "F");
        f.Should().Throw<NotSupportedException>(
            "'F' (complex64 typechar) is not a NumSharp dtype; frombuffer must be consistent with np.dtype()");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.54")]
    public void T1_54_FromBuffer_c8_RealBytes_AreSilentlyWidened()
    {
        // Build 16 bytes that, if interpreted as complex64 (2 floats), encode
        // 1.0 + 2.0i — but np.frombuffer reads them as a complex128 (2 doubles)
        // and the float bit pattern becomes garbage values.
        var floats = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var bytes = new byte[floats.Length * 4]; // 16 bytes (genuine complex64 array of length 2)
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);

        var r = np.frombuffer(bytes, "c8");

        // The current behavior silently treats the buffer as complex128 (1 element).
        // After the fix, this call should throw, OR if the API decides to support
        // c8, it should produce a 2-element complex128 array with values 1+2i and 3+4i.
        r.size.Should().Be(2,
            "c8 buffer has 2 complex64 elements; frombuffer must not silently halve to 1 complex128");
    }

    // -----------------------------------------------------------------------
    // T1.56 — np.array(Array, ndmin=1, ...) default differs from NumPy
    //   default ndmin=0.
    //   File: src/NumSharp.Core/Creation/np.array.cs:51
    //
    //   `public static NDArray array(Array array, Type dtype=null,
    //                                int ndmin=1, bool copy=true,
    //                                char order='C')`
    //
    //   NumPy signature: `np.array(object, ..., ndmin=0, ...)`.
    //
    //   Operationally: System.Array is rank>=1, so for typical inputs
    //   like `new int[]{1,2,3}` the difference is invisible. The mismatch
    //   leaks when callers explicitly compare API surface or migrate code
    //   from Python.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.56")]
    public void T1_56_NpArray_ArrayInput_DefaultNdmin_MatchesNumPy()
    {
        // Pull the (Array, Type, int, bool, char) overload and inspect the
        // `ndmin` default through reflection.
        var mi = typeof(np).GetMethod(
            nameof(np.array),
            BindingFlags.Public | BindingFlags.Static,
            null,
            new[] { typeof(Array), typeof(Type), typeof(int), typeof(bool), typeof(char) },
            null);
        mi.Should().NotBeNull("np.array(Array, Type, int, bool, char) overload");

        var ndminParam = mi!.GetParameters()[2];
        ndminParam.Name.Should().Be("ndmin");
        ndminParam.HasDefaultValue.Should().BeTrue();
        ndminParam.DefaultValue.Should().Be(0,
            "NumPy 2.x np.array default ndmin=0; NumSharp currently defaults to 1");
    }
}
