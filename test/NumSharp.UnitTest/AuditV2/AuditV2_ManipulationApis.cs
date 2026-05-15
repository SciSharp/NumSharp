using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// Audit V2 — Group 6: Manipulation / Top-Level APIs / Logic.
///
/// Each test asserts the CORRECT NumPy 2.x behavior (verified against NumPy 2.4.2 / Python 3.12).
/// Tests are marked [OpenBugs] because they fail today; remove the attribute when the bug is fixed.
///
/// Master audit: docs/plans/NDITER_BRANCH_QUALITY_AUDIT_V2.md
/// Domain report: docs/plans/audit_v2/06_manipulation_apis_logic.md
/// </summary>
[TestClass]
public class AuditV2_ManipulationApis
{
    // =====================================================================
    // T1.17 — np.expand_dims drops new axis for empty arrays
    // =====================================================================

    /// <summary>
    /// T1.17 — np.expand_dims drops the new axis when input is empty.
    ///
    /// NumPy 2.4.2: np.expand_dims(np.array([], dtype=float), 0).shape == (1, 0)
    /// NumSharp:    returns shape (0,) — the empty short-circuit at lines 7-12 returns 'a' unchanged.
    /// File: src/NumSharp.Core/Manipulation/np.expand_dims.cs:8
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.17")]
    public void T1_17_ExpandDims_EmptyArray_AddsDimension()
    {
        var e = np.array(new double[] { });
        var expanded = np.expand_dims(e, 0);

        // NumPy: shape == (1, 0)
        expanded.shape.Should().Equal(1L, 0L);
        expanded.ndim.Should().Be(2);
        expanded.size.Should().Be(0);
    }

    /// <summary>
    /// T1.17 — also broken when expanding to (1, 1, 0) via repeated calls.
    /// NumPy: np.expand_dims(np.expand_dims(e, 0), 0).shape == (1, 1, 0).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.17")]
    public void T1_17_ExpandDims_EmptyArray_NestedAdd()
    {
        var e = np.array(new double[] { });
        var expanded = np.expand_dims(np.expand_dims(e, 0), 0);
        expanded.shape.Should().Equal(1L, 1L, 0L);
    }

    // =====================================================================
    // T1.18 — np.copyto `casting` and `where` (FIXED)
    // =====================================================================

    /// <summary>
    /// T1.18 — np.copyto truncates float→int by default; NumPy raises TypeError with default casting='same_kind'.
    ///
    /// NumPy 2.4.2:
    ///   a = np.zeros(3, dtype=np.int32)
    ///   np.copyto(a, np.array([1.5, 2.5, 3.5]))
    ///   → TypeError: Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'same_kind'
    /// NumSharp: now matches — throws InvalidCastException (.NET equivalent of TypeError).
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_FloatToInt_DefaultCasting_Throws()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new double[] { 1.5, 2.5, 3.5 });

        Action act = () => np.copyto(dst, src);
        act.Should().Throw<InvalidCastException>(
            "NumPy raises TypeError under default casting='same_kind' rule when copying float to int")
            .WithMessage("*float64*int32*same_kind*");
    }

    /// <summary>
    /// T1.18 — np.copyto missing `where=` mask argument.
    ///
    /// NumPy 2.4.2:
    ///   a = np.zeros(5, dtype=np.int32)
    ///   np.copyto(a, np.array([10,20,30,40,50]), where=np.array([T,F,T,F,T]))
    ///   → a == [10, 0, 30, 0, 50]
    /// NumSharp: now exposes `where=` and writes only at masked positions.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_WhereParameter_OnlyWritesMaskedElements()
    {
        var dst = np.zeros(new Shape(5), np.int32);
        var src = np.array(new int[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new bool[] { true, false, true, false, true });

        np.copyto(dst, src, @where: mask);

        dst.GetInt32(0).Should().Be(10);
        dst.GetInt32(1).Should().Be(0);
        dst.GetInt32(2).Should().Be(30);
        dst.GetInt32(3).Should().Be(0);
        dst.GetInt32(4).Should().Be(50);
    }

    /// <summary>
    /// T1.18 — casting='unsafe' allows float→int (NumPy: truncates silently).
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_Unsafe_FloatToInt_Truncates()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new double[] { 1.5, 2.5, 3.5 });

        np.copyto(dst, src, casting: "unsafe");

        dst.GetInt32(0).Should().Be(1);
        dst.GetInt32(1).Should().Be(2);
        dst.GetInt32(2).Should().Be(3);
    }

    /// <summary>
    /// T1.18 — casting='safe' allows widening int32→int64.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_Safe_AllowsWidening()
    {
        var dst = np.zeros(new Shape(3), np.int64);
        var src = np.array(new int[] { 1, 2, 3 });

        np.copyto(dst, src, casting: "safe");

        dst.GetInt64(0).Should().Be(1);
        dst.GetInt64(1).Should().Be(2);
        dst.GetInt64(2).Should().Be(3);
    }

    /// <summary>
    /// T1.18 — casting='safe' rejects float→int (loss of precision).
    /// NumPy: TypeError; NumSharp: InvalidCastException.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_Safe_RejectsFloatToInt()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new double[] { 1.5, 2.5, 3.5 });

        Action act = () => np.copyto(dst, src, casting: "safe");
        act.Should().Throw<InvalidCastException>()
            .WithMessage("*float64*int32*safe*");
    }

    /// <summary>
    /// T1.18 — casting='no' rejects ANY dtype mismatch — int32→int64 must error.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_No_RejectsDifferentDtype()
    {
        var dst = np.zeros(new Shape(3), np.int64);
        var src = np.array(new int[] { 1, 2, 3 });

        Action act = () => np.copyto(dst, src, casting: "no");
        act.Should().Throw<InvalidCastException>()
            .WithMessage("*int32*int64*no*");
    }

    /// <summary>
    /// T1.18 — casting='no' allows identical dtype copy.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_No_AllowsSameDtype()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new int[] { 1, 2, 3 });

        np.copyto(dst, src, casting: "no");

        dst.GetInt32(0).Should().Be(1);
        dst.GetInt32(1).Should().Be(2);
        dst.GetInt32(2).Should().Be(3);
    }

    /// <summary>
    /// T1.18 — invalid casting name raises ArgumentException (NumPy: ValueError).
    /// Verifies the full set of allowed casting names is enumerated in the message.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Casting_Invalid_Throws()
    {
        var dst = np.zeros(new Shape(3), np.float64);
        var src = np.array(new double[] { 1.0, 2.0, 3.0 });

        Action act = () => np.copyto(dst, src, casting: "foo");
        act.Should().Throw<ArgumentException>()
            .WithMessage("*'no'*'equiv'*'safe'*'same_kind'*'unsafe'*foo*");
    }

    /// <summary>
    /// T1.18 — where mask broadcasts to dst shape. A 1-D mask broadcasts across
    /// rows of a 2-D dst (NumPy semantics).
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Where_BroadcastsAcrossRows()
    {
        var dst = np.array(new int[,]
        {
            { 10, 20, 30 },
            { 40, 50, 60 },
        });
        var src = np.array(new int[] { 99, 99, 99 });
        var mask = np.array(new bool[] { true, false, true }); // broadcasts across rows

        np.copyto(dst, src, @where: mask);

        var expected = np.array(new int[,]
        {
            { 99, 20, 99 },
            { 99, 50, 99 },
        });
        np.array_equal(dst, expected).Should().BeTrue();
    }

    /// <summary>
    /// T1.18 — where=null (default) behaves like where=True: full copy.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_WhereNull_DefaultIsFullCopy()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new int[] { 7, 8, 9 });

        np.copyto(dst, src);

        dst.GetInt32(0).Should().Be(7);
        dst.GetInt32(1).Should().Be(8);
        dst.GetInt32(2).Should().Be(9);
    }

    /// <summary>
    /// T1.18 — where with a 0-d scalar True/False broadcasts to whole-array copy or no-op.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Where_ScalarFalse_IsNoOp()
    {
        var dst = np.array(new int[] { 10, 20, 30 });
        var src = np.array(new int[] { 99, 99, 99 });
        var maskFalse = np.array(false);

        np.copyto(dst, src, @where: maskFalse);

        dst.GetInt32(0).Should().Be(10);
        dst.GetInt32(1).Should().Be(20);
        dst.GetInt32(2).Should().Be(30);
    }

    /// <summary>
    /// T1.18 — where with non-boolean array raises ArgumentException.
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Where_NonBoolean_Throws()
    {
        var dst = np.zeros(new Shape(3), np.int32);
        var src = np.array(new int[] { 1, 2, 3 });
        var maskInt = np.array(new int[] { 1, 0, 1 });

        Action act = () => np.copyto(dst, src, @where: maskInt);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*where*bool*int32*");
    }

    /// <summary>
    /// T1.18 — where mask combines with cross-dtype copy (cast happens only at masked positions).
    /// </summary>
    [TestMethod]
    public void T1_18_Copyto_Where_WithCast_OnlyMaskedPositionsConverted()
    {
        var dst = np.array(new long[] { 10, 20, 30, 40, 50 });
        var src = np.array(new int[] { 99, 99, 99, 99, 99 });
        var mask = np.array(new bool[] { true, false, true, false, true });

        np.copyto(dst, src, casting: "safe", @where: mask);

        dst.GetInt64(0).Should().Be(99);
        dst.GetInt64(1).Should().Be(20);
        dst.GetInt64(2).Should().Be(99);
        dst.GetInt64(3).Should().Be(40);
        dst.GetInt64(4).Should().Be(99);
    }

    // =====================================================================
    // T1.26 — np.finfo.minexp off-by-one
    // =====================================================================

    /// <summary>
    /// T1.26 — np.finfo(float32).minexp = -125; NumPy 2.x = -126.
    ///
    /// NumPy 2.4.2: np.finfo(np.float32).minexp == -126
    /// Invariant: smallest_normal == 2^minexp → 2^-126 = 1.175494351e-38 ✓
    /// NumSharp's stored value -125 violates this invariant.
    /// File: src/NumSharp.Core/APIs/np.finfo.cs:129
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.26")]
    public void T1_26_Finfo_Float32_Minexp_Is_Minus126()
    {
        var info = np.finfo(NPTypeCode.Single);
        info.minexp.Should().Be(-126, "NumPy: 2^-126 = smallest_normal for float32");
        // Verify the invariant: smallest_normal == 2^minexp
        Math.Pow(2.0, info.minexp).Should().BeApproximately(info.smallest_normal, 1e-45);
    }

    /// <summary>
    /// T1.26 — np.finfo(float64).minexp = -1021; NumPy 2.x = -1022.
    ///
    /// NumPy 2.4.2: np.finfo(np.float64).minexp == -1022
    /// Invariant: 2^-1022 = 2.2250738585072014e-308 = smallest_normal.
    /// File: src/NumSharp.Core/APIs/np.finfo.cs:145
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.26")]
    public void T1_26_Finfo_Float64_Minexp_Is_Minus1022()
    {
        var info = np.finfo(NPTypeCode.Double);
        info.minexp.Should().Be(-1022, "NumPy: 2^-1022 = smallest_normal for float64");
        Math.Pow(2.0, info.minexp).Should().BeApproximately(info.smallest_normal, 1e-320);
    }

    // =====================================================================
    // T1.49 — np.asanyarray missing IEnumerable<sbyte/Half/Complex>
    // =====================================================================

    /// <summary>
    /// T1.49 — np.asanyarray throws NotSupportedException for IEnumerable&lt;sbyte&gt;.
    ///
    /// asanyarray supports IEnumerable&lt;T&gt; for bool/byte/short/ushort/int/uint/long/ulong/char/float/double/decimal,
    /// but is missing sbyte, Half, and Complex cases (the latter two are supported as arrays only).
    /// File: src/NumSharp.Core/Creation/np.asanyarray.cs:53-64
    /// </summary>
    [TestMethod]
    public void T1_49_Asanyarray_IEnumerable_SByte()
    {
        IEnumerable<sbyte> en = new List<sbyte> { 1, 2, 3 };
        var nd = np.asanyarray(en);
        nd.dtype.Should().Be(typeof(sbyte));
        nd.size.Should().Be(3);
    }

    /// <summary>
    /// T1.49 — np.asanyarray throws NotSupportedException for IEnumerable&lt;Half&gt;.
    /// </summary>
    [TestMethod]
    public void T1_49_Asanyarray_IEnumerable_Half()
    {
        IEnumerable<Half> en = new List<Half> { (Half)1, (Half)2, (Half)3 };
        var nd = np.asanyarray(en);
        nd.dtype.Should().Be(typeof(Half));
        nd.size.Should().Be(3);
    }

    /// <summary>
    /// T1.49 — np.asanyarray throws NotSupportedException for IEnumerable&lt;Complex&gt;.
    /// </summary>
    [TestMethod]
    public void T1_49_Asanyarray_IEnumerable_Complex()
    {
        IEnumerable<Complex> en = new List<Complex> { new Complex(1, 2), new Complex(3, 4) };
        var nd = np.asanyarray(en);
        nd.dtype.Should().Be(typeof(Complex));
        nd.size.Should().Be(2);
    }

    // =====================================================================
    // T1.61 — np.copyto unwriteable dst throws wrong exception type
    // =====================================================================

    /// <summary>
    /// T1.61 — copyto into unwriteable destination throws ArgumentException (NumPy: ValueError).
    /// FIXED — np.copyto now performs the writeability check itself and throws ArgumentException
    /// with the canonical "assignment destination is read-only" message.
    /// </summary>
    [TestMethod]
    public void T1_61_Copyto_UnwriteableDst_ThrowsValueErrorEquivalent()
    {
        // Use a broadcast view to obtain an unwriteable destination.
        var basev = np.array(new double[] { 1.0 });
        var bDst = np.broadcast_to(basev, new Shape(5));
        var src = np.array(new double[] { 2.0, 3.0, 4.0, 5.0, 6.0 });

        Action act = () => np.copyto(bDst, src);
        act.Should().Throw<ArgumentException>(
            "NumPy raises ValueError on write to read-only destination; the .NET equivalent is ArgumentException")
            .WithMessage("*assignment destination is read-only*");
    }

    // =====================================================================
    // T1.62 — np.iinfo(bool) accepted; NumPy 2.x rejects
    // =====================================================================

    /// <summary>
    /// T1.62 — np.iinfo(NPTypeCode.Boolean) returns (bits=8, min=0, max=1); NumPy raises ValueError.
    ///
    /// NumPy 2.4.2: np.iinfo(np.bool_) → ValueError: Invalid integer data type 'b'.
    /// NumSharp: documented as "NumSharp extension" at np.iinfo.cs:84 — recorded as API divergence here.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.62")]
    public void T1_62_Iinfo_Bool_Rejected()
    {
        Action act = () => np.iinfo(NPTypeCode.Boolean);
        act.Should().Throw<ArgumentException>(
            "NumPy 2.x rejects bool with ValueError: 'Invalid integer data type b'");
    }

    // =====================================================================
    // T1.63 — np.iinfo(UInt64).max clamped to long.MaxValue
    // =====================================================================

    /// <summary>
    /// T1.63 — np.iinfo(UInt64).max returns long.MaxValue (9223372036854775807), not the true ulong.MaxValue.
    ///
    /// NumPy 2.4.2: np.iinfo(np.uint64).max == 18446744073709551615 (ulong.MaxValue).
    /// NumSharp: public `max` field is typed `long`, so the true value can't fit. The `maxUnsigned`
    /// field exposes the real value, but any caller using `info.max` silently gets the wrong number.
    /// File: src/NumSharp.Core/APIs/np.iinfo.cs:32 (max type) and :110 (UInt64 clamp).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.63")]
    public void T1_63_Iinfo_UInt64_Max_FullValue()
    {
        var info = np.iinfo(NPTypeCode.UInt64);
        // NumPy returns the real ulong max via .max — NumSharp can't, because the field is typed long.
        // The `maxUnsigned` field exposes the real value (passes today), but the design issue is that
        // `max` should be the full value. Until iinfo.max becomes ulong/BigInteger, this assert fails
        // because info.max == long.MaxValue (9223372036854775807) → cast to ulong is the same value,
        // not ulong.MaxValue (18446744073709551615).
        ((ulong)info.max).Should().Be(ulong.MaxValue,
            "iinfo.max should expose the true UInt64.MaxValue (18446744073709551615), not long.MaxValue (9223372036854775807)");
    }

    // =====================================================================
    // Domain-report findings (non T1.* numbered)
    // =====================================================================

    /// <summary>
    /// Section 1.6 — np.ravel('F') on F-contiguous array returns a view (no copy).
    ///
    /// NumPy 2.4.2:
    ///   aF = np.arange(12).reshape(3,4).copy(order='F')
    ///   r = np.ravel(aF, order='F')
    ///   np.shares_memory(r, aF) == True
    /// Fixed: src/NumSharp.Core/Manipulation/np.ravel.cs now takes a 1-D Alias path
    /// when the source is F-contiguous, sharing the underlying buffer.
    /// </summary>
    [TestMethod]
    public void Ravel_FContiguous_FOrder_ReturnsView()
    {
        var aF = np.arange(12).reshape(3, 4).copy('F');
        aF.Shape.IsFContiguous.Should().BeTrue("test precondition");

        var r = np.ravel(aF, 'F');

        // NumPy: shares memory (view). NumSharp currently materializes a copy.
        // Write through r and observe whether aF sees the change.
        r.SetAtIndex(999L, 0L);
        aF.GetAtIndex(0).Should().Be(999L,
            "ravel('F') of an F-contig array should return a view; NumPy: np.shares_memory(r, aF) == True");
    }

    /// <summary>
    /// Section 1.5 — np.repeat is missing the `axis` parameter.
    ///
    /// NumPy: np.repeat(a, repeats, axis=None). With axis=0, np.repeat(2x2, 2, axis=0).shape == (4,2).
    /// NumSharp: signature is repeat(NDArray, int|long|NDArray) only — no axis overload.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-repeat-axis")]
    public void Repeat_WithAxis_Implemented()
    {
        var methods = typeof(np).GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
        bool hasAxisOverload = false;
        foreach (var m in methods)
        {
            if (m.Name != "repeat") continue;
            foreach (var p in m.GetParameters())
            {
                if (p.Name == "axis") { hasAxisOverload = true; break; }
            }
        }

        hasAxisOverload.Should().BeTrue("NumPy: np.repeat(a, repeats, axis=None) requires axis support");
    }

    /// <summary>
    /// Section 1.5 — verify axis behavior. NumPy:
    ///   np.repeat(np.array([[1,2],[3,4]]), 2, axis=0) == [[1,2],[1,2],[3,4],[3,4]]
    /// NumSharp ravels the input, producing a 1-D output of length 8.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-repeat-axis")]
    public void Repeat_2D_Axis0_PreservesShape()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        // Current API has no axis parameter, so this calls the int overload which ravels first.
        // Asserting the NumPy-correct shape forces the test to fail until axis is implemented.
        var result = np.repeat(a, 2);
        result.shape.Should().Equal(new long[] { 4L, 2L },
            "NumPy: np.repeat(2x2, 2, axis=0).shape == (4, 2)");
    }

    /// <summary>
    /// Section 1.7 — np.unique missing return_index / return_inverse / return_counts / axis / equal_nan.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-unique-kwargs")]
    public void Unique_MissingKeywordArguments()
    {
        var methods = typeof(np).GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
        bool hasReturnInverse = false;
        bool hasReturnIndex = false;
        bool hasReturnCounts = false;
        bool hasAxis = false;
        bool hasEqualNan = false;

        foreach (var m in methods)
        {
            if (m.Name != "unique") continue;
            foreach (var p in m.GetParameters())
            {
                if (p.Name == "return_index") hasReturnIndex = true;
                else if (p.Name == "return_inverse") hasReturnInverse = true;
                else if (p.Name == "return_counts") hasReturnCounts = true;
                else if (p.Name == "axis") hasAxis = true;
                else if (p.Name == "equal_nan") hasEqualNan = true;
            }
        }

        // All five missing per NumPy 2.x signature: unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, equal_nan=True).
        hasReturnIndex.Should().BeTrue("NumPy: unique(..., return_index=False)");
        hasReturnInverse.Should().BeTrue("NumPy: unique(..., return_inverse=False)");
        hasReturnCounts.Should().BeTrue("NumPy: unique(..., return_counts=False)");
        hasAxis.Should().BeTrue("NumPy: unique(..., axis=None)");
        hasEqualNan.Should().BeTrue("NumPy: unique(..., equal_nan=True)");
    }

    /// <summary>
    /// Section 1.3 — np.expand_dims signature only accepts a single int axis.
    /// NumPy 2.x accepts a tuple-of-axes: np.expand_dims(np.array([1,2,3]), (0, 2)).shape == (1, 3, 1).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-expand_dims-tuple")]
    public void ExpandDims_TupleAxis_Implemented()
    {
        var methods = typeof(np).GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public);
        bool hasMultiAxisOverload = false;
        foreach (var m in methods)
        {
            if (m.Name != "expand_dims") continue;
            foreach (var p in m.GetParameters())
            {
                // Accept either int[] or System.Collections.Generic.IEnumerable<int> or a tuple type as `axis`.
                if (p.Name == "axis" && p.ParameterType != typeof(int))
                {
                    hasMultiAxisOverload = true;
                    break;
                }
            }
        }

        hasMultiAxisOverload.Should().BeTrue(
            "NumPy 2.x: np.expand_dims(a, axis) accepts an int OR a tuple/sequence of axes");
    }
}
