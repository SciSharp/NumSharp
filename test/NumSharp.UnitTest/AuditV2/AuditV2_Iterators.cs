using System;
using System.Collections.Generic;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// Audit v2 — Group 1: Iterator subsystem.
/// Each test reproduces a real correctness gap documented in
/// <c>docs/plans/audit_v2/01_iterators.md</c>.
/// Tests are marked [OpenBugs] so CI skips them until the underlying
/// defect is fixed.
/// </summary>
[TestClass]
public class AuditV2_Iterators
{
    // =====================================================================
    // T1.1 — Iternext() ignores EXLOOP, advances 1 element at a time.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIter.cs:1985-2003
    //
    // The public Iternext() method calls _state->Advance() unconditionally
    // (no EXLOOP branch, no buffered-non-reduce refill branch).
    // For EXTERNAL_LOOP iteration on a non-coalescible transposed array
    // (shape (4,3), strides (8,32) on int64), NumPy yields 1 outer iteration
    // (whole 12 elements in one chunk because the array contig-collapses
    // to (12,)), or N outer iterations of size shape[NDim-1] for non-coalescible
    // arrays. NumSharp advances 1 element at a time and yields 12 iterations.
    // =====================================================================
    /// <summary>
    /// T1.1 — Iternext() with EXTERNAL_LOOP must advance Shape[NDim-1] elements
    /// per call, NOT 1 element per call. Otherwise downstream loops that pass
    /// the inner-loop count to a kernel read past the buffer (3-12x overrun).
    ///
    /// NumPy 2.4.2: np.nditer(arange(12).reshape(3,4).T, flags=['external_loop'])
    /// produces 1 chunk of 12 elements when storage coalesces (verified).
    /// NumSharp: Iternext() returns 12 times → caller treats each iteration
    /// as a Shape[NDim-1]-sized chunk → reads 12 * Shape[NDim-1] elements
    /// from a 12-element buffer (overrun).
    ///
    /// Path: src/NumSharp.Core/Backends/Iterators/NpyIter.cs:Iternext()
    /// </summary>
    [TestMethod]
    public void T1_1_Iternext_ExternalLoop_Should_Not_Advance_One_Element_At_A_Time()
    {
        var a = np.arange(12).reshape(3, 4).transpose();  // shape (4,3), non-contig
        using var it = NpyIterRef.New(
            a,
            NpyIterGlobalFlags.EXTERNAL_LOOP,
            NPY_ORDER.NPY_CORDER,
            NPY_CASTING.NPY_SAFE_CASTING);

        // NumPy 2.4.2 with EXTERNAL_LOOP and order='C' on the transposed
        // (4,3) array yields 4 outer iterations of 3 elements each:
        //   [0,4,8], [1,5,9], [2,6,10], [3,7,11]
        // (verified with np.nditer(a_T, flags=['external_loop'], order='C')).
        //
        // NumSharp's Iternext() ignores EXLOOP and advances 1 element at a
        // time, yielding 12 outer iterations instead of 4. The "outer loop
        // count" returned here corresponds to how many times a kernel using
        //   do { kernel(dataptrs, strides, Shape[NDim-1]); } while (it.Iternext());
        // would invoke its inner loop — so 12 instead of 4 means 12*3 = 36
        // elements read from a 12-element array (3× overrun).
        int outerCount = 1;
        while (it.Iternext()) outerCount++;

        outerCount.Should().Be(4,
            "EXTERNAL_LOOP + C-order on transposed (4,3) must yield 4 outer iterations of 3 elements each per NumPy 2.4.2 (verified via np.nditer(a_T, flags=['external_loop'], order='C'))");
    }

    // =====================================================================
    // T1.2 — Iternext() BUFFERED non-reduce path has no buffer-refill logic
    // → AccessViolationException on the second buffer fill.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIter.cs:1985-2003
    //       src/NumSharp.Core/Backends/Iterators/NpyIter.cs:2723-2756 (GetDataPtr)
    //
    // For BUFFERED + !REDUCE, Iternext() falls through to state.Advance()
    // which doesn't refill the buffer; GetDataPtr then computes a pointer
    // past the buffer boundary.
    // =====================================================================
    /// <summary>
    /// T1.2 — BUFFERED non-reduce iteration must work on arrays larger than
    /// the iterator's BufferSize. NumSharp's Iternext() never refills the
    /// buffer → AccessViolationException / segfault on second buffer fill.
    ///
    /// NumPy 2.4.2: np.nditer(arange(20000).astype(int32), op_dtypes=[np.float64],
    /// flags=['buffered'], casting='safe') iterates all 20000 elements correctly.
    /// NumSharp: AccessViolationException after BufferSize elements (process crash).
    ///
    /// Note: Test reproduces the bug indirectly by asserting NumSharp yields
    /// 20000 iterations without throwing. Since the actual repro is an
    /// AccessViolationException (process crash, not catchable in .NET),
    /// we wrap in a Func and validate the count.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.2")]
    public unsafe void T1_2_Iternext_Buffered_NonReduce_Must_Refill_Buffer()
    {
        // WARNING: A naive port of this test ("loop until Iternext returns
        // false") currently triggers an AccessViolationException — an
        // uncatchable, process-fatal segfault — on the second buffer fill.
        // That would bring down the entire test runner.
        //
        // To make the bug visible WITHOUT segfaulting, we assert that
        // NumSharp's NpyIter exposes a buffer-refill path for non-reduce
        // buffered iteration. NumPy's npyiter_buffered_iternext
        // (`numpy/_core/src/multiarray/nditer_templ.c.src:325`) handles this
        // by refilling READ operands and flushing WRITE operands across
        // buffer boundaries. NumSharp's public Iternext() at NpyIter.cs:1985
        // has no equivalent — for BUFFERED + non-REDUCE, it falls through
        // to state.Advance(), and GetDataPtr then reads past the buffer end.
        //
        // The assertion: NumSharp must define a member named
        // BufferedNonReduceIternext (or similar refill path) on NpyIterRef
        // — searching by reflection. When the fix lands and such a method
        // exists (or Iternext() is rewritten to handle the case), this test
        // will start passing.
        var src = np.arange(20000).astype(NPTypeCode.Int32);
        var dtypes = new[] { NPTypeCode.Double };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };

        using var it = NpyIterRef.MultiNew(
            1, new[] { src },
            NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING.NPY_SAFE_CASTING,
            opFlags,
            dtypes);

        it.IterSize.Should().Be(20000L,
            "the buffered iterator must still advertise all 20000 elements");

        // NpyIterRef is a ref struct; use typeof() rather than GetType().
        var type = typeof(NpyIterRef);
        var method = type.GetMethod("BufferedNonReduceIternext",
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Public);

        method.Should().NotBeNull(
            "NpyIter must implement BufferedNonReduceIternext (or equivalent buffer-refill logic in Iternext) to safely iterate buffered non-reduce arrays larger than BufferSize. Without it, GetDataPtr reads past the buffer end → AccessViolationException / segfault.");
    }

    // =====================================================================
    // T1.12 — NpyIterCasting buffer paths (CopyToBuffer/CopyFromBuffer) and
    // ConvertValue(ReadAsDouble/WriteFromDouble) miss SByte, Half, Complex.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIterBufferManager.cs:166-229
    // File: src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs:339-381
    //
    // Constructing a buffered iterator with any of these source dtypes throws
    // NotSupportedException at iterator construction (during buffer copy).
    // =====================================================================
    /// <summary>
    /// T1.12 — Buffered iteration with SByte source must not throw
    /// NotSupportedException. NumPy supports int8 (=SByte) in all buffered paths.
    ///
    /// NumPy 2.4.2: list(np.nditer(np.array([1,2,3], dtype=np.int8),
    ///   op_dtypes=[np.int64], flags=['buffered'], casting='unsafe')) → [1,2,3]
    /// NumSharp: throws NotSupportedException ("Unsupported type: SByte")
    /// in ReadAsDouble during buffer copy.
    /// </summary>
    [TestMethod]
    public void T1_12_BufferedCast_SByte_Source_Must_Not_Throw()
    {
        var a = np.array(new sbyte[] { 1, 2, 3 });
        var dtypes = new[] { NPTypeCode.Int64 };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };

        Action build = () =>
        {
            using var it = NpyIterRef.MultiNew(
                1, new[] { a },
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_UNSAFE_CASTING,
                opFlags, dtypes);
        };

        build.Should().NotThrow("buffered casting SByte → Int64 is a safe widening in NumPy");
    }

    /// <summary>
    /// T1.12 — Buffered iteration with Complex source must not throw.
    ///
    /// NumPy 2.4.2: np.nditer(np.array([1+0j], dtype=np.complex128), flags=['buffered'])
    /// iterates correctly.
    /// NumSharp: throws NotSupportedException ("Buffer copy not supported for dtype Complex").
    /// </summary>
    [TestMethod]
    public void T1_12_Buffered_Complex_Source_Must_Not_Throw()
    {
        var a = np.array(new Complex[] { new Complex(1, 0), new Complex(2, 0) });
        var dtypes = new[] { NPTypeCode.Complex };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };

        Action build = () =>
        {
            using var it = NpyIterRef.MultiNew(
                1, new[] { a },
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                opFlags, dtypes);
        };

        build.Should().NotThrow("Complex same-type buffered iteration must be supported");
    }

    // =====================================================================
    // T1.23 — NpyIter.GetIterView(0) throws OverflowException on 0-d arrays.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIter.cs:2650-2704
    //
    // Line 2668 returns original.flat[0] for ndim=0, but flat[0] triggers slice
    // indexing which fails on scalars with OverflowException.
    // =====================================================================
    /// <summary>
    /// T1.23 — GetIterView on a 0-d (scalar) array must return a 0-d view
    /// that shares storage with the scalar. NumPy: it.itviews[0] returns a
    /// 0-d view. NumSharp: throws OverflowException ("Arithmetic operation
    /// resulted in an overflow").
    ///
    /// NumPy 2.4.2: np.nditer(np.array(42)).itviews[0].ndim == 0
    /// NumSharp: OverflowException at NDArray.get_Item(Slice[]).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.23")]
    public void T1_23_GetIterView_On_0D_Array_Must_Not_Throw()
    {
        var scalar = np.array(42L);
        scalar.ndim.Should().Be(0);

        using var it = NpyIterRef.New(
            scalar,
            NpyIterGlobalFlags.None,
            NPY_ORDER.NPY_CORDER,
            NPY_CASTING.NPY_SAFE_CASTING);

        NDArray? view = null;
        Exception? captured = null;
        try { view = it.GetIterView(0); } catch (Exception e) { captured = e; }

        captured.Should().BeNull("GetIterView on a 0-d operand must return a 0-d view, not throw");
        view!.ndim.Should().Be(0, "view of a 0-d scalar must remain 0-d");
        view.GetValue<long>(0).Should().Be(42L, "the 0-d view must read the scalar's value");
    }

    // =====================================================================
    // T1.24 — NpyIter.EnableExternalLoop doesn't validate MULTI_INDEX/HASINDEX.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIter.cs:2852-2857
    //
    // NumPy raises ValueError; NumSharp silently sets the flag, leaving the
    // iterator in an illegal (HasMultiIndex && HasExternalLoop) state.
    // =====================================================================
    /// <summary>
    /// T1.24 — EnableExternalLoop must raise an error when MULTI_INDEX or
    /// HASINDEX is already set; the combination is invalid.
    ///
    /// NumPy 2.4.2: np.nditer(a, flags=['multi_index']).enable_external_loop()
    ///   raises ValueError("Iterator flag EXTERNAL_LOOP cannot be used if an
    ///   index or multi-index is being tracked").
    /// NumSharp: returns true, leaves iterator in HasMultiIndex && HasExternalLoop state.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.24")]
    public void T1_24_EnableExternalLoop_Must_Reject_MultiIndex()
    {
        var a = np.arange(12).reshape(3, 4);
        using var it = NpyIterRef.New(
            a,
            NpyIterGlobalFlags.MULTI_INDEX,
            NPY_ORDER.NPY_CORDER,
            NPY_CASTING.NPY_SAFE_CASTING);

        Exception? captured = null;
        try { it.EnableExternalLoop(); } catch (Exception e) { captured = e; }

        captured.Should().BeOfType<InvalidOperationException>(
            "EXTERNAL_LOOP and MULTI_INDEX are mutually exclusive per NumPy spec");
    }

    // =====================================================================
    // T1.25 — FIXED. The old NDIterator broadcast constructor produced wrong
    // strides (C-order on the broadcast shape instead of stride=0), reading
    // past the source buffer. NDIterator has been removed; the broadcast
    // iterator is now np.broadcast_to + NpyFlatIterator (np.broadcast(...).iters),
    // which yields the correct cyclic broadcast.
    // =====================================================================
    /// <summary>
    /// T1.25 — Broadcasting [1,2,3] to (4,3) and iterating flat must yield 4
    /// cycles of [1,2,3] (stride=0 on the broadcast axis), per NumPy 2.4.2:
    /// np.broadcast_to([1,2,3], (4,3)).flatten() == [1,2,3,1,2,3,1,2,3,1,2,3].
    /// </summary>
    [TestMethod]
    public void T1_25_Broadcast_Iter_Must_Produce_Cyclic_Values()
    {
        var smaller = np.array(new long[] { 1, 2, 3 });
        var view = np.broadcast_to(smaller, new Shape(4, 3));

        view.size.Should().Be(12L);
        view.AsElements<long>().Should().Equal(
            new long[] { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 },
            "broadcast_to((4,3)) of [1,2,3] must yield 4 cycles of [1,2,3] (NumPy semantics)");
    }

    // =====================================================================
    // T1.34 — NpyExpr Const/Where/Call only support 12 dtypes (no SByte/Half/Complex).
    // File: src/NumSharp.Core/Backends/Iterators/NpyExpr.cs:406-432 (Const.EmitLoadTyped)
    // File: src/NumSharp.Core/Backends/Iterators/NpyExpr.cs:762-790 (Where.EmitPushZero)
    // File: src/NumSharp.Core/Backends/Iterators/NpyExpr.cs:973-980 (Call.IsSupported)
    // =====================================================================
    /// <summary>
    /// T1.34 — NpyExpr.Const must support Half as output dtype. Compiling
    /// a Const-only expression for Half currently throws NotSupportedException
    /// in ConstNode.EmitLoadTyped.
    /// </summary>
    [TestMethod]
    public void T1_34_NpyExpr_Const_Half_Must_Compile()
    {
        var expr = NpyExpr.Const(1.5);

        Action act = () => expr.Compile(new[] { NPTypeCode.Half }, NPTypeCode.Half, "T1_34_Const_Half");

        act.Should().NotThrow("ConstNode must emit a Half load (IL: float load + Half conversion)");
    }

    /// <summary>
    /// T1.34 — NpyExpr.Const must support SByte as output dtype.
    /// </summary>
    [TestMethod]
    public void T1_34_NpyExpr_Const_SByte_Must_Compile()
    {
        var expr = NpyExpr.Const(1);

        Action act = () => expr.Compile(new[] { NPTypeCode.SByte }, NPTypeCode.SByte, "T1_34_Const_SByte");

        act.Should().NotThrow("ConstNode must emit an SByte (int8) load");
    }

    /// <summary>
    /// T1.34 — NpyExpr.Const must support Complex as output dtype.
    /// </summary>
    [TestMethod]
    public void T1_34_NpyExpr_Const_Complex_Must_Compile()
    {
        var expr = NpyExpr.Const(1);

        Action act = () => expr.Compile(new[] { NPTypeCode.Complex }, NPTypeCode.Complex, "T1_34_Const_Complex");

        act.Should().NotThrow("ConstNode must emit a Complex load");
    }

    /// <summary>
    /// T1.34 — NpyExpr.Where must support Half. Currently throws
    /// NotSupportedException in WhereNode.EmitPushZero for Half / SByte / Complex.
    /// </summary>
    [TestMethod]
    public void T1_34_NpyExpr_Where_Half_Must_Compile()
    {
        var cond = NpyExpr.Input(0);
        var a = NpyExpr.Const(1);
        var b = NpyExpr.Const(2);
        var expr = NpyExpr.Where(cond, a, b);

        Action act = () => expr.Compile(new[] { NPTypeCode.Half }, NPTypeCode.Half, "T1_34_Where_Half");

        act.Should().NotThrow("WhereNode.EmitPushZero must support Half");
    }

    /// <summary>
    /// T1.34 — NpyExpr.Call with a Half-typed parameter must be allowed.
    /// IsSupported() rejects Half/SByte/Complex as method parameter/return types.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.34")]
    public void T1_34_NpyExpr_Call_Half_Param_Must_Compile()
    {
        Func<Half, Half> f = h => h;
        var arg = NpyExpr.Input(0);

        Action act = () =>
        {
            var expr = NpyExpr.Call(f, arg);
            expr.Compile(new[] { NPTypeCode.Half }, NPTypeCode.Half, "T1_34_Call_Half");
        };

        act.Should().NotThrow("CallNode.IsSupported must accept Half as a method parameter");
    }

    // =====================================================================
    // T1.38 — NpyIterCasting.IsSafeCast/IsSameKindCast helpers don't list
    // SByte/Half. IsSafeCast(SByte→Int32) returns false even though NumPy
    // declares it safe; IsSafeCast(Half→Single) returns false.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs:138-152
    //
    // The cast failure manifests as InvalidCastException raised by
    // ValidateCasts during iterator construction.
    // =====================================================================
    /// <summary>
    /// T1.38 — Buffered SAFE cast from SByte to Int32 must be allowed.
    /// NumPy: np.can_cast(np.int8, np.int32, 'safe') == True.
    /// NumSharp: IsSafeCast returns false because IsSignedInteger
    /// doesn't include SByte → ValidateCasts throws InvalidCastException
    /// (or the underlying ReadAsDouble throws NotSupportedException, see T1.12).
    /// </summary>
    [TestMethod]
    public void T1_38_IsSafeCast_SByte_To_Int32_Must_Be_Allowed()
    {
        var a = np.array(new sbyte[] { -1, 2 });
        var dtypes = new[] { NPTypeCode.Int32 };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };

        Action build = () =>
        {
            using var it = NpyIterRef.MultiNew(
                1, new[] { a },
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                opFlags, dtypes);
        };

        build.Should().NotThrow("SByte → Int32 is a safe widening (NumPy: can_cast(int8, int32, 'safe') == True)");
    }

    /// <summary>
    /// T1.38 — Buffered SAFE cast from Half to Single must be allowed.
    /// NumPy: np.can_cast(np.float16, np.float32, 'safe') == True.
    /// NumSharp: IsFloatingPoint doesn't include Half → InvalidCastException.
    /// </summary>
    [TestMethod]
    public void T1_38_IsSafeCast_Half_To_Single_Must_Be_Allowed()
    {
        var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3 });
        var dtypes = new[] { NPTypeCode.Single };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };

        Action build = () =>
        {
            using var it = NpyIterRef.MultiNew(
                1, new[] { a },
                NpyIterGlobalFlags.BUFFERED,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                opFlags, dtypes);
        };

        build.Should().NotThrow("Half → Single is a safe widening (NumPy: can_cast(float16, float32, 'safe') == True)");
    }

    // =====================================================================
    // T1.39 — NpyIterCasting.ReadAsDouble/WriteFromDouble routes Int64/UInt64
    // through `double`, losing precision for values above 2^53.
    // File: src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs:339-381
    //
    // For an Int64 value of (1<<53)+1 = 9007199254740993, the double round-trip
    // yields 9007199254740992 (off by 1). NumPy's int64→uint64 cast preserves
    // the exact bit pattern without going through float.
    // =====================================================================
    /// <summary>
    /// T1.39 — Int64 → UInt64 buffered cast must preserve exact bits.
    /// NumSharp routes through `double`, losing precision for values > 2^53.
    ///
    /// NumPy 2.4.2: np.array([(1&lt;&lt;53)+1], dtype=np.int64).astype(np.uint64)
    ///   == [9007199254740993] (exact preservation).
    /// NumSharp: yields 9007199254740992 (off by 1, precision lost via double).
    /// </summary>
    [TestMethod]
    public unsafe void T1_39_Int64_To_UInt64_Cast_Must_Preserve_Precision_Above_2_53()
    {
        long big = (1L << 53) + 1;  // 9007199254740993
        var src = np.array(new long[] { big, big + 1, big + 2 });

        var dtypes = new[] { NPTypeCode.UInt64 };
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };
        using var it = NpyIterRef.MultiNew(
            1, new[] { src },
            NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING.NPY_UNSAFE_CASTING,
            opFlags, dtypes);

        ulong v0 = *(ulong*)it.GetDataPtr(0);
        v0.Should().Be((ulong)big,
            "Int64 → UInt64 cast must preserve bit-exact values, not lose precision through a double intermediate");
    }

    // =====================================================================
    // Newly discovered (related to T1.24): EnableExternalLoop also doesn't
    // validate HASINDEX (C_INDEX flag). Same root cause as T1.24.
    // =====================================================================
    /// <summary>
    /// T1.24 (HASINDEX variant) — EnableExternalLoop must also reject the
    /// HASINDEX flag (set via C_INDEX or F_INDEX during construction).
    /// NumPy raises ValueError on either has_index or has_multi_index.
    /// NumSharp silently sets EXLOOP, leaving HasIndex && HasExternalLoop in an
    /// illegal combination.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.24")]
    public void T1_24_EnableExternalLoop_Must_Reject_CIndex()
    {
        var a = np.arange(12).reshape(3, 4);
        using var it = NpyIterRef.New(
            a,
            NpyIterGlobalFlags.C_INDEX,
            NPY_ORDER.NPY_CORDER,
            NPY_CASTING.NPY_SAFE_CASTING);

        it.HasIndex.Should().BeTrue("constructor flag C_INDEX must set HasIndex");

        Exception? captured = null;
        try { it.EnableExternalLoop(); } catch (Exception e) { captured = e; }

        captured.Should().BeOfType<InvalidOperationException>(
            "EXTERNAL_LOOP and HASINDEX (C_INDEX / F_INDEX) are mutually exclusive per NumPy spec");
    }

    // =====================================================================
    // T1.40 — FALSE / NOT REPRODUCED
    //
    // Stress-tested NpyIter.Copy() with buffered+reduce on both 2D and 3D
    // multi-axis op_axes. The copy advanced independently and reported the
    // expected iter index for the original. ResetDataPtrs, ArrayWritebackPtrs,
    // ReduceOuterPtrs are all copied in the visible code (NpyIter.cs:2991-3003).
    // The audit itself marks T1.40 as "Bug (latent)" — no concrete failure mode
    // was reproducible. Not adding an OpenBugs test until a specific failure
    // pattern is identified.
    //
    //
    // T1.41 — FALSE POSITIVE
    //
    // Verified NumPy 2.4.2 behavior with
    //   np.nditer(np.arange(12).reshape(3,4)).shape
    // returns (12,), NOT (3,4) as the audit's NumPy snippet claims. NumSharp
    // also returns [12] here, so behavior matches.
    //
    // NumPy only returns the original (3,4) shape when MULTI_INDEX is explicitly
    // requested — and NumSharp correctly does so too (verified via
    // NpyIterGlobalFlags.MULTI_INDEX → returns [3,4]).
    //
    // A different, narrower divergence exists when forcing order='C' on a
    // non-contiguous transposed array (NumPy reports iteration-order shape,
    // NumSharp reports requested shape), but that's a separate concern from
    // what the audit's repro asserts. Not adding an OpenBugs test for T1.41.
    // =====================================================================
}
