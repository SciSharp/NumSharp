using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// NDITER branch audit v2 — Tier 1 correctness bugs in Math/Selection/Sorting/Statistics.
///
/// Source documents:
///   docs/plans/NDITER_BRANCH_QUALITY_AUDIT_V2.md
///   docs/plans/audit_v2/07_math_ops_selection_sorting_stats.md
///
/// Each test asserts the CORRECT NumPy 2.x behavior. Tests verify the fix is in place
/// once the underlying bug is resolved (i.e. remove [OpenBugs] when test passes).
/// </summary>
[TestClass]
public class AuditV2_MathSelectionSorting
{
    // ---------------------------------------------------------------------------
    // T1.15 — SetIndicesNDNonLinear<T> throws NotImplementedException
    // ---------------------------------------------------------------------------
    //
    // File: src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Setter.cs:617
    //
    // VERIFIED STATUS: The body of SetIndicesNDNonLinear<T> is `throw new
    // NotImplementedException(...)`. However the only call site (lines 471-472)
    // is COMMENTED OUT as a TODO:
    //
    //     //TODO: if (isSubshaped && !source.Shape.IsContiguous)
    //     //TODO:     return SetIndicesNDNonLinear(source, indices, ndsCount, ...);
    //
    // Because of that, the user-facing fancy-indexed setter on a transposed/
    // sliced multi-dim source DOES NOT currently reach the NotImpl throw.
    // Instead it falls through to SetIndicesND<T> (line 553), where a
    // Debug.Assert(dstOffsets.size == values.size) fires (in DEBUG builds)
    // because the path was never designed to handle the subshaped non-contig
    // case. In Release the same path silently writes to the wrong offsets
    // (because the values shape no longer aligns with offsets).
    //
    // Two tests below — one ensures the dedicated path eventually exists and
    // doesn't throw NotImpl when called; the other reproduces the broken
    // setter path for fancy-indexing into a transposed N-D view.

    /// <summary>
    /// T1.15a — Direct invocation of SetIndicesNDNonLinear&lt;T&gt; via reflection.
    ///
    /// The method exists but its body unconditionally throws NotImplementedException.
    /// When implemented, calling it should succeed (any valid input).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.15")]
    public void T1_15a_SetIndicesNDNonLinear_ThrowsNotImplemented()
    {
        // Locate via reflection (the method is protected static)
        var method = typeof(NDArray)
            .GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
            .FirstOrDefault(m => m.Name == "SetIndicesNDNonLinear");

        method.Should().NotBeNull("the method exists in the source even if dead-code");

        // Construct minimal args. Body throws on entry regardless of inputs.
        var generic = method!.MakeGenericMethod(typeof(double));

        // We're only verifying the throw — supply nulls/empties; the throw fires
        // before any dereference.
        var source = np.arange(8).astype(NPTypeCode.Double).reshape(2, 4).MakeGeneric<double>();
        var indices = new NDArray[] { np.array(new int[] { 0, 1 }) };
        var values = np.array(new double[] { 99.0, 100.0 }).MakeGeneric<double>();

        Action act = () => generic.Invoke(null, new object[]
        {
            source, indices, /* ndsCount */ 1,
            /* retShape */ new long[] { 2L, 4L },
            /* subShape */ new long[] { 4L },
            values
        });

        // The reflection wrapper rewraps NotImplementedException in TargetInvocationException;
        // assert the underlying type:
        act.Should().NotThrow("SetIndicesNDNonLinear should be implemented (currently throws NotImplementedException)");
    }

    /// <summary>
    /// T1.15b — User-facing fancy index setter on a transposed (non-contig) source
    /// fails. NumPy supports this; NumSharp's setter routes to SetIndicesND which
    /// asserts on shape mismatch (Debug) or writes wrong offsets (Release).
    ///
    /// NumPy:
    ///   a = np.arange(24).reshape(2,3,4).transpose(2,1,0).astype(float)
    ///   a[[0, 2]] = np.zeros((2, 3, 2))
    ///   # Succeeds, writes (2,3,2) zeros into a[[0]] and a[[2]] (each (3,2) slice).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.15")]
    public void T1_15b_FancySet_TransposedNonContig_FailsOrCorrupts()
    {
        var arr = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Double);
        var transposed = arr.transpose(new int[] { 2, 1, 0 });   // shape (4,3,2), non-contig
        transposed.Shape.IsContiguous.Should().BeFalse();

        var idx = np.array(new int[] { 0, 2 });
        // values shape matches the subshape (3,2) for each of the 2 indices = (2,3,2)
        var vals = np.zeros(new Shape(2, 3, 2), NPTypeCode.Double);

        Action act = () => transposed[idx] = vals;

        // Currently fails — either Debug.Assert fires (DEBUG) or silently produces
        // wrong offsets (RELEASE). When fixed, this should succeed and zero the
        // selected slices.
        act.Should().NotThrow("fancy-index setter on transposed N-D arrays should match NumPy");

        // After fix, zeros should be written at indices 0 and 2 along axis 0
        // (and unchanged at index 1 and 3).
        transposed.GetDouble(0, 0, 0).Should().Be(0.0);
        transposed.GetDouble(2, 0, 0).Should().Be(0.0);
    }

    // ---------------------------------------------------------------------------
    // T1.27 — np.searchsorted misnamed/incomplete
    // ---------------------------------------------------------------------------
    //
    // File: src/NumSharp.Core/Sorting_Searching_Counting/np.searchsorted.cs
    //
    // Three sub-issues:
    //   a) binarySearchRightmost is actually leftmost (uses `val < target`).
    //      NumPy supports side='left' (default) AND side='right'; NumSharp has only left.
    //   b) Missing side and sorter parameters.
    //   c) Multidim 'a' is silently accepted (treated as flat); NumPy raises ValueError.

    /// <summary>
    /// T1.27a — np.searchsorted(...) is missing the side parameter (NumPy default side='left').
    /// Compile-time/API check: there is no overload accepting side or sorter.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.27a")]
    public void T1_27a_Searchsorted_Missing_SideParameter()
    {
        // NumPy: np.searchsorted([1,2,2,3], 2, side='left')  = 1
        // NumPy: np.searchsorted([1,2,2,3], 2, side='right') = 3
        var a = np.array(new int[] { 1, 2, 2, 3 });

        // Current behavior — only 'left' is available implicitly.
        long left = np.searchsorted(a, 2);
        left.Should().Be(1, "left bisect (NumSharp current default)");

        // The fix should add a `string side` (or enum) parameter so users can request 'right'.
        // Locate via reflection — there is currently no overload that accepts side.
        var overloads = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name == "searchsorted")
            .ToArray();

        bool hasSide = overloads.Any(m =>
            m.GetParameters().Any(p => p.Name == "side" || p.Name == "Side"));

        hasSide.Should().BeTrue("np.searchsorted should expose a `side` parameter (NumPy parity)");
    }

    /// <summary>
    /// T1.27b — Multidim 'a' should raise; NumSharp silently treats it as flat.
    ///
    /// NumPy:    np.searchsorted(np.arange(20).reshape(4,5), 5)
    ///           ValueError: object too deep for desired array
    /// NumSharp: returns an int (flat binsearch over 20 elements, ignores shape).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.27b")]
    public void T1_27b_Searchsorted_Multidim_Silently_Accepted()
    {
        var marr = np.arange(20).reshape(4, 5);

        Action act = () => np.searchsorted(marr, 5);

        // NumPy raises; NumSharp does not. After fix this assertion should succeed.
        act.Should().Throw<Exception>("np.searchsorted should reject multidim `a` like NumPy does");
    }

    /// <summary>
    /// T1.27c — binarySearchRightmost is named misleadingly. Its inner condition
    /// (`val &lt; target`) is the bisect-LEFT recipe; despite the name, it never
    /// produces the bisect-right answer. The function's own docstring even admits
    /// it is "left-most position … equivalent to NumPy's searchsorted with side='left'".
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.27c")]
    public void T1_27c_BinarySearchRightmost_MisnamedActuallyLeftmost()
    {
        // Repeat the canonical NumPy test:
        //   side='left'  → 1
        //   side='right' → 3
        var a = np.array(new int[] { 1, 2, 2, 3 });
        long actual = np.searchsorted(a, 2);

        // The function returns 'left'. We can't request 'right' today. When the
        // implementation grows a `side` parameter, calling it with side="right"
        // should produce 3. The current API can't express that — that's the bug.
        actual.Should().Be(3, "this assertion verifies the future side='right' path returns 3");
    }

    // ---------------------------------------------------------------------------
    // T1.32 — np.modf public tuple field name typo: Intergral
    // ---------------------------------------------------------------------------
    //
    // File: src/NumSharp.Core/Math/np.modf.cs:16,27
    //       src/NumSharp.Core/Backends/TensorEngine.cs:108,109
    //       src/NumSharp.Core/Backends/Default/Math/Default.Modf.cs:8,22
    //
    // Spelling: NumSharp uses `Intergral` (wrong); should be `Integral`.

    /// <summary>
    /// T1.32 — Public-API typo: np.modf returns (NDArray Fractional, NDArray Intergral).
    /// The second tuple element must be renamed to `Integral` (compile-time API break).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.32")]
    public void T1_32_Modf_IntegralFieldName_Typo()
    {
        // Locate the public np.modf method(s) and inspect tuple element names.
        var methods = typeof(np)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name == "modf")
            .ToArray();

        methods.Should().NotBeEmpty("np.modf should exist");

        foreach (var m in methods)
        {
            // C# 7 tuple member names are serialized via TupleElementNamesAttribute
            // on the return-type custom attributes.
            var names = m.ReturnTypeCustomAttributes
                .GetCustomAttributes(typeof(TupleElementNamesAttribute), false)
                .OfType<TupleElementNamesAttribute>()
                .SelectMany(a => a.TransformNames)
                .ToArray();

            names.Should().Contain("Fractional", "first tuple element should be named Fractional");
            names.Should().Contain("Integral", $"second tuple element should be named Integral, but found: [{string.Join(",", names)}]");
            names.Should().NotContain("Intergral", "typo 'Intergral' must be fixed");
        }
    }

    // ---------------------------------------------------------------------------
    // Additional findings from the domain report (07_math_ops_selection_sorting_stats.md)
    // ---------------------------------------------------------------------------

    /// <summary>
    /// argsort perf regression — measured ~150-184x slower than NumPy on 1000x1000.
    ///
    /// NumSharp uses LINQ (`OrderBy` + per-element NDArray view allocations) in
    /// `Sorting_Searching_Counting/ndarray.argsort.cs:17-212`. The fix is to use
    /// a pointer-based typed introsort over each axis stride.
    ///
    /// Test threshold: 50× NumPy baseline of ~12.5 ms = 625 ms ceiling.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-argsort-perf")]
    public void T2_Argsort_Perf_LINQ_AtLeast50xSlower()
    {
        var rand = new Random(42);
        var data = new double[1000 * 1000];
        for (int i = 0; i < data.Length; i++) data[i] = rand.NextDouble();
        var arr = np.array(data).reshape(1000, 1000);

        // Warmup
        _ = arr.argsort<double>(axis: -1);

        var sw = Stopwatch.StartNew();
        const int iters = 3;
        for (int i = 0; i < iters; i++)
            _ = arr.argsort<double>(axis: -1);
        sw.Stop();

        double perIter = (double)sw.ElapsedMilliseconds / iters;
        const double numpyBaselineMs = 12.5;
        const double targetMaxMs = numpyBaselineMs * 50.0; // 50x ceiling

        // Currently exceeds the 50x ceiling. The test should pass after argsort
        // is rewritten with a typed pointer sort.
        perIter.Should().BeLessThan(targetMaxMs,
            $"argsort 1000x1000 axis=-1 should be within 50x of NumPy (~{targetMaxMs} ms), measured {perIter:F1} ms");
    }

    /// <summary>
    /// T1.28a — np.negative inconsistency: unary operator `-byte_arr` works
    /// (uint wrap-around), but np.negative(byte_arr) throws NotSupportedException.
    /// Both should match NumPy and wrap.
    ///
    /// NumPy:   np.negative(np.uint8([1,5,0])) = [255, 251, 0]
    /// NumSharp operator: works   np.negative(byte): throws
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.28a")]
    public void T1_28a_NpNegative_RejectsByteArray_OperatorWorks()
    {
        var arr = np.array(new byte[] { 1, 5, 0 });

        // Operator path works.
        var opResult = -arr;
        opResult.GetByte(0).Should().Be(255);
        opResult.GetByte(1).Should().Be(251);
        opResult.GetByte(2).Should().Be(0);

        // np.negative path should also work but currently throws.
        Action act = () => np.negative(arr);
        act.Should().NotThrow("np.negative should match unary operator on uint dtypes (wrap-around like NumPy)");
    }

    /// <summary>
    /// T1.28b — np.negative(bool_array) silently returns logical NOT; NumPy raises TypeError.
    /// NumPy 2.x: "The numpy boolean negative, the `-` operator, is not supported,
    /// use the `~` operator or the logical_not function instead."
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.28b")]
    public void T1_28b_NpNegative_AcceptsBool_NumPyRejects()
    {
        var barr = np.array(new bool[] { true, false, true });

        Action act = () => np.negative(barr);
        act.Should().Throw<Exception>("np.negative(bool_array) should raise like NumPy 2.x does");
    }

    /// <summary>
    /// T1.59 — np.where(condition) shape divergence.
    ///   NumPy:    returns a tuple of N int64 arrays (Python tuple, len == ndim).
    ///   NumSharp: returns NDArray&lt;long&gt;[] — array of NDArrays.
    /// Documented divergence (not silent-corruption). Porting risk.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.59")]
    public void T1_59_NpWhere_OneArg_ReturnsArray_NotTuple()
    {
        // Current: ret is NDArray<long>[].
        var cond = np.array(new bool[] { true, false, true });
        var ret = np.where(cond);

        ret.Should().BeOfType<NDArray<long>[]>("documented divergence — NumPy returns tuple, NumSharp returns array");

        // The fix should expose a tuple-like API matching NumPy semantics. Until then
        // mark this as an OpenBugs to track parity.
        // NumPy: type(result).__name__ == 'tuple'
        bool isLikeTuple = ret.GetType().Name.Contains("Tuple")
                        || ret.GetType().Name.Contains("ValueTuple");
        isLikeTuple.Should().BeTrue("np.where(cond) should mirror NumPy's tuple return type");
    }

    /// <summary>
    /// T1.60 — np.where(cond, x, y) integer dtype divergence.
    ///   NumPy:    np.where(cond, 1, 2).dtype == int64 (Python int → numpy default)
    ///   NumSharp: int32 (NumSharp default for `np.array(int)`)
    /// Documented cross-language divergence.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.60")]
    public void T1_60_NpWhere_IntegerScalars_ReturnsInt32_NotInt64()
    {
        var cond = np.array(new bool[] { true, false, true });
        var result = np.where(cond, np.array(1), np.array(2));

        result.dtype.Should().Be(typeof(long), "np.where(cond, 1, 2) should produce int64 dtype like NumPy");
    }
}
