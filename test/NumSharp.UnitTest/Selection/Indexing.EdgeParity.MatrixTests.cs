using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

// ReSharper disable InconsistentNaming

namespace NumSharp.UnitTest.Selection;

/// <summary>
/// Edge-case indexing differential-parity test matrix.
/// Every expected shape/value was probed against NumPy 2.4.2 AND the NumSharp result
/// re-verified in a clean (--no-incremental) build (an earlier authoring pass over-
/// reported divergences because its worktree served a stale NumSharp.Core binary; all
/// claims here are re-confirmed against the committed code).
///
/// Category: 0-d arrays / scalar-array / empty / dtype-of-index / over-indexing.
///
/// Fixture:
///   A = np.arange(12, int64).reshape(3,4)   V = np.arange(5, int64)
///   B = np.arange(24, int64).reshape(2,3,4) s = np.array(5, int64)  (0-d scalar)
///
/// Decision B (NumSharp): a raw int[]/long[] as the SOLE index = FANCY; the coordinate
/// shim is nd.GetData(coords).
/// </summary>
[TestClass]
public class Indexing_EdgeParity_MatrixTests
{
    // ─── Fixture (fresh each call; no shared mutable state) ─────────────────────
    private static NDArray A() => np.arange(12L).reshape(3, 4);     // (3,4) int64
    private static NDArray V() => np.arange(5L);                     // (5,)  int64
    private static NDArray B() => np.arange(24L).reshape(2, 3, 4);  // (2,3,4) int64
    private static NDArray s() => np.array(5L);                      // ()    int64 scalar

    // ─── Case records ───────────────────────────────────────────────────────────
    public sealed record GCase(string Name, Func<NDArray> Op, int[] Shape, long[]? Vals); // Shape [<0] = ndim-0 sentinel; Vals null = shape-only
    public sealed record SCase(string Name, Func<NDArray> Op, long[] Vals);
    public sealed record ECase(string Name, Action Op);

    private static NDArray Idx<T>(params T[] data) => np.array(data);

    // ════════════════════════════════ GET — passing ════════════════════════════
    private static IEnumerable<object[]> GetCases()
    {
        // ── 0-d integer array index == a scalar int (HAS_SCALAR_ARRAY): consumes an
        //    axis, adds none. Verified clean-build NumSharp == NumPy. ──
        yield return Wrap(new("A_0dInt0",       () => A()[np.array(0L)],            new[] { 4 },    new long[] { 0, 1, 2, 3 }));
        yield return Wrap(new("A_0dInt2",       () => A()[np.array(2L)],            new[] { 4 },    new long[] { 8, 9, 10, 11 }));
        yield return Wrap(new("A_0dInt1_colon", () => A()[np.array(1L), ":"],        new[] { 4 },    new long[] { 4, 5, 6, 7 }));
        yield return Wrap(new("A_0dInt_pair",   () => A()[np.array(0L), np.array(2L)], new[] { -1 }, new long[] { 2 }));  // → 0-d scalar
        yield return Wrap(new("B_0dInt1",       () => B()[np.array(1L)],            new[] { 3, 4 }, Enumerable.Range(12, 12).Select(x => (long)x).ToArray()));
        yield return Wrap(new("A_0dInt1_newaxis", () => A()[np.array(1L), Slice.NewAxis], new[] { 1, 4 }, new long[] { 4, 5, 6, 7 }));
        yield return Wrap(new("A_newaxis_0dInt1", () => A()[Slice.NewAxis, np.array(1L)], new[] { 1, 4 }, new long[] { 4, 5, 6, 7 }));
        yield return Wrap(new("A_0dInt1_ellipsis", () => A()[np.array(1L), Slice.Ellipsis], new[] { 4 }, new long[] { 4, 5, 6, 7 }));
        yield return Wrap(new("A_ellipsis_0dInt1", () => A()[Slice.Ellipsis, np.array(1L)], new[] { 3 }, new long[] { 1, 5, 9 }));

        // ── 0-d boolean index solo (HAS_0D_BOOL): True → leading size-1 axis, False → size-0. ──
        yield return Wrap(new("A_0dTrue",  () => A()[np.array(true)],  new[] { 1, 3, 4 }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
        yield return Wrap(new("A_0dFalse", () => A()[np.array(false)], new[] { 0, 3, 4 }, new long[0]));
        yield return Wrap(new("V_0dTrue",  () => V()[np.array(true)],  new[] { 1, 5 },    new long[] { 0, 1, 2, 3, 4 }));
        yield return Wrap(new("V_0dFalse", () => V()[np.array(false)], new[] { 0, 5 },    new long[0]));

        // ── 0-d scalar base ──
        yield return Wrap(new("s_emptyTuple", () => s()[new object[0]], new[] { -1 }, new long[] { 5 }));  // s[()]
        yield return Wrap(new("s_ellipsis",   () => s()["..."],         new[] { -1 }, new long[] { 5 }));  // s[...]
        yield return Wrap(new("s_newaxis",    () => s()[Slice.NewAxis], new[] { 1 },  new long[] { 5 }));  // s[None] → (1,)

        // ── n-d array with empty tuple → the array itself ──
        yield return Wrap(new("A_emptyTuple", () => A()[new object[0]], new[] { 3, 4 }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
        yield return Wrap(new("V_emptyTuple", () => V()[new object[0]], new[] { 5 },    new long[] { 0, 1, 2, 3, 4 }));

        // ── Empty arrays / empty index ──
        yield return Wrap(new("E03_colon",         () => np.zeros(new Shape(0, 3), dtype: np.int64)[":"],                       new[] { 0, 3 }, new long[0]));
        yield return Wrap(new("E03_slice_0_0",     () => np.zeros(new Shape(0, 3), dtype: np.int64)["0:0"],                     new[] { 0, 3 }, new long[0]));
        yield return Wrap(new("E03_emptyFancyIdx", () => np.zeros(new Shape(0, 3), dtype: np.int64)[np.array(new long[] { })],  new[] { 0, 3 }, new long[0]));
        yield return Wrap(new("E03_emptyBoolMask", () => np.zeros(new Shape(0, 3), dtype: np.int64)[np.array(new bool[] { })],  new[] { 0, 3 }, new long[0]));
        yield return Wrap(new("E0_emptyFancyIdx",  () => np.arange(0L)[np.array(new long[] { })],                              new[] { 0 },    new long[0]));
        yield return Wrap(new("E0_emptyBoolMask",  () => np.arange(0L)[np.array(new bool[] { })],                              new[] { 0 },    new long[0]));
        yield return Wrap(new("A_emptyFancyIdx",   () => A()[np.array(new long[] { })],                                        new[] { 0, 4 }, new long[0]));
        yield return Wrap(new("V_emptyFancyIdx",   () => V()[np.array(new long[] { })],                                        new[] { 0 },    new long[0]));

        // ── Index dtype coverage: [1,0,1] selects rows 1,0,1 → (3,4). NumPy accepts
        //    EVERY integer dtype; all eight (signed int8/16/32/64 + unsigned 8/16/32/64)
        //    must give the identical fancy result. ──
        var fancyVals = new long[] { 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 };
        yield return Wrap(new("FancyIdx_int8",   () => A()[Idx<sbyte>(1, 0, 1)],     new[] { 3, 4 }, fancyVals));
        // negative int8 indices wrap (sign-extended to int64): [-1,-2] -> rows 2,1.
        yield return Wrap(new("FancyIdx_int8_neg", () => A()[Idx<sbyte>(-1, -2)],    new[] { 2, 4 }, new long[] { 8, 9, 10, 11, 4, 5, 6, 7 }));
        yield return Wrap(new("FancyIdx_uint8",  () => A()[Idx<byte>(1, 0, 1)],      new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_int16",  () => A()[Idx<short>(1, 0, 1)],     new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_uint16", () => A()[Idx<ushort>(1, 0, 1)],    new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_int32",  () => A()[Idx<int>(1, 0, 1)],       new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_uint32", () => A()[Idx<uint>(1u, 0u, 1u)],   new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_int64",  () => A()[Idx<long>(1L, 0L, 1L)],   new[] { 3, 4 }, fancyVals));
        yield return Wrap(new("FancyIdx_uint64", () => A()[Idx<ulong>(1ul, 0ul, 1ul)], new[] { 3, 4 }, fancyVals));

        // ── All-slice tuple equal to rank is valid ──
        yield return Wrap(new("A_allSlice2d", () => A()[Slice.All, Slice.All], new[] { 3, 4 }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
    }

    [DataTestMethod]
    [DynamicData(nameof(GetCases), DynamicDataSourceType.Method)]
    public void AssertGet(GCase tc) => CheckGet(tc);

    // ════════════════════════════════ SET — passing ════════════════════════════
    private static IEnumerable<object[]> SetCases()
    {
        // 0-d bool False set is a no-op; empty-fancy set is a no-op.
        yield return WrapS(new("A_0dFalse_set_noop",      () => { var a = A(); a[np.array(false)] = np.arange(12L, 24L).reshape(3, 4); return a; }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
        yield return WrapS(new("A_emptyFancyIdx_set_noop", () => { var a = A(); a[np.array(new long[] { })] = np.empty(new Shape(0, 4), dtype: np.int64); return a; }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));

        // Boolean-mask scalar set.
        yield return WrapS(new("A_boolMask_scalarSet", () => { var a = A(); a[a > 10] = 99L; return a; }, new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99 }));

        // 0-d bool True set: assigns the (3,4) value into all of A. NumSharp == NumPy.
        yield return WrapS(new("A_0dTrue_set_matched", () => { var a = A(); a[np.array(true)] = np.arange(12L, 24L).reshape(3, 4); return a; }, Enumerable.Range(12, 12).Select(x => (long)x).ToArray()));

        // 0-d int index set (row 0), matched (4,) value. NumSharp == NumPy.
        yield return WrapS(new("A_0dInt0_set_matched", () => { var a = A(); a[np.array(0L)] = np.array(new long[] { 99, 98, 97, 96 }); return a; }, new long[] { 99, 98, 97, 96, 4, 5, 6, 7, 8, 9, 10, 11 }));

        // uint8 single-element fancy set (row 2), matched (4,) value.
        yield return WrapS(new("A_uint8Idx_single_set", () => { var a = A(); a[Idx<byte>(2)] = np.array(new long[] { 99, 99, 99, 99 }); return a; }, new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99, 99 }));

        // uint8 multi-row fancy set (rows 1,0,1), matched (3,4) value — last write wins per row.
        yield return WrapS(new("A_uint8Idx_multiRow_set", () => { var a = A(); a[Idx<byte>(1, 0, 1)] = np.arange(100L, 112L).reshape(3, 4); return a; }, new long[] { 104, 105, 106, 107, 108, 109, 110, 111, 8, 9, 10, 11 }));
    }

    [DataTestMethod]
    [DynamicData(nameof(SetCases), DynamicDataSourceType.Method)]
    public void AssertSet(SCase tc) => CheckSet(tc);

    // ════════════════════════════════ THROW — passing ══════════════════════════
    private static IEnumerable<object[]> ThrowCases()
    {
        yield return WrapE(new("A_float32Idx_throws", () => _ = A()[np.array(new float[] { 1f, 0f, 1f })]));
        yield return WrapE(new("A_float64Idx_throws", () => _ = A()[np.array(new double[] { 1.0, 0.0, 1.0 })]));
        yield return WrapE(new("A_tooManyIntIndices_throws", () => _ = A()[(object)0, (object)0, (object)0]));
    }

    [DataTestMethod]
    [DynamicData(nameof(ThrowCases), DynamicDataSourceType.Method)]
    public void AssertThrows(ECase tc) => CheckThrows(tc);

    // ═══════════════════════════════════════════════════════════════════════════
    // BUG cases — genuine NumSharp divergences (clean-build verified). [OpenBugs]
    // ═══════════════════════════════════════════════════════════════════════════
    private static IEnumerable<object[]> BugGetCases()
    {
        // A 0-d boolean combined with a slice must ADD a leading size-1/size-0 axis
        // (HAS_0D_BOOL) and KEEP every source axis. NumSharp instead drops the axis the
        // colon should preserve, returning one rank too few.
        //   NumPy A[True, :]  → (1,3,4);  NumSharp → (1,4)
        //   NumPy A[False, :] → (0,3,4);  NumSharp → (0,4)
        //   NumPy A[:, True]  → (3,1,4);  NumSharp → (3,1)
        yield return Wrap(new("BUG_A_0dTrue_colon",  () => A()[np.array(true), ":"],  new[] { 1, 3, 4 }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
        yield return Wrap(new("BUG_A_0dFalse_colon", () => A()[np.array(false), ":"], new[] { 0, 3, 4 }, new long[0]));
        yield return Wrap(new("BUG_A_colon_0dTrue",  () => A()[":", np.array(true)],  new[] { 3, 1, 4 }, Enumerable.Range(0, 12).Select(x => (long)x).ToArray()));
    }

    [DataTestMethod]
    [OpenBugs]
    [DynamicData(nameof(BugGetCases), DynamicDataSourceType.Method)]
    public void AssertGet_Bug(GCase tc) => CheckGet(tc);

    private static IEnumerable<object[]> BugThrowCases()
    {
        // Over-indexing a 2-D array with 3 slices must raise (NumPy: too many indices).
        // NumSharp silently returns a view.
        yield return WrapE(new("BUG_A_tooManySlices_shouldThrow", () => _ = A()[":", ":", ":"]));
    }

    [DataTestMethod]
    [OpenBugs]
    [DynamicData(nameof(BugThrowCases), DynamicDataSourceType.Method)]
    public void AssertThrows_Bug(ECase tc) => CheckThrows(tc);

    // ─── Shared assertion bodies ────────────────────────────────────────────────
    private static void CheckGet(GCase tc)
    {
        NDArray r;
        try { r = tc.Op(); }
        catch (Exception ex) { Assert.Fail($"[{tc.Name}] unexpectedly threw {ex.GetType().Name}: {ex.Message}"); return; }

        if (tc.Shape is [< 0])
            Assert.AreEqual(0, r.ndim, $"[{tc.Name}] expected ndim=0 but got ndim={r.ndim}");
        else
        {
            Assert.AreEqual(tc.Shape.Length, r.ndim, $"[{tc.Name}] ndim mismatch: expected {tc.Shape.Length}, got {r.ndim}");
            for (var i = 0; i < tc.Shape.Length; i++)
                Assert.AreEqual((long)tc.Shape[i], r.Shape.dimensions[i], $"[{tc.Name}] shape[{i}] mismatch");
        }

        if (tc.Vals is not null)
        {
            var actual = r.ravel().ToArray<long>();
            CollectionAssert.AreEqual(tc.Vals, actual, $"[{tc.Name}] values mismatch. Expected [{string.Join(",", tc.Vals)}], got [{string.Join(",", actual)}]");
        }
    }

    private static void CheckSet(SCase tc)
    {
        NDArray r;
        try { r = tc.Op(); }
        catch (Exception ex) { Assert.Fail($"[{tc.Name}] unexpectedly threw {ex.GetType().Name}: {ex.Message}"); return; }
        var actual = r.ravel().ToArray<long>();
        CollectionAssert.AreEqual(tc.Vals, actual, $"[{tc.Name}] values mismatch after set. Expected [{string.Join(",", tc.Vals)}], got [{string.Join(",", actual)}]");
    }

    private static void CheckThrows(ECase tc)
    {
        // MSTest v3 Assert.ThrowsException is exact-type strict — use manual try/catch.
        bool threw = false;
        try { tc.Op(); } catch { threw = true; }
        Assert.IsTrue(threw, $"[{tc.Name}] expected an exception but none was thrown");
    }

    // ─── DynamicData wrappers ───────────────────────────────────────────────────
    private static object[] Wrap(GCase tc) => new object[] { tc };
    private static object[] WrapS(SCase tc) => new object[] { tc };
    private static object[] WrapE(ECase tc) => new object[] { tc };
}
