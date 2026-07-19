using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

// ReSharper disable InconsistentNaming

namespace NumSharp.UnitTest.Selection;

/// <summary>
/// Combinatorial advanced-indexing differential-parity matrix — the regression pins for the
/// mapping.c-parity work (docs/plans/advanced-index-combinatorial-handover.md). Every expected
/// shape/value/raise was probed against NumPy 2.4.2. These are the exact forms the seeded random
/// differential sweep (test/oracle/gen_index_oracle.py, index_random_20240626.jsonl) flagged and
/// that are now bit-identical to NumPy; they are kept as a fast, focused CI gate alongside the full
/// random corpus (Index_Random), which is GREEN and no longer [OpenBugs] now that R3 is fixed.
///
/// Buckets (see the handover):
///   R1  — value must broadcast to an EMPTY / scalar selection on assignment, else ValueError.
///   B2  — a 0-d (scalar) array rejects any axis-consuming single index ("too many indices").
///   B3  — empty advanced indices gather an empty-shaped result (no bounds-check, empty bool mask).
///   B4  — basic assignment into an EMPTY slice selection is a no-op (not a crash).
///   R2  — non-consecutive 0-d-bool/int advanced block moves to the FRONT (_get_transpose).
///   R3  — a boolean mask whose TRAILING block is EMPTY (blockSize == 0) selects nothing: it used to
///         allocate a zero-length gather/scatter buffer and write an element into it — an OOB
///         native-heap write that crashed a later GC (the flaky teardown SEGFAULT).
///
/// Fixture: A=arange(12).reshape(3,4)  V=arange(5)  B=arange(24).reshape(2,3,4)  s=array(5) (0-d)
///   ACS=A[:,::2] (3,2)  ASO=A[1:] (2,4)  ARS=A[::2] (2,4)  ANC=A[:,::-1] (3,4)
/// </summary>
[TestClass]
public class Indexing_CombinatorialParity_MatrixTests
{
    private static NDArray A() => np.arange(12L).reshape(3, 4);
    private static NDArray V() => np.arange(5L);
    private static NDArray B() => np.arange(24L).reshape(2, 3, 4);
    private static NDArray s() => np.array(5L);
    private static NDArray ACS() => A()[":", "::2"];
    private static NDArray ASO() => A()["1:"];
    private static NDArray ARS() => A()["::2"];
    private static NDArray ANC() => A()[":", "::-1"];
    private static NDArray ANR() => A()["::-1"];
    private static NDArray E03() => np.zeros(new Shape(0, 3), dtype: np.int64);

    public sealed record GCase(string Name, Func<NDArray> Op, int[] Shape, long[]? Vals);
    public sealed record SCase(string Name, Func<NDArray> Op, long[] Vals);
    public sealed record ECase(string Name, Action Op);

    // ═════════════ GET — B3 (empty advanced) + R2 (non-consec 0-d-bool placement) ═════════════
    private static IEnumerable<object[]> GetCases()
    {
        // ── B3: empty bool mask (size 0) is valid on ANY axis size, + empty fancy broadcast ──
        yield return Wrap(new("B3_ACS_emptyBool_int",  () => ACS()[np.array(new bool[] { }), (NDArray)0L],                       new[] { 0 },    new long[0]));
        yield return Wrap(new("B3_B_emptyBool_intm1",  () => B()[np.array(new bool[] { }), (NDArray)(-1L)],                      new[] { 0, 4 }, new long[0]));
        yield return Wrap(new("B3_ASO_fancy41_empty",  () => ASO()[new object[] { np.arange(4L).reshape(4, 1), np.array(new long[] { }) }], new[] { 4, 0 }, new long[0]));
        yield return Wrap(new("B3_A_zeros0bool",       () => A()[np.zeros(new Shape(0), dtype: np.@bool)],                      new[] { 0, 4 }, new long[0]));
        yield return Wrap(new("B3_A_zeros30bool",      () => A()[np.zeros(new Shape(3, 0), dtype: np.@bool)],                   new[] { 0 },    new long[0]));
        // OOB fancy values (2,3 >= size 2) are NEVER bounds-checked because the broadcast is empty.
        yield return Wrap(new("B3_ARS_oobFancy_empty", () => ARS()[new object[] { np.array(new long[] { 2, 3 }).reshape(2, 1), np.array(new long[] { }) }], new[] { 2, 0 }, new long[0]));

        // ── R2: non-consecutive advanced block (slice/newaxis separates it) -> block to FRONT ──
        yield return Wrap(new("R2_ARS_sl_T_sl_F",   () => ARS()[new object[] { new Slice(-2L, null, 1L), (NDArray)true, new Slice(0L, null, -2L), (NDArray)false }], new[] { 0, 2, 1 }, new long[0]));
        yield return Wrap(new("R2_ASO_int_sl_T",    () => ASO()[new object[] { -1, new Slice(-1L, 3L, 3L), (NDArray)true }],            new[] { 1, 0 },    new long[0]));
        yield return Wrap(new("R2_ANC_int_new_sl_F", () => ANC()[new object[] { 1, Slice.NewAxis, new Slice(-4L, -4L, 1L), (NDArray)false }], new[] { 0, 1, 0 }, new long[0]));
        // Non-empty R2: int + slice + True -> block (1,) FRONT, then slice -> (1,2) [4,5] (NOT (2,1)).
        yield return Wrap(new("R2_A_int_sl_T",      () => A()[new object[] { 1, new Slice(0L, 2L, 1L), (NDArray)true }],               new[] { 1, 2 },    new long[] { 4, 5 }));
        yield return Wrap(new("R2_A_int_new_sl_T",  () => A()[new object[] { 1, Slice.NewAxis, new Slice(0L, 2L, 1L), (NDArray)true }], new[] { 1, 1, 2 }, new long[] { 4, 5 }));

        // ── R3: a boolean mask whose TRAILING block is EMPTY (blockSize == 0 — a basic slice emptied
        //    the last axis, or the array already had a 0-length trailing axis). The selection
        //    (trueCount,)+trailing is empty, so the result is empty and NOTHING is gathered. Was
        //    handover R3: BooleanMask allocated a zero-length gather buffer and the kernel wrote an
        //    element into it — an out-of-bounds native-heap write that crashed a later GC. ──
        yield return Wrap(new("R3_ANR_mask_emptySlice", () => ANR()[new object[] { np.array(new bool[] { true, true, true }), new Slice(6L, -4L, 1L) }], new[] { 3, 0 },    new long[0]));
        yield return Wrap(new("R3_A_partMask_emptySl",  () => A()[new object[] { np.array(new bool[] { true, false, true }), new Slice(2L, 2L, 1L) }],  new[] { 2, 0 },    new long[0]));
        yield return Wrap(new("R3_A_slice_then_mask",   () => A()[":", "6:-4"][np.array(new bool[] { true, true, true })],                               new[] { 3, 0 },    new long[0]));
        yield return Wrap(new("R3_B_mask_emptySlice",   () => B()[new object[] { np.array(new bool[] { true, true }), new Slice(2L, 2L, 1L) }],           new[] { 2, 0, 4 }, new long[0]));
        yield return Wrap(new("R3_E03_get_0dTrue",      () => E03()[np.array(true)],                                                                      new[] { 1, 0, 3 }, new long[0]));
    }

    [DataTestMethod]
    [DynamicData(nameof(GetCases), DynamicDataSourceType.Method)]
    [TestCategory("FuzzMatrix")]
    public void AssertGet(GCase tc) => CheckGet(tc);

    // ═════════════ SET — B4 (empty slice no-op) ═════════════
    private static IEnumerable<object[]> SetCases()
    {
        var identity = Enumerable.Range(0, 12).Select(x => (long)x).ToArray();
        // Empty basic selection: assignment is a no-op (value broadcasts to the 0-size shape).
        yield return WrapS(new("B4_newaxis_emptySlice_int", () => { var a = A(); a[new object[] { Slice.NewAxis, new Slice(null, 0L, 3L), 2 }] = np.array(new long[] { 15 }); return a; }, identity));
        yield return WrapS(new("B4_emptySlices_ellipsis",   () => { var a = A(); a[new object[] { new Slice(-1L, -4L, 1L), new Slice(-2L, 2L, 2L), Slice.Ellipsis }] = np.array(new long[] { }); return a; }, identity));
        // B2 OK: a 0-d bool on a 0-d scalar consumes no axis -> (1,).
        yield return WrapS(new("B2_s_0dTrue_value", () => s()[np.array(true)], new long[] { 5 }));

        // R3 SET: boolean assignment into an EMPTY selection (blockSize == 0) is a no-op, NOT an OOB
        //   scatter into arr's zero-length buffer (handover R3, BooleanMaskSet). E03[True] = -7 is the
        //   exact corpus repro (set/E03/rand/7171); the (0,3) array stays empty and unchanged.
        yield return WrapS(new("R3_E03_set_0dTrue_scalar", () => { var a = E03(); a[np.array(true)] = (NDArray)(-7L); return a; }, new long[0]));
        yield return WrapS(new("R3_A_mask_emptySlice_set", () => { var a = A(); a[new object[] { np.array(new bool[] { true, true, true }), new Slice(2L, 2L, 1L) }] = (NDArray)99L; return a; }, identity));
    }

    [DataTestMethod]
    [DynamicData(nameof(SetCases), DynamicDataSourceType.Method)]
    [TestCategory("FuzzMatrix")]
    public void AssertSet(SCase tc) => CheckSet(tc);

    // ═════════════ THROW — R1 (value broadcast) + B2 (0-d over-index) ═════════════
    private static IEnumerable<object[]> ThrowCases()
    {
        // R1: value cannot broadcast to the empty / scalar selection -> ValueError.
        yield return WrapE(new("R1_slice_b0False_set",  () => { var a = A(); a[new object[] { Slice.All, (NDArray)false }] = np.array(new long[] { 4, 75 }); }));
        yield return WrapE(new("R1_scalarTarget_1dVal", () => { var a = V(); a[3] = np.array(new long[] { 78 }); }));
        yield return WrapE(new("R1_scalarTarget_multi", () => { var a = A(); a[new object[] { 0, 2 }] = np.array(new long[] { 94 }); }));
        yield return WrapE(new("R1_allFalseMask_set",   () => { var a = A(); a[np.array(new bool[] { false, false, false })] = np.array(new long[] { 93, 1, 39 }); }));
        yield return WrapE(new("R1_emptySlice_badVal",  () => { var a = A(); a[new object[] { Slice.All, new Slice(1L, 1L, 1L) }] = np.array(new long[] { 1, 2, 3 }); }));

        // B2: a 0-d (scalar) array rejects any axis-consuming index -> IndexError ("too many indices").
        yield return WrapE(new("B2_s_emptyFancy",   () => _ = s()[np.array(new long[] { })]));
        yield return WrapE(new("B2_s_fancy",        () => _ = s()[np.array(new long[] { -1, -1 })]));
        yield return WrapE(new("B2_s_boolArray",    () => _ = s()[np.array(new bool[] { false })]));
        yield return WrapE(new("B2_s_rawIntArr",    () => _ = s()[new int[] { 0 }]));
    }

    [DataTestMethod]
    [DynamicData(nameof(ThrowCases), DynamicDataSourceType.Method)]
    [TestCategory("FuzzMatrix")]
    public void AssertThrows(ECase tc) => CheckThrows(tc);

    // ─── helpers (mirror Indexing.EdgeParity.MatrixTests) ───
    private static void CheckGet(GCase tc)
    {
        NDArray r;
        try { r = tc.Op(); }
        catch (Exception ex) { Assert.Fail($"[{tc.Name}] unexpectedly threw {ex.GetType().Name}: {ex.Message}"); return; }

        Assert.AreEqual(tc.Shape.Length, r.ndim, $"[{tc.Name}] ndim mismatch: expected {tc.Shape.Length}, got {r.ndim}");
        for (var i = 0; i < tc.Shape.Length; i++)
            Assert.AreEqual((long)tc.Shape[i], r.Shape.dimensions[i], $"[{tc.Name}] shape[{i}] mismatch (expected [{string.Join(",", tc.Shape)}], got [{string.Join(",", r.shape)}])");

        if (tc.Vals is not null)
        {
            var actual = r.size == 0 ? new long[0] : r.ravel().ToArray<long>();
            CollectionAssert.AreEqual(tc.Vals, actual, $"[{tc.Name}] values mismatch. Expected [{string.Join(",", tc.Vals)}], got [{string.Join(",", actual)}]");
        }
    }

    private static void CheckSet(SCase tc)
    {
        NDArray r;
        try { r = tc.Op(); }
        catch (Exception ex) { Assert.Fail($"[{tc.Name}] unexpectedly threw {ex.GetType().Name}: {ex.Message}"); return; }
        var actual = r.size == 0 ? new long[0] : r.ravel().ToArray<long>();
        CollectionAssert.AreEqual(tc.Vals, actual, $"[{tc.Name}] values mismatch after set. Expected [{string.Join(",", tc.Vals)}], got [{string.Join(",", actual)}]");
    }

    private static void CheckThrows(ECase tc)
    {
        bool threw = false;
        try { tc.Op(); } catch { threw = true; }
        Assert.IsTrue(threw, $"[{tc.Name}] expected an exception but none was thrown");
    }

    private static object[] Wrap(GCase tc) => new object[] { tc };
    private static object[] WrapS(SCase tc) => new object[] { tc };
    private static object[] WrapE(ECase tc) => new object[] { tc };
}
