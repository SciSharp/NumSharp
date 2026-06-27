// Indexing.LayoutValueParity.MatrixTests.cs
// Memory-layout × value-semantics differential-parity matrix.
// Oracle: NumPy 2.4.2 — all expected shapes/values probed from actual Python runs.
//
// Category A:  GET  — same 7-op battery over 8 memory layouts of (3,4) int64 arange(12).
//               Layout: C-contiguous, Transposed, Row-strided, Col-strided,
//                       Neg-row, Neg-col, Sliced-offset, Broadcast.
//               Ops: int-index, slice "1:3", neg-stride "::-1", fancy Idx,
//                    bool mask, col-fancy [":", Idx], row-fancy+colon [Idx, ":"].
// Category B:  SET  — setter value-semantics (int-index, slice, bool, neg-stride,
//               scalar-broadcast, float-truncates, shape-mismatch-throws, broadcast-write-throws).
// Category C:  COPY/VIEW contract — slice→view, neg-col-slice→view, fancy→copy, bool→copy.
//
// Bug cases (NumSharp diverges from NumPy) are collected into [OpenBugs] DataTestMethod targets.
// Bug root: combined fancy-index [":", Idx] / [Idx, ":"] is not implemented for strided views.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Selection;

// ─── Case record types ──────────────────────────────────────────────────────
// GCase: a GET test — op returns an NDArray, expected shape+values pinned from NumPy.
public record GCase(string Name, Func<NDArray> Op, int[] ExpShape, long[] ExpVals)
{
    public override string ToString() => Name;
}

// SCase: a SET test — op mutates an array and returns the result; expected ravel pinned.
public record SCase(string Name, Func<NDArray> Op, long[] ExpVals)
{
    public override string ToString() => Name;
}

// ECase: an EXCEPTION test — op is expected to throw any exception.
public record ECase(string Name, Action Op)
{
    public override string ToString() => Name;
}

[TestClass]
public class Indexing_LayoutValueParity_MatrixTests
{
    // ─── Shared helpers ──────────────────────────────────────────────────────────

    /// <summary>Returns arange(12, dtype=int64).reshape(3,4) — fresh each call.</summary>
    static NDArray A() => np.arange(12L).reshape(3, 4);

    /// <summary>Builds an int64 NDArray index from the supplied values.</summary>
    static NDArray Idx(params long[] v) => np.array(v);

    // ─────────────────────────────────────────────────────────────────────────────
    // Category A — GET cases
    // ─────────────────────────────────────────────────────────────────────────────

    // NumPy oracle for each layout (shape before indexing):
    //   CC  = arange(12).reshape(3,4)           → shape (3,4) [[0..3],[4..7],[8..11]]
    //   T   = CC.T                              → shape (4,3) [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
    //   RS  = CC[::2]                           → shape (2,4) [[0..3],[8..11]]
    //   CS  = CC[:, ::2]                        → shape (3,2) [[0,2],[4,6],[8,10]]
    //   NR  = CC[::-1]                          → shape (3,4) [[8..11],[4..7],[0..3]]
    //   NC  = CC[:, ::-1]                       → shape (3,4) [[3,2,1,0],[7,6,5,4],[11,10,9,8]]
    //   SO  = CC[1:3]                           → shape (2,4) [[4..7],[8..11]]
    //   BC  = broadcast_to(arange(4),(3,4))     → shape (3,4) [[0,1,2,3]×3]

    public static IEnumerable<object[]> PassingGetCases()
    {
        // ── CC: C-contiguous (3,4) ──────────────────────────────────────────────
        // np: a[1]          → shape(4) [4,5,6,7]
        yield return G("CC_intidx",
            () => (NDArray)A()[1],
            new[] { 4 }, new long[] { 4, 5, 6, 7 });

        // np: a[1:3]        → shape(2,4) [4..7,8..11]
        yield return G("CC_slice",
            () => (NDArray)A()["1:3"],
            new[] { 2, 4 }, new long[] { 4, 5, 6, 7, 8, 9, 10, 11 });

        // np: a[::-1]       → shape(3,4) [8..11,4..7,0..3]
        yield return G("CC_negstride",
            () => (NDArray)A()["::-1"],
            new[] { 3, 4 }, new long[] { 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3 });

        // np: a[np.array([0,2])]  → shape(2,4) [0..3,8..11]
        yield return G("CC_fancy",
            () => (NDArray)A()[Idx(0L, 2L)],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 8, 9, 10, 11 });

        // np: a[(a%2)==0]   → shape(6) [0,2,4,6,8,10]
        yield return G("CC_bool",
            () => { var a = A(); return (NDArray)a[(a % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 0, 2, 4, 6, 8, 10 });

        // ── T: Transposed (4,3) ─────────────────────────────────────────────────
        // np: T[1]          → shape(3) [1,5,9]
        yield return G("T_intidx",
            () => (NDArray)A().T[1],
            new[] { 3 }, new long[] { 1, 5, 9 });

        // np: T[1:3]        → shape(2,3) [1,5,9,2,6,10]
        yield return G("T_slice",
            () => (NDArray)A().T["1:3"],
            new[] { 2, 3 }, new long[] { 1, 5, 9, 2, 6, 10 });

        // np: T[::-1]       → shape(4,3) [3,7,11,2,6,10,1,5,9,0,4,8]
        yield return G("T_negstride",
            () => (NDArray)A().T["::-1"],
            new[] { 4, 3 }, new long[] { 3, 7, 11, 2, 6, 10, 1, 5, 9, 0, 4, 8 });

        // np: T[(T%2)==0]   → shape(6) [0,4,8,2,6,10]
        yield return G("T_bool",
            () => { var t = A().T; return (NDArray)t[(t % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 0, 4, 8, 2, 6, 10 });

        // ── RS: Row-strided a[::2] → (2,4) [[0..3],[8..11]] ────────────────────
        // np: RS[1]         → shape(4) [8,9,10,11]
        yield return G("RS_intidx",
            () => (NDArray)((NDArray)A()["::2"])[1],
            new[] { 4 }, new long[] { 8, 9, 10, 11 });

        // np: RS[1:3]       → shape(1,4) [8,9,10,11]  (only 2 rows, [1:3] gives row 1)
        yield return G("RS_slice",
            () => (NDArray)((NDArray)A()["::2"])["1:3"],
            new[] { 1, 4 }, new long[] { 8, 9, 10, 11 });

        // np: RS[::-1]      → shape(2,4) [8,9,10,11,0,1,2,3]
        yield return G("RS_negstride",
            () => (NDArray)((NDArray)A()["::2"])["::-1"],
            new[] { 2, 4 }, new long[] { 8, 9, 10, 11, 0, 1, 2, 3 });

        // np: RS[(RS%2)==0] → shape(4) [0,2,8,10]
        yield return G("RS_bool",
            () => { var rs = (NDArray)A()["::2"]; return (NDArray)rs[(rs % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 4 }, new long[] { 0, 2, 8, 10 });

        // ── CS: Col-strided a[:, ::2] → (3,2) [[0,2],[4,6],[8,10]] ─────────────
        // np: CS[1]         → shape(2) [4,6]
        yield return G("CS_intidx",
            () => (NDArray)((NDArray)A()[":", "::2"])[1],
            new[] { 2 }, new long[] { 4, 6 });

        // np: CS[1:3]       → shape(2,2) [4,6,8,10]
        yield return G("CS_slice",
            () => (NDArray)((NDArray)A()[":", "::2"])["1:3"],
            new[] { 2, 2 }, new long[] { 4, 6, 8, 10 });

        // np: CS[::-1]      → shape(3,2) [8,10,4,6,0,2]
        yield return G("CS_negstride",
            () => (NDArray)((NDArray)A()[":", "::2"])["::-1"],
            new[] { 3, 2 }, new long[] { 8, 10, 4, 6, 0, 2 });

        // np: CS[(CS%2)==0] → shape(6) [0,2,4,6,8,10]
        yield return G("CS_bool",
            () => { var cs = (NDArray)A()[":", "::2"]; return (NDArray)cs[(cs % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 0, 2, 4, 6, 8, 10 });

        // ── NR: Neg-row a[::-1] → (3,4) [[8..11],[4..7],[0..3]] ─────────────────
        // np: NR[1]         → shape(4) [4,5,6,7]
        yield return G("NR_intidx",
            () => (NDArray)((NDArray)A()["::-1"])[1],
            new[] { 4 }, new long[] { 4, 5, 6, 7 });

        // np: NR[1:3]       → shape(2,4) [4,5,6,7,0,1,2,3]
        yield return G("NR_slice",
            () => (NDArray)((NDArray)A()["::-1"])["1:3"],
            new[] { 2, 4 }, new long[] { 4, 5, 6, 7, 0, 1, 2, 3 });

        // np: NR[::-1]      → shape(3,4) [0,1,2,3,4,5,6,7,8,9,10,11]
        yield return G("NR_negstride",
            () => (NDArray)((NDArray)A()["::-1"])["::-1"],
            new[] { 3, 4 }, new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });

        // np: NR[(NR%2)==0] → shape(6) [8,10,4,6,0,2]
        yield return G("NR_bool",
            () => { var nr = (NDArray)A()["::-1"]; return (NDArray)nr[(nr % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 8, 10, 4, 6, 0, 2 });

        // ── NC: Neg-col a[:, ::-1] → (3,4) [[3,2,1,0],[7,6,5,4],[11,10,9,8]] ───
        // np: NC[1]         → shape(4) [7,6,5,4]
        yield return G("NC_intidx",
            () => (NDArray)((NDArray)A()[":", "::-1"])[1],
            new[] { 4 }, new long[] { 7, 6, 5, 4 });

        // np: NC[1:3]       → shape(2,4) [7,6,5,4,11,10,9,8]
        yield return G("NC_slice",
            () => (NDArray)((NDArray)A()[":", "::-1"])["1:3"],
            new[] { 2, 4 }, new long[] { 7, 6, 5, 4, 11, 10, 9, 8 });

        // np: NC[::-1]      → shape(3,4) [11,10,9,8,7,6,5,4,3,2,1,0]
        yield return G("NC_negstride",
            () => (NDArray)((NDArray)A()[":", "::-1"])["::-1"],
            new[] { 3, 4 }, new long[] { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });

        // np: NC[(NC%2)==0] → shape(6) [2,0,6,4,10,8]
        yield return G("NC_bool",
            () => { var nc = (NDArray)A()[":", "::-1"]; return (NDArray)nc[(nc % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 2, 0, 6, 4, 10, 8 });

        // ── SO: Sliced-offset a[1:3] → (2,4) [[4..7],[8..11]] ──────────────────
        // np: SO[1]         → shape(4) [8,9,10,11]
        yield return G("SO_intidx",
            () => (NDArray)((NDArray)A()["1:3"])[1],
            new[] { 4 }, new long[] { 8, 9, 10, 11 });

        // np: SO[1:3]       → shape(1,4) [8,9,10,11]  (only 2 rows, [1:3] gives row 1)
        yield return G("SO_slice",
            () => (NDArray)((NDArray)A()["1:3"])["1:3"],
            new[] { 1, 4 }, new long[] { 8, 9, 10, 11 });

        // np: SO[::-1]      → shape(2,4) [8,9,10,11,4,5,6,7]
        yield return G("SO_negstride",
            () => (NDArray)((NDArray)A()["1:3"])["::-1"],
            new[] { 2, 4 }, new long[] { 8, 9, 10, 11, 4, 5, 6, 7 });

        // np: SO[np.array([0,1])] → shape(2,4) [4,5,6,7,8,9,10,11]
        // NOTE: row-fancy on a sliced-offset view PASSES (unlike other strided layouts).
        yield return G("SO_fancy",
            () => (NDArray)((NDArray)A()["1:3"])[Idx(0L, 1L)],
            new[] { 2, 4 }, new long[] { 4, 5, 6, 7, 8, 9, 10, 11 });

        // np: SO[(SO%2)==0] → shape(4) [4,6,8,10]
        yield return G("SO_bool",
            () => { var so = (NDArray)A()["1:3"]; return (NDArray)so[(so % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 4 }, new long[] { 4, 6, 8, 10 });

        // ── BC: Broadcast broadcast_to(arange(4),(3,4)) ─────────────────────────
        // np: BC[1]         → shape(4) [0,1,2,3]
        yield return G("BC_intidx",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))[1],
            new[] { 4 }, new long[] { 0, 1, 2, 3 });

        // np: BC[1:3]       → shape(2,4) [0,1,2,3,0,1,2,3]
        yield return G("BC_slice",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))["1:3"],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 0, 1, 2, 3 });

        // np: BC[::-1]      → shape(3,4) [0,1,2,3,0,1,2,3,0,1,2,3]
        yield return G("BC_negstride",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))["::-1"],
            new[] { 3, 4 }, new long[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });

        // np: BC[np.array([0,2])] → shape(2,4) [0,1,2,3,0,1,2,3]
        // NOTE: row-fancy on broadcast view PASSES.
        yield return G("BC_fancy",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))[Idx(0L, 2L)],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 0, 1, 2, 3 });

        // np: BC[(BC%2)==0] → shape(6) [0,2,0,2,0,2]
        yield return G("BC_bool",
            () => { var bc = np.broadcast_to(np.arange(4L), new Shape(3, 4)); return (NDArray)bc[(bc % 2 == 0).MakeGeneric<bool>()]; },
            new[] { 6 }, new long[] { 0, 2, 0, 2, 0, 2 });
    }

    [DataTestMethod]
    [DynamicData(nameof(PassingGetCases), DynamicDataSourceType.Method)]
    public void GetCase_Passes(GCase tc)
    {
        var result = tc.Op();
        AssertShape(tc, result);
        CollectionAssert.AreEqual(tc.ExpVals, result.ravel().ToArray<long>(),
            $"{tc.Name}: values mismatch");
    }

    // ─── Fancy-index-on-view GET cases (feeder) ──────────────────────────────────
    // Every fancy-on-view form below matches NumPy 2.4.2 (probed). The 5 single-fancy-
    // on-non-contiguous-view cases (T_fancy/RS_fancy/CS_fancy/NR_fancy/NC_fancy) were once
    // [OpenBugs] — FetchIndicesNDNonLinear mis-sized the result and assigned a whole
    // sub-array into a scalar slot, throwing a shape/storage error — and are now fixed
    // (the gather walks the source strides element-by-element). All asserted passing.
    public static IEnumerable<object[]> FancyOnViewGetCases()
    {
        // ── CC bugs: col-fancy and row-fancy+colon ───────────────────────────────
        // np: a[:, np.array([0,2])] → shape(3,2) [0,2,4,6,8,10]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("CC_col_fancy",
            () => (NDArray)A()[":", Idx(0L, 2L)],
            new[] { 3, 2 }, new long[] { 0, 2, 4, 6, 8, 10 });

        // np: a[np.array([0,2]), :] → shape(2,4) [0,1,2,3,8,9,10,11]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("CC_row_fancy_colon",
            () => (NDArray)A()[Idx(0L, 2L), ":"],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 8, 9, 10, 11 });

        // ── T bugs: fancy, col-fancy, row-fancy+colon ───────────────────────────
        // np: T[np.array([0,2])] → shape(2,3) [0,4,8,2,6,10]
        // NumSharp: "Given shape size does not match the size of the given storage size"
        yield return G("T_fancy",
            () => (NDArray)A().T[Idx(0L, 2L)],
            new[] { 2, 3 }, new long[] { 0, 4, 8, 2, 6, 10 });

        // np: T[:, np.array([0,2])] → shape(4,2) [0,8,1,9,2,10,3,11]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("T_col_fancy",
            () => (NDArray)A().T[":", Idx(0L, 2L)],
            new[] { 4, 2 }, new long[] { 0, 8, 1, 9, 2, 10, 3, 11 });

        // np: T[np.array([0,2]), :] → shape(2,3) [0,4,8,2,6,10]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("T_row_fancy_colon",
            () => (NDArray)A().T[Idx(0L, 2L), ":"],
            new[] { 2, 3 }, new long[] { 0, 4, 8, 2, 6, 10 });

        // ── RS bugs: fancy, col-fancy, row-fancy+colon ──────────────────────────
        // np: RS[np.array([0,1])] → shape(2,4) [0,1,2,3,8,9,10,11]
        // NumSharp: "Can't SetData to a from a shape of (4) to the target indices"
        yield return G("RS_fancy",
            () => (NDArray)((NDArray)A()["::2"])[Idx(0L, 1L)],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 8, 9, 10, 11 });

        // np: RS[:, np.array([0,2])] → shape(2,2) [0,2,8,10]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("RS_col_fancy",
            () => (NDArray)((NDArray)A()["::2"])[":", Idx(0L, 2L)],
            new[] { 2, 2 }, new long[] { 0, 2, 8, 10 });

        // np: RS[np.array([0,1]), :] → shape(2,4) [0,1,2,3,8,9,10,11]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("RS_row_fancy_colon",
            () => (NDArray)((NDArray)A()["::2"])[Idx(0L, 1L), ":"],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 8, 9, 10, 11 });

        // ── CS bugs: fancy, col-fancy, row-fancy+colon ──────────────────────────
        // np: CS[np.array([0,2])] → shape(2,2) [0,2,8,10]
        // NumSharp: "Given shape size does not match the size of the given storage size"
        yield return G("CS_fancy",
            () => (NDArray)((NDArray)A()[":", "::2"])[Idx(0L, 2L)],
            new[] { 2, 2 }, new long[] { 0, 2, 8, 10 });

        // np: CS[:, np.array([0,1])] → shape(3,2) [0,2,4,6,8,10]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("CS_col_fancy",
            () => (NDArray)((NDArray)A()[":", "::2"])[":", Idx(0L, 1L)],
            new[] { 3, 2 }, new long[] { 0, 2, 4, 6, 8, 10 });

        // np: CS[np.array([0,2]), :] → shape(2,2) [0,2,8,10]
        // NumSharp: returns wrong shape(2) [0,10] instead of shape(2,2)
        yield return G("CS_row_fancy_colon",
            () => (NDArray)((NDArray)A()[":", "::2"])[Idx(0L, 2L), ":"],
            new[] { 2, 2 }, new long[] { 0, 2, 8, 10 });

        // ── NR bugs: fancy, col-fancy, row-fancy+colon ──────────────────────────
        // np: NR[np.array([0,2])] → shape(2,4) [8,9,10,11,0,1,2,3]
        // NumSharp: "Can't SetData to a from a shape of (4) to the target indices"
        yield return G("NR_fancy",
            () => (NDArray)((NDArray)A()["::-1"])[Idx(0L, 2L)],
            new[] { 2, 4 }, new long[] { 8, 9, 10, 11, 0, 1, 2, 3 });

        // np: NR[:, np.array([0,2])] → shape(3,2) [8,10,4,6,0,2]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("NR_col_fancy",
            () => (NDArray)((NDArray)A()["::-1"])[":", Idx(0L, 2L)],
            new[] { 3, 2 }, new long[] { 8, 10, 4, 6, 0, 2 });

        // np: NR[np.array([0,2]), :] → shape(2,4) [8,9,10,11,0,1,2,3]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("NR_row_fancy_colon",
            () => (NDArray)((NDArray)A()["::-1"])[Idx(0L, 2L), ":"],
            new[] { 2, 4 }, new long[] { 8, 9, 10, 11, 0, 1, 2, 3 });

        // ── NC bugs: fancy, col-fancy, row-fancy+colon ──────────────────────────
        // np: NC[np.array([0,2])] → shape(2,4) [3,2,1,0,11,10,9,8]
        // NumSharp: "Given shape size does not match the size of the given storage size"
        yield return G("NC_fancy",
            () => (NDArray)((NDArray)A()[":", "::-1"])[Idx(0L, 2L)],
            new[] { 2, 4 }, new long[] { 3, 2, 1, 0, 11, 10, 9, 8 });

        // np: NC[:, np.array([0,2])] → shape(3,2) [3,1,7,5,11,9]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("NC_col_fancy",
            () => (NDArray)((NDArray)A()[":", "::-1"])[":", Idx(0L, 2L)],
            new[] { 3, 2 }, new long[] { 3, 1, 7, 5, 11, 9 });

        // np: NC[np.array([0,2]), :] → shape(2,4) [3,2,1,0,11,10,9,8]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("NC_row_fancy_colon",
            () => (NDArray)((NDArray)A()[":", "::-1"])[Idx(0L, 2L), ":"],
            new[] { 2, 4 }, new long[] { 3, 2, 1, 0, 11, 10, 9, 8 });

        // ── SO bugs: col-fancy and row-fancy+colon ──────────────────────────────
        // (SO_fancy with Idx(0,1) PASSES — only the colon combos fail)
        // np: SO[:, np.array([0,2])] → shape(2,2) [4,6,8,10]
        // NumSharp: returns wrong shape(2) [4,10] instead of shape(2,2)
        yield return G("SO_col_fancy",
            () => (NDArray)((NDArray)A()["1:3"])[":", Idx(0L, 2L)],
            new[] { 2, 2 }, new long[] { 4, 6, 8, 10 });

        // np: SO[np.array([0,1]), :] → shape(2,4) [4,5,6,7,8,9,10,11]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("SO_row_fancy_colon",
            () => (NDArray)((NDArray)A()["1:3"])[Idx(0L, 1L), ":"],
            new[] { 2, 4 }, new long[] { 4, 5, 6, 7, 8, 9, 10, 11 });

        // ── BC bugs: col-fancy and row-fancy+colon ──────────────────────────────
        // np: BC[:, np.array([0,2])] → shape(3,2) [0,2,0,2,0,2]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("BC_col_fancy",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))[":", Idx(0L, 2L)],
            new[] { 3, 2 }, new long[] { 0, 2, 0, 2, 0, 2 });

        // np: BC[np.array([0,2]), :] → shape(2,4) [0,1,2,3,0,1,2,3]
        // NumSharp: IncorrectShapeException "shape mismatch"
        yield return G("BC_row_fancy_colon",
            () => (NDArray)np.broadcast_to(np.arange(4L), new Shape(3, 4))[Idx(0L, 2L), ":"],
            new[] { 2, 4 }, new long[] { 0, 1, 2, 3, 0, 1, 2, 3 });
    }

    // Every fancy-on-view form (single fancy, [":", Idx], [Idx, ":"]) on every layout
    // now matches NumPy — including the single-fancy-on-strided-view cases that were
    // [OpenBugs] (T/RS/CS/NR/NC _fancy), fixed by the FetchIndicesNDNonLinear rewrite.
    [DataTestMethod]
    [DynamicData(nameof(FancyOnViewGetCases), DynamicDataSourceType.Method)]
    public void FancyOnView_Passes(GCase tc)
    {
        var result = tc.Op();
        AssertShape(tc, result);
        CollectionAssert.AreEqual(tc.ExpVals, result.ravel().ToArray<long>(),
            $"{tc.Name}: values mismatch");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Category B — SET cases
    // ─────────────────────────────────────────────────────────────────────────────

    public static IEnumerable<object[]> PassingSetCases()
    {
        // B1: integer-index set — a[1, 0] = 99 → position 4 becomes 99
        // np: a[1, 0] = 99; a.ravel() == [0,1,2,3,99,5,6,7,8,9,10,11]
        yield return S("B1_int_index_set", () =>
        {
            var a = A(); a[1, 0] = 99L; return a.ravel();
        }, new long[] { 0, 1, 2, 3, 99, 5, 6, 7, 8, 9, 10, 11 });

        // B2: slice set — a["1:2", ":"] = 99 → all of row 1 becomes 99
        // np: a[1:2, :] = 99; a.ravel() == [0,1,2,3,99,99,99,99,8,9,10,11]
        yield return S("B2_slice_set", () =>
        {
            var a = A(); a["1:2", ":"] = 99L; return a.ravel();
        }, new long[] { 0, 1, 2, 3, 99, 99, 99, 99, 8, 9, 10, 11 });

        // B3: bool mask set — a[a%2==0] = 99 → even positions become 99
        // np: mask = a%2==0; a[mask] = 99; a.ravel() == [99,1,99,3,99,5,99,7,99,9,99,11]
        yield return S("B3_bool_mask_set", () =>
        {
            var a = A();
            a[(a % 2 == 0).MakeGeneric<bool>()] = 99L;
            return a.ravel();
        }, new long[] { 99, 1, 99, 3, 99, 5, 99, 7, 99, 9, 99, 11 });

        // B4: negative-stride view set — a[::-1][0] = 99 modifies the LAST row of a
        // np: a[::-1][0] = 99; a[2,:] == [99,99,99,99]
        yield return S("B4_negstride_view_set", () =>
        {
            var a = A();
            ((NDArray)a["::-1"])[0] = 99L;
            return a.ravel();
        }, new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99, 99 });

        // B5: scalar broadcast into a row slice
        // np: a[2:3] = 0; a[2,:] == [0,0,0,0]
        yield return S("B5_scalar_bcast_row_set", () =>
        {
            var a = A(); a["2:3"] = 0L; return a.ravel();
        }, new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0 });

        // B6: float value truncates to integer (not rounded) matching NumPy same_kind cast
        // np: a = arange(6).reshape(2,3); a[0] = 3.9; a.ravel() == [3,3,3,3,4,5]
        yield return S("B6_float_truncates_to_int", () =>
        {
            var a = np.arange(6L).reshape(2, 3);
            a[0] = 3.9;
            return a.ravel();
        }, new long[] { 3, 3, 3, 3, 4, 5 });
    }

    [DataTestMethod]
    [DynamicData(nameof(PassingSetCases), DynamicDataSourceType.Method)]
    public void SetCase_Passes(SCase tc)
    {
        CollectionAssert.AreEqual(tc.ExpVals, tc.Op().ToArray<long>(),
            $"{tc.Name}: ravel mismatch");
    }

    // ─── Exception SET cases ─────────────────────────────────────────────────────

    public static IEnumerable<object[]> ThrowingSetCases()
    {
        // BE1: writing to a broadcast-view must raise (IsWriteable=false)
        // np: a.flags.writeable == False → ValueError: assignment destination is read-only
        yield return E("BE1_broadcast_readonly_write",
            () =>
            {
                var b = np.broadcast_to(np.arange(4L), new Shape(3, 4));
                b[0] = 99L;
            });

        // BE2: shape mismatch on set — assigning 5-element array to a 4-element row
        // np: a[0] = np.arange(5) → ValueError: could not broadcast input array from shape (5,) into shape (4,)
        yield return E("BE2_shape_mismatch_on_set",
            () =>
            {
                var a = A();
                a[0] = np.arange(5L);
            });
    }

    [DataTestMethod]
    [DynamicData(nameof(ThrowingSetCases), DynamicDataSourceType.Method)]
    public void SetCase_Throws(ECase tc)
    {
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw, $"{tc.Name}: expected an exception to be thrown but none was raised");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Category C — COPY vs VIEW contract
    // ─────────────────────────────────────────────────────────────────────────────

    // C1: basic slice is a VIEW — mutation propagates to the base array.
    // np: v = a[1:3]; v[0,0] = 999; a[1,0] == 999
    [TestMethod]
    public void C1_BasicSliceIsView()
    {
        var a = A();
        var view = (NDArray)a["1:3"];
        view[0, 0] = 999L;
        Assert.AreEqual(999L, a.GetInt64(1, 0),
            "a[1:3] must be a VIEW — modifying view[0,0] must update a[1,0]");
    }

    // C2: negative-column-stride slice is a VIEW.
    // np: v = a[:, ::-1]; v[0,0] = 999; a[0,3] == 999  (v[0,0] aliases a[0, last])
    [TestMethod]
    public void C2_NegColSliceIsView()
    {
        var a = A();
        var view = (NDArray)a[":", "::-1"];
        view[0, 0] = 999L;   // view[row=0, col=0] aliases a[row=0, col=3]
        Assert.AreEqual(999L, a.GetInt64(0, 3),
            "a[:, ::-1] must be a VIEW — modifying view[0,0] must update a[0,3]");
    }

    // C3: fancy indexing returns a COPY — mutation of result must not affect base.
    // np: c = a[[0,2]]; c[0,0] = 999; a[0,0] == 0
    [TestMethod]
    public void C3_FancyIndexIsCopy()
    {
        var a = A();
        var copy = (NDArray)a[Idx(0L, 2L)];
        copy[0, 0] = 999L;
        Assert.AreEqual(0L, a.GetInt64(0, 0),
            "Fancy indexing must return a COPY — base must remain unmodified");
    }

    // C4: boolean-mask indexing returns a COPY — mutation of result must not affect base.
    // np: m = a%2==0; c = a[m]; c[0] = 999; a[0,0] == 0
    [TestMethod]
    public void C4_BoolMaskIsCopy()
    {
        var a = A();
        var mask = (a % 2 == 0).MakeGeneric<bool>();
        var copy = (NDArray)a[mask];
        copy[0] = 999L;
        Assert.AreEqual(0L, a.GetInt64(0, 0),
            "Boolean-mask indexing must return a COPY — base must remain unmodified");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // DynamicData row factories
    // ─────────────────────────────────────────────────────────────────────────────

    static object[] G(string name, Func<NDArray> op, int[] shape, long[] vals)
        => new object[] { new GCase(name, op, shape, vals) };

    static object[] S(string name, Func<NDArray> op, long[] vals)
        => new object[] { new SCase(name, op, vals) };

    static object[] E(string name, Action op)
        => new object[] { new ECase(name, op) };

    // ─── Shape assertion helper ──────────────────────────────────────────────────
    // nd.shape returns long[] in NumSharp, so we compare element-by-element
    // as longs to avoid int vs long boxing-equality mismatch in CollectionAssert.
    static void AssertShape(GCase tc, NDArray result)
    {
        var actual = result.shape;  // long[]
        Assert.AreEqual(tc.ExpShape.Length, actual.Length,
            $"{tc.Name}: ndim mismatch — expected {tc.ExpShape.Length}D got {actual.Length}D");
        for (int i = 0; i < tc.ExpShape.Length; i++)
            Assert.AreEqual((long)tc.ExpShape[i], actual[i],
                $"{tc.Name}: shape[{i}] expected {tc.ExpShape[i]} got {actual[i]}");
    }
}
