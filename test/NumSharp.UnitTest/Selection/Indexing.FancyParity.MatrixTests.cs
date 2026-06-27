using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Differential parity tests for fancy (integer-array) indexing — pure
    /// advanced-index cases only (no boolean masks, no slice mix-in that isn't
    /// already covered by IndexingProbeMatrix c01–c50).
    ///
    /// Every expected shape / value was obtained by running the equivalent
    /// expression in NumPy 2.4.2 (the sole oracle). Covers:
    ///   • Single 1-D index array on V/A/B/B4 (ascending, descending, repeated,
    ///     single-element, negative wrap, mixed sign)
    ///   • 2-D and 3-D index arrays on A
    ///   • Empty index arrays
    ///   • The "decision-B equivalence": raw int[]/long[] as the sole index ==
    ///     np.array fancy (decision B, committed in nditer); coordinate-shim
    ///     GetData(coords) proves they differ
    ///   • Multiple integer arrays (pure-advanced): equal-length 1-D, 1-D vs 0-D
    ///     scalar-array broadcast, 1-D vs 2-D broadcast, all-axes selection
    ///   • Broadcast-mismatch errors (shapes that cannot broadcast)
    ///   • OOB / negative-beyond-axis errors
    ///   • Setters: matched-shape value (works), 1-D fancy scalar-set (works),
    ///     OOB set (throws)
    ///   • Copy semantics: fancy GET returns a fresh copy
    ///   • [OpenBugs] bucket: scalar / lower-rank broadcastable value assigned
    ///     into a ≥2-D subshape (the known broadcast-value fancy-set gap)
    /// </summary>
    [TestClass]
    public class Indexing_FancyParity_MatrixTests
    {
        // ── base-array factories ─────────────────────────────────────────────────
        private static NDArray V()   => np.arange(6, dtype: np.int64);                        // shape (6,)
        private static NDArray A()   => np.arange(12, dtype: np.int64).reshape(3, 4);         // shape (3,4)
        private static NDArray B()   => np.arange(24, dtype: np.int64).reshape(2, 3, 4);      // shape (2,3,4)
        private static NDArray B4()  => np.arange(120, dtype: np.int64).reshape(2, 3, 4, 5);  // shape (2,3,4,5)

        // ── index-array helpers ──────────────────────────────────────────────────
        private static NDArray IA()     => np.array(new long[] { 0, 1 });         // [0,1]
        private static NDArray IB()     => np.array(new long[] { 0, 2 });         // [0,2]
        private static NDArray IA2D()   => np.array(new long[] { 0, 1, 1, 0 }).reshape(2, 2);          // (2,2)
        private static NDArray IA3D()   => np.array(new long[] { 0, 1, 1, 0, 1, 0, 0, 1 }).reshape(2, 2, 2); // (2,2,2)
        private static NDArray IALONG() => np.array(new long[] { 1, 1, 1 });      // [1,1,1]  – repeated
        private static NDArray INEG()   => np.array(new long[] { -1, -2 });       // [-1,-2]
        private static NDArray IMIX()   => np.array(new long[] { 0, -1, 2 });     // [0,-1,2]
        private static NDArray ISINGLE()=> np.array(new long[] { 2 });            // [2]
        private static NDArray IREV()   => np.array(new long[] { 5, 4, 3, 2, 1, 0 }); // full reverse of 6-elem

        // ── case records ────────────────────────────────────────────────────────
        public sealed record GCase(string Name, Func<NDArray> Op, int[] Shape, long[] Vals);
        public sealed record ECase(string Name, Func<NDArray> Op);
        public sealed record SCase(string Name, Func<NDArray> Op, long[] Vals);

        // ── AssertGet helper ─────────────────────────────────────────────────────
        private static void AssertGet(GCase c)
        {
            var r = c.Op();
            r.shape.Select(x => (int)x).ToArray().Should().Equal(c.Shape, $"{c.Name} shape");
            if (c.Vals == null) return;       // shape-only (size > 64)
            var flat = new long[r.size];
            var f = r.flat;
            for (long i = 0; i < r.size; i++) flat[i] = Convert.ToInt64(f.GetValue(i));
            flat.Should().Equal(c.Vals, $"{c.Name} values");
        }

        // ═══════════════════ SINGLE 1-D FANCY ON V (arange(6)) ═════════════════
        // NumPy 2.4.2 probed for every case.

        // ── GET cases ────────────────────────────────────────────────────────────
        private static readonly GCase[] FancyGet =
        {
            // ─── single 1-D on V ───
            new("g01_V_asc",    () => V()[IA()],     new[]{2},    new long[]{0,1}),
            new("g02_V_desc",   () => V()[IB()["1::-1"]],  new[]{2},    new long[]{2,0}),   // [2,0] via step-slice on IB
            new("g03_V_rep",    () => V()[IALONG()], new[]{3},    new long[]{1,1,1}),
            new("g04_V_single", () => V()[ISINGLE()],new[]{1},    new long[]{2}),
            new("g05_V_neg",    () => V()[INEG()],   new[]{2},    new long[]{5,4}),
            new("g06_V_mix",    () => V()[IMIX()],   new[]{3},    new long[]{0,5,2}),
            new("g07_V_rev",    () => V()[IREV()],   new[]{6},    new long[]{5,4,3,2,1,0}),

            // ─── single 1-D on A (3,4) ───
            new("g08_A_ia",     () => A()[IA()],     new[]{2,4},  new long[]{0,1,2,3,4,5,6,7}),
            new("g09_A_ib",     () => A()[IB()],     new[]{2,4},  new long[]{0,1,2,3,8,9,10,11}),
            new("g10_A_rep",    () => A()[IALONG()], new[]{3,4},  new long[]{4,5,6,7,4,5,6,7,4,5,6,7}),
            new("g11_A_single", () => A()[ISINGLE()],new[]{1,4},  new long[]{8,9,10,11}),
            new("g12_A_neg",    () => A()[INEG()],   new[]{2,4},  new long[]{8,9,10,11,4,5,6,7}),
            new("g13_A_mix",    () => A()[IMIX()],   new[]{3,4},  new long[]{0,1,2,3,8,9,10,11,8,9,10,11}),
            new("g14_A_desc",   () => A()[IB()["1::-1"]],  new[]{2,4},  new long[]{8,9,10,11,0,1,2,3}),

            // ─── single 1-D on B (2,3,4) ───
            // B[IA()] selects batch items 0 and 1 → shape (2,3,4), size=24 → shape-only
            new("g15_B_ia",    () => B()[IA()],     new[]{2,3,4}, null),
            // B[INEG()] = B[[-1,-2]] → items 1,0 (neg wrap) → shape (2,3,4)
            new("g16_B_neg",   () => B()[INEG()],   new[]{2,3,4},
                new long[]{12,13,14,15,16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11}),

            // ─── 2-D index array on A ───
            // A[IA2D()] where IA2D is (2,2): result shape = (2,2,4)
            new("g17_A_ia2d",  () => A()[IA2D()],  new[]{2,2,4},
                new long[]{0,1,2,3,4,5,6,7,4,5,6,7,0,1,2,3}),

            // ─── 3-D index array on A ───
            // A[IA3D()] where IA3D is (2,2,2): result shape = (2,2,2,4), size=32 → shape-only
            new("g18_A_ia3d",  () => A()[IA3D()],  new[]{2,2,2,4}, null),

            // ─── empty index arrays ───
            new("g19_V_empty", () => V()[np.array(new long[]{})], new[]{0},   new long[]{}),
            new("g20_A_empty", () => A()[np.array(new long[]{})], new[]{0,4}, new long[]{}),

            // ─── 2-D index array on B → (2,2,3,4) ───
            // B[IA2D()] where IA2D is (2,2): result = (2,2,3,4), size=48 → shape-only
            new("g21_B_ia2d",  () => B()[IA2D()],  new[]{2,2,3,4}, null),

            // ─── 0-D array index (scalar array) ───
            // B4()[np.array(1L)] → same as B4()[1] → shape (3,4,5), i.e. drops the indexed axis
            new("g22_A_0d",    () => A()[np.array(new long[]{1}).reshape()], new[]{4}, new long[]{4,5,6,7}),

            // ─── multiple integer arrays (pure-advanced) ───
            // A[IA(), IB()]: point-select at (0,0) and (1,2) → shape (2,)
            new("g23_A_ia_ib",   () => A()[IA(), IB()],      new[]{2},       new long[]{0,6}),

            // B[IA(), IB()]: axis-0=[0,1], axis-1=[0,2] → rows B[0,:] and B[1,2,:] → (2,4)
            new("g24_B_ia_ib",   () => B()[IA(), IB()],      new[]{2,4},
                new long[]{0,1,2,3,20,21,22,23}),

            // B[IA(), IA(), IB()]: all 3 axes, equal-len 1-D → shape (2,)
            new("g25_B_ia_ia_ib",() => B()[IA(), IA(), IB()], new[]{2},     new long[]{0,18}),

            // B[IA(), 0-D scalar array]: 1-D vs 0-D broadcast → axis-1 selects 0 → (2,4)
            new("g26_B_ia_0d",   () => B()[IA(), np.array(new long[]{0}).reshape()], new[]{2,4},
                new long[]{0,1,2,3,12,13,14,15}),

            // B[IA2D(), IB()]: (2,2) × (2,) broadcast → result (2,2,4)
            new("g27_B_ia2d_ib", () => B()[IA2D(), IB()],    new[]{2,2,4},
                new long[]{0,1,2,3,20,21,22,23,12,13,14,15,8,9,10,11}),

            // NOTE: g28_B_ia_ib2d moved to FancyGetBug — NumSharp gives (2,1,4) instead of (2,2,4)

            // B[IA2D(), IA()]: (2,2) × (2,) → (2,2,4)
            new("g29_B_ia2d_ia", () => B()[IA2D(), IA()],    new[]{2,2,4},
                new long[]{0,1,2,3,16,17,18,19,12,13,14,15,4,5,6,7}),

            // B4()[IA(), IB()]: (2,) on axis 0 and (2,) on axis 1 → (2,4,5), size=40 → shape-only
            new("g30_B4_ia_ib",  () => B4()[IA(), IB()],     new[]{2,4,5},  null),

            // ─── decision-B equivalence: raw long[]/int[] == np.array fancy ───
            // A[new long[]{0,2}] must select rows 0 and 2 (fancy), NOT element at (0,2).
            // NumPy: shape (2,4), vals [0..3, 8..11].
            new("g31_decB_raw_long", () => A()[new long[]{0,2}], new[]{2,4},
                new long[]{0,1,2,3,8,9,10,11}),
            new("g32_decB_raw_int",  () => A()[new int[] {0,2}], new[]{2,4},
                new long[]{0,1,2,3,8,9,10,11}),

            // ─── coordinate shim differs from fancy ───
            // A.GetData(new int[]{0,2}) accesses the element at coord (0,2) = scalar 2,
            // shape () / scalar. Proves GetData stays coordinate; only nd[array] is fancy.
            // NumPy a[0,2] == 2 (scalar); NumSharp GetData returns 0-D NDArray with value 2.
            new("g33_coord_shim",    () => A().GetData(new int[]{0,2}), new int[]{}, new long[]{2}),

            // ─── negative wrap in multi-axis ───
            // B[[-1], [-1]]: single-element 1-D on axis 0 (→ row 1) and axis 1 (→ col 2)
            // → shape (1,4), vals = B[1,2,:] = [20,21,22,23]
            new("g34_B_neg_ia_ib",   () => B()[np.array(new long[]{-1}), np.array(new long[]{-1})],
                new[]{1,4}, new long[]{20,21,22,23}),

            // ─── empty multi-index → shape (0, trailing_dims) ───
            // B[empty, empty] on axes 0 and 1: broadcasting () × () → (0,), remaining axis 4-wide
            new("g35_B_empty_multi", () => B()[np.array(new long[]{}), np.array(new long[]{})],
                new[]{0,4}, new long[]{}),
        };

        // ── ERROR cases ──────────────────────────────────────────────────────────
        private static readonly ECase[] FancyGetErr =
        {
            // OOB positive: index ≥ axis length — NumSharp raises
            new("e01_V_oob_pos", () => V()[np.array(new long[]{6})]),
            new("e03_A_oob_pos", () => A()[np.array(new long[]{3})]),
            // Broadcast mismatch: (2,) vs (3,) — NumSharp raises
            new("e05_mismatch",  () => B()[np.array(new long[]{0,1}), np.array(new long[]{0,1,2})]),
        };

        // ── FIXED: negative OOB now validated (was [OpenBugs]) ───────────────
        // Was a memory-safety bug: PrepareIndexGetters wrapped a too-negative fancy
        // index (e.g. -7 on size 6 -> -1) and left a NEGATIVE offset that only the
        // upper-bound (> largestOffset) check guarded, so the gather READ OUT-OF-BOUNDS
        // memory (returned garbage). Fixed in PrepareIndexGetters: each fancy index is
        // validated to [-dim, dim-1] per axis, raising IndexError "index N is out of
        // bounds for axis A with size S" (NumPy-verbatim). Kept as a regression guard.
        private static readonly ECase[] FancyGetErrBug =
        {
            new("e02_V_oob_neg", () => V()[np.array(new long[]{-7})]),   // -7 < -6 → throws
            new("e04_A_oob_neg", () => A()[np.array(new long[]{-4})]),   // -4 < -3 → throws
        };

        // ═══════════════════ FANCY SET ══════════════════════════════════════════
        //
        // SCase: Op mutates the base array and RETURNS it; Vals = base.ravel() after.
        // NumPy is the oracle for every expected value.

        private static readonly SCase[] FancySet =
        {
            // ─── 1-D fancy on V with scalar (1-D result → scalar broadcast works) ───
            new("s01_V_scalar_set",
                () => { var v = V(); v[np.array(new long[]{1,3,5})] = (NDArray)(-1L); return v; },
                new long[]{0,-1,2,-1,4,-1}),

            // ─── 1-D fancy on V with matched 1-D value ───
            new("s02_V_matched_set",
                () => { var v = V(); v[np.array(new long[]{1,3,5})] = np.array(new long[]{10,20,30}); return v; },
                new long[]{0,10,2,20,4,30}),

            // ─── A[IA] = matched (2,4) value ───
            // A()[IA()] selects rows 0 and 1 (shape 2,4); assign exact shape.
            new("s03_A_ia_matched",
                () => { var a = A(); a[IA()] = np.arange(100, 108, dtype: np.int64).reshape(2, 4); return a; },
                new long[]{100,101,102,103,104,105,106,107,8,9,10,11}),

            // ─── A[IB] = matched (2,4) value ───
            // A()[IB()] selects rows 0 and 2.
            new("s04_A_ib_matched",
                () => { var a = A(); a[IB()] = np.arange(200, 208, dtype: np.int64).reshape(2, 4); return a; },
                new long[]{200,201,202,203,4,5,6,7,204,205,206,207}),

            // ─── multi-fancy set: B[IA, IB] = matched (2,4) ───
            new("s05_B_ia_ib_matched",
                () => { var b = B(); b[IA(), IB()] = np.arange(100, 108, dtype: np.int64).reshape(2, 4); return b; },
                new long[]{100,101,102,103,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,104,105,106,107}),

            // ─── all-axes fancy: B[IA, IA, IB] = matched (2,) ───
            // Point-assign at (0,0,0)=100 and (1,1,2)=200.
            new("s06_B_allfancy_matched",
                () => { var b = B(); b[IA(), IA(), IB()] = np.array(new long[]{100,200}); return b; },
                new long[]{100,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,200,19,20,21,22,23}),

            // ─── raw int[] set (decision-B: raw int[] is FANCY, not coordinate) ───
            // A[new int[]{0,2}] = matched → rows 0 and 2 updated.
            new("s07_decB_raw_int_set",
                () => { var a = A(); a[new int[]{0,2}] = np.arange(100, 108, dtype: np.int64).reshape(2, 4); return a; },
                new long[]{100,101,102,103,4,5,6,7,104,105,106,107}),

            // ─── raw long[] set (decision-B) ───
            new("s08_decB_raw_long_set",
                () => { var a = A(); a[new long[]{0,2}] = np.arange(200, 208, dtype: np.int64).reshape(2, 4); return a; },
                new long[]{200,201,202,203,4,5,6,7,204,205,206,207}),

            // ─── multi-fancy with BROADCAST index shapes (shared fix with g28) ───
            // B[ia.reshape(2,1), ib] broadcasts (2,1)×(2,)->(2,2); matched (2,2,4) value.
            new("s09_B_bcast_idx_matched",
                () => { var b = B(); b[IA().reshape(2,1), IB()] = np.arange(100, 116, dtype: np.int64).reshape(2,2,4); return b; },
                new long[]{100,101,102,103,4,5,6,7,104,105,106,107,108,109,110,111,16,17,18,19,112,113,114,115}),
        };

        // ── SET ERROR cases ──────────────────────────────────────────────────────
        private static readonly ECase[] FancySetErr =
        {
            // OOB set (positive)
            new("se01_A_oob_set",  () => { var a = A(); a[np.array(new long[]{3})] = (NDArray)(-1L); return a; }),
            new("se02_V_oob_set",  () => { var v = V(); v[np.array(new long[]{6})] = (NDArray)(-1L); return v; }),
            // OOB set (negative-beyond-axis) — same shared PrepareIndexGetters validation
            // as the getter; must throw BEFORE writing (was an OOB write before the fix).
            new("se03_V_oob_neg_set", () => { var v = V(); v[np.array(new long[]{-7})] = (NDArray)(-1L); return v; }),
            new("se04_A_oob_neg_set", () => { var a = A(); a[np.array(new long[]{-4})] = (NDArray)(-1L); return a; }),
        };

        // ═══════════════════ KNOWN BUGS (OpenBugs) ══════════════════════════════

        // ── FIXED: multi-fancy index broadcast (was [OpenBugs]) ──────────────────
        //
        // Two fancy index arrays of DIFFERENT shapes must broadcast together (NumPy):
        // (2,) × (2,1) -> (2,2), result (2,2)+trailing. NumSharp detected "broadcast
        // required" by SIZE equality, so equal-size-but-different-shape pairs like
        // (2,) vs (2,1) (both size 2) slipped through un-broadcast -> wrong (2,1,4).
        // Fixed: detect by SHAPE inequality and np.broadcast_arrays the index arrays.
        private static readonly GCase[] FancyGetBug =
        {
            new("g28_B_ia_ib2d", () => B()[IA(), IB().reshape(2,1)], new[]{2,2,4},
                new long[]{0,1,2,3,12,13,14,15,8,9,10,11,20,21,22,23}),
            new("g28b_B_ia2d_ib", () => B()[IA().reshape(2,1), IB()], new[]{2,2,4},
                new long[]{0,1,2,3,8,9,10,11,12,13,14,15,20,21,22,23}),
            new("g28c_A_ia2d_ib", () => A()[IA().reshape(2,1), IB()], new[]{2,2},
                new long[]{0,2,4,6}),
            new("g28d_B_2x1_1x2", () => B()[IA().reshape(2,1), IB().reshape(1,2)], new[]{2,2,4},
                new long[]{0,1,2,3,8,9,10,11,12,13,14,15,20,21,22,23}),
        };

        // ── FIXED: broadcastable value into ≥2-D subshape (was [OpenBugs]) ───
        //
        // Assigning a BROADCASTABLE value (scalar, or lower-rank) into a ≥2-D fancy
        // result used to assert in DEBUG / over-read in RELEASE: SetIndicesND block-copies
        // one contiguous subShape per offset and assumed values.size == offsets*subShape.
        // Fixed in SetIndices<T>: the value is now broadcast to the indexing-result shape
        // (num_offsets,) + subShape and materialized contiguously before the scatter,
        // matching NumPy (scalar / (subShape,) / (n,1) / (1,subShape) all stretch). Kept as
        // a passing regression guard; NumPy expected values pinned.
        private static readonly GCase[] FancySetBug =
        {
            // A[IA()] = scalar (-1) into (2,4) subshape → rows 0,1 all set to -1.
            new("bug01_A_ia_scalar_set",
                () => { var a = A(); a[IA()] = (NDArray)(-1L); return a; },
                new[]{3,4},
                new long[]{-1,-1,-1,-1,-1,-1,-1,-1,8,9,10,11}),

            // A[IA()] = (4,) array → broadcasts across both selected rows.
            new("bug02_A_ia_row_bcast",
                () => { var a = A(); a[IA()] = np.array(new long[]{10,20,30,40}); return a; },
                new[]{3,4},
                new long[]{10,20,30,40,10,20,30,40,8,9,10,11}),

            // A[IA()] = (2,1) column → each selected row filled with its scalar.
            new("bug03_A_ia_col_bcast",
                () => { var a = A(); a[IA()] = np.array(new long[]{100,200}).reshape(2,1); return a; },
                new[]{3,4},
                new long[]{100,100,100,100,200,200,200,200,8,9,10,11}),

            // A[IA()] = (1,4) row → broadcasts across the leading axis.
            new("bug04_A_ia_1row_bcast",
                () => { var a = A(); a[IA()] = np.array(new long[]{1,2,3,4}).reshape(1,4); return a; },
                new[]{3,4},
                new long[]{1,2,3,4,1,2,3,4,8,9,10,11}),

            // Duplicate selected rows + scalar (last-write-wins is moot for a constant).
            new("bug05_A_dup_scalar",
                () => { var a = A(); a[np.array(new long[]{0,2,0})] = (NDArray)(-1L); return a; },
                new[]{3,4},
                new long[]{-1,-1,-1,-1,4,5,6,7,-1,-1,-1,-1}),
        };

        // ═══════════════════ DYNAMIC DATA SOURCES ════════════════════════════════
        private static IEnumerable<object[]> Idx<T>(T[] arr, Func<T, string> name)
            => arr.Select((c, i) => new object[] { i, name(c) });

        public static IEnumerable<object[]> FancyGetData       => Idx(FancyGet,       c => c.Name);
        public static IEnumerable<object[]> FancyGetErrData    => Idx(FancyGetErr,    c => c.Name);
        public static IEnumerable<object[]> FancyGetErrBugData => Idx(FancyGetErrBug, c => c.Name);
        public static IEnumerable<object[]> FancyGetBugData    => Idx(FancyGetBug,    c => c.Name);
        public static IEnumerable<object[]> FancySetData       => Idx(FancySet,       c => c.Name);
        public static IEnumerable<object[]> FancySetErrData    => Idx(FancySetErr,    c => c.Name);
        public static IEnumerable<object[]> FancySetBugData    => Idx(FancySetBug,    c => c.Name);

        // ═══════════════════ TEST METHODS ════════════════════════════════════════

        [DataTestMethod, DynamicData(nameof(FancyGetData))]
        public void FancyGet_Get(int i, string name) => AssertGet(FancyGet[i]);

        [DataTestMethod, DynamicData(nameof(FancyGetErrData))]
        public void FancyGet_Throws(int i, string name)
        {
            Action act = () => { var _ = FancyGetErr[i].Op(); };
            act.Should().Throw<Exception>(name);
        }

        [DataTestMethod, DynamicData(nameof(FancySetData))]
        public void FancySet_Set(int i, string name)
        {
            var r = FancySet[i].Op();
            var flat = new long[r.size];
            var f = r.flat;
            for (long k = 0; k < r.size; k++) flat[k] = Convert.ToInt64(f.GetValue(k));
            flat.Should().Equal(FancySet[i].Vals, name);
        }

        [DataTestMethod, DynamicData(nameof(FancySetErrData))]
        public void FancySet_Throws(int i, string name)
        {
            Action act = () => { var _ = FancySetErr[i].Op(); };
            act.Should().Throw<Exception>(name);
        }

        // ── FIXED: GET shape with multi-fancy index broadcast (regression guard) ──
        // NumPy broadcasts the index arrays together: (2,)×(2,1) → (2,2) → (2,2,4).
        [DataTestMethod, DynamicData(nameof(FancyGetBugData))]
        public void FancyGetBug_BroadcastShape(int i, string name) => AssertGet(FancyGetBug[i]);

        // ── FIXED: negative OOB now validated (regression guard) ──────────────
        // NumPy raises IndexError for index < -axisLen; NumSharp now does too.
        [DataTestMethod, DynamicData(nameof(FancyGetErrBugData))]
        public void FancyGetErrBug_NegativeOOBNotValidated(int i, string name)
        {
            Action act = () => { var _ = FancyGetErrBug[i].Op(); };
            act.Should().Throw<Exception>(name);
        }

        // ── Copy semantics: fancy GET must be a copy ──────────────────────────
        [TestMethod]
        public void FancyCopy_GetIsCopy_MutationDoesNotAffectBase()
        {
            // NumPy: a copy is returned for fancy indexing (any mutation of the result
            // does not affect the base array).
            var a = A();
            var r = a[IA()];           // rows 0 and 1 — fresh copy

            long base00Before = Convert.ToInt64(a.flat.GetValue(0));
            r.SetValue(999L, 0, 0);    // mutate result[0,0]

            long base00After = Convert.ToInt64(a.flat.GetValue(0));
            base00After.Should().Be(base00Before,
                "fancy GET must return a copy; mutating it must not affect the base");

            Convert.ToInt64(r.flat.GetValue(0)).Should().Be(999L,
                "the result itself must reflect the mutation");
        }

        // ── Decision-B: three forms must yield identical output ───────────────
        [TestMethod]
        public void DecisionB_AllForms_ProduceSameResult()
        {
            // These three forms must all be fancy (selecting rows 0 and 2):
            //   A[np.array(new long[]{0,2})]   — NDArray index
            //   A[new long[]{0,2}]              — raw long[] sole index (decision B)
            //   A[new int[] {0,2}]              — raw int[]  sole index (decision B)
            // NumPy expected: shape (2,4), vals [0,1,2,3,8,9,10,11]
            var expected = np.array(new long[] { 0, 1, 2, 3, 8, 9, 10, 11 }).reshape(2, 4);

            var r_nd   = A()[np.array(new long[]{0,2})];
            var r_long = A()[new long[]{0,2}];
            var r_int  = A()[new int[] {0,2}];

            long[] Flat(NDArray x) => Enumerable.Range(0, (int)x.size)
                .Select(k => Convert.ToInt64(x.flat.GetValue(k))).ToArray();

            var expFlat = Flat(expected);
            Flat(r_nd  ).Should().Equal(expFlat, "NDArray-index form");
            Flat(r_long).Should().Equal(expFlat, "raw long[] form");
            Flat(r_int ).Should().Equal(expFlat, "raw int[]  form");

            r_nd  .shape.Select(x => (int)x).ToArray().Should().Equal(new[]{2,4}, "NDArray-index shape");
            r_long.shape.Select(x => (int)x).ToArray().Should().Equal(new[]{2,4}, "raw long[] shape");
            r_int .shape.Select(x => (int)x).ToArray().Should().Equal(new[]{2,4}, "raw int[]  shape");
        }

        [TestMethod]
        public void DecisionB_GetData_IsCoordinate_NotFancy()
        {
            // GetData(int[]{0,2}) is the coordinate shim — it returns the ELEMENT at
            // position (0,2) (i.e. a 0-D / scalar view with value 2), not rows 0 and 2.
            // NumPy: a[0,2] == 2 (scalar).
            var a = A();
            var scalar = a.GetData(new int[] { 0, 2 });

            // Result must be 0-D (scalar-shaped) with value 2
            scalar.ndim.Should().Be(0, "GetData(coords) returns a 0-D scalar view");
            Convert.ToInt64(scalar.GetValue(0)).Should().Be(2L,
                "coordinate (0,2) in A=arange(12).reshape(3,4) is element 2");
        }

        // ── FIXED: broadcastable value into ≥2-D subshape fancy-set ─────────
        // MATCHED-shape set (s03–s08) AND scalar/lower-rank broadcast now both work.
        [DataTestMethod, DynamicData(nameof(FancySetBugData))]
        public void FancySetBug_BroadcastableValueIntoSubshape(int i, string name)
        {
            var c = FancySetBug[i];
            var r = c.Op();
            r.shape.Select(x => (int)x).ToArray().Should().Equal(c.Shape, $"{c.Name} shape");
            var flat = new long[r.size];
            var f = r.flat;
            for (long k = 0; k < r.size; k++) flat[k] = Convert.ToInt64(f.GetValue(k));
            flat.Should().Equal(c.Vals, $"{c.Name} values");
        }

        // ── Value-shape mismatch on a ≥2-D fancy-set raises ValueError ───────
        // NumPy: ValueError: shape mismatch: value array of shape (3,) could not be
        // broadcast to indexing result of shape (2,4). NumSharp matches that text.
        [TestMethod]
        public void FancySet_ValueShapeMismatch_Throws()
        {
            Action act1 = () => { var a = A(); a[IA()] = np.array(new long[]{1,2,3}); };
            act1.Should().Throw<ValueError>()
                .WithMessage("*could not be broadcast to indexing result of shape (2,4)*");

            Action act2 = () => { var a = A(); a[IA()] = np.array(new long[]{1,2,3,4}).reshape(2,2); };
            act2.Should().Throw<ValueError>();
        }
    }
}
