// Indexing.BasicParity.MatrixTests.cs
// Exhaustive basic-indexing differential-parity test matrix.
// All expected shapes/values derived from NumPy 2.4.2 (verified by running Python probes).
// Pure basic indexing only: integer coordinates, slices, newaxis, ellipsis — NO fancy/boolean.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Selection;

[TestClass]
public class Indexing_BasicParity_MatrixTests
{
    // ───────────────────────────── Base arrays ─────────────────────────────
    // Fresh instance on each call (lambdas in SCase mutate them)
    static NDArray V()  => np.arange(6L);                        // shape (6,)
    static NDArray A()  => np.arange(12L).reshape(3, 4);         // shape (3,4)
    static NDArray B()  => np.arange(24L).reshape(2, 3, 4);      // shape (2,3,4)
    static NDArray B4() => np.arange(120L).reshape(2, 3, 4, 5);  // shape (2,3,4,5)

    // ───────────────────────────── Case types ─────────────────────────────
    // GCase: getter — Op() returns the indexed sub-array.
    //   Shape: expected shape (int[] — empty = 0-D scalar).
    //   Vals:  expected ravel values as long[]; null ⇒ shape-only (used when size > 64).
    public sealed class GCase
    {
        public string Name { get; }
        public Func<NDArray> Op { get; }
        public int[] Shape { get; }
        public long[] Vals { get; }
        public GCase(string name, Func<NDArray> op, int[] shape, long[] vals = null)
        { Name = name; Op = op; Shape = shape; Vals = vals; }
        public override string ToString() => Name;
    }

    // SCase: setter — Op() mutates base and RETURNS the base; Vals = base.ravel() after.
    public sealed class SCase
    {
        public string Name { get; }
        public Func<NDArray> Op { get; }
        public long[] Vals { get; }
        public SCase(string name, Func<NDArray> op, long[] vals)
        { Name = name; Op = op; Vals = vals; }
        public override string ToString() => Name;
    }

    // ECase: expected-throws — Op() must throw any exception.
    public sealed class ECase
    {
        public string Name { get; }
        public Func<NDArray> Op { get; }
        public ECase(string name, Func<NDArray> op)
        { Name = name; Op = op; }
        public override string ToString() => Name;
    }

    // ─────────────────────── GET cases (non-bug) ──────────────────────────
    static readonly GCase[] _getCases =
    {
        // ── Single int on V (1-D): reduces rank → 0-D scalar ──
        new GCase("V_int_first",   () => V()[0],   new int[0], new long[]{ 0 }),
        new GCase("V_int_mid",     () => V()[2],   new int[0], new long[]{ 2 }),
        new GCase("V_int_last",    () => V()[5],   new int[0], new long[]{ 5 }),
        new GCase("V_int_neg1",    () => V()[-1],  new int[0], new long[]{ 5 }),
        new GCase("V_int_negrank", () => V()[-6],  new int[0], new long[]{ 0 }),

        // ── Single int on A (2-D): reduces one axis → 1-D row ──
        new GCase("A_int0",    () => A()[0],  new[]{ 4 }, new long[]{ 0, 1, 2, 3 }),
        new GCase("A_int1",    () => A()[1],  new[]{ 4 }, new long[]{ 4, 5, 6, 7 }),
        new GCase("A_int_neg1",() => A()[-1], new[]{ 4 }, new long[]{ 8, 9, 10, 11 }),

        // ── Two-coord int on A → 0-D scalar ──
        new GCase("A_coord_00",      () => A()[0, 0],   new int[0], new long[]{ 0 }),
        new GCase("A_coord_12",      () => A()[1, 2],   new int[0], new long[]{ 6 }),
        new GCase("A_coord_23",      () => A()[2, 3],   new int[0], new long[]{ 11 }),
        new GCase("A_coord_neg1neg1",() => A()[-1, -1], new int[0], new long[]{ 11 }),
        new GCase("A_coord_neg1_0",  () => A()[-1, 0],  new int[0], new long[]{ 8 }),

        // ── Single int on B (3-D): reduces one axis → 2-D sub-matrix ──
        new GCase("B_int0",    () => B()[0],  new[]{ 3, 4 }, new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("B_int1",    () => B()[1],  new[]{ 3, 4 }, new long[]{ 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
        new GCase("B_int_neg1",() => B()[-1], new[]{ 3, 4 }, new long[]{ 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),

        // ── Two-coord int on B → 1-D row ──
        new GCase("B_coord_01", () => B()[0, 1], new[]{ 4 }, new long[]{ 4, 5, 6, 7 }),
        new GCase("B_coord_12", () => B()[1, 2], new[]{ 4 }, new long[]{ 20, 21, 22, 23 }),

        // ── Three-coord int on B → 0-D scalar ──
        new GCase("B_coord_023", () => B()[0, 2, 3], new int[0], new long[]{ 11 }),
        new GCase("B_coord_101", () => B()[1, 0, 1], new int[0], new long[]{ 13 }),

        // ── B4 (4-D) ──
        new GCase("B4_int0",       () => B4()[0],       new[]{ 3, 4, 5 }, null),   // size 60 > 64 → shape only
        new GCase("B4_coord_12",   () => B4()[1, 2],    new[]{ 4, 5 },
            new long[]{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                        110, 111, 112, 113, 114, 115, 116, 117, 118, 119 }),
        new GCase("B4_coord_012",  () => B4()[0, 1, 2], new[]{ 5 }, new long[]{ 30, 31, 32, 33, 34 }),
        new GCase("B4_coord_1234", () => B4()[1, 2, 3, 4], new int[0], new long[]{ 119 }),

        // ── Slices on V (1-D) ──
        new GCase("V_slice_all",     () => V()[":"],      new[]{ 6 }, new long[]{ 0, 1, 2, 3, 4, 5 }),
        new GCase("V_slice_1_3",     () => V()["1:3"],    new[]{ 2 }, new long[]{ 1, 2 }),
        new GCase("V_slice_step2",   () => V()["::2"],    new[]{ 3 }, new long[]{ 0, 2, 4 }),
        new GCase("V_slice_rev",     () => V()["::-1"],   new[]{ 6 }, new long[]{ 5, 4, 3, 2, 1, 0 }),
        new GCase("V_slice_2on",     () => V()["2:"],     new[]{ 4 }, new long[]{ 2, 3, 4, 5 }),
        new GCase("V_slice_to2",     () => V()[":2"],     new[]{ 2 }, new long[]{ 0, 1 }),
        new GCase("V_slice_neg2on",  () => V()["-2:"],    new[]{ 2 }, new long[]{ 4, 5 }),
        new GCase("V_slice_empty",   () => V()["1:1"],    new[]{ 0 }, new long[0]),
        new GCase("V_slice_step3",   () => V()["::3"],    new[]{ 2 }, new long[]{ 0, 3 }),
        new GCase("V_slice_clamped", () => V()["1:10"],   new[]{ 5 }, new long[]{ 1, 2, 3, 4, 5 }),
        new GCase("V_slice_rev_bnd", () => V()["3:0:-1"], new[]{ 3 }, new long[]{ 3, 2, 1 }),
        new GCase("V_slice_neg1on",  () => V()["-1:"],    new[]{ 1 }, new long[]{ 5 }),

        // ── Single-axis slices on A (2-D) ──
        new GCase("A_slice_all",    () => A()[":"],     new[]{ 3, 4 }, new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("A_slice_1_3",    () => A()["1:3"],   new[]{ 2, 4 }, new long[]{ 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("A_slice_step2",  () => A()["::2"],   new[]{ 2, 4 }, new long[]{ 0, 1, 2, 3, 8, 9, 10, 11 }),
        new GCase("A_slice_rev",    () => A()["::-1"],  new[]{ 3, 4 }, new long[]{ 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3 }),
        new GCase("A_slice_2on",    () => A()["2:"],    new[]{ 1, 4 }, new long[]{ 8, 9, 10, 11 }),
        new GCase("A_slice_to2",    () => A()[":2"],    new[]{ 2, 4 }, new long[]{ 0, 1, 2, 3, 4, 5, 6, 7 }),
        new GCase("A_slice_neg2on", () => A()["-2:"],   new[]{ 2, 4 }, new long[]{ 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("A_slice_empty",  () => A()["1:1"],   new[]{ 0, 4 }, new long[0]),

        // ── Multi-axis slices ──
        new GCase("A_col_1_3",     () => A()[":, 1:3"],    new[]{ 3, 2 }, new long[]{ 1, 2, 5, 6, 9, 10 }),
        new GCase("A_1_3_step2",   () => A()["1:3, ::2"],  new[]{ 2, 2 }, new long[]{ 4, 6, 8, 10 }),
        new GCase("A_rev_rev",     () => A()["::-1, ::-1"], new[]{ 3, 4 }, new long[]{ 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }),
        new GCase("B_last_1_3",    () => B()[":, :, 1:3"],  new[]{ 2, 3, 2 },
            new long[]{ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22 }),
        new GCase("B_rev_ax0",     () => B()["::-1, :, :"], new[]{ 2, 3, 4 },
            new long[]{ 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11 }),
        new GCase("B_1on_1_3",     () => B()["1:, 1:3"],   new[]{ 1, 2, 4 },
            new long[]{ 16, 17, 18, 19, 20, 21, 22, 23 }),

        // ── int + slice mixes ──
        new GCase("A_1_all",     () => A()["1, :"],   new[]{ 4 }, new long[]{ 4, 5, 6, 7 }),
        new GCase("A_all_2",     () => A()[":, 2"],   new[]{ 3 }, new long[]{ 2, 6, 10 }),
        new GCase("A_1_1_3",     () => A()["1, 1:3"], new[]{ 2 }, new long[]{ 5, 6 }),
        new GCase("A_neg1_all",  () => A()["-1, :"],  new[]{ 4 }, new long[]{ 8, 9, 10, 11 }),
        new GCase("A_0_1_3",     () => A()["0, 1:3"], new[]{ 2 }, new long[]{ 1, 2 }),
        new GCase("A_all_0",     () => A()[":, 0"],   new[]{ 3 }, new long[]{ 0, 4, 8 }),
        new GCase("A_rev_neg1",  () => A()["::-1, -1"], new[]{ 3 }, new long[]{ 11, 7, 3 }),
        new GCase("B_0_all_2",   () => B()["0, :, 2"], new[]{ 3 }, new long[]{ 2, 6, 10 }),
        new GCase("B_rev_1",     () => B()["::-1, 1"], new[]{ 2, 4 }, new long[]{ 16, 17, 18, 19, 4, 5, 6, 7 }),
        new GCase("B_0_1_3",     () => B()["0, 1:3"],  new[]{ 2, 4 }, new long[]{ 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("B_1_all_s2",  () => B()["1, :, ::2"], new[]{ 3, 2 }, new long[]{ 12, 14, 16, 18, 20, 22 }),

        // ── newaxis ──
        new GCase("A_na_all",   () => A()[Slice.NewAxis, Slice.All],  new[]{ 1, 3, 4 },
            new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("V_na",       () => V()[Slice.NewAxis],             new[]{ 1, 6 },
            new long[]{ 0, 1, 2, 3, 4, 5 }),
        new GCase("A_na_na",    () => A()[Slice.NewAxis, Slice.NewAxis], new[]{ 1, 1, 3, 4 }, null), // size 12 but keep shape-only for clarity
        new GCase("A_0_na",     () => A()[0, Slice.NewAxis],          new[]{ 1, 4 }, new long[]{ 0, 1, 2, 3 }),
        new GCase("A_na_0",     () => A()[Slice.NewAxis, Slice.Index(0)], new[]{ 1, 4 }, new long[]{ 0, 1, 2, 3 }),
        new GCase("A_1_3_na",   () => A()[new Slice(1,3), Slice.NewAxis], new[]{ 2, 1, 4 },
            new long[]{ 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("A_na_1_3",   () => A()[Slice.NewAxis, new Slice(1,3)], new[]{ 1, 2, 4 },
            new long[]{ 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("A_na_0_na",  () => A()[Slice.NewAxis, Slice.Index(0), Slice.NewAxis], new[]{ 1, 1, 4 },
            new long[]{ 0, 1, 2, 3 }),
        new GCase("A_na_all_na_all", () => A()[Slice.NewAxis, Slice.All, Slice.NewAxis, Slice.All],
            new[]{ 1, 3, 1, 4 }, new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),

        // ── ellipsis ──
        new GCase("B_ell_0",    () => B()[Slice.Ellipsis, Slice.Index(0)], new[]{ 2, 3 },
            new long[]{ 0, 4, 8, 12, 16, 20 }),
        new GCase("B_0_ell",    () => B()[Slice.Index(0), Slice.Ellipsis], new[]{ 3, 4 },
            new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
        new GCase("B_ell",      () => B()[Slice.Ellipsis],                 new[]{ 2, 3, 4 }, null),
        new GCase("B_ell_1_3",  () => B()[Slice.Ellipsis, new Slice(1,3)], new[]{ 2, 3, 2 },
            new long[]{ 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22 }),
        new GCase("B_ell_s2",   () => B()[Slice.Ellipsis, new Slice(null,null,2)], new[]{ 2, 3, 2 },
            new long[]{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 }),
        new GCase("B4_ell_0",   () => B4()[Slice.Ellipsis, Slice.Index(0)], new[]{ 2, 3, 4 },
            new long[]{ 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
                        60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115 }),
        new GCase("B4_0_ell",   () => B4()[Slice.Index(0), Slice.Ellipsis], new[]{ 3, 4, 5 }, null),
        new GCase("B4_01_ell",  () => B4()[Slice.Index(0), Slice.Index(1), Slice.Ellipsis], new[]{ 4, 5 },
            new long[]{ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 }),
        new GCase("B4_0_ell_0", () => B4()[Slice.Index(0), Slice.Ellipsis, Slice.Index(0)], new[]{ 3, 4 },
            new long[]{ 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 }),

        // ── ellipsis + newaxis ──
        new GCase("A_ell_na",    () => A()[Slice.Ellipsis, Slice.NewAxis],              new[]{ 3, 4, 1 }, null),
        new GCase("A_na_ell_na", () => A()[Slice.NewAxis, Slice.Ellipsis, Slice.NewAxis], new[]{ 1, 3, 4, 1 },
            new long[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }),
    };

    public static IEnumerable<object[]> GetCases =>
        _getCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(GetCases))]
    public void AssertGet(GCase tc)
    {
        NDArray result = tc.Op();

        // ── Shape check ──
        var actualShape = result.shape.Select(d => (int)d).ToArray();
        CollectionAssert.AreEqual(tc.Shape, actualShape,
            $"[{tc.Name}] shape: expected [{string.Join(",", tc.Shape)}] got [{string.Join(",", actualShape)}]");

        // ── Value check ──
        if (tc.Vals != null)
        {
            if (tc.Shape.Length == 0) // 0-D scalar
            {
                Assert.AreEqual(1, tc.Vals.Length, $"[{tc.Name}] scalar must have exactly 1 expected value");
                long actual = Convert.ToInt64(result.GetValue(new long[0]));
                Assert.AreEqual(tc.Vals[0], actual, $"[{tc.Name}] scalar value");
            }
            else
            {
                NDArray flat = result.ravel();
                Assert.AreEqual(tc.Vals.Length, (int)flat.size,
                    $"[{tc.Name}] ravel length: expected {tc.Vals.Length} got {flat.size}");
                for (int i = 0; i < tc.Vals.Length; i++)
                {
                    long actual = Convert.ToInt64(flat.GetValue(i));
                    Assert.AreEqual(tc.Vals[i], actual, $"[{tc.Name}][{i}]");
                }
            }
        }
    }

    // ─────────────────────── SET cases (non-bug) ──────────────────────────
    static readonly SCase[] _setCases =
    {
        // S1: scalar broadcast → row
        new SCase("S_A1_scalar", () => { var a = A(); a[1] = (NDArray)(-1L); return a; },
            new long[]{ 0, 1, 2, 3, -1, -1, -1, -1, 8, 9, 10, 11 }),

        // S2: scalar broadcast → column slice
        new SCase("S_Acol_scalar", () => { var a = A(); a[":, 1:3"] = (NDArray)(-1L); return a; },
            new long[]{ 0, -1, -1, 3, 4, -1, -1, 7, 8, -1, -1, 11 }),

        // S3: matched-shape assignment
        new SCase("S_A1_matched", () => { var a = A(); a[1] = np.array(new long[]{ 40, 41, 42, 43 }); return a; },
            new long[]{ 0, 1, 2, 3, 40, 41, 42, 43, 8, 9, 10, 11 }),

        // S4: reversed view scalar
        new SCase("S_Vrev_matched", () => { var v = V(); v["::-1"] = np.array(new long[]{ 10, 11, 12, 13, 14, 15 }); return v; },
            new long[]{ 15, 14, 13, 12, 11, 10 }),

        // S5: step view assignment
        new SCase("S_Vstep2_matched", () => { var v = V(); v["::2"] = np.array(new long[]{ 20, 21, 22 }); return v; },
            new long[]{ 20, 1, 21, 3, 22, 5 }),

        // S6: ellipsis broadcast scalar
        new SCase("S_Aell_scalar", () => { var a = A(); a["..."] = (NDArray)(-7L); return a; },
            new long[]{ -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7 }),

        // S7: int + slice scalar
        new SCase("S_B01_3_scalar", () => { var b = B(); b["0, 1:3"] = (NDArray)(-5L); return b; },
            new long[]{ 0, 1, 2, 3, -5, -5, -5, -5, -5, -5, -5, -5,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),

        // S8: ellipsis+int scalar (sets first element of each last-axis row)
        new SCase("S_Bell_0_scalar", () => { var b = B(); b["..., 0"] = (NDArray)(-9L); return b; },
            new long[]{ -9, 1, 2, 3, -9, 5, 6, 7, -9, 9, 10, 11,
                        -9, 13, 14, 15, -9, 17, 18, 19, -9, 21, 22, 23 }),

        // S9: step + slice matched shape
        new SCase("S_Astep2_1_3", () => { var a = A(); a["::2, 1:3"] = np.array(new long[,]{ { 40, 41 }, { 42, 43 } }); return a; },
            new long[]{ 0, 40, 41, 3, 4, 5, 6, 7, 8, 42, 43, 11 }),

        // S10: 1-D slice matched
        new SCase("S_V1_4_matched", () => { var v = V(); v["1:4"] = np.array(new long[]{ 10, 11, 12 }); return v; },
            new long[]{ 0, 10, 11, 12, 4, 5 }),

        // S11: reversed view broadcast scalar
        new SCase("S_Vrev_scalar", () => { var v = V(); v["::-1"] = (NDArray)99L; return v; },
            new long[]{ 99, 99, 99, 99, 99, 99 }),

        // S12: negative-start slice matched
        new SCase("S_Vneg3on_matched", () => { var v = V(); v["-3:"] = np.array(new long[]{ 97, 98, 99 }); return v; },
            new long[]{ 0, 1, 2, 97, 98, 99 }),

        // S13: ellipsis + step scalar
        new SCase("S_Bell_s2_scalar", () => { var b = B(); b["..., ::2"] = (NDArray)(-99L); return b; },
            new long[]{ -99, 1, -99, 3, -99, 5, -99, 7, -99, 9, -99, 11,
                        -99, 13, -99, 15, -99, 17, -99, 19, -99, 21, -99, 23 }),
    };

    public static IEnumerable<object[]> SetCases =>
        _setCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(SetCases))]
    public void AssertSet(SCase tc)
    {
        NDArray result = tc.Op();
        NDArray flat = result.ravel();
        Assert.AreEqual(tc.Vals.Length, (int)flat.size,
            $"[{tc.Name}] base ravel length mismatch");
        for (int i = 0; i < tc.Vals.Length; i++)
        {
            long actual = Convert.ToInt64(flat.GetValue(i));
            Assert.AreEqual(tc.Vals[i], actual, $"[{tc.Name}][{i}]");
        }
    }

    // ─────────────────────── ERROR cases (non-bug) ────────────────────────
    // These all DO throw in both NumPy and NumSharp.
    static readonly ECase[] _errCases =
    {
        // OOB on V
        new ECase("E_V_oob_pos",  () => V()[6]),   // index 6 >= size 6
        new ECase("E_V_oob_neg",  () => V()[-7]),  // index -7 < -6

        // OOB on A axis-0
        new ECase("E_A_oob_ax0",      () => A()[3]),    // index 3 >= 3
        new ECase("E_A_oob_ax0_neg",  () => A()[-4]),   // index -4 < -3

        // Axis-1 negative OOB: A[0,-5] (axis-1 len 4) — NumPy & NumSharp both raise.
        new ECase("E_A_oob_ax1_neg",  () => A()[0, -5]),

        // Too many indices
        new ECase("E_A_toomany", () => A()[0, 0, 0]), // ndim=2 but 3 indices

        // Multiple ellipsis — only one allowed
        new ECase("E_multi_ell", () => B()[Slice.Ellipsis, Slice.Ellipsis, Slice.Index(0)]),
    };

    public static IEnumerable<object[]> ErrCases =>
        _errCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(ErrCases))]
    public void AssertError(ECase tc)
    {
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw, $"[{tc.Name}] expected an exception (NumPy raises IndexError here)");
    }

    // ─────────────────────── FIXED: per-axis OOB now validated ─────────────
    // Was [OpenBugs]: coordinate indexing only checked the flat offset against the
    // buffer, so a per-axis OOB whose flat offset still landed inside the buffer
    // (e.g. A[0,4] → A[1,0] == 4) slipped through. Fixed in Shape.InferNegativeCoordinates
    // (every GetData coordinate path) — each index is now validated to [-dim, dim-1] and
    // raises IndexError "index N is out of bounds for axis A with size S" (NumPy-verbatim).
    // Kept as a passing regression guard.
    static readonly ECase[] _bugErrCases =
    {
        // NumPy: IndexError: index 4 is out of bounds for axis 1 with size 4
        new ECase("E_A_oob_ax1_wrapped",   () => A()[0, 4]),    // axis-1 OOB, flat offset=4 still inside buffer
        new ECase("E_A_oob_ax1_wrapped2",  () => A()[1, 4]),    // flat offset=8, also inside buffer
        new ECase("E_B_oob_ax2_wrapped",   () => B()[0, 0, 4]), // axis-2 OOB, flat offset=4 inside buffer
    };

    public static IEnumerable<object[]> BugErrCases =>
        _bugErrCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(BugErrCases))]
    public void AssertError_Bugs(ECase tc)
    {
        // NumPy throws IndexError; NumSharp now throws too (per-axis bounds check). Passing.
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw,
            $"[{tc.Name}] expected IndexError (NumPy axis-bounds check); per-axis OOB must be validated even when the flat offset is within the buffer");
    }

    // ─────────────────── FIXED: SET shape mismatch raises (was [OpenBugs]) ──────
    // NumPy raises ValueError when the value shape can't broadcast into the target.
    // NumSharp used a divisibility check (subShape.size % value.size) that let an
    // incompatible smaller value through and then copied it PARTIALLY (a[0]=[1,2] wrote
    // [1,2,_,_]); fixed to a broadcast-compatibility check in UnmanagedStorage.SetData.
    static readonly ECase[] _bugSetErrCases =
    {
        // NumPy: ValueError: could not broadcast input array from shape (2,) into shape (4,)
        new ECase("E_S_shapemismatch_row",   () => { var a = A(); a["0"] = np.array(new long[]{ 1, 2 });    return a; }), // (2,) -> (4,)
        new ECase("E_S_shapemismatch_row3",  () => { var a = A(); a["0"] = np.array(new long[]{ 1, 2, 3 }); return a; }), // (3,) -> (4,)
        new ECase("E_S_shapemismatch_row5",  () => { var a = A(); a["0"] = np.array(new long[]{ 1, 2, 3, 4, 5 }); return a; }), // (5,) -> (4,)
        new ECase("E_S_shapemismatch_whole", () => { var a = A(); a[":"] = np.array(new long[]{ 1, 2, 3, 4, 5, 6 }); return a; }), // (6,) -> (3,4)
    };

    public static IEnumerable<object[]> BugSetErrCases =>
        _bugSetErrCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(BugSetErrCases))]
    public void AssertSetError_Bugs(ECase tc)
    {
        // NumPy raises ValueError; NumSharp now does too (broadcast-compat check). Passing.
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw,
            $"[{tc.Name}] expected ValueError (setter shape mismatch); a value that cannot broadcast into the target must raise, not write partial data");
    }

    // The valid broadcasts that previously wrote PARTIAL data must now stretch correctly.
    [TestMethod]
    public void SetBroadcast_SmallerValue_TilesLikeNumPy()
    {
        // a[:] = (4,) broadcasts the row across all 3 rows (NumPy), not a partial first-row copy.
        var a = A(); a[":"] = np.array(new long[] { 10, 20, 30, 40 });
        a.ravel().ToArray<long>().Should().Equal(new long[] { 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40 });

        // a[0] = (1,) fills the row.
        var b = A(); b["0"] = np.array(new long[] { 9 });
        b.ravel().ToArray<long>().Should().Equal(new long[] { 9, 9, 9, 9, 4, 5, 6, 7, 8, 9, 10, 11 });
    }

    // ─────────────────────── Scalar-index DTYPE coverage ──────────────────────
    // NumPy indexes with a scalar of ANY integer dtype (np.uint8 .. np.uint64) like a
    // plain int: a[np.uint64(1)] == a[1]. In NumSharp the scalar binds to the Slice-based
    // indexer via Slice's implicit operators; ulong is the only integer type with no
    // implicit conversion to int/long, so it needs its own Slice operator (regression
    // guard for that fix). All eight must select row 1 (get) and assign row 1 (set).
    [TestMethod]
    public void ScalarIndex_AllIntegerDtypes_IndexLikeInt()
    {
        var expectRow1 = new long[] { 4, 5, 6, 7 };

        void Get(string name, NDArray r)
        {
            r.shape.Select(d => (int)d).ToArray().Should().Equal(new[] { 4 }, $"{name} shape");
            Enumerable.Range(0, 4).Select(i => Convert.ToInt64(r.GetValue(i))).ToArray()
                .Should().Equal(expectRow1, $"{name} values");
        }

        Get("byte",   A()[(byte)1]);
        Get("sbyte",  A()[(sbyte)1]);
        Get("short",  A()[(short)1]);
        Get("ushort", A()[(ushort)1]);
        Get("int",    A()[1]);
        Get("uint",   A()[(uint)1]);
        Get("long",   A()[1L]);
        Get("ulong",  A()[(ulong)1]);   // the gap that needed Slice's ulong operator

        void Set(string name, Func<NDArray> op)
        {
            var a = op();
            Enumerable.Range(0, 12).Select(i => Convert.ToInt64(a.GetValue(i / 4, i % 4))).ToArray()
                .Should().Equal(new long[] { 0, 1, 2, 3, -1, -1, -1, -1, 8, 9, 10, 11 }, $"{name} set");
        }

        Set("byte",  () => { var a = A(); a[(byte)1]  = (NDArray)(-1L); return a; });
        Set("uint",  () => { var a = A(); a[(uint)1]  = (NDArray)(-1L); return a; });
        Set("ulong", () => { var a = A(); a[(ulong)1] = (NDArray)(-1L); return a; });
    }

    // ─────────────────── ITuple + IEnumerable index INPUTS ────────────────────
    // C#/.NET input forms with no direct NumPy type, mapped to their Python meaning:
    //   • a C# tuple is Python's tuple index — nd[(1,2)] == nd[1,2] (coordinate/mixed);
    //   • any sequence (List<int>, ArrayList, …) coerces to a fancy index (NumPy
    //     PyArray_FROM_O). All expected values probed from NumPy 2.4.2.
    [TestMethod]
    public void TupleAndSequence_IndexInputs_NumPyParity()
    {
        long[] FlatR(NDArray x)
        {
            var f = x.ravel();
            return Enumerable.Range(0, (int)f.size).Select(i => Convert.ToInt64(f.GetValue(i))).ToArray();
        }

        // ── ITuple = Python tuple index (coordinate / mixed) ──
        var t12 = A()[(1, 2)];                       // == A[1,2] -> scalar 6
        t12.ndim.Should().Be(0, "A[(1,2)] is a 0-d scalar (tuple == coordinate)");
        Convert.ToInt64(t12.GetValue(new long[0])).Should().Be(6);

        var tmix = A()[(1, "0:2")];                  // == A[1, 0:2] -> [4,5]
        tmix.shape.Select(d => (int)d).ToArray().Should().Equal(new[] { 2 });
        FlatR(tmix).Should().Equal(new long[] { 4, 5 });

        var t1 = V()[ValueTuple.Create(5)];          // (5,) == V[5] -> scalar 5
        t1.ndim.Should().Be(0);
        Convert.ToInt64(t1.GetValue(new long[0])).Should().Be(5);

        Convert.ToInt64(A()[Tuple.Create(2, 3)].GetValue(new long[0])).Should().Be(11); // System.Tuple == A[2,3]

        // ── IEnumerable (generic + non-generic) -> fancy index ──
        FlatR(V()[new List<int> { 1, 3, 5 }]).Should().Equal(new long[] { 1, 3, 5 });
        FlatR(V()[new List<long> { 1, 3, 5 }]).Should().Equal(new long[] { 1, 3, 5 });
        FlatR(V()[new System.Collections.ArrayList { 1, 3, 5 }]).Should().Equal(new long[] { 1, 3, 5 });

        var rows = A()[new List<int> { 0, 2 }];      // list fancy on 2-D base -> rows 0,2
        rows.shape.Select(d => (int)d).ToArray().Should().Equal(new[] { 2, 4 });
        FlatR(rows).Should().Equal(new long[] { 0, 1, 2, 3, 8, 9, 10, 11 });

        FlatR(V()[new List<int>()]).Should().Equal(new long[0]); // empty list -> empty fancy

        // ── setter via tuple + sequence ──
        var s1 = A(); s1[(1, 2)] = (NDArray)(-1L);   // set element (1,2)
        FlatR(s1).Should().Equal(new long[] { 0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, 11 });

        var s2 = V(); s2[new List<int> { 1, 3, 5 }] = np.array(new long[] { 10, 30, 50 }); // fancy set via list
        FlatR(s2).Should().Equal(new long[] { 0, 10, 2, 30, 4, 50 });
    }
}
