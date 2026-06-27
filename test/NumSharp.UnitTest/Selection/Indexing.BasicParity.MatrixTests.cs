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

    // ─────────────────────── BUG: GET cases (known NumSharp divergences) ───
    // NumPy raises IndexError; NumSharp does NOT throw → marked [OpenBugs].
    static readonly ECase[] _bugErrCases =
    {
        // NumPy: IndexError: index 4 is out of bounds for axis 1 with size 4
        // NumSharp: silently returns val=4 (treats as flat-buffer index, bypasses axis check)
        new ECase("E_A_oob_ax1_wrapped",   () => A()[0, 4]),    // axis-1 OOB, flat offset=4 still inside buffer
        new ECase("E_A_oob_ax1_wrapped2",  () => A()[1, 4]),    // flat offset=8, also inside buffer
        new ECase("E_B_oob_ax2_wrapped",   () => B()[0, 0, 4]), // axis-2 OOB, flat offset=4 inside buffer
    };

    public static IEnumerable<object[]> BugErrCases =>
        _bugErrCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(BugErrCases))]
    [OpenBugs] // NumSharp doesn't throw when it should; axis-bounds check is flat-offset-only
    public void AssertError_Bugs(ECase tc)
    {
        // NumPy: throws IndexError. NumSharp: currently does not throw. This test FAILS = known bug.
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw,
            $"[{tc.Name}] expected IndexError (NumPy axis-bounds check); NumSharp does not check per-axis OOB when flat offset is within buffer");
    }

    // ─────────────────────── BUG: SET shape mismatch ─────────────────────
    // NumPy raises ValueError when value shape can't broadcast into target.
    // NumSharp silently assigns partial data.
    static readonly ECase[] _bugSetErrCases =
    {
        // NumPy: ValueError: could not broadcast input array from shape (2,) into shape (4,)
        // NumSharp: silently writes [1,2,2,3,4,5,6,7,8,9,10,11]
        new ECase("E_S_shapemismatch_row", () =>
        {
            var a = A();
            a["0"] = np.array(new long[]{ 1, 2 }); // (2,) → (4,): shape mismatch
            return a;
        }),
    };

    public static IEnumerable<object[]> BugSetErrCases =>
        _bugSetErrCases.Select(c => new object[] { c });

    [DataTestMethod]
    [DynamicData(nameof(BugSetErrCases))]
    [OpenBugs] // Setter shape mismatch doesn't throw; NumPy raises ValueError
    public void AssertSetError_Bugs(ECase tc)
    {
        // NumPy: throws ValueError. NumSharp: silently assigns partial data. This test FAILS = known bug.
        bool threw = false;
        try { tc.Op(); }
        catch { threw = true; }
        Assert.IsTrue(threw,
            $"[{tc.Name}] expected ValueError (setter shape mismatch); NumSharp does not validate broadcast compatibility");
    }
}
