using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// The fast axis argmax / argmin (<c>Default.Reduction.ArgAxis.Fast.cs</c>) and the 64-bit single-pass tournament of
/// the flat row kernel (<c>DirectILKernelGenerator.Reduction.Arg.cs</c>).
/// </summary>
/// <remarks>
/// <para>
/// The contract is NumPy's selection rule, applied in LOGICAL axis order whatever the memory layout: the first
/// occurrence of the extreme (<c>-0.0 == +0.0</c>, so the earlier wins), the first NaN winning for float argmax AND
/// argmin, strict comparison for integers, bool argmax = first True (0 when none), bool argmin = first False (0 when
/// none). Every case is checked three ways — against the per-output IL kernel the fast fold replaced
/// (<see cref="DefaultEngine.DisableFastAxisArg"/>), against an independent reference computed from a C-contiguous copy,
/// and, where the value is pinned, against NumPy 2.4.2's own output — and asserts the fast fold actually RAN
/// (<see cref="DefaultEngine.FastAxisArgRuns"/>), because a silent decline would pass every value check.
/// </para>
/// <para>
/// The layouts are chosen to hit each walk the fold picks from the strides: Rows (the axis contiguous — short rows fold
/// inline, rows of 32+ use the SIMD row kernels), Slab (a lane run tighter than the axis — vector fold for a ±1 run, a
/// reversed run flipped first, scalar fold for a wider run, chunks of 1024 lanes), AxisWalk (the axis the tightest
/// non-contiguous stride, a reversed axis included) and Zero (a broadcast axis).
/// </para>
/// </remarks>
[TestClass]
public class AxisArgFastPathTests
{
    /// <summary>
    /// Reduces <paramref name="a"/> along <paramref name="axis"/> through the fast fold and through the legacy kernel,
    /// and asserts both equal the independent reference and that the fast fold ran.
    /// </summary>
    /// <param name="a">The operand (any layout).</param>
    /// <param name="axis">The reduced axis (negative allowed).</param>
    /// <param name="isMax">Argmax (<see langword="true"/>) or argmin.</param>
    /// <param name="keepdims">Whether the reduced axis is kept as size 1.</param>
    /// <param name="what">A label for failure messages.</param>
    private static void AssertFastMatches(NDArray a, int axis, bool isMax, bool keepdims, string what)
    {
        NDArray legacy;
        DefaultEngine.DisableFastAxisArg = true;
        try
        {
            legacy = isMax ? np.argmax(a, axis, keepdims) : np.argmin(a, axis, keepdims);
        }
        finally
        {
            DefaultEngine.DisableFastAxisArg = false;
        }

        long before = DefaultEngine.FastAxisArgRuns;
        var fast = isMax ? np.argmax(a, axis, keepdims) : np.argmin(a, axis, keepdims);
        Assert.AreEqual(before + 1, DefaultEngine.FastAxisArgRuns, $"{what}: the fast fold must run");

        Assert.AreEqual(NPTypeCode.Int64, fast.typecode, $"{what}: dtype");
        CollectionAssert.AreEqual(legacy.shape, fast.shape, $"{what}: shape vs legacy");
        Assert.IsTrue(fast.Shape.IsContiguous, $"{what}: the result is C-contiguous (NumPy's always is)");
        var f = fast.ToArray<long>();
        CollectionAssert.AreEqual(legacy.ToArray<long>(), f, $"{what}: values vs the legacy kernel");
        CollectionAssert.AreEqual(Reference(a, axis, isMax), f, $"{what}: values vs the reference");
    }

    /// <summary>
    /// The independent reference: the operand copied C-contiguous (NDIter copy — no argmax code involved), then each
    /// output folded along the axis with NumPy's rule in logical order.
    /// </summary>
    /// <param name="a">The operand.</param>
    /// <param name="axis">The reduced axis (negative allowed).</param>
    /// <param name="isMax">Argmax or argmin.</param>
    /// <returns>The expected indices in C order of the reduced shape.</returns>
    private static long[] Reference(NDArray a, int axis, bool isMax)
    {
        int nd = a.ndim;
        int ax = axis < 0 ? axis + nd : axis;
        long outer = 1, inner = 1, len = a.shape[ax];
        for (int d = 0; d < ax; d++) outer *= a.shape[d];
        for (int d = ax + 1; d < nd; d++) inner *= a.shape[d];
        var c = a.copy();
        return a.typecode switch
        {
            NPTypeCode.Boolean => RefBool(c.ToArray<bool>(), outer, len, inner, isMax),
            NPTypeCode.Byte => RefInt(c.ToArray<byte>(), outer, len, inner, isMax),
            NPTypeCode.SByte => RefInt(c.ToArray<sbyte>(), outer, len, inner, isMax),
            NPTypeCode.Int16 => RefInt(c.ToArray<short>(), outer, len, inner, isMax),
            NPTypeCode.UInt16 => RefInt(c.ToArray<ushort>(), outer, len, inner, isMax),
            NPTypeCode.Char => RefInt(c.ToArray<char>(), outer, len, inner, isMax),
            NPTypeCode.Int32 => RefInt(c.ToArray<int>(), outer, len, inner, isMax),
            NPTypeCode.UInt32 => RefInt(c.ToArray<uint>(), outer, len, inner, isMax),
            NPTypeCode.Int64 => RefInt(c.ToArray<long>(), outer, len, inner, isMax),
            NPTypeCode.UInt64 => RefInt(c.ToArray<ulong>(), outer, len, inner, isMax),
            NPTypeCode.Single => RefFloat(c.ToArray<float>(), outer, len, inner, isMax),
            NPTypeCode.Double => RefFloat(c.ToArray<double>(), outer, len, inner, isMax),
            _ => throw new NotSupportedException(a.typecode.ToString()),
        };
    }

    /// <summary>Reference fold for integer lanes: strict improvement, first occurrence kept.</summary>
    /// <typeparam name="T">The integer lane type (Char included — it is an <see cref="IBinaryInteger{T}"/>).</typeparam>
    /// <param name="v">The operand's values in C order.</param>
    /// <param name="outer">Product of the dims before the axis.</param>
    /// <param name="len">The axis length.</param>
    /// <param name="inner">Product of the dims after the axis.</param>
    /// <param name="isMax">Argmax or argmin.</param>
    /// <returns>One index per output, in C order of the reduced shape.</returns>
    private static long[] RefInt<T>(T[] v, long outer, long len, long inner, bool isMax) where T : IBinaryInteger<T>
    {
        var r = new long[outer * inner];
        for (long o = 0; o < outer; o++)
            for (long i = 0; i < inner; i++)
            {
                T best = v[o * len * inner + i];
                long bi = 0;
                for (long k = 1; k < len; k++)
                {
                    T x = v[(o * len + k) * inner + i];
                    if (isMax ? x > best : x < best)
                    {
                        best = x;
                        bi = k;
                    }
                }

                r[o * inner + i] = bi;
            }

        return r;
    }

    /// <summary>Reference fold for float lanes: the first NaN wins (argmax AND argmin), else strict improvement.</summary>
    /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
    /// <param name="v">The operand's values in C order.</param>
    /// <param name="outer">Product of the dims before the axis.</param>
    /// <param name="len">The axis length.</param>
    /// <param name="inner">Product of the dims after the axis.</param>
    /// <param name="isMax">Argmax or argmin.</param>
    /// <returns>One index per output, in C order of the reduced shape.</returns>
    private static long[] RefFloat<T>(T[] v, long outer, long len, long inner, bool isMax) where T : IFloatingPointIeee754<T>
    {
        var r = new long[outer * inner];
        for (long o = 0; o < outer; o++)
            for (long i = 0; i < inner; i++)
            {
                long bi = 0;
                T best = v[o * len * inner + i];
                if (!T.IsNaN(best))
                {
                    for (long k = 1; k < len; k++)
                    {
                        T x = v[(o * len + k) * inner + i];
                        if (T.IsNaN(x))
                        {
                            bi = k;
                            break;
                        }

                        if (isMax ? x > best : x < best)
                        {
                            best = x;
                            bi = k;
                        }
                    }
                }

                r[o * inner + i] = bi;
            }

        return r;
    }

    /// <summary>Reference fold for bool lanes: argmax = first True, argmin = first False, 0 when there is none.</summary>
    /// <param name="v">The operand's values in C order.</param>
    /// <param name="outer">Product of the dims before the axis.</param>
    /// <param name="len">The axis length.</param>
    /// <param name="inner">Product of the dims after the axis.</param>
    /// <param name="isMax">Argmax (first True) or argmin (first False).</param>
    /// <returns>One index per output, in C order of the reduced shape.</returns>
    private static long[] RefBool(bool[] v, long outer, long len, long inner, bool isMax)
    {
        var r = new long[outer * inner];
        for (long o = 0; o < outer; o++)
            for (long i = 0; i < inner; i++)
            {
                long bi = 0;
                for (long k = 0; k < len; k++)
                {
                    if (v[(o * len + k) * inner + i] == isMax)
                    {
                        bi = k;
                        break;
                    }
                }

                r[o * inner + i] = bi;
            }

        return r;
    }

    /// <summary>
    /// A deterministic pool with many ties (small value range) plus the dtype's specials sprinkled in: NaN / ±0 / ±inf for
    /// floats, the extremes for integers, and a mixed True/False pattern for bool.
    /// </summary>
    /// <param name="shape">The C-contiguous shape to build.</param>
    /// <param name="tc">The dtype (one of <see cref="FastTypes"/>).</param>
    /// <param name="seed">The RNG seed, so every run tests the same values.</param>
    /// <returns>A fresh C-contiguous array.</returns>
    private static NDArray Pool(long[] shape, NPTypeCode tc, int seed)
    {
        var rng = new System.Random(seed);
        long n = 1;
        foreach (var d in shape) n *= d;
        var v = new double[n];
        bool isFloat = tc is NPTypeCode.Single or NPTypeCode.Double;
        for (long i = 0; i < n; i++)
        {
            int roll = rng.Next(60);
            v[i] = roll switch
            {
                0 when isFloat => double.NaN,
                1 when isFloat => -0.0,
                2 when isFloat => double.PositiveInfinity,
                3 when isFloat => double.NegativeInfinity,
                _ => rng.Next(0, 9),
            };
        }

        var a = np.array(v).reshape(new Shape(shape));
        if (tc == NPTypeCode.Boolean)
            return a > 4.0;
        var cast = a.astype(tc);
        if (!isFloat && tc != NPTypeCode.Boolean && n > 4)
        {
            // The integer extremes: an extreme is where the fold may stop early, so plant a few of each.
            // maxUnsigned, not max: iinfo clamps a uint64's max to long.MaxValue.
            var flat = cast.reshape(n);
            var info = np.iinfo(tc);
            flat[rng.Next((int)n)] = NDArray.Scalar(info.maxUnsigned).astype(tc);
            flat[rng.Next((int)n)] = NDArray.Scalar(info.min).astype(tc);
            flat[rng.Next((int)n)] = NDArray.Scalar(info.maxUnsigned).astype(tc);
        }

        return cast;
    }

    /// <summary>The dtypes the fast fold serves (Half / Decimal / Complex decline — see <see cref="Declines_KeepTheLegacyKernels"/>).</summary>
    private static readonly NPTypeCode[] FastTypes =
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double,
    };

    /// <summary>
    /// Every served dtype through every walk: C rows short and long (Rows), axis 0 of a C block (vector Slab), an F block
    /// (Rows on axis 0, Slab on axis 1), a reversed inner run (flipped Slab), a stride-2 inner run (scalar Slab), a
    /// reversed axis (AxisWalk with stride -1 / a Slab with a negative axis stride), a column-strided block (AxisWalk), a
    /// sliced view at a non-zero offset, and every axis of a 3-D block — argmax and argmin, keepdims both ways.
    /// </summary>
    [TestMethod]
    public void EveryDtype_EveryWalk_MatchesLegacyAndReference()
    {
        foreach (var tc in FastTypes)
        {
            var c2 = Pool(new long[] { 37, 45 }, tc, 11 + (int)tc);
            var wide = Pool(new long[] { 40, 90 }, tc, 23 + (int)tc);
            var c3 = Pool(new long[] { 4, 6, 35 }, tc, 31 + (int)tc);
            var layouts = new List<(string, NDArray)>
            {
                ("C", c2),
                ("short rows", Pool(new long[] { 60, 7 }, tc, 41 + (int)tc)),      // axis 1: rows under 32 fold inline

                ("F", np.asfortranarray(c2)),
                ("T", c2.T),
                ("reversed inner", c2[":, ::-1"]),
                ("reversed outer", c2["::-1, :"]),
                ("stride-2 inner", wide[":, ::2"]),
                ("column block at offset", wide["3:, 5:50"]),
                ("3-D C", c3),
                ("3-D F", np.asfortranarray(c3)),
                ("3-D transposed", c3.transpose(new[] { 2, 0, 1 })),
                // Rows padded to 40 but cut at 35: the outer stride (240) exceeds 6 × 35, so the two non-axis dims of
                // axis 1 must NOT coalesce into one run (merging them would read the padding).
                ("3-D sliced inner", Pool(new long[] { 4, 6, 40 }, tc, 47 + (int)tc)[":, :, :35"]),
            };
            foreach (var (name, a) in layouts)
                for (int axis = -a.ndim; axis < a.ndim; axis++)
                    foreach (bool isMax in new[] { true, false })
                        AssertFastMatches(a, axis, isMax, keepdims: axis == 0, $"{tc} {name} axis {axis} {(isMax ? "argmax" : "argmin")}");
        }
    }

    /// <summary>
    /// The Slab fold's lane chunking: 3000 lanes span three 1024-lane chunks, and lane counts around the vector width
    /// (5, 33, 1023, 1025) exercise the overlapping final vector. Monotone increasing rows make EVERY row improve every
    /// lane (the write-back path); decreasing rows never improve after row 0 (the seed path).
    /// </summary>
    [TestMethod]
    public void Slab_ChunksVectorTailsAndMonotoneRows()
    {
        foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Byte, NPTypeCode.Int16 })
        {
            foreach (long lanes in new long[] { 5, 33, 1023, 1025, 3000 })
            {
                var pool = Pool(new long[] { 9, lanes }, tc, (int)lanes);
                AssertFastMatches(pool, 0, true, false, $"{tc} {lanes} lanes argmax");
                AssertFastMatches(pool, 0, false, false, $"{tc} {lanes} lanes argmin");
            }

            var up = (np.arange(12 * 3000) / 3000).astype(tc).reshape(12, 3000);          // row k holds k: every row improves
            AssertFastMatches(up, 0, true, false, $"{tc} increasing rows argmax");
            AssertFastMatches(up, 0, false, false, $"{tc} increasing rows argmin");
            var down = up["::-1, :"].copy();
            AssertFastMatches(down, 0, true, false, $"{tc} decreasing rows argmax");
            AssertFastMatches(down, 0, false, false, $"{tc} decreasing rows argmin");
            Assert.AreEqual(11L, np.argmax(up, 0).GetAtIndex<long>(2999), $"{tc}: the last row wins every lane");
        }
    }

    /// <summary>
    /// The deciding value (a NaN, an integer extreme, the first True / False) placed at the chunk and vector boundaries of
    /// a Slab and at the start, middle and end of Rows: the fold may stop at it, so an early stop in the wrong place or a
    /// missed element shows as a wrong index.
    /// </summary>
    [TestMethod]
    public void DecidingValues_AtBoundaries()
    {
        long[] hotLanes = { 0, 3, 4, 1023, 1024, 1025, 2047, 2048, 2999 };
        // Slab: 20 rows x 3000 lanes, each hot lane decided on a different row.
        var d = np.zeros(new Shape(20, 3000), NPTypeCode.Double);
        var l = np.zeros(new Shape(20, 3000), NPTypeCode.Int64);
        var b = np.zeros(new Shape(20, 3000), NPTypeCode.Boolean);
        for (int h = 0; h < hotLanes.Length; h++)
        {
            int row = (h * 7) % 20, lane = (int)hotLanes[h];
            d[row, lane] = double.NaN;
            d[(row + 3) % 20, lane] = double.NaN;               // a second NaN later on: the first must win
            l[row, lane] = long.MaxValue;
            l[(row + 5) % 20, lane] = long.MaxValue;
            b[row, lane] = true;
        }

        foreach (bool isMax in new[] { true, false })
        {
            AssertFastMatches(d, 0, isMax, false, $"f64 NaN slab {isMax}");
            AssertFastMatches(l, 0, isMax, false, $"i64 extreme slab {isMax}");
            AssertFastMatches(b, 0, isMax, false, $"bool slab {isMax}");
            // The same data transposed puts the deciding values along contiguous rows (Rows mode).
            AssertFastMatches(np.ascontiguousarray(d.T), 1, isMax, false, $"f64 NaN rows {isMax}");
            AssertFastMatches(np.ascontiguousarray(l.T), 1, isMax, false, $"i64 extreme rows {isMax}");
            AssertFastMatches(np.ascontiguousarray(b.T), 1, isMax, false, $"bool rows {isMax}");
        }

        // Bool early exit: every lane decided on row 0 (argmax of all-True stops at once) vs one lane decided only on the
        // last row (the fold must keep going for it alone).
        var allTrue = np.ones(new Shape(50, 2000), NPTypeCode.Boolean);
        AssertFastMatches(allTrue, 0, true, false, "all-True argmax");
        AssertFastMatches(allTrue, 0, false, false, "all-True argmin (none False → 0)");
        var late = np.zeros(new Shape(50, 2000), NPTypeCode.Boolean);
        late["0, :"] = true;
        late[0, 1500] = false;
        late[49, 1500] = true;
        AssertFastMatches(late, 0, true, false, "one lane decided on the last row");
        Assert.AreEqual(49L, np.argmax(late, 0).GetAtIndex<long>(1500));
    }

    /// <summary>NumPy 2.4.2's own answers for the selection rule's edges, through C, F and transposed layouts.</summary>
    [TestMethod]
    public void NumPyPinnedValues()
    {
        // np.argmax/argmin(m, 0) = [1 2 0 2] (both), axis 1 = [2 0 1] (both): the first NaN wins either way.
        var m = np.array(new double[,] { { 1, 5, double.NaN, 2 }, { double.NaN, 7, 1, 2 }, { 3, double.NaN, 0, double.NaN } });
        foreach (var v in new[] { m, np.asfortranarray(m), np.ascontiguousarray(m.T).T })
        {
            CollectionAssert.AreEqual(new long[] { 1, 2, 0, 2 }, np.argmax(v, 0).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 2, 0, 2 }, np.argmin(v, 0).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 2, 0, 1 }, np.argmax(v, 1).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 2, 0, 1 }, np.argmin(v, 1).ToArray<long>());
        }

        // bool: argmax ax0 [2 0 1 0], argmin ax0 [0 0 0 0], argmax ax1 [0 2 0], argmin ax1 [0 0 1].
        var b = np.zeros(new Shape(3, 4), NPTypeCode.Boolean);
        b[1, 2] = true;
        b[2, 0] = true;
        CollectionAssert.AreEqual(new long[] { 2, 0, 1, 0 }, np.argmax(b, 0).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0 }, np.argmin(b, 0).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 0, 2, 0 }, np.argmax(b, 1).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 0, 0, 1 }, np.argmin(b, 1).ToArray<long>());

        // int8 extremes: argmax ax0 [2 2 1], argmin ax0 [1 0 2], argmax ax1 [0 2 0], argmin ax1 [1 0 2].
        var i8 = np.array(new sbyte[,] { { 5, -128, 3 }, { -128, 2, 127 }, { 127, 127, -1 } });
        CollectionAssert.AreEqual(new long[] { 2, 2, 1 }, np.argmax(i8, 0).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 1, 0, 2 }, np.argmin(i8, 0).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 0, 2, 0 }, np.argmax(i8, 1).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 1, 0, 2 }, np.argmin(i8, 1).ToArray<long>());

        // -0.0 == +0.0: the earlier of the two wins for both argmax and argmin.
        var z = np.array(new double[,] { { -0.0, 0.0 }, { 0.0, -0.0 } });
        CollectionAssert.AreEqual(new long[] { 0, 0 }, np.argmax(z, 0).ToArray<long>());
        CollectionAssert.AreEqual(new long[] { 0, 0 }, np.argmin(z, 1).ToArray<long>());
    }

    /// <summary>
    /// The two cheap walks: a broadcast axis (stride 0 — Zero mode, every answer 0 without reading a value) and a
    /// broadcast of a strided vector reduced along its real axis (the only non-zero run is the axis itself, so the lanes
    /// would all re-read one element — AxisWalk instead).
    /// </summary>
    [TestMethod]
    public void BroadcastAxis_AndBroadcastRuns()
    {
        var row = np.array(new double[] { 3, 9, 1, 9, 7 });
        var bc = np.broadcast_to(row, new Shape(6, 5));
        AssertFastMatches(bc, 0, true, false, "broadcast axis 0 (Zero)");
        AssertFastMatches(bc, 0, false, true, "broadcast axis 0 keepdims (Zero)");
        AssertFastMatches(bc, 1, true, false, "broadcast rows (Rows over stride-0 outer)");
        CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0, 0 }, np.argmax(bc, 0).ToArray<long>());

        var strided = np.arange(20).astype(NPTypeCode.Int32)["::2"].reshape(1, 10);   // strides (…, 2)
        var bcs = np.broadcast_to(strided, new Shape(4, 10));
        AssertFastMatches(bcs, 1, true, false, "strided vector broadcast, axis 1 (AxisWalk)");
        AssertFastMatches(bcs, 0, false, false, "strided vector broadcast, axis 0 (Zero)");
    }

    /// <summary>
    /// Declines: Half, Decimal and Complex keep their dedicated kernels, and a rank above 64 keeps the legacy kernel
    /// (the plan is stack-allocated by rank) — each still answers correctly, without the fast fold running.
    /// </summary>
    [TestMethod]
    public void Declines_KeepTheLegacyKernels()
    {
        // Argmax along `axis` must equal `want` while FastAxisArgRuns stays put — the answer came from the legacy kernel.
        void AssertDeclined(NDArray a, int axis, long[] want, string what)
        {
            long before = DefaultEngine.FastAxisArgRuns;
            CollectionAssert.AreEqual(want, np.argmax(a, axis).ToArray<long>(), what);
            Assert.AreEqual(before, DefaultEngine.FastAxisArgRuns, $"{what}: declined");
        }

        var baseVals = np.array(new double[,] { { 1, 3 }, { 2, 0 } });
        AssertDeclined(baseVals.astype(NPTypeCode.Half), 0, new long[] { 1, 0 }, "Half");
        AssertDeclined(baseVals.astype(NPTypeCode.Decimal), 0, new long[] { 1, 0 }, "Decimal");
        AssertDeclined(baseVals.astype(NPTypeCode.Complex), 0, new long[] { 1, 0 }, "Complex");

        var dims = new long[65];
        for (int i = 0; i < 65; i++) dims[i] = 1;
        dims[0] = 2;
        dims[64] = 3;
        var deep = np.arange(6).astype(NPTypeCode.Double).reshape(new Shape(dims));
        AssertDeclined(deep, 0, new long[] { 1, 1, 1 }, "rank 65");

        // The switch forces the legacy route on this thread only, and the fast fold resumes once it is cleared.
        DefaultEngine.DisableFastAxisArg = true;
        try
        {
            AssertDeclined(baseVals, 0, new long[] { 1, 0 }, "disabled");
        }
        finally
        {
            DefaultEngine.DisableFastAxisArg = false;
        }

        AssertFastMatches(baseVals, 0, true, false, "re-enabled");
    }

    /// <summary>
    /// The flat SIMD row kernels every Rows fold (and every flat argmax / argmin) runs, on TIE-HEAVY finite data: a
    /// four-value pool puts the extreme in many lanes of many blocks, which is exactly where a tournament's
    /// cross-block accumulator move (strict or not) and its lowest-index horizontal reduce decide the answer — a pool whose
    /// extreme is a lone inf or NaN never exercises either. Lengths straddle each kernel's vector block and unroll group.
    /// </summary>
    [TestMethod]
    public void RowKernels_TieHeavyRows_KeepTheFirstOccurrence()
    {
        var rng = new System.Random(4);
        foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Int16, NPTypeCode.SByte, NPTypeCode.Byte })
        {
            foreach (int n in new[] { 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4099 })
            {
                for (int rep = 0; rep < 4; rep++)
                {
                    var v = new double[n];
                    for (int i = 0; i < n; i++) v[i] = rng.Next(0, 4);
                    var a = np.array(v).astype(tc);
                    foreach (bool isMax in new[] { true, false })
                    {
                        long want = Reference(a.reshape(1, n), 1, isMax)[0];
                        Assert.AreEqual(want, isMax ? np.argmax(a) : np.argmin(a), $"{tc} flat n={n} rep={rep} {(isMax ? "argmax" : "argmin")}");
                    }
                }

                // The same kernels as a Rows fold: 6 tie-heavy rows of length n (the axis is contiguous).
                var m = new double[6 * n];
                for (int i = 0; i < m.Length; i++) m[i] = rng.Next(0, 4);
                var rows = np.array(m).astype(tc).reshape(6, n);
                AssertFastMatches(rows, 1, true, false, $"{tc} tie-heavy rows n={n} argmax");
                AssertFastMatches(rows, 1, false, false, $"{tc} tie-heavy rows n={n} argmin");
            }
        }
    }

    /// <summary>
    /// The 64-bit single-pass tournament behind the flat int64 / uint64 argmax / argmin (and every 64-bit Rows fold):
    /// lengths across the 16-element block and 4-lane vector edges, ties spread over lanes and blocks (the horizontal
    /// reduce must return the LOWEST index among equal lanes), the extremes, and uint64 values above 2^63 (an unsigned
    /// compare — a signed one would call 2^63+5 negative).
    /// </summary>
    [TestMethod]
    public void Int64Tournament_TiesExtremesAndUnsigned()
    {
        var rng = new System.Random(64);
        foreach (int n in new[] { 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000, 4099 })
        {
            for (int rep = 0; rep < 6; rep++)
            {
                var lv = new long[n];
                var uv = new ulong[n];
                for (int i = 0; i < n; i++)
                {
                    lv[i] = rng.Next(0, 5) - 2;                                   // heavy ties
                    uv[i] = (ulong)rng.Next(0, 5) + (rep % 2 == 0 ? 0UL : 1UL << 63);
                }

                if (rep == 2 && n > 3)
                {
                    lv[rng.Next(n)] = long.MaxValue;
                    lv[rng.Next(n)] = long.MinValue;
                    uv[rng.Next(n)] = ulong.MaxValue;
                }

                var la = np.array(lv);
                var ua = np.array(uv);
                // The whole vector as ONE row of the reference fold.
                long[] Ref<T>(T[] v, bool isMax) where T : IBinaryInteger<T> => RefInt(v, 1, v.Length, 1, isMax);
                Assert.AreEqual(Ref(lv, true)[0], np.argmax(la), $"int64 argmax n={n} rep={rep}");
                Assert.AreEqual(Ref(lv, false)[0], np.argmin(la), $"int64 argmin n={n} rep={rep}");
                Assert.AreEqual(Ref(uv, true)[0], np.argmax(ua), $"uint64 argmax n={n} rep={rep}");
                Assert.AreEqual(Ref(uv, false)[0], np.argmin(ua), $"uint64 argmin n={n} rep={rep}");
            }
        }

        // NumPy 2.4.2: a max at 5, 8 and 33 → 5 (lane 1 of the first block beats lane 0's later hit); extremes → 1;
        // uint64 [1, 2^63+5, 2^63+2, 2^63+5] → argmax 1, argmin 0.
        var tie = new long[40];
        tie[5] = 9;
        tie[8] = 9;
        tie[33] = 9;
        Assert.AreEqual(5L, np.argmax(np.array(tie)));
        Assert.AreEqual(5L, np.argmin(-np.array(tie)));
        Assert.AreEqual(1L, np.argmax(np.array(new long[] { 3, long.MaxValue, 1, long.MaxValue })));
        var u = np.array(new ulong[] { 1, (1UL << 63) + 5, (1UL << 63) + 2, (1UL << 63) + 5 });
        Assert.AreEqual(1L, np.argmax(u));
        Assert.AreEqual(0L, np.argmin(u));
        Assert.AreEqual(6L, np.argmax(np.arange(40) % 7));
        Assert.AreEqual(0L, np.argmin(np.arange(40) % 7));
    }
}
