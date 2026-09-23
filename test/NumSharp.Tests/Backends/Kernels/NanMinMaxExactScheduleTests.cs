using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// The NumPy-exact <c>np.nanmax</c> / <c>np.nanmin</c> schedules — on an ndarray NumPy's <c>nanmax</c> IS
/// <c>np.fmax.reduce</c> — shared by the engine and np.evaluate's <c>NanMax</c> / <c>NanMin</c> routes
/// (<c>Default.Reduction.MinMax.Exact.cs</c>, <c>DefaultEngine.Evaluate.MinMax*.cs</c>).
/// </summary>
/// <remarks>
/// <para>
/// fmax has TWO per-element rules, and which one an element meets is what these tests pin: the loop's VECTOR op
/// (<c>npyv_maxp</c> — a ±0 tie keeps the later operand, two NaNs keep the FIRST) and its SCALAR op (the CRT
/// <c>fmax</c> — a ±0 tie keeps <c>+0</c> for max / <c>-0</c> for min, two NaNs keep the LAST). An element gets the
/// vector op inside a call's vector section and the scalar op in the call's tail, in a strided row's 8-accumulator
/// unroll, and in every call NumPy's SIMD branch refuses — so the answer depends on where NumPy's inner-loop calls fall
/// (the SLAB call tiling, the flat buffer fills, the ROW mode). No all-NaN reduction ever returns the canonical NaN.
/// </para>
/// <para>
/// Every literal below was probed against NumPy 2.4.2 (win-amd64, AVX2 dispatch) with the SAME construction — the
/// script mirrors each test line by line — and NaN payloads carry an offset, so no expected NaN is the canonical one a
/// canonicalizing bug would also produce. Exact cases assert the route ENGAGED (<see cref="NDExpr.ExactMinMaxRuns"/>)
/// and that the engine and the np.evaluate bare leaf agree bit for bit: a silent decline to the old kernels would still
/// pass a value check.
/// </para>
/// </remarks>
[TestClass]
public class NanMinMaxExactScheduleTests
{
    /// <summary>A real negative zero held in a field (a <c>-0.0</c> literal is easy to lose to constant folding).</summary>
    private static readonly double NegZero = -0.0;

    /// <summary>The float32 negative zero, held in a field for the same reason.</summary>
    private static readonly float NegZeroF = -0.0f;

    /// <summary>float64 <c>-0</c> bits.</summary>
    private const ulong N0 = 0x8000000000000000UL;

    /// <summary>float64 <c>+0</c> bits.</summary>
    private const ulong P0 = 0x0UL;

    /// <summary>float32 <c>-0</c> bits.</summary>
    private const uint N0F = 0x80000000u;

    /// <summary>float32 <c>+0</c> bits.</summary>
    private const uint P0F = 0x0u;

    /// <summary>A float64 with the exact bit pattern <paramref name="bits"/>.</summary>
    /// <param name="bits">The raw IEEE-754 binary64 bits.</param>
    /// <returns>The double those bits encode.</returns>
    private static double D(ulong bits) => BitConverter.Int64BitsToDouble((long)bits);

    /// <summary>A float32 with the exact bit pattern <paramref name="bits"/>.</summary>
    /// <param name="bits">The raw IEEE-754 binary32 bits.</param>
    /// <returns>The float those bits encode.</returns>
    private static float F(uint bits) => BitConverter.Int32BitsToSingle((int)bits);

    /// <summary>
    /// A quiet float64 NaN carrying <paramref name="payload"/> — positive (<c>0x7FF8…</c>) or, with
    /// <paramref name="negative"/>, negative (<c>0xFFF8…</c>).
    /// </summary>
    /// <param name="payload">The low mantissa bits (must leave the quiet bit as the only other set bit).</param>
    /// <param name="negative">Set the sign bit.</param>
    /// <returns>The NaN.</returns>
    private static double Q64(ulong payload, bool negative = false)
        => D((negative ? 0xFFF8000000000000UL : 0x7FF8000000000000UL) | payload);

    /// <summary>A quiet positive float32 NaN carrying <paramref name="payload"/> (<c>0x7FC00000 | payload</c>).</summary>
    /// <param name="payload">The low mantissa bits (&lt; 2^22).</param>
    /// <returns>The NaN.</returns>
    private static float Q32(uint payload) => F(0x7FC00000u | payload);

    /// <summary>The raw bits of every element of a float64 result, in C order.</summary>
    /// <param name="r">A float64 array (any layout).</param>
    /// <returns>One bit pattern per element.</returns>
    private static ulong[] Bits64(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Double, r.typecode);
        return np.ascontiguousarray(r).ToArray<double>().Select(v => (ulong)BitConverter.DoubleToInt64Bits(v)).ToArray();
    }

    /// <summary>The raw bits of every element of a float32 result, in C order.</summary>
    /// <param name="r">A float32 array (any layout).</param>
    /// <returns>One bit pattern per element.</returns>
    private static uint[] Bits32(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Single, r.typecode);
        return np.ascontiguousarray(r).ToArray<float>().Select(v => (uint)BitConverter.SingleToInt32Bits(v)).ToArray();
    }

    /// <summary>
    /// The C-order positions of a float result whose sign bit is CLEAR — for a ±0 tie pattern, the elements that met
    /// the CRT scalar op of max (or the vector op of min); the rest are <c>-0</c>.
    /// </summary>
    /// <param name="r">A float32 or float64 array.</param>
    /// <returns>The positions, ascending.</returns>
    private static long[] PositiveSignPositions(NDArray r)
    {
        var flat = np.ascontiguousarray(r);
        long n = flat.size;
        var list = new System.Collections.Generic.List<long>();
        for (long i = 0; i < n; i++)
        {
            bool negative = r.typecode == NPTypeCode.Double
                ? double.IsNegative(flat.GetAtIndex<double>(i))
                : float.IsNegative(flat.GetAtIndex<float>(i));
            if (!negative)
                list.Add(i);
        }

        return list.ToArray();
    }

    /// <summary>An array of <paramref name="n"/> copies of <paramref name="bits"/>.</summary>
    /// <param name="bits">The repeated bit pattern.</param>
    /// <param name="n">The count.</param>
    /// <returns>The array.</returns>
    private static T[] Repeat<T>(T bits, int n) => Enumerable.Repeat(bits, n).ToArray();

    /// <summary>
    /// Asserts two results have the same dtype, shape and bits.
    /// </summary>
    /// <param name="expected">The reference result.</param>
    /// <param name="actual">The result to check.</param>
    /// <param name="what">A label for failure messages.</param>
    private static void AssertSameBits(NDArray expected, NDArray actual, string what)
    {
        Assert.AreEqual(expected.typecode, actual.typecode, $"{what}: dtype");
        CollectionAssert.AreEqual(expected.shape, actual.shape, $"{what}: shape");
        if (expected.typecode == NPTypeCode.Double)
            CollectionAssert.AreEqual(Bits64(expected), Bits64(actual), $"{what}: bits");
        else if (expected.typecode == NPTypeCode.Single)
            CollectionAssert.AreEqual(Bits32(expected), Bits32(actual), $"{what}: bits");
        else
            CollectionAssert.AreEqual(np.ascontiguousarray(expected).ToArray<int>(), np.ascontiguousarray(actual).ToArray<int>(), $"{what}: values");
    }

    /// <summary>
    /// The FLAT reduction through the engine (<c>np.nanmax</c> / <c>np.nanmin</c>) and the np.evaluate bare leaf,
    /// asserting both took the exact schedule and agree bit for bit; returns the engine result.
    /// </summary>
    /// <param name="x">The operand (any non-broadcast float32 / float64 layout).</param>
    /// <param name="isMax">nanmax when true, nanmin when false.</param>
    /// <param name="what">A label for failure messages.</param>
    /// <returns>The engine's 0-d result.</returns>
    private static NDArray FlatNan(NDArray x, bool isMax, string what)
    {
        int before = NDExpr.ExactMinMaxRuns;
        var engine = isMax ? np.nanmax(x) : np.nanmin(x);
        Assert.AreEqual(before + 1, NDExpr.ExactMinMaxRuns, $"{what}: the engine's flat reduction must take the exact schedule");
        var leaf = np.evaluate(isMax ? NDExpr.NanMax((NDExpr)x) : NDExpr.NanMin((NDExpr)x));
        Assert.AreEqual(before + 2, NDExpr.ExactMinMaxRuns, $"{what}: the np.evaluate bare leaf must take the exact schedule");
        Assert.AreEqual(0, engine.ndim, $"{what}: a flat result is 0-d");
        AssertSameBits(engine, leaf, $"{what}: engine vs np.evaluate");
        return engine;
    }

    /// <summary>
    /// The AXIS reduction through the engine and the np.evaluate bare leaf, asserting both took the exact schedule and
    /// agree bit for bit; returns the engine result.
    /// </summary>
    /// <param name="x">The operand (any non-broadcast float32 / float64 layout).</param>
    /// <param name="axis">The reduced axis.</param>
    /// <param name="isMax">nanmax when true, nanmin when false.</param>
    /// <param name="what">A label for failure messages.</param>
    /// <returns>The engine's result.</returns>
    private static NDArray AxisNan(NDArray x, int axis, bool isMax, string what)
    {
        int before = NDExpr.ExactMinMaxRuns;
        var engine = isMax ? np.nanmax(x, axis) : np.nanmin(x, axis);
        Assert.AreEqual(before + 1, NDExpr.ExactMinMaxRuns, $"{what}: the engine's axis reduction must take the exact schedule");
        var leaf = np.evaluate(isMax ? NDExpr.NanMax((NDExpr)x, axis) : NDExpr.NanMin((NDExpr)x, axis));
        Assert.AreEqual(before + 2, NDExpr.ExactMinMaxRuns, $"{what}: the np.evaluate bare leaf must take the exact schedule");
        AssertSameBits(engine, leaf, $"{what}: engine vs np.evaluate");
        return engine;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // AXIS — SLAB mode: every reduced index is folded in by calls tiling the non-reduced ("P") region
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// C <c>(3, 7)</c> float64 along axis 0: the P region is the 7 columns, one call of 7 per reduced index — lanes 0..3
    /// are the vector section (<c>maxp</c>: the later <c>-0</c> wins the tie), 4..6 the scalar tail (CRT <c>fmax</c>:
    /// <c>+0</c> wins). nanmin mirrors it; float32 <c>(3, 11)</c> has eight lanes and a three-element tail.
    /// </summary>
    [TestMethod]
    public void AxisSlab_OneCall_VectorSectionThenScalarTail()
    {
        var a = np.zeros(new Shape(3, 7));
        a[2] = NDArray.Scalar(NegZero);
        CollectionAssert.AreEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, Bits64(AxisNan(a, 0, true, "f64 max")));

        var b = np.full(new Shape(3, 7), NegZero);
        b[2] = NDArray.Scalar(0.0);
        CollectionAssert.AreEqual(new[] { P0, P0, P0, P0, N0, N0, N0 }, Bits64(AxisNan(b, 0, false, "f64 min")));

        var c = np.zeros(new Shape(3, 11), NPTypeCode.Single);
        c[2] = NDArray.Scalar(NegZeroF);
        CollectionAssert.AreEqual(Repeat(N0F, 8).Concat(Repeat(P0F, 3)).ToArray(), Bits32(AxisNan(c, 0, true, "f32 max")));
    }

    /// <summary>
    /// All-NaN columns keep a PAYLOAD, never the canonical NaN: the vector section keeps the FIRST NaN (<c>maxp</c>
    /// returns its first operand when the second is NaN), the tail the LAST (the CRT returns its second operand when the
    /// first is NaN). With twelve reduced indices the eight-slab fused step runs, split at the same call boundary.
    /// </summary>
    [TestMethod]
    public void AxisSlab_AllNaNColumns_VectorKeepsTheFirstNaN_TailTheLast()
    {
        var a = np.empty(new Shape(3, 7));
        a[":2"] = NDArray.Scalar(Q64(0xA0A, negative: true));
        a[2] = NDArray.Scalar(Q64(0xB0B));
        CollectionAssert.AreEqual(
            new[] { 0xFFF8000000000A0AUL, 0xFFF8000000000A0AUL, 0xFFF8000000000A0AUL, 0xFFF8000000000A0AUL, 0x7FF8000000000B0BUL, 0x7FF8000000000B0BUL, 0x7FF8000000000B0BUL },
            Bits64(AxisNan(a, 0, true, "3 rows")));

        var f = np.empty(new Shape(12, 7));
        for (int k = 0; k < 12; k++)
            f[k] = NDArray.Scalar(Q64(0x300 + (ulong)k));
        var want = Repeat(0x7FF8000000000300UL, 4).Concat(Repeat(0x7FF800000000030BUL, 3)).ToArray();
        CollectionAssert.AreEqual(want, Bits64(AxisNan(f, 0, true, "fused max")));
        CollectionAssert.AreEqual(want, Bits64(AxisNan(f, 0, false, "fused min")));

        var t = np.zeros(new Shape(12, 7));
        t[11] = NDArray.Scalar(NegZero);
        CollectionAssert.AreEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, Bits64(AxisNan(t, 0, true, "fused tie")));
    }

    /// <summary>
    /// Contiguous runs that do NOT start at P position 0: <c>(K, 3, 5)[:, :, :3]</c> folds three runs of 3 — at
    /// positions 0, 3 and 6 of one 9-position call — through the eight-slab fused step, so the run at position 6 must
    /// split where the CALL's vector section ends (position 8 is the tail), not where the run starts. K = 9 is exactly
    /// one fused group, so the deciding slab (the last) is folded by the fused step itself; K = 12 adds a one-slab
    /// remainder after the group, whose later slabs would otherwise overwrite (and so hide) a wrong fused split.
    /// </summary>
    [TestMethod]
    public void AxisSlab_FusedRunsAtNonZeroPositions_SplitAtTheCallsBoundary()
    {
        var g = np.full(new Shape(9, 3, 5), 5.0);
        var vg = g[":, :, :3"];
        vg[":8"] = NDArray.Scalar(0.0);
        vg[8] = NDArray.Scalar(NegZero);
        CollectionAssert.AreEqual(Repeat(N0, 8).Concat(new[] { P0 }).ToArray(), Bits64(AxisNan(vg, 0, true, "K=9 tie")));
        var gn = np.full(new Shape(9, 3, 5), 5.0);
        var vgn = gn[":, :, :3"];
        for (int k = 0; k < 9; k++)
            vgn[k] = NDArray.Scalar(Q64(0x600 + (ulong)k));
        var want9 = Repeat(0x7FF8000000000600UL, 8).Concat(new[] { 0x7FF8000000000608UL }).ToArray();
        CollectionAssert.AreEqual(want9, Bits64(AxisNan(vgn, 0, true, "K=9 all-NaN max")));
        CollectionAssert.AreEqual(want9, Bits64(AxisNan(vgn, 0, false, "K=9 all-NaN min")));

        var b = np.full(new Shape(12, 3, 5), 5.0);
        var v = b[":, :, :3"];
        v[":11"] = NDArray.Scalar(0.0);
        v[11] = NDArray.Scalar(NegZero);
        CollectionAssert.AreEqual(Repeat(N0, 8).Concat(new[] { P0 }).ToArray(), Bits64(AxisNan(v, 0, true, "tie")));

        var n = np.full(new Shape(12, 3, 5), 5.0);
        var vn = n[":, :, :3"];
        for (int k = 0; k < 12; k++)
            vn[k] = NDArray.Scalar(Q64(0x500 + (ulong)k));
        var want = Repeat(0x7FF8000000000500UL, 8).Concat(new[] { 0x7FF800000000050BUL }).ToArray();
        CollectionAssert.AreEqual(want, Bits64(AxisNan(vn, 0, true, "all-NaN max")));
        CollectionAssert.AreEqual(want, Bits64(AxisNan(vn, 0, false, "all-NaN min")));
    }

    /// <summary>
    /// A float32 input iterated IN PLACE with a negative innermost stride runs EVERY call through NumPy's scalar loop
    /// (<c>npyv_loadable_stride_f32</c> divides the signed stride by the unsigned <c>sizeof(float)</c>, so the SIMD
    /// branch refuses it): the whole <c>(3, 7)[:, ::-1]</c> result is the CRT answer, and a call of 20 — wider than a
    /// vector — is still scalar end to end. float64 on the same layouts keeps its vector section.
    /// </summary>
    [TestMethod]
    public void AxisSlab_Float32NegativeStride_EveryCallIsScalar()
    {
        var b = np.full(new Shape(3, 7), 5f, NPTypeCode.Single);
        var v = b[":, ::-1"];
        b[":2"] = NDArray.Scalar(0f);
        b[2] = NDArray.Scalar(NegZeroF);
        CollectionAssert.AreEqual(Repeat(P0F, 7), Bits32(AxisNan(v, 0, true, "f32 tie")));

        var b64 = np.full(new Shape(3, 7), 5.0);
        b64[":2"] = NDArray.Scalar(0.0);
        b64[2] = NDArray.Scalar(NegZero);
        CollectionAssert.AreEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, Bits64(AxisNan(b64[":, ::-1"], 0, true, "f64 tie")));

        var w = np.full(new Shape(3, 20), 5f, NPTypeCode.Single);
        var w64 = np.full(new Shape(3, 20), 5.0);
        for (int k = 0; k < 3; k++)
        {
            w[k] = NDArray.Scalar(Q32(0x40 + (uint)k));
            w64[k] = NDArray.Scalar(Q64(0x40 + (ulong)k));
        }

        CollectionAssert.AreEqual(Repeat(0x7FC00042u, 20), Bits32(AxisNan(w[":, ::-1"], 0, true, "f32 all-NaN, call of 20")));
        CollectionAssert.AreEqual(Repeat(0x7FF8000000000040UL, 20), Bits64(AxisNan(w64[":, ::-1"], 0, true, "f64 all-NaN")));
    }

    /// <summary>
    /// A BUFFERED core: <c>(2, 9, 3)[:, :, ::2]</c> does not coalesce, so NumPy's cost model grows the core over both
    /// P axes and runs ONE call over the whole 18-position region — 16 vector lanes, a two-element tail — where per-row
    /// calls of 2 would have been all scalar.
    /// </summary>
    [TestMethod]
    public void AxisSlab_BufferedCore_OneCallOverTheWholeRegion()
    {
        var b = np.full(new Shape(2, 9, 3), 5.0);
        var v = b[":, :, ::2"];
        v[0] = NDArray.Scalar(0.0);
        v[1] = NDArray.Scalar(NegZero);
        var r = AxisNan(v, 0, true, "buffered core");
        CollectionAssert.AreEqual(new long[] { 9, 2 }, r.shape);
        CollectionAssert.AreEqual(new long[] { 16, 17 }, PositiveSignPositions(r));
    }

    /// <summary>
    /// The buffer caps a call: <c>(2, 3000, 5)[:, :, ::2]</c> float32 has a 9000-position P region but NumPy buffers
    /// whole cores of 3 up to 8190 elements, so there are TWO calls — tails at 8184..8189 and 8998..8999. A larger
    /// <see cref="np.setbufsize"/> makes it one call of 9000 (a multiple of eight: no tail at all), exactly as NumPy's
    /// <c>np.setbufsize</c> does.
    /// </summary>
    [TestMethod]
    public void AxisSlab_ChunkCappedBuffer_TailAtEveryChunkEnd()
    {
        var b = np.full(new Shape(2, 3000, 5), 5f, NPTypeCode.Single);
        var v = b[":, :, ::2"];
        v[0] = NDArray.Scalar(0f);
        v[1] = NDArray.Scalar(NegZeroF);
        CollectionAssert.AreEqual(new long[] { 8184, 8185, 8186, 8187, 8188, 8189, 8998, 8999 }, PositiveSignPositions(AxisNan(v, 0, true, "bufsize 8192")));

        long old = np.setbufsize(16384);
        try
        {
            Assert.AreEqual(0, PositiveSignPositions(AxisNan(v, 0, true, "bufsize 16384")).Length);
        }
        finally
        {
            np.setbufsize(old);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // AXIS — ROW mode: the reduced axis is innermost
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// A STRIDED row (<c>base[:, ::2]</c>, rows of 17) is NumPy's 8-accumulator unroll with the CRT op throughout: a ±0
    /// tie inside one accumulator keeps <c>+0</c> (max) / <c>-0</c> (min), and an all-NaN row returns its LAST NaN. The
    /// same values as contiguous rows go through <c>simd_reduce_c</c> — <c>maxp</c> keeps the later <c>-0</c> and the
    /// seed splat keeps the FIRST NaN — so the layout alone flips both answers.
    /// </summary>
    [TestMethod]
    public void AxisRow_StridedRowIsTheCrtTree_ContiguousRowIsMaxp()
    {
        var b = np.full(new Shape(3, 34), -1.0);
        var v = b[":, ::2"];
        v.SetDouble(0.0, 0, 1);                // after-skip index 0 → accumulator m0
        v.SetDouble(NegZero, 0, 9);            // after-skip index 8 → m0 again
        for (int k = 0; k < 17; k++)
            v.SetDouble(Q64(0x100 + (ulong)k), 1, k);
        v.SetDouble(NegZero, 2, 4);
        v.SetDouble(0.0, 2, 12);
        CollectionAssert.AreEqual(new[] { P0, 0x7FF8000000000110UL, P0 }, Bits64(AxisNan(v, 1, true, "f64 strided")));
        CollectionAssert.AreEqual(new[] { N0, 0x7FF8000000000100UL, P0 }, Bits64(AxisNan(np.ascontiguousarray(v), 1, true, "f64 contiguous")));

        var b32 = np.full(new Shape(3, 34), -1f, NPTypeCode.Single);
        var v32 = b32[":, ::2"];
        v32.SetSingle(0f, 0, 1);
        v32.SetSingle(NegZeroF, 0, 9);
        for (int k = 0; k < 17; k++)
            v32.SetSingle(Q32(0x100 + (uint)k), 1, k);
        CollectionAssert.AreEqual(new[] { P0F, 0x7FC00110u, 0xBF800000u }, Bits32(AxisNan(v32, 1, true, "f32 strided")));
        CollectionAssert.AreEqual(new[] { N0F, 0x7FC00100u, 0xBF800000u }, Bits32(AxisNan(np.ascontiguousarray(v32), 1, true, "f32 contiguous")));

        var m = np.full(new Shape(3, 34), 1.0);
        var vm = m[":, ::2"];
        vm.SetDouble(NegZero, 0, 1);
        vm.SetDouble(0.0, 0, 9);
        for (int k = 0; k < 17; k++)
            vm.SetDouble(Q64(0x100 + (ulong)k, negative: true), 1, k);
        vm.SetDouble(0.0, 2, 6);
        vm.SetDouble(NegZero, 2, 14);
        CollectionAssert.AreEqual(new[] { N0, 0xFFF8000000000110UL, N0 }, Bits64(AxisNan(vm, 1, false, "f64 strided min")));
        CollectionAssert.AreEqual(new[] { P0, 0xFFF8000000000100UL, N0 }, Bits64(AxisNan(np.ascontiguousarray(vm), 1, false, "f64 contiguous min")));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // FLAT
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// One contiguous call seeded with <c>splat(x[0])</c>: <c>[+0, -0, …]</c> of length 3 never reaches a vector (the
    /// two elements after the copied one are the CRT tail: <c>+0</c>), from length 5 the vector section folds the
    /// <c>-0</c>s in (<c>maxp</c>: <c>-0</c>); a <c>+0</c> LAST always wins through the tail. nanmin mirrors it.
    /// </summary>
    [TestMethod]
    public void Flat_ContiguousCall_VectorSectionThenScalarTail()
    {
        foreach (int n in new[] { 3, 5, 7, 9, 13 })
        {
            var x = np.full(new Shape(n), NegZero);
            x.SetDouble(0.0, 0);
            var y = np.full(new Shape(n), NegZero);
            y.SetDouble(0.0, n - 1);
            CollectionAssert.AreEqual(new[] { n == 3 ? P0 : N0 }, Bits64(FlatNan(x, true, $"max +0 first n={n}")));
            CollectionAssert.AreEqual(new[] { P0 }, Bits64(FlatNan(y, true, $"max +0 last n={n}")));
            CollectionAssert.AreEqual(new[] { n == 3 ? N0 : P0 }, Bits64(FlatNan(np.negative(x), false, $"min -0 first n={n}")));
            CollectionAssert.AreEqual(new[] { N0 }, Bits64(FlatNan(np.negative(y), false, $"min -0 last n={n}")));
        }
    }

    /// <summary>
    /// An all-NaN flat call returns a payload: the seed <c>splat(x[0])</c> survives every <c>maxp</c> (a NaN second
    /// operand keeps the first), its all-NaN horizontal reduce hands back lane 0 as is, and the CRT tail then walks to
    /// the LAST NaN — so alternating payloads of length 2 / 5 / 9 / 10 give the second, the first, the first (no tail
    /// after one float32 vector) and the second.
    /// </summary>
    [TestMethod]
    public void Flat_AllNaN_ThePayloadFollowsTheCall()
    {
        float na = F(0xFFC00A0Au), nb = F(0x7FC00B0Bu);
        foreach (var (n, want) in new[] { (2, 0x7FC00B0Bu), (5, 0xFFC00A0Au), (9, 0xFFC00A0Au), (10, 0x7FC00B0Bu) })
        {
            var x = np.array(Enumerable.Range(0, n).Select(k => k % 2 == 0 ? na : nb).ToArray());
            CollectionAssert.AreEqual(new[] { want }, Bits32(FlatNan(x, true, $"max n={n}")));
            CollectionAssert.AreEqual(new[] { want }, Bits32(FlatNan(x, false, $"min n={n}")));
        }
    }

    /// <summary>
    /// A group holding a NaN does not end the NaN-free fast path for the P rules. <c>FoldGroups</c> folds a group proven
    /// NaN-free with the plain <c>vmaxp</c>, which equals NumPy's <c>maxp</c> whenever the SECOND operand is NaN-free,
    /// whatever the accumulator holds — so after a NaN group (folded with the full blended rule) the next clean group goes
    /// fast again. Three wrong versions of that branch are pinned: skipping the NaN group (A: the maximum sits inside it);
    /// folding it with the plain op (B: its NaN in vector 7 climbs the tree's second-operand path into the accumulator lane
    /// holding the earlier maximum, and the next plain group overwrites that lane with a smaller value); and letting the N
    /// rules continue too (<c>np.max</c> must still return the canonical NaN, which a plain group after the NaN group would
    /// drop). float64 has 4 lanes / 32-element groups, float32 8 lanes / 64-element groups; positions name the group /
    /// vector / lane of the elements after the copied <c>x[0]</c>. Every result was probed against NumPy 2.4.2 with the
    /// same construction.
    /// </summary>
    [TestMethod]
    public void Flat_NaNGroupThenCleanGroups_PRulesStayOnTheFastPath()
    {
        // (i * 37 % 11) * 0.25: small positive values with repeats (equal bits, so no observable tie), never NaN.
        static NDArray Base(int n, NPTypeCode t) => (np.arange(n) * 37 % 11 * 0.25).astype(t);

        // A — f64, n = 1 + 4 groups + 3: the maximum INSIDE the NaN group (group 1, vector 3 lane 2 = index 47; the NaN
        // at group 1, vector 7 lane 1 = index 62).
        var a = Base(132, NPTypeCode.Double);
        a.SetDouble(100.0, 47);
        a.SetDouble(Q64(0), 62);
        CollectionAssert.AreEqual(new[] { 0x4059000000000000UL }, Bits64(FlatNan(a, true, "A nanmax")));

        // B — the maximum BEFORE the NaN group, in the lane its last vector poisons (group 0, vector 2 lane 1 = index 10),
        // then two clean groups and a tail.
        var b = Base(132, NPTypeCode.Double);
        b.SetDouble(100.0, 10);
        b.SetDouble(Q64(0), 62);
        CollectionAssert.AreEqual(new[] { 0x4059000000000000UL }, Bits64(FlatNan(b, true, "B nanmax")));
        var bMin = Base(132, NPTypeCode.Double);
        bMin.SetDouble(-100.0, 10);
        bMin.SetDouble(Q64(0), 62);
        CollectionAssert.AreEqual(new[] { 0xC059000000000000UL }, Bits64(FlatNan(bMin, false, "B nanmin")));

        // The N rules must still leave the fast path at the NaN group: np.max propagates it as the canonical NaN.
        CollectionAssert.AreEqual(new[] { 0x7FF8000000000000UL }, Bits64(np.max(b)), "B np.max");

        // B through a streamed COMPUTED child (b * 1.0): the same schedule over the produced block.
        int before = NDExpr.ExactMinMaxRuns;
        var computed = np.evaluate(NDExpr.NanMax((NDExpr)b * 1.0));
        Assert.AreEqual(before + 1, NDExpr.ExactMinMaxRuns, "B computed child: the exact schedule must engage");
        CollectionAssert.AreEqual(new[] { 0x4059000000000000UL }, Bits64(computed), "B computed child");

        // B32 — f32, n = 1 + 3 groups + 5: the maximum at group 0, vector 2 lane 1 = index 18; the NaN at group 1,
        // vector 7 lane 1 = index 122.
        var b32 = Base(198, NPTypeCode.Single);
        b32.SetSingle(100f, 18);
        b32.SetSingle(Q32(0), 122);
        CollectionAssert.AreEqual(new[] { 0x42C80000u }, Bits32(FlatNan(b32, true, "B32 nanmax")));
        CollectionAssert.AreEqual(new[] { 0x7FC00000u }, Bits32(np.max(b32)), "B32 np.max");
    }

    /// <summary>
    /// A reversed 1-D view is ONE strided run — NumPy's 8-accumulator unroll with the CRT op everywhere — so an all-NaN
    /// run returns its LAST NaN (logical index 16); the same values contiguous return the FIRST.
    /// </summary>
    [TestMethod]
    public void Flat_ReversedRun_StridedUnrollKeepsTheLastNaN()
    {
        var b = np.empty(new Shape(17));
        for (int k = 0; k < 17; k++)
            b.SetDouble(Q64(0x200 + (ulong)k), 16 - k);   // logical k of b[::-1]
        var v = b["::-1"];
        CollectionAssert.AreEqual(new[] { 0x7FF8000000000210UL }, Bits64(FlatNan(v, true, "reversed max")));
        CollectionAssert.AreEqual(new[] { 0x7FF8000000000210UL }, Bits64(FlatNan(v, false, "reversed min")));
        CollectionAssert.AreEqual(new[] { 0x7FF8000000000200UL }, Bits64(FlatNan(np.ascontiguousarray(v), true, "contiguous")));
    }

    /// <summary>
    /// Reversed rows short enough to buffer: NumPy copies the <c>(129, 5)</c> view into ONE buffer and makes one
    /// contiguous call — an all-NaN buffer returns its FIRST NaN (iteration index 0), where per-row strided calls would
    /// have returned the last.
    /// </summary>
    [TestMethod]
    public void Flat_BufferedRows_OneContiguousCall_KeepsTheFirstNaN()
    {
        var b = np.empty(new Shape(129, 5));
        for (int i = 0; i < 129; i++)
        for (int j = 0; j < 5; j++)
            b.SetDouble(Q64(0x1000 + (ulong)(i * 5 + j)), i, 4 - j);   // logical [i, j] of b[:, ::-1]
        var v = b[":, ::-1"];
        CollectionAssert.AreEqual(new[] { 0x7FF8000000001000UL }, Bits64(FlatNan(v, true, "max")));
        CollectionAssert.AreEqual(new[] { 0x7FF8000000001000UL }, Bits64(FlatNan(v, false, "min")));
    }

    /// <summary>
    /// The buffer FILLS are the calls: a <c>(3000, 4)</c> reversed view is copied in fills of 8192 then 3808 elements.
    /// Fill 1's tail (8191 elements after the copied one: 3 for float64, 7 for float32) walks to its LAST NaN, iteration
    /// index 8191, and fill 2's seed splat keeps it; one call over the same values would end on index 11999. Halving the
    /// buffer moves it to 4095, exactly as NumPy's <c>np.setbufsize</c> does.
    /// </summary>
    [TestMethod]
    public void Flat_BufferFills_AreTheCalls()
    {
        var b = np.empty(new Shape(3000, 4));
        var b32 = np.empty(new Shape(3000, 4), NPTypeCode.Single);
        for (int i = 0; i < 3000; i++)
        for (int j = 0; j < 4; j++)
        {
            b.SetDouble(Q64(0x100000 + (ulong)(i * 4 + j)), i, 3 - j);
            b32.SetSingle(Q32(0x100000 + (uint)(i * 4 + j)), i, 3 - j);
        }

        var v = b[":, ::-1"];
        CollectionAssert.AreEqual(new[] { 0x7FF8000000101FFFUL }, Bits64(FlatNan(v, true, "f64 fills")));
        CollectionAssert.AreEqual(new[] { 0x7FF8000000102EDFUL }, Bits64(FlatNan(np.ascontiguousarray(v), true, "one call")));
        CollectionAssert.AreEqual(new[] { 0x7FD01FFFu }, Bits32(FlatNan(b32[":, ::-1"], true, "f32 fills")));

        long old = np.setbufsize(4096);
        try
        {
            CollectionAssert.AreEqual(new[] { 0x7FF8000000100FFFUL }, Bits64(FlatNan(v, true, "bufsize 4096")));
        }
        finally
        {
            np.setbufsize(old);
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // np.evaluate over a COMPUTED child
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// A streamable computed child is produced block by block, and each block's positions keep their place in NumPy's
    /// call: <c>(2, 3001, 3) · 2</c> along axis 0 is ONE call over 9003 positions, whose three-element tail — positions
    /// 9000..9002, in the stream's LAST block — is the only CRT <c>+0</c>. Float32 and float64, streamed and bare leaf.
    /// </summary>
    [TestMethod]
    public void Evaluate_StreamedChild_KeepsCallPositionsAcrossBlocks()
    {
        foreach (var t in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            var x = np.zeros(new Shape(2, 3001, 3), t);
            x[1] = t == NPTypeCode.Single ? NDArray.Scalar(NegZeroF) : NDArray.Scalar(NegZero);
            int streamed = NDExpr.StreamingReductions, exact = NDExpr.ExactMinMaxRuns;
            var r = np.evaluate(NDExpr.NanMax((NDExpr)x * 2.0, 0));
            Assert.AreEqual(streamed + 1, NDExpr.StreamingReductions, $"{t}: the computed child must stream");
            Assert.AreEqual(exact + 1, NDExpr.ExactMinMaxRuns, $"{t}: the stream must take the exact schedule");
            Assert.AreEqual(t, r.typecode);
            CollectionAssert.AreEqual(new long[] { 9000, 9001, 9002 }, PositiveSignPositions(r), $"{t} streamed");
            CollectionAssert.AreEqual(new long[] { 9000, 9001, 9002 }, PositiveSignPositions(AxisNan(x, 0, true, $"{t} leaf")));
        }
    }

    /// <summary>
    /// NumPy reduces the buffer its ufunc ALLOCATED, in K order: <c>v · 2</c> of an F-contiguous <c>v</c> is F-contiguous,
    /// so axis 0 is its innermost (contiguous ROW) axis — three-element rows, all CRT: <c>+0</c> everywhere (min:
    /// <c>-0</c>). Reducing a C-ordered copy of the same values would be a SLAB with a vector section (<c>-0</c> ×4).
    /// </summary>
    [TestMethod]
    public void Evaluate_ComputedChild_ReducesNumPysKOrderLayout()
    {
        var b = np.zeros(new Shape(7, 3));
        b[":, 2"] = NDArray.Scalar(NegZero);
        var v = b.T;
        CollectionAssert.AreEqual(Repeat(P0, 7), Bits64(np.evaluate(NDExpr.NanMax((NDExpr)v * 2.0, 0))));
        CollectionAssert.AreEqual(Repeat(P0, 7), Bits64(AxisNan(v, 0, true, "leaf")));
        CollectionAssert.AreEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, Bits64(np.nanmax(np.ascontiguousarray(v * 2.0), 0)));

        var m = np.full(new Shape(7, 3), NegZero);
        m[":, 2"] = NDArray.Scalar(0.0);
        var vm = m.T;
        CollectionAssert.AreEqual(Repeat(N0, 7), Bits64(np.evaluate(NDExpr.NanMin((NDExpr)vm * 2.0, 0))));
        CollectionAssert.AreEqual(new[] { P0, P0, P0, P0, N0, N0, N0 }, Bits64(np.nanmin(np.ascontiguousarray(vm * 2.0), 0)));
    }

    /// <summary>
    /// keepdims through np.evaluate's computed route: the flat result is <c>(1, 1)</c> and holds the one-call answer
    /// (<c>-0</c>: the vector section saw the last row's <c>-0</c>), the axis result keeps the reduced axis as 1.
    /// </summary>
    [TestMethod]
    public void Evaluate_ComputedChild_Keepdims()
    {
        var x = np.zeros(new Shape(3, 7));
        x[2] = NDArray.Scalar(NegZero);
        var flat = np.evaluate(NDExpr.NanMax((NDExpr)x * 2.0, true));
        CollectionAssert.AreEqual(new long[] { 1, 1 }, flat.shape);
        CollectionAssert.AreEqual(new[] { N0 }, Bits64(flat));
        var ax = np.evaluate(NDExpr.NanMax((NDExpr)x * 2.0, 0, keepdims: true));
        CollectionAssert.AreEqual(new long[] { 1, 7 }, ax.shape);
        CollectionAssert.AreEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, Bits64(ax));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Edges
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// fmax / fmin have no identity: an empty reduction raises NumPy's ValueError text with THEIR name — for every
    /// dtype, integers and float16 included — while a non-empty reduced axis of an empty array is the empty result.
    /// </summary>
    [TestMethod]
    public void Edges_EmptyReductions_RaiseWithTheFmaxName()
    {
        foreach (bool isMax in new[] { true, false })
        {
            string text = $"zero-size array to reduction operation {(isMax ? "fmax" : "fmin")} which has no identity";
            Func<NDArray, int?, bool, NDArray> f = isMax ? (a, ax, kd) => np.nanmax(a, ax, kd) : (a, ax, kd) => np.nanmin(a, ax, kd);

            foreach (var empty in new[] { np.array(new double[0]), np.array(new int[0]), np.array(new Half[0]), np.zeros(new Shape(0, 3)) })
            {
                var ex = Assert.ThrowsException<ArgumentException>(() => f(empty, null, false), $"{empty.dtype.name} flat");
                Assert.AreEqual(text, ex.Message);
            }

            Assert.AreEqual(text, Assert.ThrowsException<ArgumentException>(() => f(np.array(new double[0]), null, true)).Message);
            Assert.AreEqual(text, Assert.ThrowsException<ArgumentException>(() => f(np.zeros(new Shape(0, 3)), 0, false)).Message);
            CollectionAssert.AreEqual(new long[] { 0 }, f(np.zeros(new Shape(0, 3)), 1, false).shape);
            CollectionAssert.AreEqual(new long[] { 0, 1 }, f(np.zeros(new Shape(0, 3)), 1, true).shape);
            CollectionAssert.AreEqual(new long[] { 0 }, f(np.zeros(new Shape(3, 0)), 0, false).shape);
        }
    }

    /// <summary>
    /// A single element per result is returned untouched — NumPy copies it and never calls the loop, so a NaN keeps its
    /// payload: a 0-d array, a one-element vector, an extent-1 reduced axis.
    /// </summary>
    [TestMethod]
    public void Edges_SingleElement_KeepsItsPayload()
    {
        double na = Q64(0xA0A, negative: true);
        var zeroD = np.nanmax(np.array(na));
        Assert.AreEqual(0, zeroD.ndim);
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000A0AUL }, Bits64(zeroD));
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000A0AUL }, Bits64(np.nanmin(np.array(new[] { na }))));

        var one = np.nanmax(np.array(new[,] { { Q64(0x777) } }), 1);
        CollectionAssert.AreEqual(new long[] { 1 }, one.shape);
        CollectionAssert.AreEqual(new[] { 0x7FF8000000000777UL }, Bits64(one));
        CollectionAssert.AreEqual(Repeat(0x7FF8000000000555UL, 4), Bits64(np.nanmax(np.full(new Shape(1, 4), Q64(0x555)), 0)));
    }

    /// <summary>
    /// keepdims shapes (flat and per axis) and the out-of-bounds axis error, as NumPy's.
    /// </summary>
    [TestMethod]
    public void Edges_KeepdimsAndAxisBounds()
    {
        var a = np.arange(6.0).reshape(2, 3);
        var flat = np.nanmax(a, null, true);
        CollectionAssert.AreEqual(new long[] { 1, 1 }, flat.shape);
        Assert.AreEqual(5.0, flat.GetAtIndex<double>(0));
        CollectionAssert.AreEqual(new long[] { 1, 3 }, np.nanmax(a, 0, true).shape);
        CollectionAssert.AreEqual(new long[] { 2, 1 }, np.nanmin(a, 1, true).shape);
        CollectionAssert.AreEqual(new[] { 0.0, 3.0 }, np.nanmin(a, -1).ToArray<double>());
        Assert.ThrowsException<AxisError>(() => np.nanmax(a, 2));
        Assert.ThrowsException<AxisError>(() => np.nanmin(a, -3));
    }

    /// <summary>
    /// fmax / fmin of values that cannot be NaN ARE max / min: integer arrays keep their dtype and give the plain extreme
    /// on every layout, flat and per axis.
    /// </summary>
    [TestMethod]
    public void Integers_AreTheirPlainExtremes()
    {
        var baseArr = np.arange(35).reshape(5, 7) * 37 % 23 - 11;
        foreach (var x in new[] { baseArr, baseArr[":, ::-1"], baseArr.T, baseArr["::2, 1::3"] })
        {
            Assert.AreEqual(NPTypeCode.Int64, np.nanmax(x).typecode);
            Assert.AreEqual(np.max(x).GetAtIndex<long>(0), np.nanmax(x).GetAtIndex<long>(0));
            Assert.AreEqual(np.min(x).GetAtIndex<long>(0), np.nanmin(x).GetAtIndex<long>(0));
            for (int ax = 0; ax < 2; ax++)
            {
                CollectionAssert.AreEqual(np.max(x, ax).ToArray<long>(), np.nanmax(x, ax).ToArray<long>(), $"axis {ax}");
                CollectionAssert.AreEqual(np.min(x, ax).ToArray<long>(), np.nanmin(x, ax).ToArray<long>(), $"axis {ax}");
            }
        }
    }

    /// <summary>
    /// A broadcast input declines the exact schedule (a zero stride changes NpyIter's axis sort, which the ports do not
    /// model) and keeps the previous kernels — right value, bits not pinned.
    /// </summary>
    [TestMethod]
    public void Declines_BroadcastInput_KeepsTheValue()
    {
        var bc = np.broadcast_to(np.array(new[] { 1.0, double.NaN, -2.0 }), new Shape(4, 3));
        int before = NDExpr.ExactMinMaxRuns;
        Assert.AreEqual(1.0, np.nanmax(bc).GetDouble());
        Assert.AreEqual(-2.0, np.nanmin(bc, 1).GetDouble(0));
        Assert.AreEqual(before, NDExpr.ExactMinMaxRuns, "a broadcast input declines");
    }

    /// <summary>
    /// Teeth: with <see cref="NDExpr.DisableExactMinMax"/> set the engine's previous kernels run — the exact routes stop
    /// counting, and the call-tiling answer of <see cref="AxisSlab_OneCall_VectorSectionThenScalarTail"/> and the
    /// payload of an all-NaN column are no longer NumPy's. Proves those tests observe the exact schedule, not a kernel
    /// that happens to agree.
    /// </summary>
    [TestMethod]
    public void Hook_DisableExactMinMax_LosesNumPysBits()
    {
        var a = np.zeros(new Shape(3, 7));
        a[2] = NDArray.Scalar(NegZero);
        var n = np.empty(new Shape(3, 7));
        n[":2"] = NDArray.Scalar(Q64(0xA0A, negative: true));
        n[2] = NDArray.Scalar(Q64(0xB0B));
        NDExpr.DisableExactMinMax = true;
        try
        {
            int before = NDExpr.ExactMinMaxRuns;
            var tie = Bits64(np.nanmax(a, 0));
            var nan = Bits64(np.nanmax(n, 0));
            Assert.AreEqual(before, NDExpr.ExactMinMaxRuns, "the hook must decline every exact route");
            CollectionAssert.AreNotEqual(new[] { N0, N0, N0, N0, P0, P0, P0 }, tie, "the previous kernel must not reproduce the call tiling");
            Assert.IsFalse(nan.Contains(0xFFF8000000000A0AUL) && nan.Contains(0x7FF8000000000B0BUL),
                "the previous kernel must not reproduce both payloads");
        }
        finally
        {
            NDExpr.DisableExactMinMax = false;
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // float16 / complex128 — NumPy's SEQUENTIAL fmax / fmin loops (HALF_fmax / CDOUBLE_fmax)
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>The raw bits of every element of a float16 result, in C order.</summary>
    /// <param name="r">A float16 array (any layout).</param>
    /// <returns>One bit pattern per element.</returns>
    private static ushort[] Bits16(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Half, r.typecode);
        return np.ascontiguousarray(r).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray();
    }

    /// <summary>The raw bits of every element of a complex128 result, in C order — (real, imaginary) per element.</summary>
    /// <param name="r">A complex128 array (any layout).</param>
    /// <returns>Two bit patterns per element.</returns>
    private static ulong[] BitsC(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Complex, r.typecode);
        return np.ascontiguousarray(r).ToArray<System.Numerics.Complex>()
            .SelectMany(z => new[] { (ulong)BitConverter.DoubleToInt64Bits(z.Real), (ulong)BitConverter.DoubleToInt64Bits(z.Imaginary) })
            .ToArray();
    }

    /// <summary>A float16 with the exact bit pattern <paramref name="bits"/>.</summary>
    /// <param name="bits">The raw IEEE-754 binary16 bits.</param>
    /// <returns>The half those bits encode.</returns>
    private static Half H(ushort bits) => BitConverter.UInt16BitsToHalf(bits);

    /// <summary>A complex128 whose parts carry the exact bit patterns given.</summary>
    /// <param name="re">The real part's bits.</param>
    /// <param name="im">The imaginary part's bits.</param>
    /// <returns>The complex value.</returns>
    private static System.Numerics.Complex C(ulong re, ulong im) => new(D(re), D(im));

    /// <summary>
    /// The engine (<c>np.nanmax</c> / <c>np.nanmin</c>) and the np.evaluate bare leaf over a float16 / complex128
    /// operand, asserting they agree bit for bit (these dtypes take the sequential fold, not the counted exact
    /// schedules); returns the engine result.
    /// </summary>
    /// <param name="x">The operand.</param>
    /// <param name="axis">The reduced axis (null = flat).</param>
    /// <param name="isMax">nanmax when true, nanmin when false.</param>
    /// <param name="what">A label for failure messages.</param>
    /// <returns>The engine's result.</returns>
    private static NDArray Seq(NDArray x, int? axis, bool isMax, string what)
    {
        var engine = isMax ? np.nanmax(x, axis) : np.nanmin(x, axis);
        var e = axis is int ax
            ? (isMax ? NDExpr.NanMax((NDExpr)x, ax) : NDExpr.NanMin((NDExpr)x, ax))
            : (isMax ? NDExpr.NanMax((NDExpr)x) : NDExpr.NanMin((NDExpr)x));
        var leaf = np.evaluate(e);
        Assert.AreEqual(engine.typecode, leaf.typecode, $"{what}: dtype");
        CollectionAssert.AreEqual(engine.shape, leaf.shape, $"{what}: shape");
        if (engine.typecode == NPTypeCode.Half)
            CollectionAssert.AreEqual(Bits16(engine), Bits16(leaf), $"{what}: engine vs np.evaluate bits");
        else
            CollectionAssert.AreEqual(BitsC(engine), BitsC(leaf), $"{what}: engine vs np.evaluate bits");
        return engine;
    }

    /// <summary>
    /// float16: a ±0 tie keeps the EARLIER zero (IEEE <c>-0 &gt;= +0</c>, so the running value survives), and an all-NaN
    /// reduction returns its FIRST NaN verbatim — through the flat C-contiguous bit kernel and the sequential fold alike.
    /// A TRANSPOSED view is visited in memory order (NumPy's K order), so its first zero is not the logical first one; a
    /// REVERSED view is visited in logical order (<c>NPY_ITER_DONT_NEGATE_STRIDES</c>). F-contiguous axis reductions fold
    /// each column in k order: NaNs skipped, the first zero kept, an all-NaN column's first payload kept.
    /// </summary>
    [TestMethod]
    public void Half_SequentialFold_FirstZeroFirstNaN_NumPysOrder()
    {
        CollectionAssert.AreEqual(new ushort[] { 0x0000 }, Bits16(Seq(np.array(new[] { (Half)0.0, (Half)NegZero }), null, true, "C [+0,-0]")));
        CollectionAssert.AreEqual(new ushort[] { 0x8000 }, Bits16(Seq(np.array(new[] { (Half)NegZero, (Half)0.0 }), null, true, "C [-0,+0]")));
        CollectionAssert.AreEqual(new ushort[] { 0xFE0A }, Bits16(Seq(np.array(new[] { H(0xFE0A), H(0x7E0B) }), null, true, "C all-NaN")));
        CollectionAssert.AreEqual(new ushort[] { 0x8000 }, Bits16(Seq(np.array(new[] { (Half)NegZero, (Half)0.0 }), null, false, "C min [-0,+0]")));
        CollectionAssert.AreEqual(new ushort[] { 0x0000 }, Bits16(Seq(np.array(new[] { (Half)0.0, (Half)NegZero }), null, false, "C min [+0,-0]")));

        var b = np.full(new Shape(2, 3), (Half)(-1.0), NPTypeCode.Half);
        b.SetAtIndex((Half)NegZero, 1);   // b[0, 1]: memory position 1
        b.SetAtIndex((Half)0.0, 3);       // b[1, 0]: memory position 3 (logically FIRST in b.T)
        CollectionAssert.AreEqual(new ushort[] { 0x8000 }, Bits16(Seq(b.T, null, true, "transposed: memory order")));
        CollectionAssert.AreEqual(new ushort[] { 0x0000 }, Bits16(np.nanmax(np.ascontiguousarray(b.T))), "the C copy's order differs");

        var r = np.array(new[] { H(0xFE0A), H(0x7E0B) });
        CollectionAssert.AreEqual(new ushort[] { 0x7E0B }, Bits16(Seq(r["::-1"], null, true, "reversed: logical order")));
        CollectionAssert.AreEqual(new ushort[] { 0x7E0B }, Bits16(Seq(r["::-1"], null, false, "reversed min")));

        var f = np.asfortranarray(np.full(new Shape(3, 4), (Half)(-1.0), NPTypeCode.Half));
        f[":, 0"] = np.array(new[] { H(0xFE0A), H(0x7E0B), H(0xFE0A) });
        f[":, 1"] = np.array(new[] { (Half)0.0, (Half)NegZero, (Half)0.0 });
        f[":, 2"] = np.array(new[] { (Half)NegZero, (Half)0.0, (Half)NegZero });
        f[":, 3"] = np.array(new[] { H(0x7E0B), (Half)2.0, H(0xFE0A) });
        CollectionAssert.AreEqual(new ushort[] { 0xFE0A, 0x0000, 0x8000, 0x4000 }, Bits16(Seq(f, 0, true, "F axis 0 max")));
        CollectionAssert.AreEqual(new ushort[] { 0xFE0A, 0x0000, 0x8000, 0x4000 }, Bits16(Seq(f, 0, false, "F axis 0 min")));
        CollectionAssert.AreEqual(new ushort[] { 0x0000, 0x4000, 0x0000 }, Bits16(Seq(f, 1, true, "F axis 1 max")));
        CollectionAssert.AreEqual(new ushort[] { 0x0000, 0x8000, 0x0000 }, Bits16(Seq(f, 1, false, "F axis 1 min")));
    }

    /// <summary>
    /// complex128: an element with a NaN in EITHER part is skipped (it used to propagate — <c>np.amax</c>'s rule), the
    /// comparison is lexicographic, an equal pair keeps the EARLIER element (so <c>(+0, -0)</c> survives
    /// <c>(-0, +0)</c> for max AND min), an all-NaN reduction returns its first element verbatim, a transposed view is
    /// visited in memory order, and F-contiguous columns fold in k order.
    /// </summary>
    [TestMethod]
    public void Complex_SequentialFold_SkipsNaN_Lexicographic_FirstTie()
    {
        var c1 = np.array(new[] { new System.Numerics.Complex(1, 1), new System.Numerics.Complex(double.NaN, 0), new System.Numerics.Complex(2, 0), new System.Numerics.Complex(0, double.NaN) });
        CollectionAssert.AreEqual(new[] { 0x4000000000000000UL, 0x0UL }, BitsC(Seq(c1, null, true, "NaN skip max")));
        CollectionAssert.AreEqual(new[] { 0x3FF0000000000000UL, 0x3FF0000000000000UL }, BitsC(Seq(c1, null, false, "NaN skip min")));

        var c2 = np.array(new[] { new System.Numerics.Complex(1, 5), new System.Numerics.Complex(1, 7), new System.Numerics.Complex(0, 9) });
        CollectionAssert.AreEqual(new[] { 0x3FF0000000000000UL, 0x401C000000000000UL }, BitsC(Seq(c2, null, true, "lexicographic max")));
        CollectionAssert.AreEqual(new[] { 0x0UL, 0x4022000000000000UL }, BitsC(Seq(c2, null, false, "lexicographic min")));

        var c3 = np.array(new[] { C(0xFFF8000000000A0AUL, 0x0UL), C(0x0UL, 0x7FF8000000000B0BUL) });
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000A0AUL, 0x0UL }, BitsC(Seq(c3, null, true, "all-NaN max")));
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000A0AUL, 0x0UL }, BitsC(Seq(c3, null, false, "all-NaN min")));

        var c4 = np.array(new[] { C(0x0UL, N0), C(N0, 0x0UL) });
        CollectionAssert.AreEqual(new[] { 0x0UL, N0 }, BitsC(Seq(c4, null, true, "±0 tie max")));
        CollectionAssert.AreEqual(new[] { 0x0UL, N0 }, BitsC(Seq(c4, null, false, "±0 tie min")));

        var cb = np.full(new Shape(2, 3), new System.Numerics.Complex(-1, 0), NPTypeCode.Complex);
        cb.SetAtIndex(C(N0, 0x0UL), 1);   // cb[0, 1]
        cb.SetAtIndex(C(0x0UL, 0x0UL), 3); // cb[1, 0]
        CollectionAssert.AreEqual(new[] { N0, 0x0UL }, BitsC(Seq(cb.T, null, true, "transposed: memory order")));

        var cf = np.asfortranarray(np.full(new Shape(3, 2), new System.Numerics.Complex(-1, 0), NPTypeCode.Complex));
        cf[":, 0"] = np.array(new[] { new System.Numerics.Complex(double.NaN, 1), new System.Numerics.Complex(3, 1), new System.Numerics.Complex(3, 2) });
        cf[":, 1"] = np.array(new[] { C(0xFFF8000000000A0AUL, 0x0UL), C(0x0UL, 0x7FF8000000000B0BUL), C(0x7FF8000000000B0BUL, 0x7FF8000000000B0BUL) });
        CollectionAssert.AreEqual(new[] { 0x4008000000000000UL, 0x4000000000000000UL, 0xFFF8000000000A0AUL, 0x0UL }, BitsC(Seq(cf, 0, true, "F axis 0 max")));
        CollectionAssert.AreEqual(new[] { 0x4008000000000000UL, 0x3FF0000000000000UL, 0xFFF8000000000A0AUL, 0x0UL }, BitsC(Seq(cf, 0, false, "F axis 0 min")));
        CollectionAssert.AreEqual(new long[] { 1, 1 }, np.nanmax(cf, null, true).shape);
        CollectionAssert.AreEqual(new long[] { 3, 1 }, np.nanmax(cf, 1, true).shape);
    }

    /// <summary>
    /// The flat visiting order for the sequential fold (<see cref="NumPySequentialReduce.FlatVisitOrder"/>) is NpyIter's:
    /// extent &gt; 1 axes insertion-sorted by |stride| from reversed C order, an equal |stride| keeping the later axis
    /// inner, and a zero (broadcast) stride never compared — it keeps its reversed-C place.
    /// </summary>
    [TestMethod]
    public void SequentialFlatVisitOrder_IsNpyItersAxisSort()
    {
        int[] Order(long[] dims, long[] strides)
        {
            Span<int> perm = stackalloc int[dims.Length];
            int m = NumPySequentialReduce.FlatVisitOrder(dims, strides, perm);
            return perm[..m].ToArray();
        }

        CollectionAssert.AreEqual(new[] { 1, 0 }, Order(new long[] { 3, 4 }, new long[] { 4, 1 }));        // C
        CollectionAssert.AreEqual(new[] { 0, 1 }, Order(new long[] { 3, 4 }, new long[] { 1, 3 }));        // F
        CollectionAssert.AreEqual(new[] { 1, 0 }, Order(new long[] { 3, 4 }, new long[] { 4, -1 }));       // reversed columns
        CollectionAssert.AreEqual(new[] { 2, 0, 1 }, Order(new long[] { 3, 4, 5 }, new long[] { 5, 15, 1 }));
        CollectionAssert.AreEqual(new[] { 1, 0 }, Order(new long[] { 3, 4 }, new long[] { 2, -2 }));      // |stride| tie: later inner
        CollectionAssert.AreEqual(new[] { 1, 0 }, Order(new long[] { 3, 4 }, new long[] { 0, 1 }));       // broadcast rows stay outer
        CollectionAssert.AreEqual(new[] { 2, 1, 0 }, Order(new long[] { 2, 3, 2 }, new long[] { 2, 0, 1 }));  // a zero stride keeps its place
        CollectionAssert.AreEqual(new[] { 1 }, Order(new long[] { 1, 4, 1 }, new long[] { 9, 1, 9 }));     // extent-1 axes dropped
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The call-tiling port itself
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// <see cref="NumPyMinMaxReduce.SlabCallTiling.Build{T}"/> reproduces <c>npyiter_find_buffering_setup</c>'s choice for
    /// the two-operand reduction iterator: the P region and its place factors, the block and call sizes, and the float32
    /// negative-stride all-scalar case — the numbers the end-to-end tests above rely on, derived by hand from NumPy's
    /// source and confirmed by their probed results.
    /// </summary>
    [TestMethod]
    public void SlabCallTiling_Build_ReproducesNumPysCalls()
    {
        (NumPyMinMaxReduce.SlabCallTiling t, long[] place, long pRegion) Build<T>(long[] dims, long[] xs, int axis, long bufsize)
            where T : unmanaged
        {
            var place = new long[dims.Length];
            var t = NumPyMinMaxReduce.SlabCallTiling.Build<T>(dims, xs, axis, bufsize, place, out long pRegion);
            return (t, place, pRegion);
        }

        // C (3, 7) along 0: the reduce dimension is the best outer dimension — one call over the whole region.
        var r = Build<double>(new long[] { 3, 7 }, new long[] { 7, 1 }, 0, 8192);
        Assert.AreEqual((7L, 7L, false, 4), (r.t.Block, r.t.Chunk, r.t.AllScalar, r.t.Lanes));
        CollectionAssert.AreEqual(new long[] { 0, 1 }, r.place);
        Assert.AreEqual(7, r.pRegion);
        r.t.Segment(0, out long vEnd, out long cEnd);
        Assert.AreEqual((4L, 7L), (vEnd, cEnd));

        // (2, 3000, 5)[:, :, ::2] float32: not using reduce, the input buffered in whole cores of 3 → calls of 8190.
        var c = Build<float>(new long[] { 2, 3000, 3 }, new long[] { 15000, 5, 2 }, 0, 8192);
        Assert.AreEqual((9000L, 8190L, false, 8), (c.t.Block, c.t.Chunk, c.t.AllScalar, c.t.Lanes));
        CollectionAssert.AreEqual(new long[] { 0, 3, 1 }, c.place);
        c.t.Segment(8190, out vEnd, out cEnd);
        Assert.AreEqual((8998L, 9000L), (vEnd, cEnd));
        var c16 = Build<float>(new long[] { 2, 3000, 3 }, new long[] { 15000, 5, 2 }, 0, 16384);
        Assert.AreEqual((9000L, 9000L), (c16.t.Block, c16.t.Chunk));

        // (3, 7)[:, ::-1] float32: unbuffered with a negative innermost stride → every call scalar; float64 is not.
        Assert.IsTrue(Build<float>(new long[] { 3, 7 }, new long[] { 7, -1 }, 0, 8192).t.AllScalar);
        Assert.IsFalse(Build<double>(new long[] { 3, 7 }, new long[] { 7, -1 }, 0, 8192).t.AllScalar);

        // (2, 9, 3)[:, :, ::2]: the core grows over both P axes and the reduce dimension wins → one call of 18.
        var g = Build<double>(new long[] { 2, 9, 2 }, new long[] { 27, 3, 2 }, 0, 8192);
        Assert.AreEqual((18L, 18L), (g.t.Block, g.t.Chunk));
        Assert.AreEqual(18, g.pRegion);
        g.t.Segment(5, out vEnd, out cEnd);
        Assert.AreEqual((16L, 18L), (vEnd, cEnd));

        // A reduced axis that is the innermost one is ROW mode, not SLAB: a caller bug the port refuses.
        Assert.ThrowsException<InvalidOperationException>(() => Build<double>(new long[] { 3, 7 }, new long[] { 7, 1 }, 1, 8192));
    }
}
