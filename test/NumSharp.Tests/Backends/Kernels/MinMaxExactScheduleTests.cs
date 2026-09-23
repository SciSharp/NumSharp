using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// The NumPy-exact AXIS and FLAT <c>np.max</c> / <c>np.min</c> schedules (<c>Default.Reduction.MinMax.Exact.cs</c>),
/// shared by the engine (<c>np.max</c> / <c>np.min</c> / <c>np.ptp</c>) and np.evaluate's bare-leaf routes.
/// </summary>
/// <remarks>
/// <para>
/// For floats the ORDER NumPy folds in is observable exactly where the answer is a ±0 (which zero sign survives a tie)
/// or a NaN (a NaN that reached a lane reduce comes back as the canonical <c>+NaN</c>, one met only by a scalar op keeps
/// its payload). Every literal below was probed against NumPy 2.4.2 (win-amd64, AVX2 dispatch) with the same array
/// recipe — <c>np.full</c> plus a few set elements, then a view — and most were CHOSEN to discriminate between the
/// schedules NumPy could plausibly run: per-row calls vs buffered contiguous chunks vs one call over the whole array.
/// </para>
/// <para>
/// Every exact case also asserts the route ENGAGED (<see cref="NDExpr.ExactMinMaxRuns"/>) and that the engine and the
/// np.evaluate bare leaf agree bit for bit — a silent decline to the old kernels would still pass a value check.
/// </para>
/// </remarks>
[TestClass]
public class MinMaxExactScheduleTests
{
    /// <summary>
    /// A real negative zero held in a field — a <c>-0.0</c> literal folded into an expression is easy to lose to a
    /// constant fold that normalizes it.
    /// </summary>
    private static readonly double NegZero = -0.0;

    /// <summary>NumPy's canonical float64 NaN (<c>0x7FF8…</c> — POSITIVE, unlike .NET's <see cref="double.NaN"/>).</summary>
    private const ulong CanonicalNaN64 = 0x7FF8000000000000UL;

    /// <summary>A float64 with the exact bit pattern <paramref name="bits"/>.</summary>
    /// <param name="bits">The raw IEEE-754 binary64 bits.</param>
    /// <returns>The double those bits encode.</returns>
    private static double D(ulong bits) => BitConverter.Int64BitsToDouble((long)bits);

    /// <summary>A float32 with the exact bit pattern <paramref name="bits"/>.</summary>
    /// <param name="bits">The raw IEEE-754 binary32 bits.</param>
    /// <returns>The float those bits encode.</returns>
    private static float F(uint bits) => BitConverter.Int32BitsToSingle((int)bits);

    /// <summary>The raw bits of every element of a float64 result, in C order.</summary>
    /// <param name="r">A float64 array (any layout).</param>
    /// <returns>One bit pattern per element.</returns>
    private static ulong[] Bits64(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Double, r.typecode);
        var v = np.ascontiguousarray(r).ToArray<double>();
        var b = new ulong[v.Length];
        for (int i = 0; i < v.Length; i++)
            b[i] = (ulong)BitConverter.DoubleToInt64Bits(v[i]);
        return b;
    }

    /// <summary>The raw bits of every element of a float32 result, in C order.</summary>
    /// <param name="r">A float32 array (any layout).</param>
    /// <returns>One bit pattern per element.</returns>
    private static uint[] Bits32(NDArray r)
    {
        Assert.AreEqual(NPTypeCode.Single, r.typecode);
        var v = np.ascontiguousarray(r).ToArray<float>();
        var b = new uint[v.Length];
        for (int i = 0; i < v.Length; i++)
            b[i] = (uint)BitConverter.SingleToInt32Bits(v[i]);
        return b;
    }

    /// <summary>
    /// The FLAT reduction through the engine (<c>np.max</c> / <c>np.min</c>) and the np.evaluate bare leaf, asserting both
    /// took the exact schedule and agree, and returning the engine result.
    /// </summary>
    /// <param name="x">The operand (any non-broadcast layout of a served dtype).</param>
    /// <param name="isMax">Max when true, min when false.</param>
    /// <param name="what">A label for failure messages.</param>
    /// <returns>The engine's 0-d result.</returns>
    private static NDArray Flat(NDArray x, bool isMax, string what)
    {
        int before = NDExpr.ExactMinMaxRuns;
        var engine = isMax ? np.max(x) : np.min(x);
        Assert.AreEqual(before + 1, NDExpr.ExactMinMaxRuns, $"{what}: the engine's flat reduction must take the exact schedule");
        var leaf = np.evaluate(isMax ? NDExpr.Max((NDExpr)x) : NDExpr.Min((NDExpr)x));
        Assert.AreEqual(before + 2, NDExpr.ExactMinMaxRuns, $"{what}: the np.evaluate bare leaf must take the exact schedule");
        Assert.AreEqual(0, engine.ndim, $"{what}: a flat result is 0-d");
        Assert.AreEqual(engine.GetAtIndex(0).GetType(), leaf.GetAtIndex(0).GetType(), $"{what}: dtype");
        if (engine.typecode == NPTypeCode.Double)
            CollectionAssert.AreEqual(Bits64(engine), Bits64(leaf), $"{what}: engine vs np.evaluate bits");
        else if (engine.typecode == NPTypeCode.Single)
            CollectionAssert.AreEqual(Bits32(engine), Bits32(leaf), $"{what}: engine vs np.evaluate bits");
        else
            Assert.AreEqual(engine.GetAtIndex(0), leaf.GetAtIndex(0), $"{what}: engine vs np.evaluate value");
        return engine;
    }

    /// <summary>
    /// The AXIS reduction through the engine and the np.evaluate bare leaf, asserting both took the exact schedule and
    /// agree bit for bit, and returning the engine result.
    /// </summary>
    /// <param name="x">The operand (any non-broadcast layout of a served dtype).</param>
    /// <param name="axis">The reduced axis.</param>
    /// <param name="isMax">Max when true, min when false.</param>
    /// <param name="what">A label for failure messages.</param>
    /// <returns>The engine's result.</returns>
    private static NDArray Axis(NDArray x, int axis, bool isMax, string what)
    {
        int before = NDExpr.ExactMinMaxRuns;
        var engine = isMax ? np.max(x, axis) : np.min(x, axis);
        Assert.AreEqual(before + 1, NDExpr.ExactMinMaxRuns, $"{what}: the engine's axis reduction must take the exact schedule");
        var leaf = np.evaluate(isMax ? NDExpr.Max((NDExpr)x, axis) : NDExpr.Min((NDExpr)x, axis));
        Assert.AreEqual(before + 2, NDExpr.ExactMinMaxRuns, $"{what}: the np.evaluate bare leaf must take the exact schedule");
        CollectionAssert.AreEqual(engine.shape, leaf.shape, $"{what}: shape");
        if (engine.typecode == NPTypeCode.Double)
            CollectionAssert.AreEqual(Bits64(engine), Bits64(leaf), $"{what}: engine vs np.evaluate bits");
        else if (engine.typecode == NPTypeCode.Single)
            CollectionAssert.AreEqual(Bits32(engine), Bits32(leaf), $"{what}: engine vs np.evaluate bits");
        return engine;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // AXIS
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// ROW mode over REVERSED rows: each row is one strided call — NumPy's 8-accumulator unroll, run here as vector lanes
    /// fed by a reversed load. Rows of 40 give 39 elements after the copied one: four lane groups plus a tail, so the
    /// ties land in specific accumulators and the pairwise combine decides them; a NaN in the lane section keeps its
    /// payload (no lane reduce on this path). float64 and float32 (one vector of eight 4-byte lanes).
    /// </summary>
    [TestMethod]
    public void AxisRowStrided_LaneTiesAndPayload_MatchNumPy()
    {
        // Logical index k of row i of base[:, ::-1] is base[i, 39 - k].
        var b = np.full(new Shape(3, 40), -1.0);
        b.SetDouble(0.0, 0, 39 - 3);
        b.SetDouble(NegZero, 0, 39 - 11);                          // same lane (after-skip 2 and 10): the later wins
        b.SetDouble(NegZero, 1, 39 - 3);
        b.SetDouble(0.0, 1, 39 - 4);                               // lanes 2 and 3: the (2,3) combine decides
        b.SetDouble(D(0xFFF8000000000123), 2, 39 - 9);             // lane section: payload kept
        var r = Axis(b[":, ::-1"], 1, isMax: true, "f64 reversed rows");
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL, 0x0UL, 0xFFF8000000000123UL }, Bits64(r));

        var bf = b.astype(NPTypeCode.Single);
        bf.SetSingle(F(0xFFC00123), 2, 39 - 9);
        var rf = Axis(bf[":, ::-1"], 1, isMax: true, "f32 reversed rows");
        CollectionAssert.AreEqual(new[] { 0x80000000U, 0x0U, 0xFFC00123U }, Bits32(rf));
    }

    /// <summary>
    /// ROW mode over CONTIGUOUS rows of 100 (99 after the copied element: three 8-vector groups and a tail). Row 0 is
    /// NaN-free — the whole row takes the plain-max fast fold, and the ±0 ties across groups must still come out as
    /// NumPy's; row 1's NaN sits in group 1 — the fast fold must hand that group to NumPy's blended rule, and the lane
    /// reduce returns the canonical NaN; row 2's NaN sits in the scalar tail, so its payload survives.
    /// </summary>
    [TestMethod]
    public void AxisRowContiguous_FastFoldThenNaNGroup_MatchNumPy()
    {
        var c = np.full(new Shape(3, 100), 1.0);
        c.SetDouble(NegZero, 0, 5);
        c.SetDouble(0.0, 0, 37);
        c.SetDouble(NegZero, 0, 70);
        c.SetDouble(D(0xFFF8000000000456), 1, 50);
        c.SetDouble(D(0xFFF8000000000789), 2, 99);
        var r = Axis(c, 1, isMax: false, "f64 contiguous rows");
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL, CanonicalNaN64, 0xFFF8000000000789UL }, Bits64(r));
    }

    /// <summary>
    /// SLAB mode (axis 0 of a C block): every call is elementwise, so each output is the plain sequential fold — the LAST
    /// equal zero wins and the FIRST NaN (with its payload) is kept.
    /// </summary>
    [TestMethod]
    public void AxisSlab_SequentialFold_MatchNumPy()
    {
        var s = np.full(new Shape(5, 7), -1.0);
        s.SetDouble(0.0, 1, 0);
        s.SetDouble(NegZero, 3, 0);
        s.SetDouble(NegZero, 1, 1);
        s.SetDouble(0.0, 4, 1);
        s.SetDouble(D(0xFFF8000000000AAA), 2, 2);
        s.SetDouble(D(0x7FF8000000000BBB), 3, 2);
        var r = Axis(s, 0, isMax: true, "slab");
        CollectionAssert.AreEqual(
            new[] { 0x8000000000000000UL, 0x0UL, 0xFFF8000000000AAAUL, 0xBFF0000000000000UL, 0xBFF0000000000000UL, 0xBFF0000000000000UL, 0xBFF0000000000000UL },
            Bits64(r));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // FLAT
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// A reversed 1-D view coalesces to ONE strided run: NumPy's 8-accumulator unroll over the whole of it, so a NaN in the
    /// lane section keeps its payload (a contiguous model would canonicalize it), and a ±0 tie inside one accumulator goes
    /// to the later element.
    /// </summary>
    [TestMethod]
    public void Flat_ReversedRun_StridedUnroll_MatchNumPy()
    {
        var v = np.full(new Shape(40), -1.0);
        v.SetDouble(D(0xFFF8000000000CCC), 39 - 17);
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000CCCUL }, Bits64(Flat(v["::-1"], isMax: true, "reversed NaN")));

        var w = np.full(new Shape(40), 1.0);
        w.SetDouble(NegZero, 39 - 2);
        w.SetDouble(0.0, 39 - 10);
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(w["::-1"], isMax: false, "reversed tie")));
    }

    /// <summary>
    /// A stepped view (<c>base[:, ::2]</c>) coalesces to ONE strided run of stride 2 across the rows (<c>2 · 10 == 20</c>),
    /// so the NaN keeps its payload — the per-row and contiguous models both disagree.
    /// </summary>
    [TestMethod]
    public void Flat_SteppedView_CoalescesToOneStridedRun()
    {
        var st = np.full(new Shape(4, 20), 1.0);
        st.SetDouble(D(0xFFF8000000000FFF), 2, 6);
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000FFFUL }, Bits64(Flat(st[":, ::2"], isMax: true, "stepped")));
    }

    /// <summary>
    /// Reversed rows that do NOT coalesce, short enough to buffer: NumPy copies the (129, 5) view into ONE buffer of
    /// 645 elements and reduces it with one contiguous call, so a NaN at logical [0, 1] reaches the vector section and
    /// comes back canonical — where per-row strided calls would have kept its payload.
    /// </summary>
    [TestMethod]
    public void Flat_BufferedRows_OneContiguousCall_CanonicalizesNaN()
    {
        var bb = np.full(new Shape(129, 5), 1.0);
        bb.SetDouble(D(0xFFF8000000000DDD), 0, 4 - 1);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64 }, Bits64(Flat(bb[":, ::-1"], isMax: true, "buffered")));
    }

    /// <summary>
    /// Reversed rows LONGER than half the buffer (5000 &gt; 8192 / 2): NumPy's cost model keeps the input unbuffered and
    /// makes one strided call per row, so the NaN at logical [1, 4999] keeps its payload — buffering would have
    /// canonicalized it.
    /// </summary>
    [TestMethod]
    public void Flat_RowsLongerThanHalfTheBuffer_StayUnbuffered()
    {
        var ub = np.full(new Shape(2, 5000), 1.0);
        ub.SetDouble(D(0xFFF8000000000EEE), 1, 0);
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000EEEUL }, Bits64(Flat(ub[":, ::-1"], isMax: true, "row by row")));
    }

    /// <summary>
    /// The buffer CHUNK boundary is observable: a (3000, 4) reversed view is copied in fills of 8192 then 3808 elements,
    /// each one contiguous call seeded with the running result. <c>+0</c> at iteration index 8191 (the last element of
    /// fill 1) and <c>-0</c> at 8192 (the first of fill 2) give NumPy <c>+0</c>; the SAME values as one contiguous array
    /// (a single call) give <c>-0</c>, and so would per-row calls.
    /// </summary>
    [TestMethod]
    public void Flat_BufferChunkBoundary_DecidesTheTie()
    {
        var t = np.full(new Shape(3000, 4), -1.0);
        t.SetDouble(0.0, 8191 / 4, 3 - 8191 % 4);
        t.SetDouble(NegZero, 8192 / 4, 3 - 8192 % 4);
        var view = t[":, ::-1"];
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(view, isMax: true, "chunk boundary")));
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL }, Bits64(Flat(np.ascontiguousarray(view), isMax: true, "same values, one call")));
    }

    /// <summary>
    /// Every buffer fill after the first re-seeds <c>splat(running result)</c>: a NaN met in the scalar TAIL of fill 1
    /// keeps its payload through that call, and the next contiguous call turns it canonical.
    /// </summary>
    [TestMethod]
    public void Flat_NextBufferFill_CanonicalizesARunningNaN()
    {
        var tb = np.full(new Shape(3000, 4), 1.0);
        tb.SetDouble(D(0xFFF8000000000333), 2047, 3 - 3);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64 }, Bits64(Flat(tb[":, ::-1"], isMax: true, "running NaN")));
    }

    /// <summary>
    /// The chunking follows <see cref="np.setbufsize"/> exactly as NumPy's follows <c>np.setbufsize</c>: <c>+0</c> at
    /// iteration index 4095 and <c>-0</c> at 4096 share one fill at the default 8192 (NumPy: <c>-0</c>) but straddle the
    /// boundary at 4096 (NumPy: <c>+0</c>).
    /// </summary>
    [TestMethod]
    public void Flat_ChunkingFollowsBufsize()
    {
        var t = np.full(new Shape(3000, 4), -1.0);
        t.SetDouble(0.0, 4095 / 4, 3 - 4095 % 4);
        t.SetDouble(NegZero, 4096 / 4, 3 - 4096 % 4);
        var view = t[":, ::-1"];
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL }, Bits64(Flat(view, isMax: true, "bufsize 8192")));

        long old = np.setbufsize(4096);
        try
        {
            CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(view, isMax: true, "bufsize 4096")));
        }
        finally
        {
            np.setbufsize(old);
        }
    }

    /// <summary>
    /// An F-contiguous block is one dense block: one contiguous call in MEMORY order, so a NaN at memory index 5 (inside
    /// the vector section) comes back canonical.
    /// </summary>
    [TestMethod]
    public void Flat_FOrderDense_OneMemoryOrderCall()
    {
        var fo = np.asfortranarray(np.full(new Shape(6, 7), 2.0));
        fo.SetDouble(D(0xFFF8000000000111), 5, 0);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64 }, Bits64(Flat(fo, isMax: true, "F dense")));
    }

    /// <summary>
    /// A single element (here (1, 1)) is returned untouched — NumPy never calls the loop, so a NaN keeps its payload.
    /// </summary>
    [TestMethod]
    public void Flat_SingleElement_ReturnsItsBitsUntouched()
    {
        var one = np.full(new Shape(1, 1), 0.0);
        one.SetDouble(D(0xFFF8000000000222), 0, 0);
        CollectionAssert.AreEqual(new[] { 0xFFF8000000000222UL }, Bits64(Flat(one, isMax: true, "single")));
    }

    /// <summary>
    /// Integer flat reductions are order-free, so every layout must give the exact extreme — checked against a naive scan
    /// of a C-contiguous copy over dense, reversed, stepped, buffered, row-by-row and 3-D mixed layouts, for every served
    /// integer width.
    /// </summary>
    [TestMethod]
    public void Flat_Integers_EveryLayout_ExactExtremes()
    {
        foreach (var t in new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64 })
        {
            var baseArr = (np.arange(12000L) * 7919 % 251 - 120).astype(t);   // wraps for unsigned: still a fixed pattern
            var b2 = baseArr.reshape(3000, 4);
            var b3 = baseArr.reshape(10, 40, 30);
            foreach (var (name, x) in new[]
                     {
                         ("C", (NDArray)b2), ("reversed(buffered)", b2[":, ::-1"]), ("stepped", b2[":, ::2"]), ("rev1d", baseArr["::-1"]),
                         ("F", np.asfortranarray(b2)), ("3d-negmid", b3[":, ::-1, :"]), ("rows>half-buffer", baseArr.reshape(2, 6000)[":, ::-1"]),
                     })
            {
                var dense = np.ascontiguousarray(x).astype(NPTypeCode.Double).ToArray<double>();
                double mx = double.MinValue, mn = double.MaxValue;
                foreach (var e in dense)
                {
                    mx = System.Math.Max(mx, e);   // System.: NumSharp.Tests.Math is a sibling namespace
                    mn = System.Math.Min(mn, e);
                }

                Assert.AreEqual(mx, Convert.ToDouble(Flat(x, isMax: true, $"{t} {name} max").GetAtIndex(0)), $"{t} {name} max");
                Assert.AreEqual(mn, Convert.ToDouble(Flat(x, isMax: false, $"{t} {name} min").GetAtIndex(0)), $"{t} {name} min");
            }
        }
    }

    /// <summary>
    /// keepdims keeps the (1, …, 1) shape with the same bits, and a flat <c>np.ptp</c> over a buffered layout composes
    /// the two exact reductions.
    /// </summary>
    [TestMethod]
    public void Flat_KeepdimsAndPtp()
    {
        var bb = np.full(new Shape(129, 5), 1.0);
        bb.SetDouble(-3.5, 17, 2);
        bb.SetDouble(8.25, 90, 4);
        var view = bb[":, ::-1"];
        var k = np.max(view, keepdims: true);
        CollectionAssert.AreEqual(new long[] { 1, 1 }, k.shape);
        Assert.AreEqual(8.25, k.GetDouble(0, 0));
        Assert.AreEqual(11.75, np.ptp(view).GetDouble());
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Boundaries: each case changes NumPy's answer if a lane, a fill or a flush moves by one element
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// Rows mode (2, 5000) reversed: <c>+0</c> at logical [0, 8] and <c>-0</c> at [0, 9] land in accumulator lanes 7
    /// and 0 after the copied element, and the unroll's combine makes lane 7 win — NumPy <c>+0</c>. Including the copied
    /// element in the first call (a one-element lane shift) would put them in lanes 0 and 1 and give <c>-0</c>.
    /// </summary>
    [TestMethod]
    public void Flat_RowsMode_FirstCallSkipsTheCopiedElement()
    {
        var t = np.full(new Shape(2, 5000), -1.0);
        t.SetDouble(0.0, 0, 4999 - 8);
        t.SetDouble(NegZero, 0, 4999 - 9);
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(t[":, ::-1"], isMax: true, "rows lane shift")));
    }

    /// <summary>
    /// Buffered (129, 5) reversed: <c>+0</c> at iteration index 32 and <c>-0</c> at 33 sit in group 0's last vector and
    /// group 1's first after the copied element; the lane reduce makes <c>+0</c> win (NumPy). A first fill that did not
    /// skip the copied element would put both in one vector and give <c>-0</c>.
    /// </summary>
    [TestMethod]
    public void Flat_BufferedFirstFill_SkipsTheCopiedElement()
    {
        var t = np.full(new Shape(129, 5), -1.0);
        t.SetDouble(0.0, 32 / 5, 4 - 32 % 5);
        t.SetDouble(NegZero, 33 / 5, 4 - 33 % 5);
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(t[":, ::-1"], isMax: true, "buffered lane shift")));
    }

    /// <summary>
    /// A buffer fill holds WHOLE cores: base (2000, 5, 2) viewed <c>[:, 1:4, ::-1]</c> has a core of 6 (two axes that do
    /// not coalesce), so a fill is 6 · ⌊8192 / 6⌋ = 8190 elements, not 8192. <c>+0</c> at iteration 8189 and <c>-0</c> at
    /// 8190 straddle that boundary — NumPy <c>+0</c> — while the same values in one contiguous call give <c>-0</c>.
    /// </summary>
    [TestMethod]
    public void Flat_BufferFill_IsAWholeNumberOfCores()
    {
        var b = np.full(new Shape(2000, 5, 2), -1.0);
        void Put(int idx, double val) => b.SetDouble(val, idx / 6, 1 + idx % 6 / 2, 1 - idx % 2);
        Put(8189, 0.0);
        Put(8190, NegZero);
        var v = b[":, 1:4, ::-1"];
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(v, isMax: true, "fill rounding")));
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL }, Bits64(Flat(np.ascontiguousarray(v), isMax: true, "one call")));
    }

    /// <summary>
    /// A fill never crosses the outer dimension's end: base (4, 91, 101) viewed <c>[:, :90, :100]</c> buffers blocks of
    /// 9000 (core 100 × 90) as fills of 8100 then 900. <c>+0</c> at iteration 8999 (the last of block 0) and <c>-0</c> at
    /// 9000 (the first of block 1) are in different fills — NumPy <c>+0</c>; one contiguous call gives <c>-0</c>.
    /// </summary>
    [TestMethod]
    public void Flat_BufferFill_StopsAtTheOuterBlockEnd()
    {
        var b = np.full(new Shape(4, 91, 101), -1.0);
        void Put(int idx, double val) => b.SetDouble(val, idx / 9000, idx % 9000 / 100, idx % 100);
        Put(8999, 0.0);
        Put(9000, NegZero);
        var v = b[":, :90, :100"];
        CollectionAssert.AreEqual(new[] { 0x0UL }, Bits64(Flat(v, isMax: true, "block end")));
        CollectionAssert.AreEqual(new[] { 0x8000000000000000UL }, Bits64(Flat(np.ascontiguousarray(v), isMax: true, "one call")));
    }

    /// <summary>
    /// The NaN-free fast fold must hand EVERY group holding a NaN to NumPy's blended rule, whichever of its eight vectors
    /// the NaN is in (a NaN in a first operand would be lost by a plain max), and must not start from a NaN seed. Rows of
    /// 100: row q holds a NaN in vector q of group 1; a ninth case seeds the row with a NaN. All canonical in NumPy.
    /// </summary>
    [TestMethod]
    public void AxisRowContiguous_FastFold_NeverSwallowsANaN()
    {
        var c = np.full(new Shape(8, 100), 1.0);
        for (int q = 0; q < 8; q++)
            c.SetDouble(D(0xFFF8000000000500UL + (ulong)q), q, 1 + 32 + 4 * q);
        var r = Axis(c, 1, isMax: false, "NaN in each vector");
        foreach (var bits in Bits64(r))
            Assert.AreEqual(CanonicalNaN64, bits, "every row reaches the lane reduce with a NaN lane");

        var seed = np.full(new Shape(1, 100), 1.0);
        seed.SetDouble(D(0xFFF8000000000600), 0, 0);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64 }, Bits64(Axis(seed, 1, isMax: true, "NaN seed")));
    }

    /// <summary>
    /// A contiguous call SHORTER than one vector still re-seeds <c>splat(running result)</c>: the (4097, 2) reversed view
    /// fills 8192 then 2 elements, the NaN in fill 1's scalar tail keeps its payload there, and the 2-element fill turns
    /// it canonical; likewise a 2-element row whose first element is NaN.
    /// </summary>
    [TestMethod]
    public void ShortContiguousCalls_CanonicalizeTheSeed()
    {
        var t = np.full(new Shape(4097, 2), 1.0);
        t.SetDouble(D(0xFFF8000000000700), 8191 / 2, 1 - 8191 % 2);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64 }, Bits64(Flat(t[":, ::-1"], isMax: true, "2-element fill")));

        var r = np.full(new Shape(3, 2), 1.0);
        r.SetDouble(D(0xFFF8000000000800), 0, 0);
        CollectionAssert.AreEqual(new[] { CanonicalNaN64, 0x3FF0000000000000UL, 0x3FF0000000000000UL },
            Bits64(Axis(r, 1, isMax: true, "2-element rows")));
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Routes, hooks, declines
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// <see cref="NDExpr.DisableExactMinMax"/> sends the engine's flat and axis reductions back to the previous kernels:
    /// no exact run is counted, and the value (not the bits) is unchanged.
    /// </summary>
    [TestMethod]
    public void DisableHook_RestoresThePreviousKernels()
    {
        var a = (np.arange(600.0) * 7919 % 101 - 50).reshape(20, 30)[":, ::-1"];
        double flat = np.max(a).GetDouble(), axis0 = np.max(a, 0).GetDouble(3);
        NDExpr.DisableExactMinMax = true;
        try
        {
            int before = NDExpr.ExactMinMaxRuns;
            Assert.AreEqual(flat, np.max(a).GetDouble());
            Assert.AreEqual(axis0, np.max(a, 0).GetDouble(3));
            Assert.AreEqual(before, NDExpr.ExactMinMaxRuns, "the hook must skip every exact schedule");
        }
        finally
        {
            NDExpr.DisableExactMinMax = false;
        }
    }

    /// <summary>
    /// What the exact schedules decline keeps its previous kernel, with the right value: a broadcast input (a zero
    /// stride changes NpyIter's axis sort) and the dtypes without a NumPy SIMD loop to reproduce.
    /// </summary>
    [TestMethod]
    public void Declines_BroadcastAndUnservedDtypes_KeepTheirValue()
    {
        var bc = np.broadcast_to(np.array(new[] { 1.0, 5.0, -2.0 }), new Shape(4, 3));
        int before = NDExpr.ExactMinMaxRuns;
        Assert.AreEqual(5.0, np.max(bc).GetDouble());
        Assert.AreEqual(-2.0, np.min(bc, 1).GetDouble(0));
        Assert.AreEqual(before, NDExpr.ExactMinMaxRuns, "a broadcast input declines");

        var h = np.array(new[] { (Half)1, (Half)7, (Half)(-3) })["::-1"];
        Assert.AreEqual((Half)7, np.max(h).GetAtIndex<Half>(0));
        var bo = np.array(new[] { false, true, false })["::-1"];
        Assert.AreEqual(true, np.max(bo).GetAtIndex<bool>(0));
        Assert.AreEqual(before, NDExpr.ExactMinMaxRuns, "Half / Boolean decline");
    }

    // ---------------------------------------------------------------------------------------------------------------
    // The iterator port itself
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// <see cref="NumPyMinMaxReduce.FlatIteratorAxes"/> reproduces NpyIter's axis order and coalescing: |stride| order
    /// innermost first, extent-1 axes dropped, SIGNED coalescing (a fully reversed block is one run of stride -1, a
    /// stepped one one run of stride 2), and non-coalescable neighbours left apart.
    /// </summary>
    [TestMethod]
    public void FlatIteratorAxes_OrderAndCoalescing()
    {
        (long[] d, long[] s) Run(long[] dims, long[] strides)
        {
            Span<long> cd = stackalloc long[dims.Length], cs = stackalloc long[dims.Length];
            int c = NumPyMinMaxReduce.FlatIteratorAxes(dims, strides, cd, cs);
            return (cd[..c].ToArray(), cs[..c].ToArray());
        }

        var r = Run(new long[] { 3, 4 }, new long[] { -4, -1 });                  // base[::-1, ::-1]
        CollectionAssert.AreEqual(new long[] { 12 }, r.d);
        CollectionAssert.AreEqual(new long[] { -1 }, r.s);
        r = Run(new long[] { 4, 10 }, new long[] { 20, 2 });                      // base(4, 20)[:, ::2]
        CollectionAssert.AreEqual(new long[] { 40 }, r.d);
        CollectionAssert.AreEqual(new long[] { 2 }, r.s);
        r = Run(new long[] { 129, 5 }, new long[] { 5, -1 });                     // base[:, ::-1]
        CollectionAssert.AreEqual(new long[] { 5, 129 }, r.d);
        CollectionAssert.AreEqual(new long[] { -1, 5 }, r.s);
        r = Run(new long[] { 7, 6 }, new long[] { 1, 7 });                        // F (7, 6): memory order coalesces
        CollectionAssert.AreEqual(new long[] { 42 }, r.d);
        CollectionAssert.AreEqual(new long[] { 1 }, r.s);
        r = Run(new long[] { 4, 1, 6, 35 }, new long[] { 210, 999, -35, 1 });     // extent-1 dropped, negmid kept apart
        CollectionAssert.AreEqual(new long[] { 35, 6, 4 }, r.d);
        CollectionAssert.AreEqual(new long[] { 1, -35, 210 }, r.s);
        r = Run(new long[] { 1, 1 }, new long[] { 5, 7 });                        // one element
        Assert.AreEqual(0, r.d.Length);
        r = Run(new long[] { 3, 4 }, new long[] { 2, -2 });                       // equal |stride|: the LATER axis stays inner
        CollectionAssert.AreEqual(new long[] { 4, 3 }, r.d);
        CollectionAssert.AreEqual(new long[] { -2, 2 }, r.s);
    }

    /// <summary>
    /// <see cref="NumPyMinMaxReduce.FlatBufferingDim"/> reproduces <c>npyiter_find_buffering_setup</c>'s choice: rows of
    /// 5 and 4 are buffered with their neighbours (the core grows while the size fits), a row longer than half the buffer
    /// stays unbuffered, a larger buffer size admits it, and a three-axis layout grows the core over two dimensions.
    /// </summary>
    /// <remarks>
    /// The <c>(4096, 2, 3)</c> case sits on the loop's stop test: the running size is EXACTLY the buffer size when
    /// dimension 2 is considered, and NumPy's <c>size &gt;= maximum_size</c> stops there. A <c>&gt;</c> would take
    /// dimension 2 (core 8192) — which fills the same 8192-element buffers, so no reduction result can tell the two
    /// apart; the pin guards the port of the cost model itself, which the fill arithmetic downstream relies on.
    /// </remarks>
    [TestMethod]
    public void FlatBufferingDim_ReproducesNumPysCostModel()
    {
        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 5, 129 }, 8192, out long core, out long size));
        Assert.AreEqual(5, core);
        Assert.AreEqual(645, size);

        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 4, 3000 }, 8192, out core, out size));
        Assert.AreEqual(4, core);
        Assert.AreEqual(12000, size);                    // capped by the caller to 4 · ⌊8192 / 4⌋ = 8192 per fill

        Assert.AreEqual(0, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 5000, 2 }, 8192, out core, out size));
        Assert.AreEqual(1, core);
        Assert.AreEqual(5000, size);
        Assert.AreEqual(0, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 4097, 3 }, 8192, out _, out _));   // 2·4097 > 8192
        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 4096, 3 }, 8192, out _, out _));   // 2·4096 <= 8192
        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 5000, 2 }, 16384, out _, out _));  // a larger buffer admits it

        Assert.AreEqual(2, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 35, 6, 4 }, 8192, out core, out size));
        Assert.AreEqual(210, core);
        Assert.AreEqual(840, size);
        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 100, 90, 4 }, 8192, out core, out size));  // 9000 >= 8192 stops at dim 2
        Assert.AreEqual(100, core);
        Assert.AreEqual(9000, size);

        // size == 8192 exactly when dimension 2 comes up: ">=" stops (see remarks).
        Assert.AreEqual(1, NumPyMinMaxReduce.FlatBufferingDim(new long[] { 4096, 2, 3 }, 8192, out core, out size));
        Assert.AreEqual(4096, core);
        Assert.AreEqual(8192, size);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Lane order of the strided unroll
    // ---------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// A 1-D view whose LOGICAL order is <paramref name="logical"/> and whose element stride is <paramref name="s"/>
    /// (any sign, any magnitude), over a fresh base whose unused gaps hold <c>7</c> — the gaps must never be read, and
    /// a 7 surviving into a min/max of zeros would show it.
    /// </summary>
    /// <param name="logical">The element values in logical order.</param>
    /// <param name="s">The view's element stride (nonzero).</param>
    /// <param name="t">float64 or float32.</param>
    /// <returns>The strided view.</returns>
    private static NDArray StridedView(double[] logical, int s, NPTypeCode t)
    {
        int n = logical.Length, a = System.Math.Abs(s);
        var baseArr = np.full(new Shape(a * n), 7.0, t);
        // base[::s] walks from the START for s > 0 and from the END for s < 0.
        for (int k = 0; k < n; k++)
        {
            long at = s > 0 ? (long)k * s : a * n - 1 - (long)k * a;
            if (t == NPTypeCode.Double)
                baseArr.SetAtIndex(logical[k], at);
            else
                baseArr.SetAtIndex((float)logical[k], at);
        }

        return baseArr[$"::{s}"];
    }

    /// <summary>
    /// The combine order of the strided 8-accumulator unroll, pinned for EVERY lane pair: NumPy folds a strided run into
    /// eight accumulators and combines them <c>(0,1)(2,3)(4,5)(6,7) → (01,23)(45,67) → (0123,4567)</c>, after the copied
    /// <c>x[0]</c> and before the scalar tail, the second operand winning a tie. Because that lane op is associative, the
    /// result is the rule "the zero LATEST in the order x[0], lane 0 … lane 7, tail wins a ±0 tie" — probed against
    /// NumPy 2.4.2 on all 1,056 cases this test replays (every pair a &lt; b of the eight lanes × both zero orders ×
    /// strides -1 (the reversed load), 2 and -3 (the gathers) × max / min × float64 / float32, plus x[0] against each
    /// lane and each lane against the tail): 0 disagreements. Here the lanes are Vector256 lanes, so this is what
    /// catches a lane extracted into the wrong accumulator or a reversed load whose lane permute is off.
    /// </summary>
    [TestMethod]
    public void Flat_StridedRun_LaterLaneWinsEveryTie()
    {
        const int Chain = 43;                                  // five whole lane groups + a three-element tail
        const int L = 1 + Chain;                               // + the copied x[0]
        int cases = 0;
        foreach (var t in new[] { NPTypeCode.Double, NPTypeCode.Single })
        foreach (int s in new[] { -1, 2, -3 })
        foreach (bool isMax in new[] { true, false })
        {
            double fill = isMax ? -1.0 : 1.0;                  // away from zero on the losing side

            void Check(double[] logical, bool wantNegative, string what)
            {
                var r = Flat(StridedView(logical, s, t), isMax, $"{t} s={s} {(isMax ? "max" : "min")} {what}");
                double v = t == NPTypeCode.Double ? r.GetAtIndex<double>(0) : r.GetAtIndex<float>(0);
                Assert.AreEqual(0.0, v, $"{t} s={s} {what}: a zero must win");
                Assert.AreEqual(wantNegative, double.IsNegative(v), $"{t} s={s} {(isMax ? "max" : "min")} {what}: the later zero's sign");
                cases++;
            }

            for (int a = 0; a < 8; a++)
            for (int b = a + 1; b < 8; b++)
            foreach (bool plusFirst in new[] { true, false })
            {
                var logical = new double[L];
                Array.Fill(logical, fill);
                logical[1 + 16 + a] = plusFirst ? 0.0 : NegZero;   // group 2, lane a
                logical[1 + 16 + b] = plusFirst ? NegZero : 0.0;   // group 2, lane b — the later one
                Check(logical, plusFirst, $"lanes {a}<{b} plusFirst={plusFirst}");
            }

            for (int lane = 0; lane < 8; lane++)
            foreach (bool plusFirst in new[] { true, false })
            {
                var logical = new double[L];
                Array.Fill(logical, fill);
                logical[0] = plusFirst ? 0.0 : NegZero;             // the copied x[0] comes first …
                logical[1 + 8 + lane] = plusFirst ? NegZero : 0.0;  // … every lane after it
                Check(logical, plusFirst, $"x0 vs lane {lane}");

                logical = new double[L];
                Array.Fill(logical, fill);
                logical[1 + 8 + lane] = plusFirst ? 0.0 : NegZero;  // a lane …
                logical[1 + 41] = plusFirst ? NegZero : 0.0;        // … before the scalar tail
                Check(logical, plusFirst, $"lane {lane} vs tail");
            }
        }

        Assert.AreEqual(1056, cases, "every probed case replayed");
    }
}
