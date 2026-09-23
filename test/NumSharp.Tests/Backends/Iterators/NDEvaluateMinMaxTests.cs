using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// The NumPy-exact flat <c>Min</c> / <c>Max</c> of np.evaluate (DefaultEngine.Evaluate.MinMax.cs): NumPy 2.4.2's
    /// contiguous reduction schedule reproduced lane for lane, so the result BITS — which zero sign survives a ±0 tie,
    /// the canonical +NaN for a NaN reaching the vector section, a tail NaN's payload — equal NumPy's, on every route the
    /// host takes (the bare contiguous leaf in place, a streamed computed child, a materialized one). The literal cases
    /// below were probed against NumPy 2.4.2 (win-amd64, AVX2 dispatch); the random cases hold all routes AND an
    /// independent test-side model of the schedule to identical bits.
    /// </summary>
    [TestClass]
    public class NDEvaluateMinMaxTests
    {
        /// <summary>
        /// A real negative zero held in a field — a <c>-0.0</c> literal folded into an expression is easy to lose to a
        /// constant-fold that normalizes it, so the tie cases read it from here.
        /// </summary>
        private static readonly double NegZero = -0.0;

        /// <summary>A float64 with the exact bit pattern <paramref name="bits"/> (for NaNs of chosen sign / payload).</summary>
        /// <param name="bits">The raw IEEE-754 binary64 bits.</param>
        /// <returns>The double those bits encode.</returns>
        private static double NaN64(ulong bits) => BitConverter.Int64BitsToDouble((long)bits);

        /// <summary>A float32 with the exact bit pattern <paramref name="bits"/> (for NaNs of chosen sign / payload).</summary>
        /// <param name="bits">The raw IEEE-754 binary32 bits.</param>
        /// <returns>The float those bits encode.</returns>
        private static float NaN32(uint bits) => BitConverter.Int32BitsToSingle((int)bits);

        /// <summary>
        /// Evaluate <c>Max</c> / <c>Min</c> of <paramref name="x"/> on the three exact routes — the bare leaf (in place),
        /// <c>x * 1</c> streamed, and <c>x * 1</c> materialized (streaming disabled) — asserting each engaged the exact
        /// schedule, and return the three raw bit patterns.
        /// </summary>
        /// <param name="x">A C-contiguous float64 or float32 array.</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <returns>The raw result bits of (leaf, streamed, materialized).</returns>
        private static (long leaf, long stream, long mat) AllRoutes(NDArray x, bool isMax)
        {
            var one = np.ones(x.Shape, x.typecode);
            NDExpr Wrap(NDExpr e) => isMax ? NDExpr.Max(e) : NDExpr.Min(e);
            long Bits(NDArray r) => r.typecode == NPTypeCode.Double
                ? BitConverter.DoubleToInt64Bits(r.GetAtIndex<double>(0))
                : BitConverter.SingleToInt32Bits(r.GetAtIndex<float>(0));

            NDExpr.ExactMinMaxRuns = 0;
            NDExpr.StreamingReductions = 0;
            long leaf = Bits(np.evaluate(Wrap((NDExpr)x)));
            Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "the bare contiguous leaf must take the exact schedule in place");
            Assert.AreEqual(0, NDExpr.StreamingReductions, "the in-place route streams nothing");

            long stream = Bits(np.evaluate(Wrap((NDExpr)x * (NDExpr)one)));
            Assert.AreEqual(2, NDExpr.ExactMinMaxRuns, "a contiguous computed child must take the exact schedule");
            Assert.AreEqual(1, NDExpr.StreamingReductions, "... streamed");

            long mat;
            try
            {
                NDExpr.DisableStreamingReduce = true;
                mat = Bits(np.evaluate(Wrap((NDExpr)x * (NDExpr)one)));
            }
            finally
            {
                NDExpr.DisableStreamingReduce = false;
            }

            Assert.AreEqual(3, NDExpr.ExactMinMaxRuns, "with streaming off the float child is materialized and still exact");
            return (leaf, stream, mat);
        }

        /// <summary>
        /// Literal cases probed against NumPy 2.4.2: a ±0 tie decided by the LANE STRUCTURE (a sequential scan picks the
        /// other sign), tail-only ties, the canonical +NaN for a vector-section NaN (and for a NaN first element once
        /// anything follows it), and a tail / lone NaN keeping its payload — for max and min, float64 and float32.
        /// </summary>
        [TestMethod]
        public void NumPyProbedBits_LaneStructuredTiesAndNaNs()
        {
            // f64 n=33: x0 then ONE 8-vector group. +0 at stream index 3 (lane 3 of v0), -0 at 28 (lane 0 of v7):
            // NumPy says +0, where "the last zero wins" (a sequential scan) would say -0.
            void F64(int i, double si, int j, double sj, bool isMax, long expect)
            {
                var x = new double[33];
                Array.Fill(x, isMax ? -1.0 : 1.0);
                x[1 + i] = si;
                x[1 + j] = sj;
                var (leaf, stream, mat) = AllRoutes(np.array(x), isMax);
                Assert.AreEqual(expect, leaf, $"leaf {(isMax ? "max" : "min")} ({i}:{si},{j}:{sj})");
                Assert.AreEqual(expect, stream, "stream");
                Assert.AreEqual(expect, mat, "materialized");
            }

            const long Pos = 0L, Neg = unchecked((long)0x8000000000000000UL);
            F64(3, 0.0, 28, NegZero, isMax: true, Pos);
            F64(3, NegZero, 28, 0.0, isMax: true, Neg);
            F64(0, 0.0, 31, NegZero, isMax: true, Neg);
            F64(28, 0.0, 3, NegZero, isMax: true, Neg);
            F64(3, NegZero, 28, 0.0, isMax: false, Neg);
            F64(3, 0.0, 28, NegZero, isMax: false, Pos);
            F64(0, NegZero, 31, 0.0, isMax: false, Pos);

            // Tail only (n = 3 < one vector): the scalar op, second operand wins the tie.
            Assert.AreEqual(Neg, AllRoutes(np.array(new[] { -1.0, 0.0, NegZero }), true).leaf);
            Assert.AreEqual(Pos, AllRoutes(np.array(new[] { -1.0, NegZero, 0.0 }), true).leaf);

            // A NaN that reaches the vector section comes back as NumPy's canonical POSITIVE quiet NaN.
            var vn = new double[40];
            Array.Fill(vn, 1.0);
            vn[5] = NaN64(0xFFF8000000000001);
            var r = AllRoutes(np.array(vn), true);
            Assert.AreEqual(0x7FF8000000000000L, r.leaf);
            Assert.AreEqual(0x7FF8000000000000L, r.stream);
            Assert.AreEqual(0x7FF8000000000000L, r.mat);
            Assert.AreEqual(0x7FF8000000000000L, AllRoutes(np.array(vn), false).leaf);

            // A NaN met only in the scalar tail keeps its payload (n = 35: one group of 32 after x0, then 2 tail).
            var tn = new double[35];
            Array.Fill(tn, 1.0);
            tn[34] = NaN64(0xFFF800000000BEEF);
            Assert.AreEqual(unchecked((long)0xFFF800000000BEEFUL), AllRoutes(np.array(tn), true).stream);
            Assert.AreEqual(unchecked((long)0xFFF800000000BEEFUL), AllRoutes(np.array(tn), false).leaf);

            // A lone element is returned untouched (NumPy's loop returns before touching anything); a NaN first element
            // with anything after it seeds every lane and comes back canonical.
            Assert.AreEqual(unchecked((long)0xFFF800000000BEEFUL), AllRoutes(np.array(new[] { NaN64(0xFFF800000000BEEF) }), true).leaf);
            Assert.AreEqual(0x7FF8000000000000L, AllRoutes(np.array(new[] { NaN64(0xFFF800000000BEEF), 1.0 }), true).leaf);

            // f32 n=65: x0 then ONE 8-vector group of 64 (8 lanes).
            void F32(int i, float si, int j, float sj, long expect)
            {
                var x = new float[65];
                Array.Fill(x, -1f);
                x[1 + i] = si;
                x[1 + j] = sj;
                var (leaf, stream, mat) = AllRoutes(np.array(x), true);
                Assert.AreEqual(expect, leaf, $"f32 leaf ({i}:{si},{j}:{sj})");
                Assert.AreEqual(expect, stream, "f32 stream");
                Assert.AreEqual(expect, mat, "f32 materialized");
            }

            F32(7, 0f, 56, -0f, 0L);
            F32(7, -0f, 56, 0f, unchecked((int)0x80000000));
            F32(0, 0f, 63, -0f, unchecked((int)0x80000000));

            // Across the STREAM's scratch-block boundary (1024 float64 / 2048 float32 elements): one zero in lane 0 of
            // block 0, the other in lane 1 of block 1. NumPy's single-buffer schedule keeps them in their own lanes, so
            // the lane-1 zero decides; a stream that restarted the schedule per block (horizontal reduce + re-splat)
            // would smear the block-0 zero over every lane and report the other sign.
            foreach (int n in new[] { 1 + 1024 + 64, 1 + 3000 })
            {
                foreach (var (sa, sb, expectMax) in new[] { (0.0, NegZero, Neg), (NegZero, 0.0, Pos) })
                {
                    var x = new double[n];
                    Array.Fill(x, -1.0);
                    x[1] = sa;
                    x[1 + 1025] = sb;
                    var (leaf, stream, mat) = AllRoutes(np.array(x), true);
                    Assert.AreEqual(expectMax, leaf, $"cross-block leaf n={n}");
                    Assert.AreEqual(expectMax, stream, $"cross-block stream n={n}");
                    Assert.AreEqual(expectMax, mat, $"cross-block materialized n={n}");
                    for (int i = 0; i < n; i++) x[i] = -x[i];
                    Assert.AreEqual(expectMax == Neg ? Pos : Neg, AllRoutes(np.array(x), false).stream, $"cross-block min stream n={n}");
                }
            }

            foreach (int n in new[] { 1 + 2048 + 128, 1 + 5000 })
            {
                foreach (var (sa, sb, expectMax) in new[] { (0f, -0f, unchecked((int)0x80000000)), (-0f, 0f, 0) })
                {
                    var x = new float[n];
                    Array.Fill(x, -1f);
                    x[1] = sa;
                    x[1 + 2049] = sb;
                    var (leaf, stream, mat) = AllRoutes(np.array(x), true);
                    Assert.AreEqual((long)expectMax, leaf, $"f32 cross-block leaf n={n}");
                    Assert.AreEqual((long)expectMax, stream, $"f32 cross-block stream n={n}");
                    Assert.AreEqual((long)expectMax, mat, $"f32 cross-block materialized n={n}");
                }
            }
            var fn = new float[70];
            Array.Fill(fn, 1f);
            fn[3] = NaN32(0xFFC00001);
            Assert.AreEqual(0x7FC00000L, AllRoutes(np.array(fn), true).leaf);
        }

        /// <summary>
        /// Random tie- and NaN-heavy data across the group (32 / 64) and scratch-block (1024 / 2048) boundaries: every
        /// route must produce the same bits as an INDEPENDENT test-side model of NumPy's schedule (lane arrays, no
        /// vectors), for float64 and float32, max and min.
        /// </summary>
        [TestMethod]
        public void Routes_MatchAnIndependentModelOfTheSchedule()
        {
            var rng = new System.Random(20260923);
            foreach (int n in new[] { 1, 2, 3, 4, 5, 8, 31, 32, 33, 34, 63, 64, 65, 66, 100, 1024, 1025, 1026, 2049, 2050, 5000, 20001 })
            foreach (int mode in new[] { 0, 1, 2 })
            {
                var d = new double[n];
                var f = new float[n];
                for (int i = 0; i < n; i++)
                {
                    double v = mode switch
                    {
                        0 => rng.Next(2) == 0 ? 0.0 : NegZero,                                   // all ±0
                        1 => rng.Next(40) == 0 ? (rng.Next(2) == 0 ? 0.0 : NegZero) : -1.0 - rng.NextDouble(), // sparse ±0 extremes
                        _ => rng.Next(250) == 0 ? NaN64(rng.Next(2) == 0 ? 0xFFF8000000000001UL : 0x7FF8000000000123UL)
                                                : (rng.Next(2) == 0 ? 0.0 : NegZero),             // NaN-laced ±0
                    };
                    d[i] = v;
                    f[i] = double.IsNaN(v) ? NaN32(v < 0 || BitConverter.DoubleToInt64Bits(v) < 0 ? 0xFFC00001u : 0x7FC00123u) : (float)v;
                }

                foreach (bool isMax in new[] { true, false })
                {
                    long want = BitConverter.DoubleToInt64Bits(Model(d, isMax));
                    var r = AllRoutes(np.array(d), isMax);
                    Assert.AreEqual(want, r.leaf, $"f64 leaf n={n} mode={mode} max={isMax}");
                    Assert.AreEqual(want, r.stream, $"f64 stream n={n} mode={mode} max={isMax}");
                    Assert.AreEqual(want, r.mat, $"f64 materialized n={n} mode={mode} max={isMax}");

                    long want32 = BitConverter.SingleToInt32Bits(Model(f, isMax));
                    var r32 = AllRoutes(np.array(f), isMax);
                    Assert.AreEqual(want32, r32.leaf, $"f32 leaf n={n} mode={mode} max={isMax}");
                    Assert.AreEqual(want32, r32.stream, $"f32 stream n={n} mode={mode} max={isMax}");
                    Assert.AreEqual(want32, r32.mat, $"f32 materialized n={n} mode={mode} max={isMax}");
                }
            }
        }

        /// <summary>
        /// Test-side model of NumPy 2.4.2's <c>simd_reduce_c_{max,min}</c> (AVX2: 32-byte vectors, 8-vector groups,
        /// canonical +NaN horizontal step, SSE scalar tail) written over plain lane arrays — independent of the
        /// engine's vector code, so an engine bug cannot hide behind a shared helper.
        /// </summary>
        /// <typeparam name="T">double or float.</typeparam>
        /// <param name="x">The contiguous data.</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <returns>NumPy's result.</returns>
        private static T Model<T>(T[] x, bool isMax) where T : unmanaged, IFloatingPointIeee754<T>
        {
            T P(T a, T b) => isMax ? (a > b ? a : b) : (a < b ? a : b);
            T N(T a, T b) => T.IsNaN(a) ? a : P(a, b);

            int n = x.Length;
            if (n == 1)
                return x[0];
            int vstep = 32 / System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            int wstep = vstep * 8;
            var acc = new T[vstep];
            Array.Fill(acc, x[0]);
            int i = 1;
            for (; n - i >= wstep; i += wstep)
            {
                for (int l = 0; l < vstep; l++)
                {
                    T r01 = N(x[i + l], x[i + vstep + l]);
                    T r23 = N(x[i + 2 * vstep + l], x[i + 3 * vstep + l]);
                    T r45 = N(x[i + 4 * vstep + l], x[i + 5 * vstep + l]);
                    T r67 = N(x[i + 6 * vstep + l], x[i + 7 * vstep + l]);
                    acc[l] = N(acc[l], N(N(r01, r23), N(r45, r67)));
                }
            }

            for (; n - i >= vstep; i += vstep)
                for (int l = 0; l < vstep; l++)
                    acc[l] = N(acc[l], x[i + l]);

            T r;
            if (Array.Exists(acc, T.IsNaN))
            {
                r = typeof(T) == typeof(double)
                    ? (T)(object)BitConverter.Int64BitsToDouble(0x7FF8000000000000)
                    : (T)(object)BitConverter.Int32BitsToSingle(0x7FC00000);
            }
            else
            {
                for (int width = vstep; width > 1; width >>= 1)
                    for (int l = 0, h = width >> 1; l < h; l++)
                        acc[l] = P(acc[l], acc[l + h]);
                r = acc[0];
            }

            for (; i < n; i++)
                r = N(r, x[i]);
            return r;
        }

        /// <summary>
        /// Integer children are order-free (no NaN, no tie that differs in bits): the exact routes (in place, streamed)
        /// must equal <c>np.amax</c> / <c>np.amin</c> for every width, extremes included; a non-streamable integer
        /// child is left to the (already exact) fold.
        /// </summary>
        [TestMethod]
        public void Integers_AreExactOnEveryWidth()
        {
            var rng = new System.Random(7);
            foreach (var tc in new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
                                       NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64 })
            foreach (int n in new[] { 1, 7, 33, 257, 3000, 70001 })
            {
                var raw = new long[n];
                for (int i = 0; i < n; i++) raw[i] = rng.Next(-120, 121);
                if (n > 5)
                {
                    // The dtype's own extremes, away from the edges, so the lane / tail split is exercised. The int64 →
                    // narrower astype is modular, so -1 lands on the unsigned maxima (uint64 included).
                    (long hi, long lo) = tc switch
                    {
                        NPTypeCode.SByte => (sbyte.MaxValue, sbyte.MinValue),
                        NPTypeCode.Int16 => (short.MaxValue, short.MinValue),
                        NPTypeCode.Int32 => (int.MaxValue, int.MinValue),
                        NPTypeCode.Int64 => (long.MaxValue, long.MinValue),
                        _ => (-1L, 0L),
                    };
                    raw[n / 3] = hi;
                    raw[2 * n / 3] = lo;
                }

                var a = np.array(raw).astype(tc);

                var one = np.ones(a.Shape, tc);
                foreach (bool isMax in new[] { true, false })
                {
                    var want = isMax ? np.amax(a) : np.amin(a);
                    NDExpr.ExactMinMaxRuns = 0;
                    var leaf = np.evaluate(isMax ? NDExpr.Max((NDExpr)a) : NDExpr.Min((NDExpr)a));
                    var stream = np.evaluate(isMax ? NDExpr.Max((NDExpr)a * (NDExpr)one) : NDExpr.Min((NDExpr)a * (NDExpr)one));
                    Assert.AreEqual(2, NDExpr.ExactMinMaxRuns, $"{tc} n={n}: both contiguous routes are exact-schedule");
                    Assert.AreEqual(tc, leaf.typecode);
                    Assert.AreEqual(want.GetAtIndex(0), leaf.GetAtIndex(0), $"{tc} leaf n={n} max={isMax}");
                    Assert.AreEqual(want.GetAtIndex(0), stream.GetAtIndex(0), $"{tc} stream n={n} max={isMax}");
                }
            }
        }

        /// <summary>
        /// Where the host must NOT impose the contiguous schedule: a bare strided / broadcast leaf (NumPy reduces those
        /// with its scalar 8-accumulator unroll), an integer child that does not stream (the fold is exact), and every
        /// dtype outside the served set (Half, Complex, Boolean) — all fold, with the value still right. A strided FLOAT
        /// computed child is materialized (NumPy materializes it too) and does take the exact schedule.
        /// </summary>
        [TestMethod]
        public void Declines_AndTheMaterializeRoute()
        {
            var big = np.arange(4000).astype(NPTypeCode.Double) % 97 - 50;
            var strided = big["::3"];
            NDExpr.ExactMinMaxRuns = 0;
            var s = np.evaluate(NDExpr.Max((NDExpr)strided));
            Assert.AreEqual(0, NDExpr.ExactMinMaxRuns, "a bare strided leaf keeps the fold");
            Assert.AreEqual((double)np.amax(strided), s.GetAtIndex<double>(0));

            NDExpr.StreamingReductions = 0;
            var sc = np.evaluate(NDExpr.Max((NDExpr)strided * 2.0));
            Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "a strided float computed child is materialized, then exact");
            Assert.AreEqual(0, NDExpr.StreamingReductions);
            Assert.AreEqual((double)np.amax(strided * 2.0), sc.GetAtIndex<double>(0));

            var istrided = (np.arange(4000).astype(NPTypeCode.Int32) % 97)["::3"];
            NDExpr.ExactMinMaxRuns = 0;
            var ic = np.evaluate(NDExpr.Min((NDExpr)istrided * 2));
            Assert.AreEqual(0, NDExpr.ExactMinMaxRuns, "a non-streamable integer child keeps the (exact) fold");
            Assert.AreEqual((int)np.amin(istrided * 2), ic.GetAtIndex<int>(0));

            var row = np.arange(50).astype(NPTypeCode.Double);
            var col = np.arange(40).astype(NPTypeCode.Double).reshape(40, 1);
            NDExpr.ExactMinMaxRuns = 0;
            var bc = np.evaluate(NDExpr.Max((NDExpr)col * (NDExpr)row));
            Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "a broadcast float product is materialized, then exact");
            Assert.AreEqual(39.0 * 49.0, bc.GetAtIndex<double>(0));

            foreach (var tc in new[] { NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Boolean })
            {
                var a = (np.arange(100) % 5).astype(tc);
                NDExpr.ExactMinMaxRuns = 0;
                var r = np.evaluate(NDExpr.Max((NDExpr)a));
                Assert.AreEqual(0, NDExpr.ExactMinMaxRuns, $"{tc} is not served");
                Assert.AreEqual(np.amax(a).GetAtIndex(0), r.GetAtIndex(0), $"{tc} value");
            }

            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Max((NDExpr)np.zeros(new Shape(0), NPTypeCode.Double))),
                "an empty max still raises (no identity) — the exact route only runs for n > 0");
        }

        /// <summary>
        /// The exact route honours the flat reduction's result plumbing: <c>keepdims</c> returns shape <c>(1,)*ndim</c>,
        /// an <c>out=</c> of a wider dtype receives the converted value and is returned as is, and an F-contiguous 2-D
        /// leaf reduces its memory in place.
        /// </summary>
        [TestMethod]
        public void KeepdimsOutAndFContiguous()
        {
            var m = (np.arange(6 * 7).astype(NPTypeCode.Single) % 11 - 5).reshape(6, 7);
            var kd = np.evaluate(NDExpr.Max((NDExpr)m, keepdims: true));
            CollectionAssert.AreEqual(new long[] { 1, 1 }, kd.shape);
            Assert.AreEqual(5f, kd.GetAtIndex<float>(0));

            var o = np.zeros(Shape.NewScalar(), NPTypeCode.Double);
            var ro = np.evaluate(NDExpr.Min((NDExpr)m * 2f), @out: o);
            Assert.AreSame(o, ro);
            Assert.AreEqual(-10.0, o.GetAtIndex<double>(0));

            var fm = np.asfortranarray(m);
            NDExpr.ExactMinMaxRuns = 0;
            var fr = np.evaluate(NDExpr.Max((NDExpr)fm));
            Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "an F-contiguous leaf is one dense block — in place");
            Assert.AreEqual(5f, fr.GetAtIndex<float>(0));

            // The in-place route must start at the view's logical element 0: an F column block keeps a NON-zero offset
            // (a contiguous row slice would re-seat the address instead). Columns 0..1 — outside the window — hold 100,
            // so a scan that ignored the offset would report it.
            var w = np.zeros(new Shape(4, 10), NPTypeCode.Double);
            w[":, :2"] = 100.0;
            w[":, 2:"] = (np.arange(32).astype(NPTypeCode.Double) % 6).reshape(4, 8);
            var win = np.asfortranarray(w)[":, 2:10"];
            Assert.IsTrue(win.Shape.IsFContiguous && win.Shape.offset == 8, "precondition: F-contiguous at offset 8");
            NDExpr.ExactMinMaxRuns = 0;
            Assert.AreEqual(5.0, np.evaluate(NDExpr.Max((NDExpr)win)).GetAtIndex<double>(0));
            Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "the offset window is dense — in place");
        }
    }
}
