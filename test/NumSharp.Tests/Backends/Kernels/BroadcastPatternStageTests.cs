using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The element-wise kernel paths np.clip's iterator route exercised into existence, each pinned against an
    ///     independent route that computes the same values:
    ///     <list type="bullet">
    ///       <item>the N-ary broadcast-pattern STAGES (DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs) of the 1-D
    ///       shell, the 2-D block kernel and the where= 2-D block kernel — every C/S pattern of three inputs, where S is an
    ///       input broadcast along the loop (inner byte stride 0);</item>
    ///       <item>the hardware-gather route's broadcast stages — a strided input next to 0-d ones, for clip and for the
    ///       binary ufuncs that share the shell;</item>
    ///       <item>the flat clip kernel's output-alignment peel and its non-temporal (streaming) store variant;</item>
    ///       <item>NDIter's allocation-light broadcast of operands whose shapes differ.</item>
    ///     </list>
    ///     The reference for a kernel path is the SAME call with every operand materialized full-size and contiguous — the
    ///     all-contiguous loop that predates the stages — so a stage that read the wrong operand, mis-hoisted a broadcast or
    ///     stored to the wrong lane fails. Values never tie a bound and hold no NaN, so the float loop NumPy would pick
    ///     (const vs general) cannot change a result.
    /// </summary>
    [TestClass]
    public class BroadcastPatternStageTests
    {
        /// <summary>
        /// The dtypes the pattern tests sweep: each has a vector clip body, and together they cover the 2-, 4- and 8-byte
        /// lanes, signed and unsigned integers and both float widths (the const loop's float32 / float64 paths).
        /// </summary>
        private static readonly DType[] SimdDtypes = { np.int16, np.int32, np.uint32, np.int64, np.float32, np.float64 };

        /// <summary>
        /// Random whole values in [lo, lo + span), plus <paramref name="frac"/> for a float dtype. The tests give the
        /// elements and the bounds different fractions (0.5 and 0.25), so no float element ever ties a bound: a tie is
        /// where NumPy's const and general float loops differ (an integer tie changes nothing).
        /// </summary>
        /// <param name="rnd">The seeded generator of the calling test, so every run draws the same values.</param>
        /// <param name="dt">The dtype of the result.</param>
        /// <param name="shape">The result shape; an empty array gives a 0-d result.</param>
        /// <param name="lo">The smallest whole value.</param>
        /// <param name="span">The number of whole values to draw from.</param>
        /// <param name="frac">The fraction added to every value of a float dtype (ignored for integer dtypes).</param>
        /// <returns>A fresh C-contiguous array of <paramref name="dt"/> with <paramref name="shape"/>.</returns>
        private static NDArray Data(System.Random rnd, DType dt, long[] shape, int lo, int span, double frac)
        {
            long n = shape.Aggregate(1L, (a, b) => a * b);
            var v = new double[n];
            bool isFloat = dt == np.float32 || dt == np.float64;
            for (long i = 0; i < n; i++)
                v[i] = lo + rnd.Next(span) + (isFloat ? frac : 0);
            return np.array(v).reshape(shape).astype(dt);
        }

        /// <summary>NumPy's right-aligned broadcast of operand shapes. It does not check compatibility: the tests only pass shapes that broadcast.</summary>
        /// <param name="shapes">The operand shapes (an empty array is a 0-d operand).</param>
        /// <returns>The broadcast shape: per axis, the extent of the last operand that is not 1 there, else 1.</returns>
        private static long[] BroadcastShape(params long[][] shapes)
        {
            int nd = shapes.Max(s => s.Length);
            var b = new long[nd];
            for (int k = 0; k < nd; k++)
            {
                long d = 1;
                foreach (var s in shapes)
                {
                    int j = s.Length - nd + k;
                    if (j >= 0 && s[j] != 1)
                        d = s[j];
                }
                b[k] = d;
            }
            return b;
        }

        /// <summary>
        /// One operand broadcast to the call's full shape and copied into a fresh C-contiguous array: the reference
        /// operand, which takes the all-contiguous loop that predates the stages.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="shape">The full broadcast shape of the call.</param>
        /// <returns>A fresh, writeable C-contiguous array of <paramref name="a"/>'s dtype.</returns>
        private static NDArray Full(NDArray a, long[] shape) => np.broadcast_to(a, new Shape(shape)).copy();

        /// <summary>
        /// Asserts that two results have the same dtype, shape and values. The values are compared after a cast to
        /// float64, which is exact for every dtype these tests use (all integers here are far below 2^53).
        /// </summary>
        /// <param name="got">The result of the route under test.</param>
        /// <param name="want">The result of the reference route.</param>
        /// <param name="what">The case description that a failure message names.</param>
        /// <exception cref="AssertFailedException">The dtype, the shape or a value differs (AwesomeAssertions raises the
        /// test framework's failure type).</exception>
        private static void SameResult(NDArray got, NDArray want, string what)
        {
            got.dtype.Should().Be(want.dtype, what);
            got.shape.Should().Equal(want.shape, what);
            got.astype(np.float64).ToArray<double>().Should().Equal(want.astype(np.float64).ToArray<double>(), what);
        }

        /// <summary>
        ///     Every C/S pattern of (x, lo, hi) over a 2-D (rows, cols) iteration — a full array (C), a row (C along the
        ///     inner axis, broadcast along the outer one), a column (S: one value per row) and a 0-d scalar (S) — through
        ///     the 2-D block kernel's stages, plain, into an out=, and under a per-element and a per-row where= mask.
        /// </summary>
        [TestMethod]
        public void TwoDimensional_EveryBroadcastPattern_MatchesMaterializedOperands()
        {
            var rnd = new System.Random(7);
            string[] forms = { "C", "R", "S", "K" };
            foreach (var dt in SimdDtypes)
            foreach (long cols in new long[] { 1, 3, 4, 5, 8, 9, 17, 33 })
            {
                const long rows = 3;
                long[] full = { rows, cols };
                long[] Shape(string f) => f switch { "C" => full, "R" => new[] { cols }, "S" => Array.Empty<long>(), _ => new[] { rows, 1L } };
                foreach (var fx in forms)
                foreach (var flo in forms)
                foreach (var fhi in forms)
                {
                    var x = Data(rnd, dt, Shape(fx), 1, 99, 0.5);
                    var lo = Data(rnd, dt, Shape(flo), 20, 20, 0.25);
                    var hi = Data(rnd, dt, Shape(fhi), 60, 20, 0.25);
                    string tag = $"{dt.name} cols={cols} x={fx} lo={flo} hi={fhi}";
                    long[] b = BroadcastShape(Shape(fx), Shape(flo), Shape(fhi));

                    SameResult(np.clip(x, lo, hi), np.clip(Full(x, b), Full(lo, b), Full(hi, b)), tag + " plain");

                    var o1 = np.empty(new NumSharp.Shape(b), dt);
                    var o2 = np.empty(new NumSharp.Shape(b), dt);
                    np.clip(x, lo, hi, o1);
                    np.clip(Full(x, b), Full(lo, b), Full(hi, b), o2);
                    SameResult(o1, o2, tag + " out");

                    long count = b.Aggregate(1L, (p, q) => p * q);
                    var mask = np.array(Enumerable.Range(0, (int)count).Select(i => i % 3 != 1).ToArray()).reshape(b);
                    var w1 = np.full(new NumSharp.Shape(b), 7, dt);
                    var w2 = np.full(new NumSharp.Shape(b), 7, dt);
                    np.clip(x, lo, hi, w1, where: mask);
                    np.clip(Full(x, b), Full(lo, b), Full(hi, b), w2, where: mask);
                    SameResult(w1, w2, tag + " where");

                    if (b.Length == 2 && b[1] == cols)
                    {
                        var rowMask = np.array(new[] { true, false, true }).reshape(rows, 1);
                        var r1 = np.full(new NumSharp.Shape(b), 7, dt);
                        var r2 = np.full(new NumSharp.Shape(b), 7, dt);
                        np.clip(x, lo, hi, r1, where: rowMask);
                        np.clip(Full(x, b), Full(lo, b), Full(hi, b), r2, where: rowMask);
                        SameResult(r1, r2, tag + " row where");
                    }
                }
            }
        }

        /// <summary>
        ///     The 1-D shell's stages: an out= of another dtype buffers the iteration, so the 0-d bounds reach the kernel as
        ///     stride-0 operands next to the contiguous input — every C/S pattern of a 1-D call, across the unroll /
        ///     remainder / tail boundaries — and an out overlapping the input takes the copy route.
        /// </summary>
        [TestMethod]
        public void OneDimensional_BufferedAndOverlapCalls_MatchMaterializedOperands()
        {
            var rnd = new System.Random(11);
            foreach (var dt in SimdDtypes)
            foreach (long n in new long[] { 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 65 })
            foreach (var fx in new[] { "C", "S" })
            foreach (var flo in new[] { "C", "S" })
            foreach (var fhi in new[] { "C", "S" })
            {
                if (fx == "S" && flo == "S" && fhi == "S")
                    continue;
                long[] full = { n };
                long[] Shape(string f) => f == "C" ? full : Array.Empty<long>();
                var x = Data(rnd, dt, Shape(fx), 1, 99, 0.5);
                var lo = Data(rnd, dt, Shape(flo), 20, 20, 0.25);
                var hi = Data(rnd, dt, Shape(fhi), 60, 20, 0.25);
                string tag = $"{dt.name} n={n} x={fx} lo={flo} hi={fhi}";

                var castDt = dt == np.float64 ? np.float32 : np.float64;
                var c1 = np.empty(new NumSharp.Shape(full), castDt);
                var c2 = np.empty(new NumSharp.Shape(full), castDt);
                np.clip(x, lo, hi, c1);
                np.clip(Full(x, full), Full(lo, full), Full(hi, full), c2);
                SameResult(c1, c2, tag + " out-cast");
            }

            // Overlap: out is the input shifted by one element, so the iterator copies before the kernel runs.
            foreach (var dt in SimdDtypes)
            {
                var buf = Data(rnd, dt, new long[] { 40 }, 1, 99, 0.5);
                var lo = Data(rnd, dt, Array.Empty<long>(), 30, 1, 0.25);
                var hi = Data(rnd, dt, Array.Empty<long>(), 70, 1, 0.25);
                var want = np.clip(buf["0:39"].copy(), lo, hi);
                np.clip(buf["0:39"], lo, hi, buf["1:40"]);
                SameResult(buf["1:40"].copy(), want, $"{dt.name} overlap");
            }
        }

        /// <summary>
        ///     The gather route's broadcast stages: a strided (::2) or reversed (::-2) input next to 0-d inputs — clip and the
        ///     binary ufuncs add / maximum, into a contiguous and into a strided (scattered) out.
        /// </summary>
        [TestMethod]
        public void Gather_StridedInputWithBroadcastInputs_MatchesMaterializedOperands()
        {
            var rnd = new System.Random(13);
            NDArray Form(string f, DType dt, long n, int lo, int span, double frac) => f switch
            {
                "C" => Data(rnd, dt, new[] { n }, lo, span, frac),
                "G" => Data(rnd, dt, new[] { 2 * n }, lo, span, frac)["::2"],
                "R" => Data(rnd, dt, new[] { 2 * n }, lo, span, frac)["::-2"],
                _ => Data(rnd, dt, Array.Empty<long>(), lo, span, frac),
            };
            string[] forms = { "C", "G", "R", "S" };
            foreach (var dt in new[] { np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64 })
            foreach (long n in new long[] { 1, 3, 8, 9, 33, 130 })
            {
                long[] full = { n };
                foreach (var fx in forms)
                foreach (var flo in forms)
                foreach (var fhi in forms)
                {
                    if (!(fx is "G" or "R") && !(flo is "G" or "R") && !(fhi is "G" or "R"))
                        continue;
                    var x = Form(fx, dt, n, 1, 99, 0.5);
                    var lo = Form(flo, dt, n, 20, 20, 0.25);
                    var hi = Form(fhi, dt, n, 60, 20, 0.25);
                    string tag = $"clip {dt.name} n={n} x={fx} lo={flo} hi={fhi}";
                    var want = np.clip(Full(x, full), Full(lo, full), Full(hi, full));
                    SameResult(np.clip(x, lo, hi), want, tag);
                    var buf = np.zeros(new Shape(2 * n), dt);
                    np.clip(x, lo, hi, buf["::2"]);
                    SameResult(buf["::2"].copy(), want, tag + " out-strided");
                }

                foreach (var fa in forms)
                foreach (var fb in forms)
                {
                    if (!(fa is "G" or "R") && !(fb is "G" or "R"))
                        continue;
                    var a = Form(fa, dt, n, 1, 50, 0.5);
                    var b = Form(fb, dt, n, 1, 50, 0.25);
                    var want = np.add(Full(a, full), Full(b, full));
                    var o1 = np.zeros(new Shape(n), dt);
                    np.add(a, b, o1);
                    SameResult(o1, want, $"add {dt.name} n={n} a={fa} b={fb}");
                    var wantMax = np.maximum(Full(a, full), Full(b, full));
                    var buf = np.zeros(new Shape(2 * n), dt);
                    np.maximum(a, b, buf["::2"]);
                    SameResult(buf["::2"].copy(), wantMax, $"maximum {dt.name} n={n} a={fa} b={fb} out-strided");
                }
            }
        }

        /// <summary>
        ///     The flat clip kernel's output-alignment peel: every output alignment (views at element offsets 0..7) and
        ///     sizes around the peel bound give the values of an independent route (maximum then minimum).
        /// </summary>
        [TestMethod]
        public void FlatKernel_EveryOutputAlignment_MatchesMaximumThenMinimum()
        {
            var rnd = new System.Random(17);
            foreach (var dt in SimdDtypes)
            foreach (long n in new long[] { 1, 2, 3, 5, 8, 13, 31, 32, 33, 100 })
            foreach (int offset in Enumerable.Range(0, 8))
            {
                var x = Data(rnd, dt, new[] { n }, 1, 99, 0.5);
                var lo = Data(rnd, dt, Array.Empty<long>(), 30, 1, 0.25);
                var hi = Data(rnd, dt, Array.Empty<long>(), 70, 1, 0.25);
                var buf = np.zeros(new Shape(n + 8), dt);
                var o = buf[$"{offset}:{offset + n}"];
                np.clip(x, lo, hi, o);
                SameResult(o.copy(), np.minimum(np.maximum(x, lo), hi), $"{dt.name} n={n} offset={offset}");
            }
        }

        /// <summary>
        ///     The flat clip kernel's streaming-store variant (forced on through the test hook) writes the same bytes as the
        ///     plain variant — two scalar bounds, array bounds, one-sided clips, misaligned outputs and the sizes around
        ///     the peel and unroll boundaries.
        /// </summary>
        [TestMethod]
        public void FlatKernel_StreamingStores_MatchPlainStores()
        {
            var rnd = new System.Random(19);
            try
            {
                foreach (var dt in SimdDtypes)
                foreach (long n in new long[] { 1, 3, 7, 16, 17, 64, 65, 1000 })
                foreach (int offset in new[] { 0, 1, 3 })
                {
                    var x = Data(rnd, dt, new[] { n }, 1, 99, 0.5);
                    var loS = Data(rnd, dt, Array.Empty<long>(), 30, 1, 0.25);
                    var hiS = Data(rnd, dt, Array.Empty<long>(), 70, 1, 0.25);
                    var loA = Data(rnd, dt, new[] { n }, 20, 20, 0.25);
                    var hiA = Data(rnd, dt, new[] { n }, 60, 20, 0.25);
                    foreach (var (lo, hi, what) in new (NDArray, NDArray, string)[]
                             { (loS, hiS, "scalar"), (loA, hiA, "array"), (loS, null, "min-only"), (null, hiS, "max-only") })
                    {
                        NDArray Run(bool streaming)
                        {
                            DirectILKernelGenerator.ClipStreamingOverride = streaming;
                            var buf = np.zeros(new Shape(n + 4), dt);
                            var o = buf[$"{offset}:{offset + n}"];
                            np.clip(x, lo, hi, o);
                            return o.copy();
                        }
                        SameResult(Run(true), Run(false), $"{dt.name} n={n} offset={offset} {what}");
                    }
                }
            }
            finally
            {
                DirectILKernelGenerator.ClipStreamingOverride = null;
            }
        }

        /// <summary>
        ///     NDIter's broadcast of operands whose shapes differ: the direct right-aligned pass gives the broadcast shape,
        ///     and a mismatch still raises the established exception and message (it falls through to the original route).
        /// </summary>
        [TestMethod]
        public void NDIter_BroadcastOfDifferingShapes()
        {
            long[] ShapeOf(params NDArray[] ops)
            {
                var flags = Enumerable.Repeat(NDIterPerOpFlags.READONLY, ops.Length).ToArray();
                using var it = NDIterRef.MultiNew(ops.Length, ops, NDIterGlobalFlags.MULTI_INDEX, NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING, flags, null);
                return it.Shape.ToArray();
            }

            ShapeOf(np.zeros(new Shape(3, 1)), np.zeros(new Shape(4))).Should().Equal(3, 4);
            ShapeOf(np.array(1.0).reshape(Array.Empty<long>()), np.zeros(new Shape(2, 3)), np.zeros(new Shape(3))).Should().Equal(2, 3);
            ShapeOf(np.zeros(new Shape(2, 0)), np.zeros(new Shape(1))).Should().Equal(2, 0);

            Action mismatch = () => ShapeOf(np.zeros(new Shape(3)), np.zeros(new Shape(4)));
            mismatch.Should().Throw<IncorrectShapeException>().WithMessage("shape mismatch: objects cannot be broadcast to a single shape");
            Action emptyMismatch = () => ShapeOf(np.zeros(new Shape(2, 0)), np.zeros(new Shape(3)));
            emptyMismatch.Should().Throw<IncorrectShapeException>().WithMessage("shape mismatch: objects cannot be broadcast to a single shape");
        }
    }
}
