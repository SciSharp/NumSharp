using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// Plan P2 M3 — the streaming (no-temp) exact reductions (DefaultEngine.Evaluate.Stream.cs). The contract is
    /// that streaming is byte-identical to the M1/M2 materialize-then-reduce path it replaces for contiguous
    /// inputs (every non-NaN bit, the sign of zero included; a NaN stays NaN — which of two DISTINCT NaN payloads
    /// survives is not contractual, as the oracle tokenizes NaN, because the JIT may commute a float add), that it
    /// actually ENGAGES where it applies, and that every other layout falls back to the materialize path.
    /// </summary>
    [TestClass]
    public class NDEvaluateStreamingTests
    {
        /// <summary>Evaluate <paramref name="make"/> with streaming on and off; assert equality and whether it streamed.</summary>
        /// <param name="make">Builds the tree to evaluate (rebuilt per call — the structural cache serves both).</param>
        /// <param name="expectStream">Whether the streaming path must have engaged.</param>
        private static void AssertSameBothWays(Func<NDExpr> make, bool expectStream)
        {
            NDArray want, got;
            try
            {
                NDExpr.DisableStreamingReduce = true;
                want = np.evaluate(make());
            }
            finally
            {
                NDExpr.DisableStreamingReduce = false;
            }

            NDExpr.StreamingReductions = 0;
            got = np.evaluate(make());
            Assert.AreEqual(expectStream, NDExpr.StreamingReductions > 0, "streaming engagement");
            AssertBitsEqualNaNTokenized(want, got);
        }

        /// <summary>Raw-bit equality of two float/complex arrays with NaN tokenized (any NaN equals any NaN).</summary>
        /// <param name="want">The materialize-path result.</param>
        /// <param name="got">The streaming-path result.</param>
        private static void AssertBitsEqualNaNTokenized(NDArray want, NDArray got)
        {
            Assert.AreEqual(want.typecode, got.typecode);
            CollectionAssert.AreEqual(want.shape, got.shape);
            var w = want.flatten();
            var g = got.flatten();
            for (long i = 0; i < w.size; i++)
            {
                switch (w.typecode)
                {
                    case NPTypeCode.Double:
                        AssertDouble(w.GetAtIndex<double>(i), g.GetAtIndex<double>(i), i);
                        break;
                    case NPTypeCode.Single:
                        float p = w.GetAtIndex<float>(i), q = g.GetAtIndex<float>(i);
                        Assert.IsTrue((float.IsNaN(p) && float.IsNaN(q)) || BitConverter.SingleToInt32Bits(p) == BitConverter.SingleToInt32Bits(q),
                            $"element {i}: {p:R} vs {q:R}");
                        break;
                    case NPTypeCode.Complex:
                        var cw = w.GetAtIndex<Complex>(i);
                        var cg = g.GetAtIndex<Complex>(i);
                        AssertDouble(cw.Real, cg.Real, i);
                        AssertDouble(cw.Imaginary, cg.Imaginary, i);
                        break;
                    default:
                        Assert.Fail($"unexpected dtype {w.typecode}");
                        break;
                }
            }
        }

        private static void AssertDouble(double p, double q, long i)
            => Assert.IsTrue((double.IsNaN(p) && double.IsNaN(q)) || BitConverter.DoubleToInt64Bits(p) == BitConverter.DoubleToInt64Bits(q),
                $"element {i}: {p:R} vs {q:R}");

        /// <summary>A deterministic wide-magnitude pool (so pairwise and any other order genuinely differ) with specials sprinkled in.</summary>
        private static NDArray Pool(long[] shape, NPTypeCode tc, int seed, bool benign = false)
        {
            var rng = new System.Random(seed);
            long n = 1;
            foreach (var d in shape) n *= d;
            double Next()
            {
                if (benign) return 0.5 + rng.NextDouble();
                return rng.Next(50) switch
                {
                    0 => -0.0,
                    1 => double.NaN,
                    2 => double.PositiveInfinity,
                    3 => 5e-324 * rng.Next(1, 100),
                    _ => (rng.NextDouble() - 0.5) * System.Math.Pow(10, rng.Next(-10, 11)),
                };
            }

            if (tc == NPTypeCode.Complex)
            {
                var c = new Complex[n];
                for (long i = 0; i < n; i++) c[i] = new Complex(Next(), Next());
                return np.array(c).reshape(shape);
            }

            var d64 = new double[n];
            for (long i = 0; i < n; i++) d64[i] = Next();
            var a = np.array(d64).reshape(shape);
            return tc == NPTypeCode.Double ? a : a.astype(tc);
        }

        [TestMethod]
        public void FlatSumMean_StreamsAndMatchesMaterialize_AcrossPairwiseBoundaries()
        {
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Complex })
            foreach (long n in new long[] { 1, 7, 8, 127, 128, 129, 513, 1023, 1024, 1025, 2049, 4097, 8193, 100003 })
            {
                var a = Pool(new[] { n }, tc, (int)n);
                var b = Pool(new[] { n }, tc, (int)n + 1);
                AssertSameBothWays(() => NDExpr.Sum((NDExpr)a * b), expectStream: true);
                AssertSameBothWays(() => NDExpr.Mean((NDExpr)a + (NDExpr)b * a), expectStream: true);
            }
        }

        [TestMethod]
        public void FlatProd_StreamsAndMatchesMaterialize()
        {
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single })
            foreach (long n in new long[] { 1, 9, 1500, 5000 })
            {
                var a = Pool(new[] { n }, tc, 7 * (int)n, benign: true);
                AssertSameBothWays(() => NDExpr.Prod((NDExpr)a * a), expectStream: true);
            }
        }

        [TestMethod]
        public void Flat_FContiguousOffsetSliceAndParameter_Stream_MixedAndStrided_FallBack()
        {
            var c = Pool(new long[] { 37, 129 }, NPTypeCode.Double, 11);
            var f = np.asfortranarray(Pool(new long[] { 37, 129 }, NPTypeCode.Double, 12));
            var f2 = np.asfortranarray(Pool(new long[] { 37, 129 }, NPTypeCode.Double, 13));
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)f * f2), expectStream: true);    // all-F: memory order
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)c * f), expectStream: false);    // mixed order → materialize

            var big = Pool(new long[] { 6000 }, NPTypeCode.Double, 14);
            var s1 = big["17:4017"];
            var s2 = big["900:4900"];
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)s1 * s2), expectStream: true);  // contiguous views at offsets
            var st = big["::2"];
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)st * st), expectStream: false);  // strided → materialize

            var k = NDArray.Scalar(1.25);                                                  // 0-d → hoisted parameter
            AssertSameBothWays(() => NDExpr.Mean((NDExpr)c * k), expectStream: true);
        }

        [TestMethod]
        public void Axis_PinnedRowsAndSlabs_StreamAndMatchMaterialize()
        {
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Complex })
            foreach (var shape in new[]
                     {
                         new long[] { 9 }, new long[] { 7, 1 }, new long[] { 1, 7 }, new long[] { 4, 1500 },
                         new long[] { 5000, 3 }, new long[] { 3, 5000 }, new long[] { 2, 3, 700 }, new long[] { 40, 3, 70 },
                     })
            {
                var a = Pool(shape, tc, shape.Length * 100 + (int)shape[0]);
                var b = Pool(shape, tc, shape.Length * 100 + (int)shape[0] + 1);
                for (int axis = 0; axis < shape.Length; axis++)
                {
                    int ax = axis;
                    AssertSameBothWays(() => NDExpr.Sum((NDExpr)a * b, ax), expectStream: true);
                    AssertSameBothWays(() => NDExpr.Mean((NDExpr)a * b, ax - shape.Length), expectStream: true);
                    if (tc != NPTypeCode.Complex)
                    {
                        var p = Pool(shape, tc, 5 + ax, benign: true);
                        AssertSameBothWays(() => NDExpr.Prod((NDExpr)p * p, ax), expectStream: true);
                    }
                }
            }
        }

        [TestMethod]
        public void Axis_FContiguousInputs_StreamAsTheTransposedCReduction()
        {
            // A transposed C array is F-contiguous, and NumPy reduces the F-contiguous child it materializes in
            // MEMORY order — i.e. reducing axis k of the F view is reducing axis nd-1-k of the underlying C data. The
            // F-aware stream must therefore equal the C stream of the transposed problem, transposed back, bit for
            // bit (the NumPy .npy oracle pinned both on the session probe: 80/80 strict-F Sum/Mean/Prod cases). The
            // old route here was the 4-accumulator fold (E1-excused), so there is no materialize twin to diff against.
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Complex })
            foreach (var shape in new[] { new long[] { 30, 40 }, new long[] { 4, 5, 67 }, new long[] { 9, 1, 33 } })
            {
                var c1 = Pool(shape, tc, shape.Length * 31 + (int)shape[0]);
                var c2 = Pool(shape, tc, shape.Length * 31 + (int)shape[0] + 1);
                var f1 = c1.T;   // F-contiguous views of the same memory
                var f2 = c2.T;
                int nd = shape.Length;
                for (int k = 0; k < nd; k++)
                {
                    NDExpr.StreamingReductions = 0;
                    var fSum = np.evaluate(NDExpr.Sum((NDExpr)f1 * f2, k));
                    Assert.IsTrue(NDExpr.StreamingReductions > 0, $"{tc} {string.Join("x", shape)} axis {k}: the F sum must stream");
                    var cSum = np.evaluate(NDExpr.Sum((NDExpr)c1 * c2, nd - 1 - k));
                    AssertBitsEqualNaNTokenized(cSum.T, fSum);

                    var fMean = np.evaluate(NDExpr.Mean((NDExpr)f1 * f2, k));
                    var cMean = np.evaluate(NDExpr.Mean((NDExpr)c1 * c2, nd - 1 - k));
                    AssertBitsEqualNaNTokenized(cMean.T, fMean);

                    if (tc != NPTypeCode.Complex)
                    {
                        var p = Pool(shape, tc, 77 + k, benign: true);
                        var fProd = np.evaluate(NDExpr.Prod((NDExpr)p.T * p.T, k));
                        var cProd = np.evaluate(NDExpr.Prod((NDExpr)p * p, nd - 1 - k));
                        AssertBitsEqualNaNTokenized(cProd.T, fProd);
                    }
                }
            }
        }

        [TestMethod]
        public void Axis_MixedOrderInputs_FallBack()
        {
            // One C and one F operand share no memory order, so neither stream applies: the materialize route runs.
            var c = Pool(new long[] { 30, 40 }, NPTypeCode.Double, 21);
            var f = np.asfortranarray(Pool(new long[] { 30, 40 }, NPTypeCode.Double, 22));
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)c * f, 0), expectStream: false);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)c * f, 1), expectStream: false);
        }

        [TestMethod]
        public void WeightedAverage_CContiguous_StreamsAndMatchesMaterialize()
        {
            // Σ(v·w) / Σ(w) with both sums streamed (no product / weights temporaries) must equal the materialize
            // route byte for byte over C-contiguous operands — flat, every axis, keepdims — incl. an int·int average
            // (multiplied at float64, never wrapping) and a fused values child.
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Complex })
            foreach (var shape in new[] { new long[] { 1 }, new long[] { 129 }, new long[] { 37, 129 }, new long[] { 4, 5, 67 } })
            {
                var v = Pool(shape, tc, 3 + shape.Length);
                var w = Pool(shape, tc, 4 + shape.Length, benign: true);
                AssertSameBothWays(() => NDExpr.Average((NDExpr)v, (NDExpr)w), expectStream: true);
                AssertSameBothWays(() => NDExpr.Average((NDExpr)v * v, (NDExpr)w, keepdims: true), expectStream: true);
                for (int ax = 0; ax < shape.Length; ax++)
                {
                    int k = ax;
                    AssertSameBothWays(() => NDExpr.Average((NDExpr)v, (NDExpr)w, k), expectStream: true);
                }
            }

            var iv = np.arange(1000).astype(NPTypeCode.Int32).reshape(10, 100) * 100000;
            var iw = (np.arange(1000).astype(NPTypeCode.Int32) % 7 + 1).reshape(10, 100) * 30000;
            AssertSameBothWays(() => NDExpr.Average((NDExpr)iv, (NDExpr)iw), expectStream: true);
            AssertSameBothWays(() => NDExpr.Average((NDExpr)iv, (NDExpr)iw, 1), expectStream: true);
        }

        [TestMethod]
        public void WeightedAverage_FContiguous_EqualsTheTransposedCAverage()
        {
            // NumPy's np.average multiplies in K order (an F product for F inputs) and sums in MEMORY order, so a
            // flat F average is the flat sum over the F memory walk and an axis-k F average is the axis-(nd-1-k)
            // average of the underlying C data. The streams must reproduce exactly that; the old materialize route
            // copied F inputs to C and summed in C order (NumPy-divergent — 6 of 9 F cases on the .npy oracle).
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Complex })
            foreach (var shape in new[] { new long[] { 37, 129 }, new long[] { 4, 5, 67 } })
            {
                var v = Pool(shape, tc, 50 + shape.Length);
                var w = Pool(shape, tc, 60 + shape.Length, benign: true);
                int nd = shape.Length;

                // Flat: the F memory walk of v.T is the C memory walk of v, so the flat F average equals the flat C
                // average over the SAME data (NumPy: np.average(a.T, weights=w.T) == np.average(a, weights=w)).
                NDExpr.StreamingReductions = 0;
                var fFlat = np.evaluate(NDExpr.Average((NDExpr)v.T, (NDExpr)w.T));
                Assert.IsTrue(NDExpr.StreamingReductions > 0, "the flat F average must stream");
                AssertBitsEqualNaNTokenized(np.evaluate(NDExpr.Average((NDExpr)v, (NDExpr)w)), fFlat);

                for (int k = 0; k < nd; k++)
                {
                    NDExpr.StreamingReductions = 0;
                    var fAx = np.evaluate(NDExpr.Average((NDExpr)v.T, (NDExpr)w.T, k));
                    Assert.IsTrue(NDExpr.StreamingReductions > 0, $"the F axis-{k} average must stream");
                    var cAx = np.evaluate(NDExpr.Average((NDExpr)v, (NDExpr)w, nd - 1 - k));
                    AssertBitsEqualNaNTokenized(cAx.T, fAx);
                }
            }
        }

        [TestMethod]
        public void NegativeZeroSignContract_MatchesMaterialize()
        {
            // An all -0.0 child: the flat/PINNED sum is 0 + (-0.0 fold) = +0.0 (NumPy's +0.0); the engine's
            // size-1-axis reduction COPIES (-0.0) — streaming reproduces both behaviors of the materialize path.
            var z = np.full(new Shape(4, 3), -0.0);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)z * 1.0), expectStream: true);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)z * 1.0, 0), expectStream: true);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)z * 1.0, 1), expectStream: true);
            var z1 = np.full(new Shape(4, 1), -0.0);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)z1 * 1.0, 1), expectStream: true);
            Assert.IsTrue(BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Sum((NDExpr)z * 1.0)).GetAtIndex<double>(0)) == 0L,
                "flat sum of -0.0 is +0.0 (NumPy parity)");
        }
    }
}
