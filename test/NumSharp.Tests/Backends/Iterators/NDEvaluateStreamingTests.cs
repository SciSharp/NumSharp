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
        public void Axis_TransposedInput_FallsBackToMaterialize()
        {
            var a = Pool(new long[] { 30, 40 }, NPTypeCode.Double, 21).T;
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)a * a, 0), expectStream: false);
            AssertSameBothWays(() => NDExpr.Sum((NDExpr)a * a, 1), expectStream: false);
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
