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

        /// <summary>
        /// Plan lever 3: the streamed flat <c>Any</c> / <c>All</c> / <c>CountNonzero</c> must equal the scalar fold
        /// kernel byte for byte at every deciding-element position (first, last, block boundaries, none), across
        /// C / F / offset layouts, bool-array leaves and 0-d parameters — and must decline strided / broadcast operands.
        /// </summary>
        [TestMethod]
        public void BoolFolds_AnyAllCountNonzero_StreamAndMatchTheFoldKernel()
        {
            // Plan lever 3: flat Any / All / CountNonzero over a bool child stream through the child's SIMD kernel +
            // vectorized block scans. OR / AND / an integer count are order-independent, so the streamed answer must
            // equal the scalar fold kernel's for every position of the deciding element — first, last, block
            // boundaries (8192 bools per scratch block), none, all — and every shared contiguous layout.
            foreach (long n in new long[] { 1, 7, 8191, 8192, 8193, 20000, 100003 })
            foreach (long hot in new long[] { -1, 0, n / 2, n - 1, 8191, 8192 })
            {
                if (hot >= n) continue;
                // x has exactly one element > 0 at `hot` (none when hot == -1); y is x's complement pattern.
                var x = np.zeros(new Shape(n), NPTypeCode.Double) - 1.0;
                if (hot >= 0) x[hot] = 2.0;
                var y = np.zeros(new Shape(n), NPTypeCode.Double) + 1.0;
                if (hot >= 0) y[hot] = -3.0;
                var xs = x; var ys = y;
                AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)xs > (NDExpr)NDArray.Scalar(0.0)), expectStream: true);
                AssertSameBothWaysExact(() => NDExpr.All((NDExpr)ys > (NDExpr)NDArray.Scalar(0.0)), expectStream: true);
                AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)xs > (NDExpr)ys), expectStream: true);
            }

            // All-true / all-false, F and offset layouts, and a many-true count across blocks.
            var m = (np.arange(3 * 5001).astype(NPTypeCode.Double) % 7 - 3.0).reshape(3, 5001);
            var mf = np.asfortranarray(m);
            var big = np.arange(30000).astype(NPTypeCode.Int32) % 5;
            var off = big["123:29000"];
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)m), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)mf), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)off), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)off), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)off), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)off + 1), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)off * 0), expectStream: true);

            // Declines: a strided operand, a broadcast pair.
            var st = big["::3"];
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)st), expectStream: false);
            var row = np.arange(5001).astype(NPTypeCode.Double);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)m > (NDExpr)row), expectStream: false);

            // A bare bool ARRAY leaf: folded in place (no kernel, no scratch copy), across what would be many blocks.
            var flags = (np.arange(20000) % 3) == 0;
            var noFlags = np.zeros(new Shape(20000), NPTypeCode.Boolean);
            var allFlags = np.ones(new Shape(20000), NPTypeCode.Boolean);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)flags), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)noFlags), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)flags), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)allFlags), expectStream: true);

            // The in-place scan must start at the view's logical element 0. A contiguous slice re-seats Address (offset
            // 0), but an F column block keeps a NON-zero offset: columns 0..6 are all True and columns 3993..3999 all
            // False, so a scan that ignored the offset would read 42 extra Trues in place of 42 Falses.
            var bm = np.zeros(new Shape(6, 4000), NPTypeCode.Boolean);
            bm[":, :7"] = true;
            bm[":, 7:3993"] = ((np.arange(6 * 3986) % 5) == 0).reshape(6, 3986);
            var bfo = np.asfortranarray(bm)[":, 7:4000"];
            Assert.IsTrue(bfo.Shape.IsFContiguous && bfo.Shape.offset == 42, "precondition: F-contiguous at offset 42");
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)bfo), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)np.asfortranarray(np.ones(new Shape(3, 50), NPTypeCode.Boolean))[":, 2:50"]),
                expectStream: true);
            Assert.AreEqual(np.count_nonzero(bfo), np.evaluate(NDExpr.CountNonzero((NDExpr)bfo)).GetAtIndex<long>(0),
                "the streamed count agrees with np.count_nonzero");
        }

        /// <summary>
        /// A flat reduction over a TRANSPOSED dense block (neither C- nor F-contiguous in logical order) must reduce in
        /// MEMORY order: NumPy's K-order iterator coalesces a dense block into one inner loop over memory (probed on
        /// 2.4.2: 0 / 400 misses for memory order, 231 / 400 for the logical C order the old materialize route used).
        /// So the transposed view's flat Sum / Mean / Prod / Min / Max — as a bare leaf and as a fused product of two
        /// identically-permuted operands — must carry the same bits as the same reduction over the C-contiguous base.
        /// </summary>
        [TestMethod]
        public void Flat_PermutedDenseOperands_ReduceInMemoryOrder()
        {
            var rng = new System.Random(4242);
            NDArray Wide(NPTypeCode tc)
            {
                var d = new double[5 * 6 * 7];
                for (int i = 0; i < d.Length; i++) d[i] = (rng.NextDouble() - 0.5) * System.Math.Pow(10, rng.Next(-6, 7));
                var a = np.array(d).reshape(5, 6, 7);
                return tc == NPTypeCode.Double ? a : a.astype(tc);
            }

            NDArray Benign(NPTypeCode tc)
            {
                var d = new double[5 * 6 * 7];
                for (int i = 0; i < d.Length; i++) d[i] = 0.75 + 0.5 * rng.NextDouble();
                var a = np.array(d).reshape(5, 6, 7);
                return tc == NPTypeCode.Double ? a : a.astype(tc);
            }

            long Bits(NDArray r) => r.typecode == NPTypeCode.Double
                ? BitConverter.DoubleToInt64Bits(r.GetAtIndex<double>(0))
                : BitConverter.SingleToInt32Bits(r.GetAtIndex<float>(0));

            int logicalOrderDiffers = 0;
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single })
            foreach (var perm in new[] { new[] { 2, 0, 1 }, new[] { 1, 0, 2 }, new[] { 0, 2, 1 }, new[] { 1, 2, 0 } })
            {
                var b1 = Wide(tc);
                var b2 = Wide(tc);
                var bp = Benign(tc);
                var t1 = b1.transpose(perm);
                var t2 = b2.transpose(perm);
                var tp = bp.transpose(perm);
                Assert.IsFalse(t1.Shape.IsContiguous || t1.Shape.IsFContiguous, "precondition: a permuted, non-C/F view");

                foreach (var (name, make, baseMake) in new (string, Func<NDExpr>, Func<NDExpr>)[]
                {
                    ("sum leaf", () => NDExpr.Sum((NDExpr)t1), () => NDExpr.Sum((NDExpr)b1)),
                    ("mean leaf", () => NDExpr.Mean((NDExpr)t1), () => NDExpr.Mean((NDExpr)b1)),
                    ("sum product", () => NDExpr.Sum((NDExpr)t1 * (NDExpr)t2), () => NDExpr.Sum((NDExpr)b1 * (NDExpr)b2)),
                    ("prod leaf", () => NDExpr.Prod((NDExpr)tp), () => NDExpr.Prod((NDExpr)bp)),
                    ("max product", () => NDExpr.Max((NDExpr)t1 * (NDExpr)t2), () => NDExpr.Max((NDExpr)b1 * (NDExpr)b2)),
                    ("min leaf", () => NDExpr.Min((NDExpr)t1), () => NDExpr.Min((NDExpr)b1)),
                })
                {
                    NDExpr.StreamingReductions = 0;
                    NDExpr.ExactMinMaxRuns = 0;
                    long got = Bits(np.evaluate(make()));
                    Assert.IsTrue(NDExpr.StreamingReductions > 0 || NDExpr.ExactMinMaxRuns > 0,
                        $"{tc} {string.Join(",", perm)} {name}: the permuted dense block must take the memory-order route");
                    Assert.AreEqual(Bits(np.evaluate(baseMake())), got, $"{tc} ({string.Join(",", perm)}) {name}: memory order");
                }

                // Teeth: the logical-order sum (what the materialize route computed) must differ somewhere.
                if (Bits(np.evaluate(NDExpr.Sum((NDExpr)np.ascontiguousarray(t1)))) != Bits(np.evaluate(NDExpr.Sum((NDExpr)t1))))
                    logicalOrderDiffers++;
            }

            Assert.IsTrue(logicalOrderDiffers > 0, "the pools must make summation order observable, or the test has no teeth");
        }

        /// <summary>
        /// A flat <c>Mean</c> of an INTEGER child is NumPy's <c>add.reduce(child, dtype=float64)</c>: a buffered cast,
        /// so a pairwise sum per <c>np.getbufsize()</c> chunk added from +0.0, then the divide. The chunk length is
        /// observable — the literal bits below were probed on NumPy 2.4.2 for splitmix64 uint64 data at the default
        /// 8192 AND after <c>np.setbufsize(4096)</c> (n = 20001 / 50001 differ between the two; n = 20000 does not) —
        /// and every route (streamed, materialized) must reproduce both.
        /// </summary>
        [TestMethod]
        public void IntegerMean_ChunksLikeNumPysBufferedCast()
        {
            NDArray SplitMix(int n)
            {
                var v = new ulong[n];
                for (int i = 0; i < n; i++)
                {
                    ulong z = unchecked((ulong)i * 0x9E3779B97F4A7C15UL + 0x9E3779B97F4A7C15UL);
                    z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
                    z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
                    v[i] = z ^ (z >> 31);
                }

                return np.array(v);
            }

            long MeanBits(NDArray a) => BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Mean((NDExpr)a)).GetAtIndex<double>(0));

            foreach (var (n, at8192, at4096) in new[]
                     {
                         (20001, 0x43dfd95523477bdeL, 0x43dfd95523477bdcL),
                         (50001, 0x43dff538dac8aaf0L, 0x43dff538dac8aaefL),
                         (20000, 0x43dfd9901e043787L, 0x43dfd9901e043787L),
                     })
            {
                var a = SplitMix(n);
                NDExpr.StreamingReductions = 0;
                Assert.AreEqual(at8192, MeanBits(a), $"n={n} default bufsize (streamed)");
                Assert.IsTrue(NDExpr.StreamingReductions > 0, "a contiguous integer child streams the float64 cast");

                try
                {
                    NDExpr.DisableStreamingReduce = true;
                    Assert.AreEqual(at8192, MeanBits(a), $"n={n} default bufsize (materialized)");
                }
                finally
                {
                    NDExpr.DisableStreamingReduce = false;
                }

                long old = np.setbufsize(4096);
                try
                {
                    Assert.AreEqual(at4096, MeanBits(a), $"n={n} bufsize 4096 — the chunking follows np.getbufsize()");
                }
                finally
                {
                    np.setbufsize(old);
                }
            }

            // Bool / narrow integers take the same route; their sums are exact, so the value is the plain mean.
            var flags = (np.arange(30001) % 3) == 0;
            Assert.AreEqual(10001.0 / 30001.0, np.evaluate(NDExpr.Mean((NDExpr)flags)).GetAtIndex<double>(0));
            var i8 = (np.arange(1000) % 256 - 128).astype(NPTypeCode.SByte);
            Assert.AreEqual((double)np.mean(i8), np.evaluate(NDExpr.Mean((NDExpr)i8)).GetAtIndex<double>(0));
        }

        /// <summary>
        /// <c>NDExprProgram.NonzeroBoolOperandProgram</c> — the substitution that lets the bool fold stream <c>x</c>'s
        /// SIMD kernel instead of the scalar int64-typed <c>x != 0</c> — must resolve ONLY when the reduction child is
        /// the factories' nonzero test over a Boolean <c>x</c>; a numeric <c>x</c>, a different child or a
        /// non-reduction must stay null so no genuine comparison is ever rewritten.
        /// </summary>
        [TestMethod]
        public void BoolFolds_NonzeroOperand_ResolvesOnlyForABoolTestedOperand()
        {
            // The factories build `x != 0`; for a bool x the fold streams x itself (its SIMD kernel) — the pattern must
            // resolve for exactly that shape and stay null everywhere the child is not the identity of x.
            var a = np.arange(64).astype(NPTypeCode.Double);
            var b = np.arange(64).astype(NPTypeCode.Double) % 5;
            var flags = (np.arange(64) % 3) == 0;

            var cmp = NDExpr.CountNonzero((NDExpr)a > (NDExpr)b).GetProgram(null).NonzeroBoolOperandProgram;
            Assert.IsNotNull(cmp, "count_nonzero(a > b): a > b is bool, so it is the streamed operand");
            Assert.AreEqual(NPTypeCode.Boolean, cmp.ResultType);
            Assert.IsNotNull(NDExpr.Any((NDExpr)flags).GetProgram(null).NonzeroBoolOperandProgram, "any(bool array)");
            Assert.IsNotNull(NDExpr.All((NDExpr)a < 3.0).GetProgram(null).NonzeroBoolOperandProgram, "all(a < 3.0)");

            // A numeric x keeps the child `x != 0` (x's values are not bools); a non-reduction or a non-pattern child
            // has nothing to substitute.
            Assert.IsNull(NDExpr.CountNonzero((NDExpr)a).GetProgram(null).NonzeroBoolOperandProgram, "count_nonzero(f64)");
            Assert.IsNull(NDExpr.Any((NDExpr)a * 2.0).GetProgram(null).NonzeroBoolOperandProgram, "any(a * 2.0)");
            Assert.IsNull(NDExpr.Sum((NDExpr)a > (NDExpr)b).GetProgram(null).NonzeroBoolOperandProgram, "sum(a > b) has no != 0 test");
            Assert.IsNull(((NDExpr)a > (NDExpr)b).GetProgram(null).NonzeroBoolOperandProgram, "not a reduction");

            // A hand-spelled test against a DIFFERENT literal is a genuine comparison, not the identity: substituting
            // `flags` for `flags != 1` would count the complement. Both the resolution and the value must hold.
            // (NDExpr defines no ==/!= operators — reference equality stays intact — so the tests use the factory.)
            Assert.IsNull(NDExpr.Sum(NDExpr.NotEqual((NDExpr)flags, NDExpr.Const(1))).GetProgram(null).NonzeroBoolOperandProgram,
                "sum(flags != 1)");
            Assert.IsNull(NDExpr.Sum(NDExpr.NotEqual(NDExpr.Const(0), (NDExpr)flags)).GetProgram(null).NonzeroBoolOperandProgram,
                "sum(0 != flags) — the literal on the left is not the factories' spelling");
            Assert.IsNull(NDExpr.Sum(NDExpr.NotEqual((NDExpr)flags, NDExpr.Const(0.0))).GetProgram(null).NonzeroBoolOperandProgram,
                "sum(flags != 0.0) — only the integer literal the factories emit is recognized");
            Assert.IsNotNull(NDExpr.Sum(NDExpr.NotEqual((NDExpr)flags, NDExpr.Const(0))).GetProgram(null).NonzeroBoolOperandProgram,
                "sum(flags != 0) IS the pattern");
            var big = (np.arange(20000) % 3) == 0;
            AssertSameBothWaysExact(() => NDExpr.Sum(NDExpr.NotEqual((NDExpr)big, NDExpr.Const(1))), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.Sum(NDExpr.NotEqual((NDExpr)big, NDExpr.Const(0))), expectStream: true);
        }

        /// <summary>
        /// A deterministic sparse float64 pool: each element is 1.0 with probability 1/<paramref name="oneIn"/>, else 0.0 —
        /// so an axis Any over it mixes True and False outputs (and All over its complement), which a dense pool would not.
        /// </summary>
        /// <param name="shape">The pool's shape.</param>
        /// <param name="seed">The generator seed.</param>
        /// <param name="oneIn">The inverse density of the 1.0 elements.</param>
        /// <returns>A fresh C-contiguous float64 array.</returns>
        private static NDArray SparsePool(long[] shape, int seed, int oneIn)
        {
            var rng = new System.Random(seed);
            long n = 1;
            foreach (var d in shape) n *= d;
            var v = new double[n];
            for (long i = 0; i < n; i++) v[i] = rng.Next(oneIn) == 0 ? 1.0 : 0.0;
            return np.array(v).reshape(shape);
        }

        /// <summary>
        /// Plan lever 3, AXIS form: the streamed axis <c>Any</c> / <c>All</c> / <c>CountNonzero</c> / <c>Sum(bool)</c> must
        /// equal the seeded axis fold byte for byte across the row walk (reduced axis contiguous — short rows many per
        /// block and rows longer than a block), the slab walk (outer axis — short slabs many per block, slabs longer than a
        /// block, counts across more than 255 slabs and exactly at the byte-counter flush boundary), every axis of a 3-D
        /// block, C and F walks, computed children (with a 0-d parameter) and bare bool leaves (read in place), keepdims
        /// and out= — and must decline strided, broadcast, mixed-order and empty reductions.
        /// </summary>
        [TestMethod]
        public void AxisBoolFolds_AnyAllCount_StreamAndMatchTheAxisFold()
        {
            var shapes = new[]
            {
                new long[] { 700 }, new long[] { 20000 },                        // 1-D reduce-all: one short / one long row
                new long[] { 1000, 100 }, new long[] { 5, 20000 },               // slabs (inner 100) / long rows (> 8192)
                new long[] { 300, 9000 },                                        // slabs longer than a block (9000 > 8192)
                new long[] { 255, 3 }, new long[] { 256, 3 }, new long[] { 511, 2 }, // counts at the 255-slab flush edges
                new long[] { 4, 5, 67 }, new long[] { 4, 300, 7 },               // every axis of a 3-D block
            };
            var half = NDArray.Scalar(0.5);   // 0-d → a hoisted parameter of the computed child
            foreach (var shape in shapes)
            {
                int nd = shape.Length;
                var a = SparsePool(shape, 1000 + (int)shape[0], oneIn: 97);
                var b = SparsePool(shape, 2000 + (int)shape[0], oneIn: 5);
                var mask = a > 0.5;                                            // a bare bool leaf
                var fa = np.asfortranarray(a);
                var fb = np.asfortranarray(b);
                var fmask = np.asfortranarray(mask);
                for (int axis = 0; axis < nd; axis++)
                {
                    int ax = axis;
                    // Computed children (C and F walks), the 0-d parameter included.
                    AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)a > (NDExpr)half, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.All((NDExpr)a < (NDExpr)half, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)a > (NDExpr)b, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)fa > (NDExpr)fb, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.All((NDExpr)fa <= (NDExpr)fb, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)fb, ax), expectStream: true);
                    // Bare bool leaves, read in place: Any / All / CountNonzero and a plain Sum of the bools.
                    AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)mask, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.All((NDExpr)fmask, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)mask, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.Sum((NDExpr)fmask, ax), expectStream: true);
                }
            }

            // DENSE counts: a byte counter holds 255 additions, so an all-true column longer than that is what exposes a
            // late flush (the sparse pools above never put 256 Trues between two flushes). 255 / 256 / 1000 slabs, and the
            // longer-than-a-block slab walk (inner 9000), as a computed child and as a bool leaf.
            foreach (var shape in new[] { new long[] { 255, 5 }, new long[] { 256, 5 }, new long[] { 1000, 3 }, new long[] { 600, 9000 } })
            {
                var ones = np.ones(new Shape(shape), NPTypeCode.Double);
                var onesMask = np.ones(new Shape(shape), NPTypeCode.Boolean);
                var fOnes = np.asfortranarray(np.ones(new Shape(shape[1], shape[0]), NPTypeCode.Double));
                AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)ones, 0), expectStream: true);
                AssertSameBothWaysExact(() => NDExpr.Sum((NDExpr)onesMask, 0), expectStream: true);
                AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)fOnes, 1), expectStream: true);   // F walk: slabs
                Assert.AreEqual(shape[0], np.evaluate(NDExpr.CountNonzero((NDExpr)ones, 0)).GetAtIndex<long>(shape[1] - 1),
                    $"{shape[0]}x{shape[1]}: every column counts all {shape[0]} Trues");
            }

            // NumSharp's own unfused reductions agree (an independent cross-check of the fold both sides above share).
            var m2 = SparsePool(new long[] { 1000, 100 }, 77, oneIn: 400) > 0.5;
            for (int axis = 0; axis < 2; axis++)
            {
                Assert.IsTrue(np.array_equal(np.any(m2, axis), np.evaluate(NDExpr.Any((NDExpr)m2, axis))), $"any axis {axis}");
                Assert.IsTrue(np.array_equal(np.all(!m2, axis), np.evaluate(NDExpr.All(!(NDExpr)m2, axis))), $"all axis {axis}");
                Assert.IsTrue(np.array_equal(np.count_nonzero(m2, axis), np.evaluate(NDExpr.CountNonzero((NDExpr)m2, axis))),
                    $"count_nonzero axis {axis}");
            }
        }

        /// <summary>
        /// The deciding element of an axis <c>Any</c> / <c>All</c> at every position that matters to the stream — the first
        /// and last element of a row, both sides of a scratch-block boundary (8192 bools) of a long row, a row with none —
        /// and the slab walk's early exit, both when every output decides early (the walk stops) and when one output never
        /// decides (the walk must run to the end), so a premature stop or a missed element shows as a wrong output.
        /// </summary>
        [TestMethod]
        public void AxisBoolFolds_DecidingElementAtEveryPosition_AndTheSlabEarlyExit()
        {
            // Long rows (reduced axis contiguous, 20000 > the 8192-bool block): one hot element per row, at a different
            // position in each row; row 0 has none.
            long len = 20000;
            long[] hot = { -1, 0, 1, 8191, 8192, 8193, 16383, 16384, len / 2, len - 1 };
            var x = np.zeros(new Shape(hot.Length, len), NPTypeCode.Double);
            for (int r = 0; r < hot.Length; r++)
                if (hot[r] >= 0) x[r, (int)hot[r]] = 1.0;
            var xs = x;
            var xt = np.asfortranarray(x.T.copy()).T;   // the same values, reached through an F walk of the transpose
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)xs, 1), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)xs < 0.5, 1), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)xs, 1), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)xt, 1), expectStream: true);
            var anyRows = np.evaluate(NDExpr.Any((NDExpr)xs, 1));
            for (int r = 0; r < hot.Length; r++)
                Assert.AreEqual(hot[r] >= 0, anyRows.GetAtIndex<bool>(r), $"row {r} (hot at {hot[r]})");

            // Slabs (outer axis 0 reduced, 3000 x 50): every column decided within the first rows → the walk stops early.
            var early = np.zeros(new Shape(3000, 50), NPTypeCode.Double);
            for (int c = 0; c < 50; c++) early[(c * 7) % 90, c] = 1.0;
            var es = early;
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)es, 0), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)es < 0.5, 0), expectStream: true);
            Assert.IsTrue(np.evaluate(NDExpr.Any((NDExpr)es, 0)).GetAtIndex<bool>(49), "column 49 decided at row 73");

            // …and the same walk with ONE column that never decides (its hot element is on the very last row): stopping
            // before the end would report column 13 False.
            var late = early.copy();
            late[":, 13"] = 0.0;
            late[2999, 13] = 1.0;
            var ls = late;
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)ls, 0), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)ls < 0.5, 0), expectStream: true);
            Assert.IsTrue(np.evaluate(NDExpr.Any((NDExpr)ls, 0)).GetAtIndex<bool>(13), "column 13's True is on the last row");
            Assert.IsFalse(np.evaluate(NDExpr.All((NDExpr)ls < 0.5, 0)).GetAtIndex<bool>(13), "…so column 13 is not all-below");

            // Slabs longer than a block (inner 9000): column 8999 decides only on the last slab.
            var wide = np.zeros(new Shape(40, 9000), NPTypeCode.Double);
            wide[":, :8999"] = 1.0;
            wide[39, 8999] = 1.0;
            var ws = wide;
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)ws, 0), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)ws, 0), expectStream: true);
            Assert.IsTrue(np.evaluate(NDExpr.Any((NDExpr)ws, 0)).GetAtIndex<bool>(8999), "the last slab decides column 8999");
        }

        /// <summary>
        /// The streamed axis bool fold's result plumbing: keepdims reshapes it, <c>out=</c> receives it (the same instance
        /// returned), an F walk's F-contiguous result reads back in logical order, a bool leaf reached through an F column
        /// block at a NON-zero offset is scanned from its logical element 0 — and strided, broadcast, mixed-order and
        /// empty reductions decline to the fold.
        /// </summary>
        [TestMethod]
        public void AxisBoolFolds_KeepdimsOutOffsetAndDeclines()
        {
            var a = SparsePool(new long[] { 60, 40 }, 5, oneIn: 23);
            var b = SparsePool(new long[] { 60, 40 }, 6, oneIn: 3);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)a > (NDExpr)b, 0, keepdims: true), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)a, 1, keepdims: true), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)a <= (NDExpr)b, -1, keepdims: true), expectStream: true);

            // out=: the streamed result is copied into the caller's array, which is what np.evaluate returns.
            var dst = np.full(new Shape(40), -7L);
            var want = np.count_nonzero(a, 0);
            NDExpr.StreamingReductions = 0;
            var got = np.evaluate(NDExpr.CountNonzero((NDExpr)a, 0), @out: dst);
            Assert.IsTrue(NDExpr.StreamingReductions > 0, "the count streams");
            Assert.AreSame(dst, got, "out= is returned");
            Assert.IsTrue(np.array_equal(want, dst), "out= holds the counts");

            // An F walk of a 3-D block: the F-contiguous result reads back in logical order.
            var c3 = SparsePool(new long[] { 6, 7, 8 }, 9, oneIn: 11);
            var f3 = np.asfortranarray(c3);
            for (int axis = 0; axis < 3; axis++)
            {
                int ax = axis;
                AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)f3, ax), expectStream: true);
                Assert.IsTrue(np.array_equal(np.count_nonzero(c3, ax), np.evaluate(NDExpr.CountNonzero((NDExpr)f3, ax))),
                    $"F count axis {ax} equals the C count");
            }

            // A bool leaf in an F column block at a NON-zero offset: columns 0..6 of the base are all True and are NOT
            // part of the view, so a scan that ignored the offset would report them.
            var bm = np.zeros(new Shape(6, 400), NPTypeCode.Boolean);
            bm[":, :7"] = true;
            bm[":, 7:393"] = ((np.arange(6 * 386) % 5) == 0).reshape(6, 386);
            var bfo = np.asfortranarray(bm)[":, 7:400"];
            Assert.IsTrue(bfo.Shape.IsFContiguous && bfo.Shape.offset == 42, "precondition: F-contiguous at offset 42");
            for (int axis = 0; axis < 2; axis++)
            {
                int ax = axis;
                AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)bfo, ax), expectStream: true);
                AssertSameBothWaysExact(() => NDExpr.All((NDExpr)bfo, ax), expectStream: true);
                Assert.IsTrue(np.array_equal(np.count_nonzero(bfo, ax), np.evaluate(NDExpr.CountNonzero((NDExpr)bfo, ax))),
                    $"offset-view count axis {ax}");
            }

            // Declines: a strided operand, a broadcast pair, mixed C / F, an empty reduced axis, an empty output.
            var st = SparsePool(new long[] { 60, 80 }, 12, oneIn: 9)[":, ::2"];
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)st, 0), expectStream: false);
            var row = SparsePool(new long[] { 40 }, 13, oneIn: 2);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)a > (NDExpr)row, 0), expectStream: false);
            var fb = np.asfortranarray(b);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)a > (NDExpr)fb, 1), expectStream: false);
            var e0 = np.zeros(new Shape(0, 5), NPTypeCode.Double);
            AssertSameBothWaysExact(() => NDExpr.All((NDExpr)e0, 0), expectStream: false);
            AssertSameBothWaysExact(() => NDExpr.CountNonzero((NDExpr)e0, 0), expectStream: false);
            var e1 = np.zeros(new Shape(5, 0), NPTypeCode.Double);
            AssertSameBothWaysExact(() => NDExpr.Any((NDExpr)e1, 0), expectStream: false);
        }

        /// <summary>
        /// The streamed fused ArgMax / ArgMin (<c>DefaultEngine.Evaluate.ArgStream.cs</c>) over COMPUTED children of every
        /// served dtype family: flat (one long row), rows (the reduced axis innermost — many short rows per produce block,
        /// and rows longer than a block folded chunk by chunk) and slabs (an outer axis — narrow slabs many rows per block,
        /// and slabs wider than a 1024-lane chunk), every axis of 2-D / 3-D blocks, keepdims both ways — byte-identical to
        /// the materialize route it replaces, and to the engine argmax of the materialized child.
        /// </summary>
        [TestMethod]
        public void ArgReduce_ComputedChildren_StreamAndMatchMaterialize()
        {
            var shapes = new[]
            {
                new long[] { 700 }, new long[] { 5000 },                          // one short row / a row of five f64 blocks
                new long[] { 1000, 100 }, new long[] { 5, 5000 },                 // many rows per block / rows longer than one
                new long[] { 300, 3000 },                                          // slabs wider than a lane chunk
                new long[] { 4, 5, 67 }, new long[] { 4, 300, 7 },                // every axis of a 3-D block
            };
            var half = NDArray.Scalar(0.5);   // 0-d → a hoisted parameter of the computed child
            foreach (var shape in shapes)
            {
                int nd = shape.Length;
                var a = Pool(shape, NPTypeCode.Double, 40 + (int)shape[0]);                 // NaN / ±inf / -0 laced
                var b = Pool(shape, NPTypeCode.Double, 50 + (int)shape[0], benign: true);   // specials-free
                var bf = b.astype(NPTypeCode.Single);
                var bi = (b * 5).astype(NPTypeCode.Int32);                                  // 2..7: many ties
                var bl = (b * 5).astype(NPTypeCode.Int64);
                var bu = (b * 5).astype(NPTypeCode.Byte);
                var children = new (string, Func<NDExpr>)[]
                {
                    ("f64 NaN-laced", () => (NDExpr)a * (NDExpr)b),
                    ("f64 benign", () => (NDExpr)b - (NDExpr)half),
                    ("f32", () => (NDExpr)bf * (NDExpr)bf),
                    ("i32 ties", () => (NDExpr)bi - 3),
                    ("i64 ties", () => (NDExpr)bl * (NDExpr)bl),
                    ("u8 wrap", () => (NDExpr)bu + (NDExpr)bu),
                    ("bool", () => (NDExpr)b > (NDExpr)half),
                };
                foreach (var (name, child) in children)
                {
                    var c = child;
                    AssertSameBothWaysExact(() => NDExpr.ArgMax(c()), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.ArgMin(c()), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.ArgMax(c(), keepdims: true), expectStream: true);
                    for (int axis = -nd; axis < nd; axis++)
                    {
                        int ax = axis;
                        AssertSameBothWaysExact(() => NDExpr.ArgMax(c(), ax), expectStream: true);
                        AssertSameBothWaysExact(() => NDExpr.ArgMin(c(), ax, keepdims: ax >= 0), expectStream: true);
                    }

                    // An independent cross-check: the engine argmax over the materialized child.
                    using var materialized = np.evaluate(c());
                    for (int axis = 0; axis < nd; axis++)
                        Assert.IsTrue(np.array_equal(np.argmax(materialized, axis), np.evaluate(NDExpr.ArgMax(c(), axis))),
                            $"{name} {string.Join("x", shape)} axis {axis}");
                    Assert.AreEqual(np.argmin(materialized), np.evaluate(NDExpr.ArgMin(c())).GetAtIndex<long>(0), $"{name} flat");
                }
            }
        }

        /// <summary>
        /// A row longer than the produce block is folded chunk by chunk: the chunk winners must combine to the GLOBAL first
        /// occurrence (a tie in a later chunk never replaces an earlier one), and the fold must stop at — and only at — a
        /// deciding value (the first NaN for argmax AND argmin, an integer extreme, the first True / False), wherever
        /// it falls, including a row with no deciding value at all.
        /// </summary>
        [TestMethod]
        public void ArgReduce_LongRowChunks_KeepTheFirstOccurrence()
        {
            // The flat argmax/argmin must stream, equal the materialize route byte for byte, and be the pinned index.
            void AssertIndex(Func<NDExpr> make, long want, string what)
            {
                AssertSameBothWaysExact(make, expectStream: true);
                Assert.AreEqual(want, np.evaluate(make()).GetAtIndex<long>(0), what);
            }

            // One row of 5000 doubles = five 1024-element blocks.
            var x = np.zeros(new Shape(5000), NPTypeCode.Double);
            x[1000] = 9.0;
            x[2100] = 9.0;
            AssertIndex(() => NDExpr.ArgMax((NDExpr)x * 1.0), 1000, "a tie across chunks keeps chunk 0's");
            AssertIndex(() => NDExpr.ArgMin((NDExpr)x * -1.0), 1000, "…for argmin too");
            x[3000] = double.NaN;
            x[4000] = double.NaN;
            AssertIndex(() => NDExpr.ArgMax((NDExpr)x * 1.0), 3000, "the first NaN wins argmax");
            AssertIndex(() => NDExpr.ArgMin((NDExpr)x * 1.0), 3000, "…and argmin");
            x[5] = double.NaN;
            AssertIndex(() => NDExpr.ArgMax((NDExpr)x * 1.0), 5, "a NaN in chunk 0 decides the row");

            // int64: the extreme in chunk 2 and chunk 3 (2048-element blocks at 8 bytes = 1024) → chunk 2's.
            var l = np.zeros(new Shape(5000), NPTypeCode.Int64);
            l[2500] = long.MaxValue;
            l[3500] = long.MaxValue;
            AssertIndex(() => NDExpr.ArgMax((NDExpr)l + 0), 2500, "int64 extreme, first chunk holding it");
            AssertIndex(() => NDExpr.ArgMin(-(NDExpr)l), 2500, "int64 negated extreme");

            // Bool rows of 20000 = three 8192-element blocks: the first True only in the last block, none at all, and
            // argmin's first False.
            var m = np.zeros(new Shape(3, 20000), NPTypeCode.Double);
            m[0, 16384] = 1.0;
            m[0, 19999] = 1.0;
            m["2, :"] = 1.0;
            m[2, 9000] = 0.0;
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)m > 0.5, 1), expectStream: true);
            AssertSameBothWaysExact(() => NDExpr.ArgMin((NDExpr)m > 0.5, 1), expectStream: true);
            CollectionAssert.AreEqual(new long[] { 16384, 0, 0 }, np.evaluate(NDExpr.ArgMax((NDExpr)m > 0.5, 1)).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 0, 0, 9000 }, np.evaluate(NDExpr.ArgMin((NDExpr)m > 0.5, 1)).ToArray<long>());
            AssertIndex(() => NDExpr.ArgMax((NDExpr)m["1, :"] > 0.5), 0, "no True at all → 0");

            // A slab whose lanes decide on different rows, one never: the lane-chunk walk may stop only when all have.
            var s = np.zeros(new Shape(40, 3000), NPTypeCode.Double);
            for (int c = 0; c < 3000; c += 7) s[(c * 13) % 40, c] = 1.0;
            s[":, 2999"] = 0.0;
            s[39, 2999] = 1.0;
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)s > 0.5, 0), expectStream: true);
            Assert.AreEqual(39L, np.evaluate(NDExpr.ArgMax((NDExpr)s > 0.5, 0)).GetAtIndex<long>(2999));
        }

        /// <summary>
        /// The route choice and the result plumbing: a BARE leaf streams in every layout (the engine's fast fold reads it in
        /// place); a computed child over F / strided / broadcast / mixed-order operands, a Half / Decimal child and an
        /// empty child keep the materialize route (same answer, same exception); <c>out=</c> — int64 or a same-kind float64
        /// — receives the streamed result and is what comes back; NumPy's pinned values hold.
        /// </summary>
        [TestMethod]
        public void ArgReduce_LeavesDeclinesOutAndPinnedValues()
        {
            var c = Pool(new long[] { 40, 30 }, NPTypeCode.Double, 3);
            var f = np.asfortranarray(c);
            var leaves = new[] { c, f, c[":, ::-1"], c["::2, :"], np.broadcast_to(c["0, :"], new Shape(40, 30)) };
            foreach (var leaf in leaves)
            {
                var lv = leaf;
                AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)lv), expectStream: true);
                for (int axis = 0; axis < 2; axis++)
                {
                    int ax = axis;
                    AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)lv, ax), expectStream: true);
                    AssertSameBothWaysExact(() => NDExpr.ArgMin((NDExpr)lv, ax, keepdims: true), expectStream: true);
                }
            }

            // Declines: the child's memory order is not its logical C order, or an operand broadcasts.
            var row = Pool(new long[] { 30 }, NPTypeCode.Double, 4);
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)f * 2.0, 0), expectStream: false);
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)c[":, ::2"] * 2.0, 1), expectStream: false);
            AssertSameBothWaysExact(() => NDExpr.ArgMin((NDExpr)c + (NDExpr)f, 1), expectStream: false);
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)c + (NDExpr)row, 0), expectStream: false);
            // Dtypes the fold does not serve.
            var h = Pool(new long[] { 40, 30 }, NPTypeCode.Double, 5, benign: true).astype(NPTypeCode.Half);
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)h + (NDExpr)h, 1), expectStream: false);
            var dm = Pool(new long[] { 40, 30 }, NPTypeCode.Double, 6, benign: true).astype(NPTypeCode.Decimal);
            AssertSameBothWaysExact(() => NDExpr.ArgMin((NDExpr)dm * 2, 0), expectStream: false);
            // Empty: NumPy's error from the materialize route, whichever route was tried.
            var e0 = np.zeros(new Shape(0, 5), NPTypeCode.Double);
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.ArgMax((NDExpr)e0 * 2.0)));
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.ArgMax((NDExpr)e0 * 2.0, 0)));
            AssertSameBothWaysExact(() => NDExpr.ArgMax((NDExpr)e0 * 2.0, 1), expectStream: false);

            // out=: the streamed result lands in the caller's array, which is what comes back.
            var benign = Pool(new long[] { 40, 30 }, NPTypeCode.Double, 7, benign: true);
            var dst = np.full(new Shape(30), -7L);
            NDExpr.StreamingReductions = 0;
            var got = np.evaluate(NDExpr.ArgMax((NDExpr)benign * 2.0, 0), @out: dst);
            Assert.IsTrue(NDExpr.StreamingReductions > 0, "the argmax streams");
            Assert.AreSame(dst, got, "out= is returned");
            CollectionAssert.AreEqual(np.argmax(benign * 2.0, 0).ToArray<long>(), dst.ToArray<long>());
            var dstf = np.zeros(new Shape(40), NPTypeCode.Double);
            np.evaluate(NDExpr.ArgMin((NDExpr)benign * 2.0, 1), @out: dstf);
            CollectionAssert.AreEqual(np.argmin(benign * 2.0, 1).astype(NPTypeCode.Double).ToArray<double>(), dstf.ToArray<double>());

            // NumPy 2.4.2: argmax/argmin(m, 0) = [1 2 0 2] and axis 1 = [2 0 1] — the first NaN wins either way.
            var m = np.array(new double[,] { { 1, 5, double.NaN, 2 }, { double.NaN, 7, 1, 2 }, { 3, double.NaN, 0, double.NaN } });
            CollectionAssert.AreEqual(new long[] { 1, 2, 0, 2 }, np.evaluate(NDExpr.ArgMax((NDExpr)m * 1.0, 0)).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 2, 0, 2 }, np.evaluate(NDExpr.ArgMin((NDExpr)m * 1.0, 0)).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 2, 0, 1 }, np.evaluate(NDExpr.ArgMax((NDExpr)m * 1.0, 1)).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 2, 0, 1 }, np.evaluate(NDExpr.ArgMin((NDExpr)m * 1.0, 1)).ToArray<long>());
        }

        /// <summary>
        /// A child that throws part-way through the stream (a <c>Call</c> node) raises the same exception on both routes —
        /// the stream never turns a failure into a fallback, and the half-written result is released rather than returned.
        /// </summary>
        [TestMethod]
        public void ArgReduce_ThrowingChild_PropagatesTheSameException()
        {
            var x = np.arange(6000).astype(NPTypeCode.Double).reshape(3, 2000);
            Func<double, double> boom = v => v > 4500 ? throw new InvalidOperationException("boom " + v) : v;
            foreach (int? axis in new int?[] { null, 0, 1 })
            {
                foreach (bool disable in new[] { false, true })
                {
                    NDExpr.DisableStreamingReduce = disable;
                    try
                    {
                        var ex = Assert.ThrowsException<InvalidOperationException>(() => np.evaluate(axis is int ax
                            ? NDExpr.ArgMax(NDExpr.Call(boom, (NDExpr)x), ax)
                            : NDExpr.ArgMax(NDExpr.Call(boom, (NDExpr)x))));
                        Assert.AreEqual("boom 4501", ex.Message, $"axis {axis} streaming {!disable}");
                    }
                    finally
                    {
                        NDExpr.DisableStreamingReduce = false;
                    }
                }
            }
        }

        /// <summary>
        /// <see cref="AssertSameBothWays"/> for results of ANY dtype (bool / int64 here): evaluate with streaming off
        /// and on, assert engagement, and compare dtype, shape and raw bytes.
        /// </summary>
        /// <param name="make">Builds the tree to evaluate.</param>
        /// <param name="expectStream">Whether the streaming path must have engaged.</param>
        private static void AssertSameBothWaysExact(Func<NDExpr> make, bool expectStream)
        {
            NDArray want;
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
            var got = np.evaluate(make());
            Assert.AreEqual(want.typecode, got.typecode, "dtype");
            CollectionAssert.AreEqual(want.shape, got.shape, "shape");
            CollectionAssert.AreEqual(np.ascontiguousarray(want).Unsafe.ReadOnlyBytes().ToArray(),
                np.ascontiguousarray(got).Unsafe.ReadOnlyBytes().ToArray(), "raw bytes");
            Assert.AreEqual(expectStream, NDExpr.StreamingReductions > 0, "streaming engagement");
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
