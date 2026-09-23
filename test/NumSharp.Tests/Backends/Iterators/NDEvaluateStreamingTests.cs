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
