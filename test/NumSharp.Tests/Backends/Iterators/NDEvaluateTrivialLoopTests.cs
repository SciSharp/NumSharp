using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// The trivial-loop dispatch of np.evaluate's elementwise pass (DefaultEngine.Evaluate.Trivial.cs — NumPy's
    /// try_trivial_single_output_loop). The contract: a trivially iterable call (identical dims, one shared
    /// contiguous order or 1-D, 0-d inputs by a zero stride, no cast, a provided out overlapping inputs only
    /// exactly) runs the fused kernel once with no iterator and produces the SAME bytes, dtype, shape and
    /// memory layout as the NDIter pass; every other call declines and runs the NDIter pass unchanged. Each
    /// case evaluates the call both ways (<see cref="NDExpr.DisableTrivialLoop"/>) and asserts whether the
    /// shortcut engaged (<see cref="NDExpr.TrivialLoopRuns"/>), so a silently dead fast path fails too.
    /// </summary>
    [TestClass]
    public class NDEvaluateTrivialLoopTests
    {
        /// <summary>
        /// Run <paramref name="call"/> with the trivial loop disabled and enabled; assert identical results and
        /// whether the shortcut served the second run.
        /// </summary>
        /// <param name="call">Builds its operands FRESH (deterministically) and evaluates — rebuilt per run because
        /// an out= / aliasing call mutates its arrays. Returns the array to compare (the result, or the base an
        /// in-place call wrote through).</param>
        /// <param name="expectTrivial">Whether the trivial loop must have served the enabled run.</param>
        private static void AssertSameBothWays(Func<NDArray> call, bool expectTrivial)
        {
            NDArray want;
            try
            {
                NDExpr.DisableTrivialLoop = true;
                want = call();
            }
            finally
            {
                NDExpr.DisableTrivialLoop = false;
            }

            NDExpr.TrivialLoopRuns = 0;
            var got = call();
            int runs = NDExpr.TrivialLoopRuns;
            // Values first: a wrong answer is reported as one even when the engagement expectation is also off.
            AssertIdentical(want, got);
            Assert.AreEqual(expectTrivial, runs > 0, "trivial-loop engagement");
        }

        /// <summary>Same dtype, shape, C/F contiguity flags and raw bytes (logical C order).</summary>
        /// <param name="want">The NDIter-pass result.</param>
        /// <param name="got">The trivial-loop result.</param>
        private static void AssertIdentical(NDArray want, NDArray got)
        {
            Assert.AreEqual(want.typecode, got.typecode, "dtype");
            CollectionAssert.AreEqual(want.shape, got.shape, "shape");
            Assert.AreEqual(want.Shape.IsContiguous, got.Shape.IsContiguous, "C-contiguity");
            Assert.AreEqual(want.Shape.IsFContiguous, got.Shape.IsFContiguous, "F-contiguity");
            if (want.size == 0)
                return;
            var w = np.ascontiguousarray(want).Unsafe.ReadOnlyBytes().ToArray();
            var g = np.ascontiguousarray(got).Unsafe.ReadOnlyBytes().ToArray();
            CollectionAssert.AreEqual(w, g, "raw bytes");
        }

        /// <summary>A deterministic value pool of <paramref name="tc"/> with <paramref name="shape"/> (C-contiguous).</summary>
        /// <param name="shape">The dims.</param>
        /// <param name="tc">The dtype.</param>
        /// <param name="seed">The value seed.</param>
        private static NDArray Pool(long[] shape, NPTypeCode tc, int seed)
        {
            var rng = new System.Random(seed);
            long n = 1;
            foreach (var d in shape) n *= d;
            if (tc == NPTypeCode.Complex)
            {
                var c = new Complex[n];
                for (long i = 0; i < n; i++) c[i] = new Complex(rng.Next(-50, 50) * 0.25, rng.Next(-50, 50) * 0.5);
                return np.array(c).reshape(shape);
            }

            var d64 = new double[n];
            for (long i = 0; i < n; i++)
                d64[i] = tc == NPTypeCode.Boolean ? rng.Next(2) : rng.Next(-60, 60) * 0.75;
            var a = np.array(d64).reshape(shape);
            return tc == NPTypeCode.Double ? a : a.astype(tc);
        }

        [TestMethod]
        public void Contiguous_COrF_EveryRank_EngagesAndMatchesIterator()
        {
            foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32, NPTypeCode.Int64,
                         NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Boolean })
            foreach (var shape in new[]
                     {
                         new long[] { 1 }, new long[] { 37 }, new long[] { 3, 4 }, new long[] { 1, 9 }, new long[] { 9, 1 },
                         new long[] { 2, 3, 5 }, new long[] { 4, 1, 6 },
                     })
            {
                int seed = (int)tc * 100 + shape.Length * 10 + (int)shape[0];
                var sh = shape;
                // C-contiguous operands → a fresh C result.
                AssertSameBothWays(() =>
                {
                    var a = Pool(sh, tc, seed);
                    var b = Pool(sh, tc, seed + 1);
                    return np.evaluate((NDExpr)a * b + a);
                }, expectTrivial: true);
                // F-contiguous operands → a fresh F result (the strict-F heuristic), still one flat loop.
                AssertSameBothWays(() =>
                {
                    var a = np.asfortranarray(Pool(sh, tc, seed + 2));
                    var b = np.asfortranarray(Pool(sh, tc, seed + 3));
                    return np.evaluate((NDExpr)a * b + b);
                }, expectTrivial: true);
            }
        }

        [TestMethod]
        public void TransposedAndOffsetViews_ShareOneOrder_Engage()
        {
            // A transposed C-contiguous 2-D array IS F-contiguous: two of them share the F order.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 5, 7 }, NPTypeCode.Double, 1).T;
                var b = Pool(new long[] { 5, 7 }, NPTypeCode.Double, 2).T;
                return np.evaluate((NDExpr)a - b * a);
            }, expectTrivial: true);
            // Contiguous row blocks at a non-zero element offset.
            AssertSameBothWays(() =>
            {
                var big = Pool(new long[] { 10, 6 }, NPTypeCode.Single, 3);
                var x = big["2:5"];
                var y = big["6:9"];
                return np.evaluate((NDExpr)x * y + 1.5f);
            }, expectTrivial: true);
        }

        [TestMethod]
        public void OneDim_AnyStride_Engages()
        {
            Func<NDArray> Make(Func<NDArray, NDArray> view0, Func<NDArray, NDArray> view1) => () =>
            {
                var big = Pool(new long[] { 60 }, NPTypeCode.Double, 7);
                return np.evaluate((NDExpr)view0(big) * view1(big) + view0(big));
            };

            AssertSameBothWays(Make(x => x["::2"], x => x["1::2"]), expectTrivial: true);           // positive strides
            AssertSameBothWays(Make(x => x["::-2"], x => x["0:30"]), expectTrivial: true);          // a negative stride
            AssertSameBothWays(Make(x => x["5:35"],                                                   // a 1-D stride-0 broadcast input
                x => np.broadcast_to(x["3:4"], new Shape(30))), expectTrivial: true);
        }

        [TestMethod]
        public void ParametersAndZeroDOperands_Engage()
        {
            // A 0-d array beside an N-d operand is a hoisted parameter (aux block).
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 3, 4 }, NPTypeCode.Double, 11);
                return np.evaluate((NDExpr)a * NDArray.Scalar(2.5) + a);
            }, expectTrivial: true);
            // Every input 0-d: nothing is hoisted, the operands ride the kernel at a zero stride, 0-d result.
            AssertSameBothWays(() => np.evaluate((NDExpr)NDArray.Scalar(1.5) * NDArray.Scalar(-3.0) + NDArray.Scalar(0.25)),
                expectTrivial: true);
            // A positional compiled handle hoists nothing, so its 0-d operand rides the kernel at a zero stride.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 2, 9 }, NPTypeCode.Double, 12);
                var h = (NDExpr.Input(0) * NDExpr.Input(1) + NDExpr.Input(0)).Compile(NPTypeCode.Double, NPTypeCode.Double);
                return h.Evaluate(new[] { a, NDArray.Scalar(4.0) });
            }, expectTrivial: true);
        }

        [TestMethod]
        public void NotTriviallyIterable_Declines_AndStillMatches()
        {
            // A genuine broadcast between N-d operands.
            AssertSameBothWays(() =>
            {
                var m = Pool(new long[] { 3, 4 }, NPTypeCode.Double, 21);
                var row = Pool(new long[] { 4 }, NPTypeCode.Double, 22);
                return np.evaluate((NDExpr)m * row + m);
            }, expectTrivial: false);
            // A strided N-d operand.
            AssertSameBothWays(() =>
            {
                var m = Pool(new long[] { 4, 8 }, NPTypeCode.Double, 23);
                var s = m[":, ::2"];
                return np.evaluate((NDExpr)s * s + 1.0);
            }, expectTrivial: false);
            // C- and F-only operands disagree on the order.
            AssertSameBothWays(() =>
            {
                var c = Pool(new long[] { 3, 5 }, NPTypeCode.Double, 24);
                var f = np.asfortranarray(Pool(new long[] { 3, 5 }, NPTypeCode.Double, 25));
                return np.evaluate((NDExpr)c * f);
            }, expectTrivial: false);
            // A dtype= request (the output becomes a buffered cast operand).
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 3, 4 }, NPTypeCode.Double, 26);
                return np.evaluate((NDExpr)a * a, dtype: np.float32);
            }, expectTrivial: false);
            // A where= mask (the masked driver).
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 12 }, NPTypeCode.Double, 27);
                var o = np.zeros(new Shape(12), NPTypeCode.Double);
                return np.evaluate((NDExpr)a * 2.0, @out: o, where: a > 0.0);
            }, expectTrivial: false);
        }

        [TestMethod]
        public void OrderKeyword_FreshLayoutConflict_Declines_OneDimEngages()
        {
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 3, 5 }, NPTypeCode.Double, 31);
                return np.evaluate((NDExpr)a + a, order: 'F');   // F result from C inputs → the iterator transposes
            }, expectTrivial: false);
            AssertSameBothWays(() =>
            {
                var a = np.asfortranarray(Pool(new long[] { 3, 5 }, NPTypeCode.Double, 32));
                return np.evaluate((NDExpr)a + a, order: 'C');   // C result from F inputs
            }, expectTrivial: false);
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 15 }, NPTypeCode.Double, 33);
                return np.evaluate((NDExpr)a + a, order: 'F');   // 1-D is both orders
            }, expectTrivial: true);
        }

        [TestMethod]
        public void Out_ExactAliasEngages_OtherOverlapDeclines_CopySemanticsKept()
        {
            // In place (out IS an input): every element is read before the same element is written.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 4, 6 }, NPTypeCode.Double, 41);
                np.evaluate((NDExpr)a * 2.0 + a, @out: a);
                return a;
            }, expectTrivial: true);
            // A provided out that overlaps nothing, C and F.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 4, 6 }, NPTypeCode.Double, 42);
                var o = np.zeros(new Shape(4, 6), NPTypeCode.Double);
                return np.evaluate((NDExpr)a * a, @out: o);
            }, expectTrivial: true);
            AssertSameBothWays(() =>
            {
                var a = np.asfortranarray(Pool(new long[] { 4, 6 }, NPTypeCode.Double, 43));
                var o = np.asfortranarray(np.zeros(new Shape(4, 6), NPTypeCode.Double));
                return np.evaluate((NDExpr)a * a, @out: o);
            }, expectTrivial: true);
            // A shifted self-overlap with the OUTPUT ahead of the input: run in place, each write would clobber the
            // element the next step reads (a cascade across vector blocks), so the iterator must copy the input first
            // (COPY_IF_OVERLAP). This is the case that corrupts values if the overlap guard is lost.
            AssertSameBothWays(() =>
            {
                var base0 = Pool(new long[] { 257 }, NPTypeCode.Double, 48);
                np.evaluate((NDExpr)base0[":-1"] * 2.0 + 1.0, @out: base0["1:"]);
                return base0;
            }, expectTrivial: false);
            // The mirror image (input ahead of output) happens to be safe in place, but it is still a non-exact overlap,
            // so the trivial loop declines it and the iterator's copy runs — same bytes either way.
            AssertSameBothWays(() =>
            {
                var base1 = Pool(new long[] { 21 }, NPTypeCode.Double, 44);
                np.evaluate((NDExpr)base1["1:"] * 2.0, @out: base1[":-1"]);
                return base1;
            }, expectTrivial: false);
            // An out whose order disagrees with the inputs'.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 4, 6 }, NPTypeCode.Double, 45);
                var o = np.asfortranarray(np.zeros(new Shape(4, 6), NPTypeCode.Double));
                return np.evaluate((NDExpr)a * a, @out: o);
            }, expectTrivial: false);
            // A dtype-mismatched out (a same_kind float64 → float32 cast).
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 4, 6 }, NPTypeCode.Double, 46);
                var o = np.zeros(new Shape(4, 6), NPTypeCode.Single);
                return np.evaluate((NDExpr)a * a, @out: o);
            }, expectTrivial: false);
            // A 0-d iterator operand (positional handle — no parameter hoisting) that lives INSIDE the out: the
            // iterator copies it, so every element sees its ORIGINAL value, not the one written at index 2.
            AssertSameBothWays(() =>
            {
                var a = Pool(new long[] { 8 }, NPTypeCode.Double, 47);
                var k = a[2];
                Assert.AreEqual(0, k.ndim);
                var h = (NDExpr.Input(0) + NDExpr.Input(1)).Compile(NPTypeCode.Double, NPTypeCode.Double);
                h.Evaluate(new[] { a, k }, @out: a);
                return a;
            }, expectTrivial: false);
        }

        [TestMethod]
        public void Empty_Engages_ReturnsTheEmptyResult()
        {
            AssertSameBothWays(() =>
            {
                var a = np.zeros(new Shape(0, 3), NPTypeCode.Double);
                return np.evaluate((NDExpr)a * 2.0 + a);
            }, expectTrivial: true);
        }

        [TestMethod]
        public void ReadOnlyOut_RaisesTheSameErrorEitherWay()
        {
            string Message()
            {
                var a = Pool(new long[] { 5 }, NPTypeCode.Double, 51);
                var ro = np.broadcast_to(NDArray.Scalar(0.0), new Shape(5));
                try
                {
                    np.evaluate((NDExpr)a * a, @out: ro);
                }
                catch (Exception e)
                {
                    return e.GetType().Name + ": " + e.Message;
                }

                return "no exception";
            }

            string want;
            try
            {
                NDExpr.DisableTrivialLoop = true;
                want = Message();
            }
            finally
            {
                NDExpr.DisableTrivialLoop = false;
            }

            Assert.AreNotEqual("no exception", want);
            Assert.AreEqual(want, Message());
        }
    }
}
