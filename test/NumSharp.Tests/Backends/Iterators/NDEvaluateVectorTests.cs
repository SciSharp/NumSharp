using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// The "vector v2" contract of np.evaluate (docs/plans/ndexpr-evaluate.md Phase 1): every
    /// tree with a comparison / where / min-max / predicate / logical node or a bool operand now
    /// vectorizes (lane masks), and the vector path must reproduce the SCALAR body byte for byte —
    /// the scalar body is the one the evaluate.jsonl oracle tier holds to NumPy. Each case runs
    /// the same tree twice (vector kernel, then <see cref="NDExpr.ForceScalar"/>) over sizes that
    /// exercise the 4x-unrolled block, the one-vector remainder and the scalar tail for every lane
    /// width, across contiguous / F / strided / broadcast / 0-d operands, and compares bytes.
    /// </summary>
    [TestClass]
    public class NDEvaluateVectorTests
    {
        private const int N = 131;   // not a multiple of any lane count: unrolled x4 + remainder + tail

        // ---- data -------------------------------------------------------------------------

        private static double[] FloatPool(int n)
        {
            var edge = new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.0, 0.0, 1.0, -1.0, 0.5, 0.25, 1.9, -1.9, 127.0, 1e20, -1e-20, 3.0 };
            var r = new double[n];
            for (int i = 0; i < n; i++) r[i] = i % 5 == 0 ? edge[(i / 5) % edge.Length] : ((i * 37 % 101) - 50) / 7.0;
            return r;
        }

        private static long[] IntPool(int n)
        {
            var r = new long[n];
            for (int i = 0; i < n; i++) r[i] = ((i * 53) % 257) - 128;
            return r;
        }

        private static NDArray Floats(NPTypeCode tc, int n, int seed = 0)
        {
            var p = FloatPool(n);
            if (seed != 0) for (int i = 0; i < n; i++) p[i] = p[(i + seed) % n] * (seed % 2 == 0 ? 1.0 : -0.75);
            return np.array(p).astype(tc);
        }

        private static NDArray Ints(NPTypeCode tc, int n, int seed = 0)
        {
            var p = IntPool(n);
            if (seed != 0) for (int i = 0; i < n; i++) p[i] = p[(i + seed) % n];
            return np.array(p).astype(tc);
        }

        private static NDArray Mask(int n, int period)
            => np.array(Enumerable.Range(0, n).Select(i => (i % period) < period / 2 || i % 7 == 0).ToArray());

        // ---- the oracle: the scalar body -------------------------------------------------

        private static byte[] Bytes(NDArray r)
        {
            var c = np.ascontiguousarray(r);
            if (!(c.Shape.IsContiguous && c.Shape.offset == 0)) c = c.copy();
            var buf = new byte[c.size * c.dtypesize];
            unsafe { fixed (byte* d = buf) Buffer.MemoryCopy((byte*)c.Address, d, buf.Length, buf.Length); }
            return buf;
        }

        private static void AssertVectorMatchesScalar(string what, Func<NDExpr> build)
        {
            var vec = np.evaluate(build());
            NDArray sca;
            NDExpr.ForceScalar = true;
            try { sca = np.evaluate(build()); }
            finally { NDExpr.ForceScalar = false; }

            Assert.AreEqual(sca.typecode, vec.typecode, $"{what}: dtype");
            CollectionAssert.AreEqual(sca.shape, vec.shape, $"{what}: shape");
            var vb = Bytes(vec);
            var sb = Bytes(sca);
            Assert.AreEqual(sb.Length, vb.Length, $"{what}: byte length");
            for (int i = 0; i < sb.Length; i++)
            {
                if (sb[i] != vb[i])
                {
                    long elem = i / vec.dtypesize;
                    Assert.Fail($"{what}: byte {i} (element {elem}) differs — scalar {sca.GetAtIndex(elem)} vs vector {vec.GetAtIndex(elem)}");
                }
            }
        }

        // Half joins the float lanes: its PURE ARITHMETIC trees vectorize (Vector256<ushort>
        // widen-compute-narrow, Phase 5) and must match the scalar body; every other Half tree
        // (comparison / where / min-max / transcendental / unary) stays scalar, so vec==scalar is
        // then trivially the same scalar code — still a worthwhile pin that the plan gate is right.
        private static readonly NPTypeCode[] FloatLanes = { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Half };
        private static readonly NPTypeCode[] IntLanes = { NPTypeCode.Int64, NPTypeCode.Int32, NPTypeCode.Int16, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.UInt32 };

        /// <summary>The lanes whose data comes from the adversarial FLOAT pool (NaN / ±inf / exponent-gap).</summary>
        private static bool IsFloatLane(NPTypeCode lane)
            => lane == NPTypeCode.Single || lane == NPTypeCode.Double || lane == NPTypeCode.Half;

        // ---- shapes of the tree ------------------------------------------------------------

        private static IEnumerable<(string name, Func<NDArray, NDArray, NDArray, NDArray, NDArray, NDExpr> build)> Trees()
        {
            // a, b, c: lane-dtype operands; m1, m2: bool masks
            yield return ("a*b+c", (a, b, c, m1, m2) => (NDExpr)a * b + c);
            yield return ("(a-b)/(a+b)", (a, b, c, m1, m2) => (NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
            yield return ("a>b", (a, b, c, m1, m2) => NDExpr.Greater(a, b));
            yield return ("a>=0.5", (a, b, c, m1, m2) => NDExpr.GreaterEqual(a, 0.5));
            yield return ("a==b", (a, b, c, m1, m2) => NDExpr.Equal(a, b));
            yield return ("a!=b", (a, b, c, m1, m2) => NDExpr.NotEqual(a, b));
            yield return ("a<=b", (a, b, c, m1, m2) => NDExpr.LessEqual(a, b));
            yield return ("where(a>b,a,b)", (a, b, c, m1, m2) => NDExpr.Where(NDExpr.Greater(a, b), a, b));
            yield return ("max(a,b)", (a, b, c, m1, m2) => NDExpr.Max(a, b));
            yield return ("min(a,b)", (a, b, c, m1, m2) => NDExpr.Min(a, b));
            yield return ("clamp(a,b,c)", (a, b, c, m1, m2) => NDExpr.Clamp(a, NDExpr.Min(b, c), NDExpr.Max(b, c)));
            yield return ("(a>0.2)&(b<0.8)", (a, b, c, m1, m2) => NDExpr.Greater(a, 0.2) & NDExpr.Less(b, 0.8));
            yield return ("(a>0.2)|(b<0.8)", (a, b, c, m1, m2) => NDExpr.Greater(a, 0.2) | NDExpr.Less(b, 0.8));
            yield return ("(a>b)^(b>c)", (a, b, c, m1, m2) => NDExpr.Greater(a, b) ^ NDExpr.Greater(b, c));
            yield return ("!(a>b)", (a, b, c, m1, m2) => !NDExpr.Greater(a, b));
            yield return ("!a", (a, b, c, m1, m2) => !NDExpr.Arr(a));
            yield return ("a*(a>2)", (a, b, c, m1, m2) => (NDExpr)a * NDExpr.Greater(a, 2));
            yield return ("(a>2)+b", (a, b, c, m1, m2) => NDExpr.Greater(a, 2) + NDExpr.Arr(b));
            yield return ("leaky", (a, b, c, m1, m2) => NDExpr.Where(NDExpr.Greater(a, 0.5), a, (NDExpr)a * 0.01));
            yield return ("where(a,b,c)", (a, b, c, m1, m2) => NDExpr.Where(a, b, c));
            yield return ("where(m1,a,b)", (a, b, c, m1, m2) => NDExpr.Where(m1, a, b));
            yield return ("where(m1,1,0)", (a, b, c, m1, m2) => NDExpr.Where(m1, (NDExpr)a * 0 + 1, (NDExpr)a * 0));
            yield return ("a*m1", (a, b, c, m1, m2) => (NDExpr)a * m1);
            yield return ("a+m1", (a, b, c, m1, m2) => (NDExpr)a + m1);
            yield return ("m1&!m2", (a, b, c, m1, m2) => NDExpr.Arr(m1) & !NDExpr.Arr(m2));
            yield return ("m1|m2", (a, b, c, m1, m2) => NDExpr.Arr(m1) | m2);
            yield return ("m1^m2", (a, b, c, m1, m2) => NDExpr.Arr(m1) ^ m2);
            yield return ("m1+m2", (a, b, c, m1, m2) => NDExpr.Add(m1, m2));
            yield return ("m1*m2", (a, b, c, m1, m2) => NDExpr.Multiply(m1, m2));
            yield return ("m1==m2", (a, b, c, m1, m2) => NDExpr.Equal(m1, m2));
            yield return ("m1<m2", (a, b, c, m1, m2) => NDExpr.Less(m1, m2));
            yield return ("m1>=m2", (a, b, c, m1, m2) => NDExpr.GreaterEqual(m1, m2));
            yield return ("where(m1,m2,!m1)", (a, b, c, m1, m2) => NDExpr.Where(m1, m2, !NDExpr.Arr(m1)));
            yield return ("max(m1,m2)", (a, b, c, m1, m2) => NDExpr.Max(m1, m2));
            yield return ("where(m1&(a>b),a,-b)", (a, b, c, m1, m2) => NDExpr.Where(NDExpr.Arr(m1) & NDExpr.Greater(a, b), a, -NDExpr.Arr(b)));
            yield return ("isnan(a)", (a, b, c, m1, m2) => NDExpr.IsNaN(a));
            yield return ("isinf(a)", (a, b, c, m1, m2) => NDExpr.IsInf(a));
            yield return ("isfinite(a)", (a, b, c, m1, m2) => NDExpr.IsFinite(a));
            yield return ("where(isnan(a),0,a)", (a, b, c, m1, m2) => NDExpr.Where(NDExpr.IsNaN(a), (NDExpr)a * 0, a));
            yield return ("abs(a)", (a, b, c, m1, m2) => NDExpr.Abs(a));
            yield return ("-a", (a, b, c, m1, m2) => -NDExpr.Arr(a));
            yield return ("floor(a)", (a, b, c, m1, m2) => NDExpr.Floor(a));
            yield return ("trunc(a)+a", (a, b, c, m1, m2) => NDExpr.Truncate(a) + NDExpr.Arr(a));
            yield return ("square(a)-b", (a, b, c, m1, m2) => NDExpr.Square(a) - NDExpr.Arr(b));
            yield return ("a+2", (a, b, c, m1, m2) => (NDExpr)a + 2);
            yield return ("a*true", (a, b, c, m1, m2) => (NDExpr)a * true);
            yield return ("where(a>b,true,false)", (a, b, c, m1, m2) => NDExpr.Where(NDExpr.Greater(a, b), NDExpr.Const(true), NDExpr.Const(false)));
        }

        private static bool Applicable(string name, NPTypeCode lane)
        {
            bool isFloat = IsFloatLane(lane);
            if (!isFloat && (name.Contains("0.5") || name.Contains("0.2") || name == "leaky" || name.Contains("(a-b)/(a+b)")))
                return false;   // float literals / true_divide would type the tree off the integer lane (scalar anyway)
            if (!isFloat && (name.StartsWith("isnan") || name.StartsWith("isinf") || name.StartsWith("isfinite") || name.StartsWith("where(isnan")))
                return true;    // constant masks on integer lanes — still worth the byte compare
            return true;
        }

        private void RunAll(string layout, Func<NPTypeCode, (NDArray a, NDArray b, NDArray c, NDArray m1, NDArray m2)> make)
        {
            var failures = new List<string>();
            foreach (var lane in FloatLanes.Concat(IntLanes))
            {
                var (a, b, c, m1, m2) = make(lane);
                foreach (var (name, build) in Trees())
                {
                    if (!Applicable(name, lane)) continue;
                    try { AssertVectorMatchesScalar($"{layout}/{lane}/{name}", () => build(a, b, c, m1, m2)); }
                    catch (Exception e) { failures.Add($"{layout}/{lane}/{name}: {e.Message}"); }
                }
            }
            Assert.AreEqual(0, failures.Count, string.Join("\n", failures));
        }

        private static (NDArray, NDArray, NDArray, NDArray, NDArray) Make(NPTypeCode lane, int n, Func<NDArray, NDArray> shape, Func<NDArray, NDArray> maskShape = null)
        {
            bool isFloat = IsFloatLane(lane);
            NDArray a = shape(isFloat ? Floats(lane, n) : Ints(lane, n));
            NDArray b = shape(isFloat ? Floats(lane, n, 3) : Ints(lane, n, 3));
            NDArray c = shape(isFloat ? Floats(lane, n, 8) : Ints(lane, n, 8));
            maskShape ??= shape;
            return (a, b, c, maskShape(Mask(n, 6)), maskShape(Mask(n, 10)));
        }

        // ---- layouts --------------------------------------------------------------------------

        [TestMethod]
        public void Contiguous1D_VectorMatchesScalar()
            => RunAll("contig1d", lane => Make(lane, N, x => x));

        // 149 = 9×16 + 5 for 4-lane f64: unrolled blocks + ONE remainder vector + a tail (131 hits no
        // remainder vector at 4 lanes); the other lane widths get their own remainder/tail splits.
        [TestMethod]
        public void Contiguous1D_RemainderVector_VectorMatchesScalar()
            => RunAll("contig1d_149", lane => Make(lane, 149, x => x));

        [TestMethod]
        public void Contiguous2D_VectorMatchesScalar()
            => RunAll("contig2d", lane => Make(lane, 7 * 19, x => x.reshape(7, 19)));

        [TestMethod]
        public void FContiguous2D_VectorMatchesScalar()
            => RunAll("fcontig2d", lane => Make(lane, 7 * 19, x => x.reshape(19, 7).T.copy(order: 'F')));

        [TestMethod]
        public void Strided1D_VectorMatchesScalar()
            => RunAll("strided", lane => Make(lane, 2 * N, x => x["::2"]));

        [TestMethod]
        public void NegativeStride1D_VectorMatchesScalar()
            => RunAll("negstride", lane => Make(lane, N, x => x["::-1"]));

        [TestMethod]
        public void BroadcastColumn_VectorMatchesScalar()
        {
            // a: (7,19); b, c: (7,1) columns -> stride-0 along the inner axis (the hoisted-broadcast SIMD loop);
            // masks: (7,19) and a (7,1) column mask.
            RunAll("bcast", lane =>
            {
                bool isFloat = IsFloatLane(lane);
                NDArray a = (isFloat ? Floats(lane, 133) : Ints(lane, 133)).reshape(7, 19);
                NDArray b = (isFloat ? Floats(lane, 7, 3) : Ints(lane, 7, 3)).reshape(7, 1);
                NDArray c = (isFloat ? Floats(lane, 7, 8) : Ints(lane, 7, 8)).reshape(7, 1);
                return (a, b, c, Mask(133, 6).reshape(7, 19), Mask(7, 4).reshape(7, 1));
            });
        }

        [TestMethod]
        public void ZeroDOperand_VectorMatchesScalar()
        {
            RunAll("scalar0d", lane =>
            {
                bool isFloat = IsFloatLane(lane);
                NDArray a = isFloat ? Floats(lane, N) : Ints(lane, N);
                NDArray b = (isFloat ? Floats(lane, 1, 3) : Ints(lane, 1, 3)).reshape();
                NDArray c = isFloat ? Floats(lane, N, 8) : Ints(lane, N, 8);
                return (a, b, c, Mask(N, 6), np.array(true).reshape());
            });
        }

        // ---- targeted pins ---------------------------------------------------------------------

        [TestMethod]
        public void BoolOutput_PackedLanes_MatchNumPySemantics()
        {
            // f64 (4 lanes), f32 (8), i16 (16), i8 (32) outputs through the mask -> byte pack.
            var f = np.array(new double[] { 1, double.NaN, 3, 4, -0.0, 0.0, 7, 8, 9 });
            var r = np.evaluate(NDExpr.Greater(f, 3.5));
            Assert.AreEqual(NPTypeCode.Boolean, r.typecode);
            CollectionAssert.AreEqual(new[] { false, false, false, true, false, false, true, true, true }, Enumerable.Range(0, 9).Select(i => r.GetBoolean(i)).ToArray());

            var ne = np.evaluate(NDExpr.NotEqual(f, f));          // NaN != NaN is True
            Assert.IsTrue(ne.GetBoolean(1));
            Assert.IsFalse(ne.GetBoolean(0));

            var i8 = np.arange(40).astype(NPTypeCode.SByte) - (sbyte)20;
            var neg = np.evaluate(NDExpr.Less(i8, 0));
            for (int i = 0; i < 40; i++) Assert.AreEqual(i < 20, neg.GetBoolean(i), $"[{i}]");

            var u8 = np.arange(40).astype(NPTypeCode.Byte);
            var big = np.evaluate(NDExpr.Greater(u8, 200));
            Assert.IsFalse(Enumerable.Range(0, 40).Any(i => big.GetBoolean(i)));
        }

        [TestMethod]
        public void VectorPlan_DecidesLaneAndScalarFallback()
        {
            // homogeneous f64 + masks -> lane Double
            var x = NDExpr.Input(0);
            Assert.IsTrue(Plan(NDExpr.Where(NDExpr.Greater(x, 0.5), x, x * 2), new[] { NPTypeCode.Double }, out var lane) && lane == NPTypeCode.Double);
            // f64 with a STREAMED bool operand -> lane Double (mask input), but only where the host can
            // widen bool bytes to 8-byte lane masks (the x86 SSE4.1/AVX2 sign-extend). On ARM64 the plan
            // must go scalar, or the kernel throws PlatformNotSupportedException when it runs (this
            // was the macOS-arm64 CI failure of every *_VectorMatchesScalar test above).
            bool boolInputVectorizes = DirectILKernelGenerator.FusedBoolInputMasksAvailable(NPTypeCode.Double);
            var whereBool = NDExpr.Where(NDExpr.Input(1), NDExpr.Input(0), NDExpr.Input(0));
            var doubleAndBool = new[] { NPTypeCode.Double, NPTypeCode.Boolean };
            Assert.AreEqual(boolInputVectorizes, Plan(whereBool, doubleAndBool, out lane), "streamed bool input: vector iff the host has the byte->lane expansion");
            if (boolInputVectorizes)
                Assert.AreEqual(NPTypeCode.Double, lane);
            // ...the same tree with the bool as a hoisted 0-d PARAMETER is a constant mask built from
            // portable Zero/AllBitsSet, so it vectorizes on every 128/256-bit host, ARM64 included.
            Assert.AreEqual(DirectILKernelGenerator.FusedBoolLanesAvailable, Plan(whereBool, doubleAndBool, out lane, isParam: new[] { false, true }), "bool parameter needs no x86 expansion");
            if (DirectILKernelGenerator.FusedBoolLanesAvailable)
                Assert.AreEqual(NPTypeCode.Double, lane);
            // ...and a bool OUTPUT alone is packed portably, so a comparison stays vectorized there too.
            Assert.AreEqual(DirectILKernelGenerator.FusedBoolLanesAvailable, Plan(NDExpr.Greater(NDExpr.Input(0), 0.5), new[] { NPTypeCode.Double }, out lane), "bool output needs no x86 expansion");
            // all-bool -> byte mode
            Assert.IsTrue(Plan(NDExpr.Input(0) & !NDExpr.Input(1), new[] { NPTypeCode.Boolean, NPTypeCode.Boolean }, out lane) && lane == NPTypeCode.Boolean);
            // P5.2: mixed lanes now take the MIXED-WIDTH plan on a 256-bit AVX2 host — lane = the
            // root dtype, the narrow operand rides a partial Vector128 and the edge widens exactly;
            // elsewhere (no AVX2 / 512-bit) they keep the pre-P5.2 scalar fallback.
            if (DirectILKernelGenerator.FusedMixedWidthAvailable)
            {
                // i4 + f8 -> mixed at lane f8 (the acceptance pair: i4 loads V128<int>, widens i4->f8)
                Assert.IsTrue(PlanMixed(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Double }, out lane, out var mixed) && lane == NPTypeCode.Double && mixed);
                // int true_divide types the DIVIDE node to f64 over int32 leaves -> mixed at f8 too
                Assert.IsTrue(PlanMixed(NDExpr.Input(0) / NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, out lane, out mixed) && lane == NPTypeCode.Double && mixed);
                // f4 + f8 -> mixed (float->double widen)
                Assert.IsTrue(PlanMixed(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Single, NPTypeCode.Double }, out lane, out mixed) && lane == NPTypeCode.Double && mixed);
                // i1 + u1 -> i2: BOTH children widen (sign- and zero-extend)
                Assert.IsTrue(PlanMixed(NDExpr.Input(0) * NDExpr.Input(1), new[] { NPTypeCode.SByte, NPTypeCode.Byte }, out lane, out mixed) && lane == NPTypeCode.Int16 && mixed);
                // i8 + f8 -> the int64->float64 edge ROUNDS (no exact vector widen) -> stays scalar
                Assert.IsFalse(PlanMixed(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Int64, NPTypeCode.Double }, out _, out _));
                // i1 + i4 -> the sbyte container would be 64-bit under an i4 lane (ratio 4) -> scalar
                Assert.IsFalse(PlanMixed(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.SByte, NPTypeCode.Int32 }, out _, out _));
                // a comparison root (bool-typed node) keeps a mixed tree scalar in this increment
                Assert.IsFalse(PlanMixed(NDExpr.Greater(NDExpr.Input(0), NDExpr.Input(1)), new[] { NPTypeCode.Int32, NPTypeCode.Double }, out _, out _));
                // a bool operand keeps a mixed tree scalar in this increment
                Assert.IsFalse(PlanMixed(NDExpr.Input(0) + NDExpr.Input(1) + NDExpr.Input(2), new[] { NPTypeCode.Boolean, NPTypeCode.Int32, NPTypeCode.Double }, out _, out _));
                // a unary node in a mixed tree has no mixed emit -> scalar
                Assert.IsFalse(PlanMixed(NDExpr.Negate(NDExpr.Input(0)) + NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Double }, out _, out _));
                // uniform-input tree lifted by a weak float literal (i4 + 2.5 -> f8) -> mixed
                Assert.IsTrue(PlanMixed(NDExpr.Input(0) + 2.5, new[] { NPTypeCode.Int32 }, out lane, out mixed) && lane == NPTypeCode.Double && mixed);
            }
            else
            {
                Assert.IsFalse(Plan(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Double }, out _));
                Assert.IsFalse(Plan(NDExpr.Input(0) / NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, out _));
            }
            // Half ARITHMETIC vectorizes (Phase 5, Vector256<ushort> widen-compute-narrow) on a 256-bit
            // AVX2 host — but only arithmetic: a Half comparison / transcendental keeps its tree scalar.
            if (DirectILKernelGenerator.FusedHalfArithAvailable)
            {
                Assert.IsTrue(Plan(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Half, NPTypeCode.Half }, out lane) && lane == NPTypeCode.Half);
                Assert.IsFalse(Plan(NDExpr.Sqrt(NDExpr.Input(0)), new[] { NPTypeCode.Half }, out _));
                Assert.IsFalse(Plan(NDExpr.Greater(NDExpr.Input(0), NDExpr.Input(1)), new[] { NPTypeCode.Half, NPTypeCode.Half }, out _));
            }
            // Decimal / Complex lanes never vectorize.
            Assert.IsFalse(Plan(NDExpr.Input(0) + NDExpr.Input(0), new[] { NPTypeCode.Complex }, out _));
            // a Call node keeps its tree scalar
            Assert.IsFalse(Plan(NDExpr.Call<double, double>(System.Math.Sqrt, NDExpr.Input(0)) + NDExpr.Input(0), new[] { NPTypeCode.Double }, out _));
        }

        /// <summary>
        /// Resolve <paramref name="tree"/>'s NumPy types and ask the v2 planner whether it vectorizes
        /// on THIS host — through the uniform OR the P5.2 mixed-width plan; use
        /// <see cref="PlanMixed"/> to also learn which one answered.
        /// </summary>
        /// <param name="tree">The expression tree to plan.</param>
        /// <param name="inputs">Every input's dtype, in input order.</param>
        /// <param name="lane">Receives the planned lane dtype (Empty when the plan fails).</param>
        /// <param name="isParam">Per input, whether it is a hoisted 0-d parameter; null = all streamed.</param>
        /// <returns>The planner's verdict: true = vector kernel, false = scalar shell.</returns>
        private static bool Plan(NDExpr tree, NPTypeCode[] inputs, out NPTypeCode lane, bool[] isParam = null)
            => PlanMixed(tree, inputs, out lane, out _, isParam);

        /// <summary>
        /// <see cref="Plan"/>, also reporting whether the verdict came from the P5.2 mixed-width plan.
        /// </summary>
        /// <param name="tree">The expression tree to plan.</param>
        /// <param name="inputs">Every input's dtype, in input order.</param>
        /// <param name="lane">Receives the planned lane dtype (the root dtype for a mixed plan; Empty when the plan fails).</param>
        /// <param name="mixedWidth">Receives true when the tree took the mixed-width plan.</param>
        /// <param name="isParam">Per input, whether it is a hoisted 0-d parameter; null = all streamed.</param>
        /// <returns>The planner's verdict: true = vector kernel, false = scalar shell.</returns>
        private static bool PlanMixed(NDExpr tree, NPTypeCode[] inputs, out NPTypeCode lane, out bool mixedWidth, bool[] isParam = null)
        {
            var resolved = tree.ResolveNumPyTypes(inputs, out var types);
            _ = resolved;
            return NDExprVectorPlan.TryPlan(tree, inputs, types, out lane, out mixedWidth, isParam);
        }

        /// <summary>
        /// The truth table of <see cref="DirectILKernelGenerator.InlineMaskCreationSupported"/>, stated
        /// against the ISA it actually depends on: 1-byte lanes are portable at every width, 2/4/8-byte
        /// lanes need AVX2 (256-bit) or SSE4.1 (128-bit), 512-bit has no wide-lane expansion. On an
        /// ARM64 host every x86 ISA reports unsupported, so only the 1-byte row is true there.
        /// </summary>
        [TestMethod]
        public void InlineMaskCreationSupported_TracksTheX86SignExtendIsa()
        {
            foreach (int bits in new[] { 128, 256, 512 })
                Assert.IsTrue(DirectILKernelGenerator.InlineMaskCreationSupported(bits, 1), $"1-byte lanes are portable at {bits} bits");
            foreach (int size in new[] { 2, 4, 8 })
            {
                Assert.AreEqual(System.Runtime.Intrinsics.X86.Avx2.IsSupported, DirectILKernelGenerator.InlineMaskCreationSupported(256, size), $"{size}-byte lanes at 256 bits need AVX2");
                Assert.AreEqual(System.Runtime.Intrinsics.X86.Sse41.IsSupported, DirectILKernelGenerator.InlineMaskCreationSupported(128, size), $"{size}-byte lanes at 128 bits need SSE4.1");
                Assert.IsFalse(DirectILKernelGenerator.InlineMaskCreationSupported(512, size), $"{size}-byte lanes have no 512-bit expansion");
            }
            Assert.IsFalse(DirectILKernelGenerator.InlineMaskCreationSupported(128, 3), "no expansion for a 3-byte lane");
            Assert.IsFalse(DirectILKernelGenerator.InlineMaskCreationSupported(128, 16), "no expansion for a 16-byte lane");
        }

        // ---- P5.2 mixed-width ("lane groups") ---------------------------------------------------
        //
        // A tree over TWO distinct dtypes (or lifted past its inputs by a weak literal) now
        // vectorizes: lane = the root dtype, half-lane operands/nodes ride partial Vector128s,
        // every edge an exact widening. The vector path must reproduce the SCALAR body byte for
        // byte — the scalar body being what the evaluate.jsonl tier holds to NumPy — across every
        // admitted dtype pair, both widen directions, layouts and vector/tail boundary sizes.

        /// <summary>The admitted ratio-2 dtype pairs the mixed sweep drives (operand A narrow, B wide, plus both-widen pairs).</summary>
        private static readonly (NPTypeCode a, NPTypeCode b)[] MixedPairs =
        {
            (NPTypeCode.Int32, NPTypeCode.Double),
            (NPTypeCode.UInt32, NPTypeCode.Double),      // the sign-bias u4→f8 edge
            (NPTypeCode.Single, NPTypeCode.Double),
            (NPTypeCode.Int32, NPTypeCode.Single),       // both widen → f8
            (NPTypeCode.Int32, NPTypeCode.Int64),
            (NPTypeCode.UInt32, NPTypeCode.UInt64),
            (NPTypeCode.Int32, NPTypeCode.UInt32),       // both widen → i8
            (NPTypeCode.Int16, NPTypeCode.Single),
            (NPTypeCode.UInt16, NPTypeCode.Single),
            (NPTypeCode.Int16, NPTypeCode.Int32),
            (NPTypeCode.UInt16, NPTypeCode.UInt32),
            (NPTypeCode.Int16, NPTypeCode.UInt16),       // both widen → i4
            (NPTypeCode.SByte, NPTypeCode.Byte),         // both widen → i2
            (NPTypeCode.SByte, NPTypeCode.Int16),
            (NPTypeCode.Byte, NPTypeCode.UInt16),
        };

        /// <summary>
        /// Mixed trees over operands (a, b) of DIFFERENT dtypes: arithmetic + bitwise compositions,
        /// a leaf consumed at two dtypes (a in both the narrow product and the wide sum), and the
        /// weak-literal forms. Ops outside the mixed op set (comparisons, where, min/max) are also
        /// present — they must fall back to the scalar plan and still match trivially.
        /// </summary>
        private static IEnumerable<(string name, Func<NDArray, NDArray, NDExpr> build)> MixedTrees(bool intPair)
        {
            yield return ("a+b", (a, b) => (NDExpr)a + b);
            yield return ("a-b", (a, b) => (NDExpr)a - b);
            yield return ("a*b", (a, b) => (NDExpr)a * b);
            yield return ("a/b", (a, b) => (NDExpr)a / b);
            yield return ("a*b+a", (a, b) => (NDExpr)a * b + a);            // `a` read at its own AND the wide dtype
            yield return ("a*2+b", (a, b) => (NDExpr)a * 2 + b);            // weak literal wraps at a's dtype (the acceptance tree)
            yield return ("(a+b)*(a-b)", (a, b) => ((NDExpr)a + b) * ((NDExpr)a - b));
            yield return ("b+a*a", (a, b) => (NDExpr)b + (NDExpr)a * a);    // narrow product wraps, then widens
            if (intPair)
                yield return ("a&b|a", (a, b) => ((NDExpr)a & b) | a);      // bitwise through widen edges
            yield return ("a>b", (a, b) => NDExpr.Greater(a, b));           // bool root → scalar fallback, still must match
            yield return ("where(a>b,b,a+b)", (a, b) => NDExpr.Where(NDExpr.Greater(a, b), NDExpr.Arr(b), (NDExpr)a + b));
        }

        [TestMethod]
        public void MixedWidth_VectorMatchesScalar()
        {
            var failures = new List<string>();
            foreach (var (ta, tb) in MixedPairs)
            {
                bool intPair = !IsFloatLane(ta) && !IsFloatLane(tb);
                foreach (var (layout, n, wrap) in new (string, int, Func<NDArray, NDArray>)[]
                {
                    ("contig", N, x => x),
                    ("contig149", 149, x => x),
                    ("strided", 2 * N, x => x["::2"]),
                    ("negstride", N, x => x["::-1"]),
                })
                {
                    NDArray MakeOp(NPTypeCode tc, int seed)
                        => wrap(IsFloatLane(tc) ? Floats(tc, n, seed) : Ints(tc, n, seed));

                    var a = MakeOp(ta, 0);
                    var b = MakeOp(tb, 3);
                    foreach (var (name, build) in MixedTrees(intPair))
                    {
                        try { AssertVectorMatchesScalar($"mixed/{layout}/{ta}+{tb}/{name}", () => build(a, b)); }
                        catch (Exception e) { failures.Add($"mixed/{layout}/{ta}+{tb}/{name}: {e.Message}"); }
                    }

                    // The wide operand as a 0-d PARAMETER (hoisted into the kernel aux block) and as
                    // a broadcast column — the two stride-0 representations of the same value.
                    var b0d = MakeOp(tb, 3)["0"].reshape();
                    try { AssertVectorMatchesScalar($"mixed/{layout}/{ta}+{tb}0d/a*b+a", () => (NDExpr)a * b0d + a); }
                    catch (Exception e) { failures.Add($"mixed/{layout}/{ta}+{tb}0d: {e.Message}"); }
                    var a0d = MakeOp(ta, 0)["0"].reshape();
                    try { AssertVectorMatchesScalar($"mixed/{layout}/{ta}0d+{tb}/a*b+a", () => (NDExpr)a0d * b + b); }
                    catch (Exception e) { failures.Add($"mixed/{layout}/{ta}0d+{tb}: {e.Message}"); }
                }
            }

            Assert.AreEqual(0, failures.Count, string.Join("\n", failures));
        }

        [TestMethod]
        public void MixedWidth_ValuePins_MatchNumPy()
        {
            // Every expected value below is the literal NumPy 2.4.2 output (probed).
            // i4*2 WRAPS at int32 BEFORE promoting to f8 — the per-node-dtype contract.
            var a = np.array(new[] { 2_000_000_000, -2_000_000_000, 7, -1 });
            var b = np.array(new[] { 0.5, 1.5, 2.5, 3.5 });
            var r = np.evaluate((NDExpr)a * 2 + b);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            CollectionAssert.AreEqual(new[] { -294967295.5, 294967297.5, 16.5, 1.5 },
                Enumerable.Range(0, 4).Select(i => r.GetDouble(i)).ToArray());

            // u4 → f8 is value-exact up to uint.MaxValue (the sign-bias widen must not lose a bit).
            var u = np.array(new[] { 4294967295u, 2147483648u, 3000000000u, 0u });
            r = np.evaluate((NDExpr)u + b);
            CollectionAssert.AreEqual(new[] { 4294967295.5, 2147483649.5, 3000000002.5, 3.5 },
                Enumerable.Range(0, 4).Select(i => r.GetDouble(i)).ToArray());

            // i1 * u1 → i2: BOTH children widen (sign- and zero-extend) before the int16 multiply.
            var i1 = np.array(new sbyte[] { -128, 100, -7, 127 });
            var u1 = np.array(new byte[] { 2, 3, 255, 255 });
            r = np.evaluate((NDExpr)i1 * u1);
            Assert.AreEqual(NPTypeCode.Int16, r.typecode);
            CollectionAssert.AreEqual(new short[] { -256, 300, -1785, 32385 },
                Enumerable.Range(0, 4).Select(i => r.GetAtIndex<short>(i)).ToArray());

            // i4 + u4 → i8 (NEP50's signed×unsigned same-size promotion).
            var i4 = np.array(new[] { -1, 5, -100, 7 });
            var u4 = np.array(new[] { 4294967295u, 10u, 3u, 2u });
            r = np.evaluate((NDExpr)i4 + u4);
            Assert.AreEqual(NPTypeCode.Int64, r.typecode);
            CollectionAssert.AreEqual(new long[] { 4294967294, 15, -97, 9 },
                Enumerable.Range(0, 4).Select(i => r.GetInt64(i)).ToArray());

            // i4 / i4 → f8 (true divide lifts a UNIFORM int tree to a mixed-width kernel).
            var num = np.array(new[] { 7, -7, 9, 1 });
            var den = np.array(new[] { 2, 2, 4, 8 });
            r = np.evaluate((NDExpr)num / den);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            CollectionAssert.AreEqual(new[] { 3.5, -3.5, 2.25, 0.125 },
                Enumerable.Range(0, 4).Select(i => r.GetDouble(i)).ToArray());

            // i2 + f4 → f4 (a HALF-LANE pair whose lane is float32, sub-word widen chain).
            var i2 = np.array(new short[] { 1000, -32768, 77, 3 });
            var f4 = np.array(new[] { 0.5f, 0.25f, -1.5f, 2.0f });
            r = np.evaluate((NDExpr)i2 + f4);
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
            CollectionAssert.AreEqual(new[] { 1000.5f, -32767.75f, 75.5f, 5.0f },
                Enumerable.Range(0, 4).Select(i => r.GetSingle(i)).ToArray());
        }

        [TestMethod]
        public void MixedWidth_ComposesWithOutWhereAndDtype()
        {
            var a = np.array(Enumerable.Range(0, 40).ToArray());                       // int32
            var b = np.array(Enumerable.Range(0, 40).Select(i => i * 0.5).ToArray());  // float64
            var expr = (NDExpr)a * 2 + b;

            // out= at the natural dtype
            var dst = np.full(new Shape(40), -1.0);
            var r = np.evaluate(expr, @out: dst);
            Assert.AreSame(dst, r);
            Assert.AreEqual(2 * 2 + 1.0, r.GetDouble(2));

            // where= masks the write; masked-off slots keep prior contents
            var prior = np.full(new Shape(40), -1.0);
            var mask = np.array(Enumerable.Range(0, 40).Select(i => i % 2 == 0).ToArray());
            r = np.evaluate((NDExpr)a * 2 + b, @out: prior, where: mask);
            Assert.AreEqual(0.0, r.GetDouble(0));
            Assert.AreEqual(-1.0, r.GetDouble(1));
            Assert.AreEqual(4 * 2 + 2.0, r.GetDouble(4));

            // dtype= casts the fused result (compute at f8, store f4)
            r = np.evaluate((NDExpr)a * 2 + b, dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
            Assert.AreEqual((float)(3 * 2 + 1.5), r.GetSingle(3));
        }
    }
}
