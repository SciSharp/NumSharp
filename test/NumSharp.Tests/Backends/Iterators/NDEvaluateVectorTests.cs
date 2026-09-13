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

        private static readonly NPTypeCode[] FloatLanes = { NPTypeCode.Double, NPTypeCode.Single };
        private static readonly NPTypeCode[] IntLanes = { NPTypeCode.Int64, NPTypeCode.Int32, NPTypeCode.Int16, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.UInt32 };

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
            bool isFloat = lane == NPTypeCode.Single || lane == NPTypeCode.Double;
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
            bool isFloat = lane == NPTypeCode.Single || lane == NPTypeCode.Double;
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
                bool isFloat = lane == NPTypeCode.Single || lane == NPTypeCode.Double;
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
                bool isFloat = lane == NPTypeCode.Single || lane == NPTypeCode.Double;
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
            // f64 with a bool operand -> lane Double (mask input)
            Assert.IsTrue(Plan(NDExpr.Where(NDExpr.Input(1), NDExpr.Input(0), NDExpr.Input(0)), new[] { NPTypeCode.Double, NPTypeCode.Boolean }, out lane) && lane == NPTypeCode.Double);
            // all-bool -> byte mode
            Assert.IsTrue(Plan(NDExpr.Input(0) & !NDExpr.Input(1), new[] { NPTypeCode.Boolean, NPTypeCode.Boolean }, out lane) && lane == NPTypeCode.Boolean);
            // mixed lanes (i4 + f8) -> scalar
            Assert.IsFalse(Plan(NDExpr.Input(0) + NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Double }, out _));
            // int true_divide types the node to f64 off the int32 lane -> scalar
            Assert.IsFalse(Plan(NDExpr.Input(0) / NDExpr.Input(1), new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, out _));
            // Half / Decimal / Complex lanes never vectorize
            Assert.IsFalse(Plan(NDExpr.Input(0) + NDExpr.Input(0), new[] { NPTypeCode.Half }, out _));
            Assert.IsFalse(Plan(NDExpr.Input(0) + NDExpr.Input(0), new[] { NPTypeCode.Complex }, out _));
            // a Call node keeps its tree scalar
            Assert.IsFalse(Plan(NDExpr.Call<double, double>(System.Math.Sqrt, NDExpr.Input(0)) + NDExpr.Input(0), new[] { NPTypeCode.Double }, out _));
        }

        private static bool Plan(NDExpr tree, NPTypeCode[] inputs, out NPTypeCode lane)
        {
            var resolved = tree.ResolveNumPyTypes(inputs, out var types);
            _ = resolved;
            return NDExprVectorPlan.TryPlan(tree, inputs, types, out lane);
        }
    }
}
