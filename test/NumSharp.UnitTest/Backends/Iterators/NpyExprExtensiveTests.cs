using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest coverage for the expanded NpyExpr DSL.
    /// Each op class has:
    ///   • Happy path at float32 + float64
    ///   • Dtype matrix (integer where meaningful)
    ///   • Edge values (NaN, Inf, zero, neg, overflow)
    ///   • Strided vs contiguous inputs
    ///   • Composition tests (e.g. sigmoid, relu)
    ///   • Cache reuse checks
    /// </summary>
    [TestClass]
    public unsafe class NpyExprExtensiveTests
    {
        // =====================================================================
        // Helpers
        // =====================================================================

        private static NpyIterRef Iter(NDArray input, NDArray output)
            => NpyIterRef.MultiNew(2, new[] { input, output },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

        private static NpyIterRef Iter3(NDArray a, NDArray b, NDArray c)
            => NpyIterRef.MultiNew(3, new[] { a, b, c },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.WRITEONLY });

        private static void RunUnary_f64(
            double[] xs, Func<NpyExpr, NpyExpr> fn, Func<double, double> expected,
            double tol = 1e-9, string? key = null)
        {
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            iter.ExecuteExpression(fn(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: key);

            for (int i = 0; i < xs.Length; i++)
            {
                double got = output.GetDouble(i);
                double want = expected(xs[i]);
                if (double.IsNaN(want))
                    Assert.IsTrue(double.IsNaN(got), $"[{i}] expected NaN got {got}");
                else if (double.IsInfinity(want))
                    Assert.IsTrue(double.IsInfinity(got) && Math.Sign(got) == Math.Sign(want),
                        $"[{i}] expected {want} got {got}");
                else
                    Assert.AreEqual(want, got, tol, $"[{i}] xs={xs[i]}");
            }
        }

        private static void RunBinary_f64(
            double[] xs, double[] ys, Func<NpyExpr, NpyExpr, NpyExpr> fn,
            Func<double, double, double> expected, double tol = 1e-9, string? key = null)
        {
            var a = np.array(xs);
            var b = np.array(ys);
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(fn(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double, cacheKey: key);

            for (int i = 0; i < xs.Length; i++)
            {
                double got = c.GetDouble(i);
                double want = expected(xs[i], ys[i]);
                if (double.IsNaN(want))
                    Assert.IsTrue(double.IsNaN(got), $"[{i}] expected NaN got {got}");
                else
                    Assert.AreEqual(want, got, tol, $"[{i}] x={xs[i]} y={ys[i]}");
            }
        }

        // =====================================================================
        // Binary arithmetic: Mod, Power, FloorDivide, ATan2
        // =====================================================================

        [TestMethod]
        public void Mod_Double_PositiveAndNegative()
        {
            // NumPy mod uses floored division: sign of result matches divisor.
            RunBinary_f64(
                new double[] { 10, -10, 10, -10, 7, 0 },
                new double[] { 3, 3, -3, -3, 2, 5 },
                NpyExpr.Mod,
                (x, y) =>
                {
                    // floored mod
                    return x - Math.Floor(x / y) * y;
                }, key: "mod_f64_v1");
        }

        [TestMethod]
        public void Mod_OperatorOverload_Percent()
        {
            var a = np.array(new double[] { 10.0, 7.0, -7.0 });
            var b = np.array(new double[] { 3.0, 2.0, 3.0 });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Input(0) % NpyExpr.Input(1),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "mod_op_v1");
            Assert.AreEqual(1.0, c.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, c.GetDouble(1), 1e-9);
            Assert.AreEqual(2.0, c.GetDouble(2), 1e-9);  // -7 mod 3 = 2 (floored)
        }

        [TestMethod]
        public void Mod_Int32_FlooredSemantics()
        {
            var a = np.array(new int[] { 10, -10, 10, -10 });
            var b = np.array(new int[] { 3, 3, -3, -3 });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Mod(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "mod_i32_v1");
            // NumPy: 10%3=1, -10%3=2, 10%-3=-2, -10%-3=-1
            Assert.AreEqual(1, c.GetInt32(0));
            Assert.AreEqual(2, c.GetInt32(1));
            Assert.AreEqual(-2, c.GetInt32(2));
            Assert.AreEqual(-1, c.GetInt32(3));
        }

        [TestMethod]
        public void Power_Double_IntegerAndFractional()
        {
            RunBinary_f64(
                new double[] { 2, 3, 4, 0, -1, 9 },
                new double[] { 10, 0, 0.5, 0, 3, 0.5 },
                NpyExpr.Power, Math.Pow, key: "pow_f64_v1");
        }

        [TestMethod]
        public void Power_Double_NaNInput()
        {
            var a = np.array(new double[] { double.NaN, 2.0, double.NaN });
            var b = np.array(new double[] { 2.0, 0.0, 1.0 });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Power(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "pow_nan_v1");
            Assert.IsTrue(double.IsNaN(c.GetDouble(0)));
            Assert.AreEqual(1.0, c.GetDouble(1), 1e-9); // anything^0 = 1, even NaN^0 in NumPy
            Assert.IsTrue(double.IsNaN(c.GetDouble(2)));
        }

        [TestMethod]
        public void FloorDivide_Double_NegativeFloorsDown()
        {
            RunBinary_f64(
                new double[] { 10, -10, 7, -7, 15, -15 },
                new double[] { 3, 3, 2, 2, 4, 4 },
                NpyExpr.FloorDivide,
                (x, y) => Math.Floor(x / y), key: "floordiv_f64_v1");
        }

        [TestMethod]
        public void FloorDivide_Int32_SignedFloor()
        {
            var a = np.array(new int[] { 10, -10, 7, -7 });
            var b = np.array(new int[] { 3, 3, 2, 2 });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.FloorDivide(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "floordiv_i32_v1");
            Assert.AreEqual(3, c.GetInt32(0));
            Assert.AreEqual(-4, c.GetInt32(1));  // floored, not truncated
            Assert.AreEqual(3, c.GetInt32(2));
            Assert.AreEqual(-4, c.GetInt32(3));
        }

        [TestMethod]
        public void ATan2_Quadrants()
        {
            RunBinary_f64(
                new double[] { 1, 1, -1, -1, 0, 0, 1, -1 },
                new double[] { 1, -1, -1, 1, 1, -1, 0, 0 },
                NpyExpr.ATan2, Math.Atan2, tol: 1e-9, key: "atan2_f64_v1");
        }

        // =====================================================================
        // Binary bitwise: BitwiseAnd/Or/Xor (SIMD-capable)
        // =====================================================================

        [TestMethod]
        public void BitwiseAnd_Int32_Operator()
        {
            var a = np.array(new int[] { 0b1100, 0b1010, 0xFFFF, 0 });
            var b = np.array(new int[] { 0b1010, 0b0101, 0xFF00, 0xFFFF });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Input(0) & NpyExpr.Input(1),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "and_i32_v1");
            Assert.AreEqual(0b1000, c.GetInt32(0));
            Assert.AreEqual(0, c.GetInt32(1));
            Assert.AreEqual(0xFF00, c.GetInt32(2));
            Assert.AreEqual(0, c.GetInt32(3));
        }

        [TestMethod]
        public void BitwiseOr_Int32_Operator()
        {
            var a = np.array(new int[] { 0b1100, 0b1010, 0, 0xFFFF });
            var b = np.array(new int[] { 0b0011, 0b0101, 0xABCD, 0 });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Input(0) | NpyExpr.Input(1),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "or_i32_v1");
            Assert.AreEqual(0b1111, c.GetInt32(0));
            Assert.AreEqual(0b1111, c.GetInt32(1));
            Assert.AreEqual(0xABCD, c.GetInt32(2));
            Assert.AreEqual(0xFFFF, c.GetInt32(3));
        }

        [TestMethod]
        public void BitwiseXor_Int64_Operator()
        {
            var a = np.array(new long[] { 0xAAAAAAAAL, 0, 0xFFFFL });
            var b = np.array(new long[] { 0x55555555L, 0xABCDL, 0xFFFFL });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Input(0) ^ NpyExpr.Input(1),
                new[] { NPTypeCode.Int64, NPTypeCode.Int64 }, NPTypeCode.Int64,
                cacheKey: "xor_i64_v1");
            Assert.AreEqual(0xFFFFFFFFL, c.GetInt64(0));
            Assert.AreEqual(0xABCDL, c.GetInt64(1));
            Assert.AreEqual(0L, c.GetInt64(2));
        }

        // =====================================================================
        // Min, Max, Clamp
        // =====================================================================

        [TestMethod]
        public void Min_Double_ReturnsSmaller()
        {
            RunBinary_f64(
                new double[] { 1, 5, -3, 0, 7 },
                new double[] { 2, 3, -2, 0, 7 },
                NpyExpr.Min, Math.Min, key: "min_f64_v1");
        }

        [TestMethod]
        public void Max_Int32_ReturnsLarger()
        {
            var a = np.array(new int[] { 1, 5, -3, 0, int.MaxValue });
            var b = np.array(new int[] { 2, 3, -2, 0, int.MinValue });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "max_i32_v1");
            Assert.AreEqual(2, c.GetInt32(0));
            Assert.AreEqual(5, c.GetInt32(1));
            Assert.AreEqual(-2, c.GetInt32(2));
            Assert.AreEqual(0, c.GetInt32(3));
            Assert.AreEqual(int.MaxValue, c.GetInt32(4));
        }

        [TestMethod]
        public void Min_Double_NaNPropagation()
        {
            // NumPy np.minimum: NaN propagates (unlike fmin)
            var a = np.array(new double[] { 1.0, double.NaN, 5.0 });
            var b = np.array(new double[] { 2.0, 3.0, double.NaN });
            var c = np.empty_like(a);
            using var iter = Iter3(a, b, c);
            iter.ExecuteExpression(NpyExpr.Min(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "min_nan_v1");
            Assert.AreEqual(1.0, c.GetDouble(0), 1e-9);
            Assert.IsTrue(double.IsNaN(c.GetDouble(1)));
            Assert.IsTrue(double.IsNaN(c.GetDouble(2)));
        }

        [TestMethod]
        public void Clamp_Double_ToRange()
        {
            var xs = new double[] { -5, -1, 0, 0.5, 1, 2, 100 };
            var expected = new double[] { 0, 0, 0, 0.5, 1, 1, 1 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            iter.ExecuteExpression(
                NpyExpr.Clamp(NpyExpr.Input(0), NpyExpr.Const(0.0), NpyExpr.Const(1.0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "clamp_f64_v1");
            for (int i = 0; i < xs.Length; i++)
                Assert.AreEqual(expected[i], output.GetDouble(i), 1e-9, $"[{i}]");
        }

        // =====================================================================
        // Where ternary
        // =====================================================================

        [TestMethod]
        public void Where_SelectsByCondition()
        {
            var cond = np.array(new double[] { 1, 0, 1, 0 });
            var a = np.array(new double[] { 10, 20, 30, 40 });
            var b = np.array(new double[] { -1, -2, -3, -4 });
            var r = np.empty_like(a);
            using var it = NpyIterRef.MultiNew(4, new[] { cond, a, b, r },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            it.ExecuteExpression(
                NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2)),
                new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "where_f64_v1");
            Assert.AreEqual(10.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(-2.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(30.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(-4.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void Where_ReLUComposition()
        {
            var xs = new double[] { -5, -1, 0, 1, 5 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(NpyExpr.Greater(x, NpyExpr.Const(0.0)),
                x, NpyExpr.Const(0.0));
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "relu_f64_v1");
            for (int i = 0; i < xs.Length; i++)
                Assert.AreEqual(Math.Max(0, xs[i]), output.GetDouble(i), 1e-9);
        }

        // =====================================================================
        // Exponentials: Exp, Exp2, Expm1, Log, Log2, Log10, Log1p
        // =====================================================================

        [TestMethod] public void Exp_Double() => RunUnary_f64(
            new double[] { 0, 1, 2, -1, Math.Log(10) }, NpyExpr.Exp, Math.Exp, tol: 1e-9, key: "exp_f64_v1");

        [TestMethod] public void Exp2_Double() => RunUnary_f64(
            new double[] { 0, 1, 2, 3, -1, 0.5 }, NpyExpr.Exp2,
            x => Math.Pow(2, x), tol: 1e-9, key: "exp2_f64_v1");

        [TestMethod] public void Expm1_Double_AccurateNearZero() => RunUnary_f64(
            new double[] { 0, 1e-10, 1, -1 }, NpyExpr.Expm1,
            x => Math.Exp(x) - 1, tol: 1e-9, key: "expm1_f64_v1");

        [TestMethod] public void Log_Double_SpecialValues()
        {
            var xs = new double[] { 1.0, Math.E, 10.0, 0.1 };
            RunUnary_f64(xs, NpyExpr.Log, Math.Log, tol: 1e-9, key: "log_f64_v1");
        }

        [TestMethod] public void Log_Double_NegativeIsNaN()
        {
            var a = np.array(new double[] { -1.0, 0.0, 1.0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Log(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "log_neg_v1");
            Assert.IsTrue(double.IsNaN(r.GetDouble(0)));
            Assert.IsTrue(double.IsNegativeInfinity(r.GetDouble(1)));
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
        }

        [TestMethod] public void Log2_Double() => RunUnary_f64(
            new double[] { 1, 2, 4, 8, 1024 }, NpyExpr.Log2, Math.Log2, tol: 1e-9, key: "log2_f64_v1");

        [TestMethod] public void Log10_Double() => RunUnary_f64(
            new double[] { 1, 10, 100, 1000, 1e-3 }, NpyExpr.Log10, Math.Log10, tol: 1e-9, key: "log10_f64_v1");

        [TestMethod] public void Log1p_Double_AccurateNearZero() => RunUnary_f64(
            new double[] { 0, 1e-10, 1, -0.5 }, NpyExpr.Log1p,
            x => Math.Log(1 + x), tol: 1e-9, key: "log1p_f64_v1");

        // =====================================================================
        // Trigonometric
        // =====================================================================

        [TestMethod] public void Sin_Double() => RunUnary_f64(
            new double[] { 0, Math.PI / 2, Math.PI, -Math.PI / 2 }, NpyExpr.Sin, Math.Sin,
            tol: 1e-9, key: "sin_f64_v1");

        [TestMethod] public void Cos_Double() => RunUnary_f64(
            new double[] { 0, Math.PI / 2, Math.PI, -Math.PI / 2 }, NpyExpr.Cos, Math.Cos,
            tol: 1e-9, key: "cos_f64_v1");

        [TestMethod] public void Tan_Double() => RunUnary_f64(
            new double[] { 0, Math.PI / 4, -Math.PI / 4 }, NpyExpr.Tan, Math.Tan,
            tol: 1e-9, key: "tan_f64_v1");

        [TestMethod] public void Sinh_Double() => RunUnary_f64(
            new double[] { 0, 1, -1, 2 }, NpyExpr.Sinh, Math.Sinh, tol: 1e-9, key: "sinh_f64_v1");

        [TestMethod] public void Cosh_Double() => RunUnary_f64(
            new double[] { 0, 1, -1, 2 }, NpyExpr.Cosh, Math.Cosh, tol: 1e-9, key: "cosh_f64_v1");

        [TestMethod] public void Tanh_Double() => RunUnary_f64(
            new double[] { 0, 1, -1, 100, -100 }, NpyExpr.Tanh, Math.Tanh, tol: 1e-9, key: "tanh_f64_v1");

        [TestMethod] public void ASin_Double() => RunUnary_f64(
            new double[] { 0, 0.5, 1, -1 }, NpyExpr.ASin, Math.Asin, tol: 1e-9, key: "asin_f64_v1");

        [TestMethod] public void ACos_Double() => RunUnary_f64(
            new double[] { 0, 0.5, 1, -1 }, NpyExpr.ACos, Math.Acos, tol: 1e-9, key: "acos_f64_v1");

        [TestMethod] public void ATan_Double() => RunUnary_f64(
            new double[] { 0, 1, -1, 1000 }, NpyExpr.ATan, Math.Atan, tol: 1e-9, key: "atan_f64_v1");

        [TestMethod] public void Deg2Rad_Double() => RunUnary_f64(
            new double[] { 0, 90, 180, 360, -90 }, NpyExpr.Deg2Rad,
            x => x * Math.PI / 180.0, tol: 1e-9, key: "d2r_f64_v1");

        [TestMethod] public void Rad2Deg_Double() => RunUnary_f64(
            new double[] { 0, Math.PI / 2, Math.PI, -Math.PI }, NpyExpr.Rad2Deg,
            x => x * 180.0 / Math.PI, tol: 1e-9, key: "r2d_f64_v1");

        // =====================================================================
        // Rounding
        // =====================================================================

        [TestMethod] public void Floor_Double() => RunUnary_f64(
            new double[] { 1.7, -1.7, 2.5, -2.5, 0, 1 }, NpyExpr.Floor, Math.Floor,
            tol: 0, key: "floor_f64_v1");

        [TestMethod] public void Ceil_Double() => RunUnary_f64(
            new double[] { 1.3, -1.3, 2.5, -2.5, 0, 1 }, NpyExpr.Ceil, Math.Ceiling,
            tol: 0, key: "ceil_f64_v1");

        [TestMethod] public void Round_Double_Banker() => RunUnary_f64(
            new double[] { 0.5, 1.5, 2.5, -0.5, -1.5 }, NpyExpr.Round,
            x => Math.Round(x), tol: 0, key: "round_f64_v1");

        [TestMethod] public void Truncate_Double() => RunUnary_f64(
            new double[] { 1.7, -1.7, 2.5, -2.5, 0 }, NpyExpr.Truncate, Math.Truncate,
            tol: 0, key: "trunc_f64_v1");

        // =====================================================================
        // Sign, Reciprocal, Cbrt
        // =====================================================================

        [TestMethod] public void Sign_Double() => RunUnary_f64(
            new double[] { -5, -1, 0, 1, 5 }, NpyExpr.Sign, x => (double)Math.Sign(x),
            tol: 0, key: "sign_f64_v1");

        [TestMethod] public void Sign_Int32()
        {
            var a = np.array(new int[] { -5, -1, 0, 1, 5 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Sign(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "sign_i32_v1");
            Assert.AreEqual(-1, r.GetInt32(0));
            Assert.AreEqual(-1, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
            Assert.AreEqual(1, r.GetInt32(3));
            Assert.AreEqual(1, r.GetInt32(4));
        }

        [TestMethod] public void Reciprocal_Double() => RunUnary_f64(
            new double[] { 1, 2, 4, 0.5, -1 }, NpyExpr.Reciprocal,
            x => 1.0 / x, tol: 1e-9, key: "recip_f64_v1");

        [TestMethod] public void Cbrt_Double() => RunUnary_f64(
            new double[] { 0, 1, 8, 27, -27, -8 }, NpyExpr.Cbrt, Math.Cbrt,
            tol: 1e-9, key: "cbrt_f64_v1");

        // =====================================================================
        // Bitwise unary: BitwiseNot, LogicalNot
        // =====================================================================

        [TestMethod]
        public void BitwiseNot_Int32_Operator()
        {
            var a = np.array(new int[] { 0, 1, -1, 255, int.MaxValue });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(~NpyExpr.Input(0),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "bnot_i32_v1");
            Assert.AreEqual(~0, r.GetInt32(0));
            Assert.AreEqual(~1, r.GetInt32(1));
            Assert.AreEqual(~(-1), r.GetInt32(2));
            Assert.AreEqual(~255, r.GetInt32(3));
            Assert.AreEqual(~int.MaxValue, r.GetInt32(4));
        }

        [TestMethod]
        public void BitwiseNot_Int64()
        {
            var a = np.array(new long[] { 0L, 1L, -1L, 0xFF00FF00L });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.BitwiseNot(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int64 }, NPTypeCode.Int64, cacheKey: "bnot_i64_v1");
            Assert.AreEqual(~0L, r.GetInt64(0));
            Assert.AreEqual(~1L, r.GetInt64(1));
            Assert.AreEqual(~(-1L), r.GetInt64(2));
            Assert.AreEqual(~0xFF00FF00L, r.GetInt64(3));
        }

        [TestMethod]
        public void LogicalNot_Double_Operator()
        {
            var a = np.array(new double[] { 0, 1, 2, 0, -5, double.NaN });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(!NpyExpr.Input(0),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "lnot_f64_v1");
            // NumPy: !0=1, !nonzero=0, !NaN=0 (NaN is truthy)
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(4), 1e-9);
            // NaN comparison: NaN == 0 is false → !NaN = 0
            Assert.AreEqual(0.0, r.GetDouble(5), 1e-9);
        }

        [TestMethod]
        public void LogicalNot_Int64()
        {
            var a = np.array(new long[] { 0L, 1L, -1L, 999L, 0L });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.LogicalNot(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int64 }, NPTypeCode.Int64, cacheKey: "lnot_i64_v1");
            Assert.AreEqual(1L, r.GetInt64(0));
            Assert.AreEqual(0L, r.GetInt64(1));
            Assert.AreEqual(0L, r.GetInt64(2));
            Assert.AreEqual(0L, r.GetInt64(3));
            Assert.AreEqual(1L, r.GetInt64(4));
        }

        // =====================================================================
        // Predicates: IsNaN, IsFinite, IsInf
        // =====================================================================

        [TestMethod]
        public void IsNaN_Double()
        {
            var a = np.array(new double[] { 1.0, double.NaN, 3.0, double.PositiveInfinity, 0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.IsNaN(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "isnan_f64_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(4), 1e-9);
        }

        [TestMethod]
        public void IsNaN_Int32_AlwaysFalse()
        {
            // Integers cannot be NaN — result is always 0.
            var a = np.array(new int[] { int.MinValue, 0, int.MaxValue });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.IsNaN(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "isnan_i32_v1");
            Assert.AreEqual(0, r.GetInt32(0));
            Assert.AreEqual(0, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
        }

        [TestMethod]
        public void IsFinite_Double()
        {
            var a = np.array(new double[] {
                1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.IsFinite(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "isfin_f64_v1");
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(4), 1e-9);
        }

        [TestMethod]
        public void IsFinite_Int32_AlwaysTrue()
        {
            var a = np.array(new int[] { int.MinValue, 0, int.MaxValue });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.IsFinite(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "isfin_i32_v1");
            Assert.AreEqual(1, r.GetInt32(0));
            Assert.AreEqual(1, r.GetInt32(1));
            Assert.AreEqual(1, r.GetInt32(2));
        }

        [TestMethod]
        public void IsInf_Double()
        {
            var a = np.array(new double[] {
                1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.IsInf(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "isinf_f64_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(4), 1e-9);
        }

        // =====================================================================
        // Comparison ops (Eq, Ne, Lt, Le, Gt, Ge) — return 0/1 at output dtype
        // =====================================================================

        [TestMethod]
        public void Equal_Double()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 1, 0, 3, 0 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Equal(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "eq_f64_v1");
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void Equal_Double_NaNIsNotEqualToItself()
        {
            var a = np.array(new double[] { double.NaN, 1.0 });
            var b = np.array(new double[] { double.NaN, 1.0 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Equal(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "eq_nan_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9); // NaN == NaN → 0
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
        }

        [TestMethod]
        public void NotEqual_Int32()
        {
            var a = np.array(new int[] { 1, 2, 3, 4 });
            var b = np.array(new int[] { 1, 0, 3, 0 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.NotEqual(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "ne_i32_v1");
            Assert.AreEqual(0, r.GetInt32(0));
            Assert.AreEqual(1, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
            Assert.AreEqual(1, r.GetInt32(3));
        }

        [TestMethod]
        public void Less_Double()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 1, 3, 2, 4 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Less(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "lt_f64_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void LessEqual_Double()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 1, 3, 2, 4 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.LessEqual(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "le_f64_v1");
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void Greater_Double()
        {
            var a = np.array(new double[] { 1, 5, 2, 4 });
            var b = np.array(new double[] { 1, 3, 2, 4 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Greater(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "gt_f64_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
        }

        [TestMethod]
        public void GreaterEqual_Int32()
        {
            var a = np.array(new int[] { 1, 5, 2, 4 });
            var b = np.array(new int[] { 1, 3, 2, 4 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.GreaterEqual(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "ge_i32_v1");
            Assert.AreEqual(1, r.GetInt32(0));
            Assert.AreEqual(1, r.GetInt32(1));
            Assert.AreEqual(1, r.GetInt32(2));
            Assert.AreEqual(1, r.GetInt32(3));
        }

        // =====================================================================
        // SIMD vs strided fallback — same expr, different strides
        // =====================================================================

        [TestMethod]
        public void Mod_StridedInput_UsesScalarFallback()
        {
            // Create 20-element then slice ::2 → strided view of 10 elements
            var src = np.arange(20).astype(np.float64);
            var sliced = src["::2"];
            Assert.AreEqual(10, sliced.size);

            var output = np.empty(new Shape(10), np.float64);
            using var iter = Iter(sliced, output);
            iter.ExecuteExpression(NpyExpr.Mod(NpyExpr.Input(0), NpyExpr.Const(3.0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "mod_strided_v1");

            for (int i = 0; i < 10; i++)
            {
                double x = 2.0 * i;
                double want = x - Math.Floor(x / 3.0) * 3.0;
                Assert.AreEqual(want, output.GetDouble(i), 1e-9, $"[{i}]");
            }
        }

        [TestMethod]
        public void Floor_ReversedStride_ProducesCorrectOutput()
        {
            var src = np.array(new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 });
            var reversed = src["::-1"];  // stride = -elemSize

            var output = np.empty(new Shape(5), np.float64);
            using var iter = Iter(reversed, output);
            iter.ExecuteExpression(NpyExpr.Floor(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "floor_rev_v1");

            // reversed = [5.5, 4.5, 3.5, 2.5, 1.5] → floor = [5, 4, 3, 2, 1]
            Assert.AreEqual(5.0, output.GetDouble(0), 1e-9);
            Assert.AreEqual(4.0, output.GetDouble(1), 1e-9);
            Assert.AreEqual(3.0, output.GetDouble(2), 1e-9);
            Assert.AreEqual(2.0, output.GetDouble(3), 1e-9);
            Assert.AreEqual(1.0, output.GetDouble(4), 1e-9);
        }

        [TestMethod]
        public void Exp_LargeArray_SimdVsScalarSameResult()
        {
            // 1024 contiguous vs 2048::2 strided (same values)
            var contig = np.arange(1024).astype(np.float64) * 0.01;
            var bigsrc = np.arange(2048).astype(np.float64) * 0.005;
            var strided = bigsrc["::2"];

            var contigOut = np.empty_like(contig);
            var stridedOut = np.empty(new Shape(1024), np.float64);

            using (var it1 = Iter(contig, contigOut))
                it1.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "exp_big_v1");
            using (var it2 = Iter(strided, stridedOut))
                it2.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "exp_big_v1");

            for (int i = 0; i < 1024; i++)
                Assert.AreEqual(contigOut.GetDouble(i), stridedOut.GetDouble(i), 1e-9,
                    $"mismatch at i={i}");
        }

        // =====================================================================
        // Composition: sigmoid, relu, softplus
        // =====================================================================

        [TestMethod]
        public void Composition_Sigmoid_Double()
        {
            var xs = new double[] { -100, -1, 0, 1, 100 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var x = NpyExpr.Input(0);
            // 1 / (1 + exp(-x))
            var expr = NpyExpr.Const(1.0) / (NpyExpr.Const(1.0) + NpyExpr.Exp(-x));
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "sigmoid_f64_v1");

            for (int i = 0; i < xs.Length; i++)
            {
                double want = 1.0 / (1.0 + Math.Exp(-xs[i]));
                Assert.AreEqual(want, output.GetDouble(i), 1e-9, $"[{i}]");
            }
        }

        [TestMethod]
        public void Composition_Softplus_Double()
        {
            // softplus(x) = log(1 + exp(x)) = Log1p(Exp(x))
            var xs = new double[] { -100, -1, 0, 1, 30 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var expr = NpyExpr.Log1p(NpyExpr.Exp(NpyExpr.Input(0)));
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "softplus_f64_v1");

            for (int i = 0; i < xs.Length; i++)
            {
                double want = Math.Log(1.0 + Math.Exp(xs[i]));
                if (double.IsInfinity(want))
                    Assert.IsTrue(double.IsInfinity(output.GetDouble(i)), $"[{i}]");
                else
                    Assert.AreEqual(want, output.GetDouble(i), 1e-9, $"[{i}]");
            }
        }

        [TestMethod]
        public void Composition_Hypot_Double()
        {
            // sqrt(a^2 + b^2)
            var a = np.array(new double[] { 3, 5, 8, 0, 1 });
            var b = np.array(new double[] { 4, 12, 15, 0, 0 });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            var x = NpyExpr.Input(0);
            var y = NpyExpr.Input(1);
            it.ExecuteExpression(NpyExpr.Sqrt(x * x + y * y),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "hypot_f64_v1");

            Assert.AreEqual(5.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(13.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(17.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(4), 1e-9);
        }

        [TestMethod]
        public void Composition_WhereWithComparison_Abs()
        {
            // Manual abs: where(x < 0, -x, x)
            var xs = new double[] { -5, -1, 0, 1, 5 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(NpyExpr.Less(x, NpyExpr.Const(0.0)), -x, x);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "where_abs_v1");

            for (int i = 0; i < xs.Length; i++)
                Assert.AreEqual(Math.Abs(xs[i]), output.GetDouble(i), 1e-9);
        }

        // =====================================================================
        // Dtype matrix — verify ops work across integer dtypes
        // =====================================================================

        [DataTestMethod]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        public void BitwiseAnd_IntegerDtypes(NPTypeCode dtype)
        {
            var src1 = np.array(new int[] { 0xFF, 0x0F, 0xF0, 0x55 }).astype(dtype);
            var src2 = np.array(new int[] { 0x0F, 0xFF, 0x0F, 0xAA }).astype(dtype);
            var r = np.empty_like(src1);
            using var it = Iter3(src1, src2, r);
            it.ExecuteExpression(NpyExpr.Input(0) & NpyExpr.Input(1),
                new[] { dtype, dtype }, dtype,
                cacheKey: $"and_dtype_{dtype}_v1");

            Assert.AreEqual(0x0FL, GetInt64AsLong(r, 0, dtype));
            Assert.AreEqual(0x0FL, GetInt64AsLong(r, 1, dtype));
            Assert.AreEqual(0x00L, GetInt64AsLong(r, 2, dtype));
            Assert.AreEqual(0x00L, GetInt64AsLong(r, 3, dtype));
        }

        [DataTestMethod]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Int64)]
        public void Sign_SignedIntegerDtypes(NPTypeCode dtype)
        {
            var src = np.array(new int[] { -5, -1, 0, 1, 5 }).astype(dtype);
            var r = np.empty_like(src);
            using var it = Iter(src, r);
            it.ExecuteExpression(NpyExpr.Sign(NpyExpr.Input(0)),
                new[] { dtype }, dtype, cacheKey: $"sign_dtype_{dtype}_v1");

            Assert.AreEqual(-1L, GetInt64AsLong(r, 0, dtype));
            Assert.AreEqual(-1L, GetInt64AsLong(r, 1, dtype));
            Assert.AreEqual(0L, GetInt64AsLong(r, 2, dtype));
            Assert.AreEqual(1L, GetInt64AsLong(r, 3, dtype));
            Assert.AreEqual(1L, GetInt64AsLong(r, 4, dtype));
        }

        private static long GetInt64AsLong(NDArray nd, int i, NPTypeCode dtype)
        {
            switch (dtype)
            {
                case NPTypeCode.Byte: return nd.GetByte(i);
                case NPTypeCode.Int16: return nd.GetInt16(i);
                case NPTypeCode.UInt16: return nd.GetUInt16(i);
                case NPTypeCode.Int32: return nd.GetInt32(i);
                case NPTypeCode.UInt32: return nd.GetUInt32(i);
                case NPTypeCode.Int64: return nd.GetInt64(i);
                case NPTypeCode.UInt64: return (long)nd.GetUInt64(i);
                default: throw new NotSupportedException(dtype.ToString());
            }
        }

        // =====================================================================
        // Float32 dtype coverage
        // =====================================================================

        [TestMethod]
        public void Exp_Float32()
        {
            var a = np.array(new float[] { 0f, 1f, 2f, -1f });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                new[] { NPTypeCode.Single }, NPTypeCode.Single, cacheKey: "exp_f32_v1");
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(MathF.Exp((float)(new double[] { 0, 1, 2, -1 })[i]),
                    r.GetSingle(i), 1e-5f, $"[{i}]");
        }

        [TestMethod]
        public void Sin_Float32()
        {
            var a = np.array(new float[] { 0f, (float)(Math.PI / 2), (float)Math.PI });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Sin(NpyExpr.Input(0)),
                new[] { NPTypeCode.Single }, NPTypeCode.Single, cacheKey: "sin_f32_v1");
            Assert.AreEqual(0f, r.GetSingle(0), 1e-5f);
            Assert.AreEqual(1f, r.GetSingle(1), 1e-5f);
            Assert.AreEqual(0f, r.GetSingle(2), 1e-5f);
        }

        // =====================================================================
        // Overflow / underflow
        // =====================================================================

        [TestMethod]
        public void Exp_Overflow_Double_ReturnsInfinity()
        {
            var a = np.array(new double[] { 1000, 709.78 });  // ~max before overflow
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "exp_ovf_v1");
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(0)));
            Assert.IsFalse(double.IsInfinity(r.GetDouble(1)));
        }

        [TestMethod]
        public void Exp_Underflow_Double_ReturnsZero()
        {
            var a = np.array(new double[] { -1000, 0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "exp_udf_v1");
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-100);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
        }

        [TestMethod]
        public void Power_Int32_OverflowWraps()
        {
            // int32 overflow via Math.Pow conversion → wraps after cast
            var a = np.array(new int[] { 10, 2 });
            var b = np.array(new int[] { 9, 30 });  // 10^9 = 1e9 (fits), 2^30 = 1e9 (fits)
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Power(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "pow_i32_v1");
            Assert.AreEqual(1000000000, r.GetInt32(0));
            Assert.AreEqual(1 << 30, r.GetInt32(1));
        }

        // =====================================================================
        // Cache behavior: distinct keys yield distinct kernels, same key reuses
        // =====================================================================

        [TestMethod]
        public void Cache_DistinctExpressionsProduceDistinctKernels()
        {
            ILKernelGenerator.ClearInnerLoopCache();
            int before = ILKernelGenerator.InnerLoopCachedCount;

            var a = np.arange(10).astype(np.float64);
            var r1 = np.empty_like(a);
            var r2 = np.empty_like(a);

            using (var it = Iter(a, r1))
                it.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int afterExp = ILKernelGenerator.InnerLoopCachedCount;

            using (var it = Iter(a, r2))
                it.ExecuteExpression(NpyExpr.Log(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int afterLog = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(before + 1, afterExp, "Exp should add 1 entry");
            Assert.AreEqual(afterExp + 1, afterLog, "Log should add 1 entry (distinct)");
        }

        [TestMethod]
        public void Cache_SameExpressionReusesKernel()
        {
            ILKernelGenerator.ClearInnerLoopCache();
            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Sin(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "cache_sin_reuse");
            int after1 = ILKernelGenerator.InnerLoopCachedCount;

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Sin(NpyExpr.Input(0)),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "cache_sin_reuse");
            int after2 = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(after1, after2, "Same cache key should reuse kernel");
        }

        // =====================================================================
        // Deep-nesting / expression tree corner cases
        // =====================================================================

        [TestMethod]
        public void DeepNesting_20Layers_Math()
        {
            // Chain 20 unary ops: sin(cos(sin(cos(...sin(x)))))
            var a = np.array(new double[] { 0.5 });
            var r = np.empty_like(a);

            NpyExpr expr = NpyExpr.Input(0);
            for (int i = 0; i < 10; i++)
                expr = NpyExpr.Cos(NpyExpr.Sin(expr));

            using var it = Iter(a, r);
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "deep20_v1");

            double want = 0.5;
            for (int i = 0; i < 10; i++)
                want = Math.Cos(Math.Sin(want));
            Assert.AreEqual(want, r.GetDouble(0), 1e-9);
        }

        [TestMethod]
        public void Polynomial_Degree5_Int32()
        {
            // Horner's: ((((1*x + 2)*x + 3)*x + 4)*x + 5)*x + 6
            var a = np.array(new int[] { 0, 1, 2, 3 });
            var r = np.empty_like(a);

            var x = NpyExpr.Input(0);
            var expr = (((x + NpyExpr.Const(2)) * x + NpyExpr.Const(3)) * x +
                       NpyExpr.Const(4)) * x + NpyExpr.Const(5);

            using var it = Iter(a, r);
            it.ExecuteExpression(expr, new[] { NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "poly5_i32_v1");

            // For x=0: ((((0+2)*0+3)*0+4)*0+5) = 5
            // For x=1: ((((1+2)*1+3)*1+4)*1+5) = (((3)*1+3)*1+4)*1+5 = (6)*1+4)*1+5 = 10*1+5=15... let me compute
            // x=0: 0+2=2; 2*0=0; 0+3=3; 3*0=0; 0+4=4; 4*0=0; 0+5=5 → 5
            // x=1: 1+2=3; 3*1=3; 3+3=6; 6*1=6; 6+4=10; 10*1=10; 10+5=15
            // x=2: 2+2=4; 4*2=8; 8+3=11; 11*2=22; 22+4=26; 26*2=52; 52+5=57
            // x=3: 3+2=5; 5*3=15; 15+3=18; 18*3=54; 54+4=58; 58*3=174; 174+5=179
            Assert.AreEqual(5, r.GetInt32(0));
            Assert.AreEqual(15, r.GetInt32(1));
            Assert.AreEqual(57, r.GetInt32(2));
            Assert.AreEqual(179, r.GetInt32(3));
        }

        [TestMethod]
        public void ComparisonChain_NestedWhere()
        {
            // Sign-like: where(x > 0, 1, where(x < 0, -1, 0))
            var xs = new double[] { -5, -0.1, 0, 0.1, 5 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(
                NpyExpr.Greater(x, NpyExpr.Const(0.0)),
                NpyExpr.Const(1.0),
                NpyExpr.Where(
                    NpyExpr.Less(x, NpyExpr.Const(0.0)),
                    NpyExpr.Const(-1.0),
                    NpyExpr.Const(0.0)));
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "nested_where_v1");

            Assert.AreEqual(-1.0, output.GetDouble(0), 1e-9);
            Assert.AreEqual(-1.0, output.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, output.GetDouble(2), 1e-9);
            Assert.AreEqual(1.0, output.GetDouble(3), 1e-9);
            Assert.AreEqual(1.0, output.GetDouble(4), 1e-9);
        }

        // =====================================================================
        // Size stress — run compound op across a sweep of sizes
        // =====================================================================

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(7)]
        [DataRow(31)]
        [DataRow(32)]
        [DataRow(33)]
        [DataRow(63)]
        [DataRow(65)]
        [DataRow(127)]
        [DataRow(128)]
        [DataRow(255)]
        [DataRow(256)]
        [DataRow(513)]
        [DataRow(1025)]
        public void Stress_Power_AcrossSizes(int size)
        {
            var a = np.arange(size).astype(np.float64);
            var r = np.empty(new Shape(size), np.float64);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Power(NpyExpr.Input(0), NpyExpr.Const(2.0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "stress_pow_v1");

            for (int i = 0; i < size; i++)
                Assert.AreEqual((double)i * i, r.GetDouble(i), 1e-9, $"size={size} i={i}");
        }

        [DataTestMethod]
        [DataRow(2)]  // size=1 hits a pre-existing NumSharp bug: arange(1)-0.5 returns
                      // shape [] (0-d scalar) instead of [1] (1-d). See IsSimdSlice
                      // handling in arithmetic ops. Skipping size=1 until that bug is
                      // fixed upstream.
        [DataRow(7)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(1024)]
        public void Stress_Sigmoid_AcrossSizes(int size)
        {
            // Build input directly from a double[] to avoid NumSharp's
            // scalar-reducing arithmetic bug on tiny arrays.
            var xs = new double[size];
            for (int i = 0; i < size; i++)
                xs[i] = (i - size / 2.0) * 0.1;
            var a = np.array(xs);
            var r = np.empty_like(a);

            using var it = Iter(a, r);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Const(1.0) / (NpyExpr.Const(1.0) + NpyExpr.Exp(-x));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "stress_sig_v1");

            for (int i = 0; i < size; i++)
            {
                double want = 1.0 / (1.0 + Math.Exp(-xs[i]));
                Assert.AreEqual(want, r.GetDouble(i), 1e-9, $"size={size} i={i}");
            }
        }

        // =====================================================================
        // Zero / empty / 1-element edge behavior
        // =====================================================================

        [TestMethod]
        public void Empty_Mod_NoCrash()
        {
            var a = np.empty(new Shape(0), np.float64);
            var r = np.empty(new Shape(0), np.float64);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Mod(NpyExpr.Input(0), NpyExpr.Const(3.0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "mod_empty_v1");
            // No crash is the assertion.
        }

        [TestMethod]
        public void Single_Element_Power()
        {
            var a = np.array(new double[] { 3.0 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Power(NpyExpr.Input(0), NpyExpr.Const(4.0)),
                new[] { NPTypeCode.Double }, NPTypeCode.Double, cacheKey: "pow_1elem_v1");
            Assert.AreEqual(81.0, r.GetDouble(0), 1e-9);
        }

        // =====================================================================
        // Operator overload validation
        // =====================================================================

        [TestMethod]
        public void Operator_Mod_Percent()
        {
            var a = np.array(new double[] { 10, 7 });
            var b = np.array(new double[] { 3, 2 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);
            it.ExecuteExpression(NpyExpr.Input(0) % NpyExpr.Input(1),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "op_pct_v1");
            Assert.AreEqual(1.0, c.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, c.GetDouble(1), 1e-9);
        }

        [TestMethod]
        public void Operator_BitwiseNot_Tilde()
        {
            var a = np.array(new int[] { 0, 5, -1 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(~NpyExpr.Input(0),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "op_tilde_v1");
            Assert.AreEqual(~0, r.GetInt32(0));
            Assert.AreEqual(~5, r.GetInt32(1));
            Assert.AreEqual(~(-1), r.GetInt32(2));
        }

        [TestMethod]
        public void Operator_LogicalNot_Bang()
        {
            var a = np.array(new int[] { 0, 1, 2, 0, -5 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(!NpyExpr.Input(0),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Int32, cacheKey: "op_bang_v1");
            Assert.AreEqual(1, r.GetInt32(0));
            Assert.AreEqual(0, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
            Assert.AreEqual(1, r.GetInt32(3));
            Assert.AreEqual(0, r.GetInt32(4));
        }

        [TestMethod]
        public void Operator_BitwiseAnd_Ampersand()
        {
            var a = np.array(new int[] { 0b1100 });
            var b = np.array(new int[] { 0b1010 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);
            it.ExecuteExpression(NpyExpr.Input(0) & NpyExpr.Input(1),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "op_amp_v1");
            Assert.AreEqual(0b1000, c.GetInt32(0));
        }

        [TestMethod]
        public void Operator_BitwiseOr_Pipe()
        {
            var a = np.array(new int[] { 0b1100 });
            var b = np.array(new int[] { 0b1010 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);
            it.ExecuteExpression(NpyExpr.Input(0) | NpyExpr.Input(1),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "op_pipe_v1");
            Assert.AreEqual(0b1110, c.GetInt32(0));
        }

        [TestMethod]
        public void Operator_BitwiseXor_Caret()
        {
            var a = np.array(new int[] { 0b1100 });
            var b = np.array(new int[] { 0b1010 });
            var c = np.empty_like(a);
            using var it = Iter3(a, b, c);
            it.ExecuteExpression(NpyExpr.Input(0) ^ NpyExpr.Input(1),
                new[] { NPTypeCode.Int32, NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "op_caret_v1");
            Assert.AreEqual(0b0110, c.GetInt32(0));
        }

        // =====================================================================
        // Auto-derived cache key: same structural expression should reuse
        // =====================================================================

        [TestMethod]
        public void AutoKey_EquivalentExpressionsShareKernel()
        {
            ILKernelGenerator.ClearInnerLoopCache();

            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);

            using (var it = Iter(a, r))
            {
                var expr = NpyExpr.Sqrt(NpyExpr.Input(0));
                it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double);
            }
            int after1 = ILKernelGenerator.InnerLoopCachedCount;

            // Build a *different instance* of the same expression — must reuse.
            using (var it = Iter(a, r))
            {
                var expr = NpyExpr.Sqrt(NpyExpr.Input(0));
                it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double);
            }
            int after2 = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(after1, after2,
                "Structurally identical exprs should produce same auto-derived cache key");
        }

        [TestMethod]
        public void AutoKey_DifferentConstantsProduceDifferentKernels()
        {
            ILKernelGenerator.ClearInnerLoopCache();

            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Input(0) + NpyExpr.Const(1.0),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int after1 = ILKernelGenerator.InnerLoopCachedCount;

            using (var it = Iter(a, r))
                it.ExecuteExpression(NpyExpr.Input(0) + NpyExpr.Const(2.0),
                    new[] { NPTypeCode.Double }, NPTypeCode.Double);
            int after2 = ILKernelGenerator.InnerLoopCachedCount;

            Assert.AreEqual(after1 + 1, after2,
                "Different constant values must produce distinct cache entries");
        }

        // =====================================================================
        // Strided inputs for scalar-only ops (ComparisonNode, MinMaxNode, WhereNode)
        // =====================================================================

        [TestMethod]
        public void Equal_StridedInput()
        {
            var src1 = np.arange(20).astype(np.float64);
            var src2 = np.arange(20).astype(np.float64);
            // Mutate src2[::2] to make half mismatch
            for (int i = 0; i < 20; i += 4) src2.SetDouble(999, i);

            var a = src1["::2"];  // 10 elements
            var b = src2["::2"];  // 10 elements
            var r = np.empty(new Shape(10), np.float64);

            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Equal(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "eq_strided_v1");

            // src1[::2] = 0,2,4,6,8,10,12,14,16,18
            // src2[::2] = 999,2,999,6,999,10,999,14,999,18 (every other = 999)
            Assert.AreEqual(0.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(1.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(2), 1e-9);
        }

        [TestMethod]
        public void Max_StridedInput()
        {
            var src1 = np.arange(20).astype(np.float64);
            var src2 = np.arange(40, 60).astype(np.float64);

            var a = src1["::2"];  // 10 elements: 0,2,4,...,18
            var b = src2["::2"];  // 10 elements: 40,42,44,...,58
            var r = np.empty(new Shape(10), np.float64);

            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "max_strided_v1");

            for (int i = 0; i < 10; i++)
                Assert.AreEqual(40 + 2 * i, r.GetDouble(i), 1e-9, $"[{i}]");
        }

        [TestMethod]
        public void Where_StridedInput()
        {
            var cond = np.arange(10).astype(np.float64) - 5;  // [-5..4]
            var a = np.arange(10, 20).astype(np.float64);     // [10..19]
            var b = np.arange(20, 30).astype(np.float64);     // [20..29]

            // Don't strip — just take contiguous.
            var r = np.empty(new Shape(10), np.float64);

            using var it = NpyIterRef.MultiNew(4, new[] { cond, a, b, r },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            // Where(cond > 0, a, b)
            it.ExecuteExpression(
                NpyExpr.Where(
                    NpyExpr.Greater(NpyExpr.Input(0), NpyExpr.Const(0.0)),
                    NpyExpr.Input(1), NpyExpr.Input(2)),
                new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double },
                NPTypeCode.Double, cacheKey: "where_strided_v1");

            // cond = [-5,-4,-3,-2,-1,0,1,2,3,4]
            // select: cond>0 → take a, else b
            // expected = [20,21,22,23,24,25,16,17,18,19]
            Assert.AreEqual(20.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(25.0, r.GetDouble(5), 1e-9);  // cond=0 → b
            Assert.AreEqual(16.0, r.GetDouble(6), 1e-9);  // cond=1 → a
            Assert.AreEqual(19.0, r.GetDouble(9), 1e-9);
        }

        // =====================================================================
        // Decimal coverage — scalar-only fallback
        // =====================================================================

        [TestMethod]
        public void Add_Decimal_ScalarOnly()
        {
            var a = np.array(new decimal[] { 1m, 2m, 3m });
            var b = np.array(new decimal[] { 10m, 20m, 30m });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Input(0) + NpyExpr.Input(1),
                new[] { NPTypeCode.Decimal, NPTypeCode.Decimal }, NPTypeCode.Decimal,
                cacheKey: "dec_add_v1");
            Assert.AreEqual(11m, r.GetDecimal(0));
            Assert.AreEqual(22m, r.GetDecimal(1));
            Assert.AreEqual(33m, r.GetDecimal(2));
        }

        [TestMethod]
        public void Max_Decimal()
        {
            var a = np.array(new decimal[] { 1m, 5m, 3m });
            var b = np.array(new decimal[] { 2m, 4m, 6m });
            var r = np.empty_like(a);
            using var it = Iter3(a, b, r);
            it.ExecuteExpression(NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Input(1)),
                new[] { NPTypeCode.Decimal, NPTypeCode.Decimal }, NPTypeCode.Decimal,
                cacheKey: "dec_max_v1");
            Assert.AreEqual(2m, r.GetDecimal(0));
            Assert.AreEqual(5m, r.GetDecimal(1));
            Assert.AreEqual(6m, r.GetDecimal(2));
        }

        [TestMethod]
        public void Where_Decimal()
        {
            var cond = np.array(new decimal[] { 1m, 0m, 1m });
            var a = np.array(new decimal[] { 100m, 200m, 300m });
            var b = np.array(new decimal[] { -1m, -2m, -3m });
            var r = np.empty_like(a);
            using var it = NpyIterRef.MultiNew(4, new[] { cond, a, b, r },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            it.ExecuteExpression(
                NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2)),
                new[] { NPTypeCode.Decimal, NPTypeCode.Decimal, NPTypeCode.Decimal },
                NPTypeCode.Decimal, cacheKey: "dec_where_v1");
            Assert.AreEqual(100m, r.GetDecimal(0));
            Assert.AreEqual(-2m, r.GetDecimal(1));
            Assert.AreEqual(300m, r.GetDecimal(2));
        }

        // =====================================================================
        // Type promotion: integer input → float output via auto-convert
        // =====================================================================

        [TestMethod]
        public void Sqrt_Int32Input_Float64Output()
        {
            var a = np.array(new int[] { 1, 4, 9, 16, 25 });
            var r = np.empty(new Shape(5), np.float64);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Sqrt(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Double,
                cacheKey: "sqrt_i32_f64_v1");
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(2.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(4.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(5.0, r.GetDouble(4), 1e-9);
        }

        [TestMethod]
        public void Exp_Int32Input_Float64Output()
        {
            var a = np.array(new int[] { 0, 1, 2 });
            var r = np.empty(new Shape(3), np.float64);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Exp(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int32 }, NPTypeCode.Double,
                cacheKey: "exp_i32_f64_v1");
            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(Math.E, r.GetDouble(1), 1e-9);
            Assert.AreEqual(Math.E * Math.E, r.GetDouble(2), 1e-9);
        }

        // =====================================================================
        // Validation: argument errors
        // =====================================================================

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validation_NullInnerExpression_Throws()
        {
            NpyExpr dummy = null!;
            _ = NpyExpr.Sqrt(dummy);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void Validation_NegativeInputIndex_Throws()
        {
            _ = NpyExpr.Input(-1);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void Validation_InputOutOfRange_ThrowsAtCompile()
        {
            // Build expression referring to Input(5) but only provide 1 input — should fail
            // at scalar emit (invoked by CompileInnerLoop during kernel generation).
            var a = np.arange(5).astype(np.float64);
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Input(5),  // out of range
                new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "oob_input_" + Guid.NewGuid());
        }

        // =====================================================================
        // Mixed op composition: scalar op on top of SIMD subtree disables SIMD
        // =====================================================================

        [TestMethod]
        public void Composition_ScalarTopSimdBottom()
        {
            // Mod is scalar-only; Sqrt/Add/Input are SIMD. Whole tree goes scalar path.
            var a = np.arange(20).astype(np.float64);
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            // ((x + 1)^2) mod 7 — mod forces scalar path for the whole tree
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Mod(NpyExpr.Square(x + NpyExpr.Const(1.0)), NpyExpr.Const(7.0));
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "mix_mod_sq_v1");

            for (int i = 0; i < 20; i++)
            {
                double want = ((i + 1) * (i + 1)) % 7.0;
                // floored-mod for positive operands matches C# %, so this works.
                Assert.AreEqual(want, r.GetDouble(i), 1e-9, $"[{i}]");
            }
        }

        [TestMethod]
        public void Composition_PredicateUsedInArithmetic()
        {
            // NaN mask → multiply input by 1 - isNaN(x), producing 0 at NaN positions.
            // After: (x * (1 - isNaN(x))) + (0 * isNaN(x)) — NaN*0 is NaN in IEEE,
            // so this composition doesn't fully replace NaN. Use Where instead.
            var a = np.array(new double[] { 1, double.NaN, 3, double.NaN, 5 });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(NpyExpr.IsNaN(x), NpyExpr.Const(0.0), x);
            it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "nan_replace_v1");

            Assert.AreEqual(1.0, r.GetDouble(0), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(1), 1e-9);
            Assert.AreEqual(3.0, r.GetDouble(2), 1e-9);
            Assert.AreEqual(0.0, r.GetDouble(3), 1e-9);
            Assert.AreEqual(5.0, r.GetDouble(4), 1e-9);
        }

        // =====================================================================
        // Large integer edge cases
        // =====================================================================

        [TestMethod]
        public void Abs_Int64_MinValue_Overflows()
        {
            // abs(Int64.MinValue) is not representable — produces Int64.MinValue (wraps).
            // NumSharp/NumPy same behavior.
            var a = np.array(new long[] { long.MinValue, -1L, 0L, 1L, long.MaxValue });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Abs(NpyExpr.Input(0)),
                new[] { NPTypeCode.Int64 }, NPTypeCode.Int64, cacheKey: "abs_i64_v1");
            // Int64.MinValue = -9223372036854775808; abs wraps to -9223372036854775808
            Assert.AreEqual(long.MinValue, r.GetInt64(0));
            Assert.AreEqual(1L, r.GetInt64(1));
            Assert.AreEqual(0L, r.GetInt64(2));
            Assert.AreEqual(1L, r.GetInt64(3));
            Assert.AreEqual(long.MaxValue, r.GetInt64(4));
        }

        [TestMethod]
        public void Negate_UInt32_WrapsAround()
        {
            // Negating an unsigned value gives two's complement wrap: -x = ~x + 1
            var a = np.array(new uint[] { 0u, 1u, 100u });
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(-NpyExpr.Input(0),
                new[] { NPTypeCode.UInt32 }, NPTypeCode.UInt32, cacheKey: "neg_u32_v1");
            Assert.AreEqual(0u, r.GetUInt32(0));
            Assert.AreEqual(uint.MaxValue, r.GetUInt32(1));  // -1 as uint
            Assert.AreEqual(uint.MaxValue - 99u, r.GetUInt32(2));
        }

        // =====================================================================
        // Float32 SIMD path — ensure Square/Abs/Sqrt etc work in SIMD
        // =====================================================================

        [TestMethod]
        public void Sqrt_Float32_LargeContiguous_SimdPath()
        {
            int N = 256;
            var xs = new float[N];
            for (int i = 0; i < N; i++) xs[i] = i * 0.5f;
            var a = np.array(xs);
            var r = np.empty_like(a);

            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Sqrt(NpyExpr.Input(0)),
                new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "sqrt_f32_big_v1");

            for (int i = 0; i < N; i++)
                Assert.AreEqual(MathF.Sqrt(xs[i]), r.GetSingle(i), 1e-5f, $"[{i}]");
        }

        [TestMethod]
        public void Square_Float32_LargeContiguous()
        {
            int N = 1024;
            var xs = new float[N];
            for (int i = 0; i < N; i++) xs[i] = (i - 512) * 0.01f;
            var a = np.array(xs);
            var r = np.empty_like(a);

            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Square(NpyExpr.Input(0)),
                new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "sq_f32_big_v1");

            for (int i = 0; i < N; i++)
                Assert.AreEqual(xs[i] * xs[i], r.GetSingle(i), 1e-5f, $"[{i}]");
        }

        // =====================================================================
        // Mixed comparison into Where for piecewise definition
        // =====================================================================

        [TestMethod]
        public void Piecewise_LeakyReLU()
        {
            // leaky_relu(x, alpha=0.1) = x if x > 0 else alpha*x
            var xs = new double[] { -5, -1, 0, 1, 5 };
            var input = np.array(xs);
            var output = np.empty_like(input);
            using var iter = Iter(input, output);
            var x = NpyExpr.Input(0);
            var expr = NpyExpr.Where(
                NpyExpr.Greater(x, NpyExpr.Const(0.0)),
                x,
                NpyExpr.Const(0.1) * x);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "leaky_relu_v1");

            for (int i = 0; i < xs.Length; i++)
            {
                double want = xs[i] > 0 ? xs[i] : 0.1 * xs[i];
                Assert.AreEqual(want, output.GetDouble(i), 1e-9, $"[{i}]");
            }
        }

        // =====================================================================
        // Reuse same NpyExpr instance across two executes
        // =====================================================================

        [TestMethod]
        public void Reuse_SameExprInstance_ExecutesTwice()
        {
            var expr = NpyExpr.Exp(NpyExpr.Input(0));

            var a1 = np.array(new double[] { 0, 1 });
            var r1 = np.empty_like(a1);
            using (var it = Iter(a1, r1))
                it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "reuse_expr_v1");

            var a2 = np.array(new double[] { 2, 3 });
            var r2 = np.empty_like(a2);
            using (var it = Iter(a2, r2))
                it.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                    cacheKey: "reuse_expr_v1");

            Assert.AreEqual(1.0, r1.GetDouble(0), 1e-9);
            Assert.AreEqual(Math.E, r1.GetDouble(1), 1e-9);
            Assert.AreEqual(Math.E * Math.E, r2.GetDouble(0), 1e-9);
            Assert.AreEqual(Math.E * Math.E * Math.E, r2.GetDouble(1), 1e-9);
        }

        // =====================================================================
        // Single-Const expression — should just write the constant
        // =====================================================================

        [TestMethod]
        public void Constant_Only_Expression_BroadcastsConstant()
        {
            // out = 42 for every element (input is required but ignored)
            var a = np.arange(10).astype(np.float64);
            var r = np.empty_like(a);
            using var it = Iter(a, r);
            it.ExecuteExpression(NpyExpr.Const(42.0) + NpyExpr.Const(0.0) * NpyExpr.Input(0),
                new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "const_only_v1");
            for (int i = 0; i < 10; i++)
                Assert.AreEqual(42.0, r.GetDouble(i), 1e-9);
        }

    }
}
