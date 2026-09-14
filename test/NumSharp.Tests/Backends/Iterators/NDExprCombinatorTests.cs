using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// Gate for the C6 macro / decision combinators (NDExpr.Combinators.cs). Each combinator is a pure
    /// composition of primitive nodes, so its parity contract is its composition's NumPy chain; these
    /// tests are the metamorphic gate — the fused <c>np.evaluate(NDExpr.X(...))</c> is compared to the
    /// eager <c>np.*</c> reference (or a hand-verified NumPy 2.4.2 output). No Python at test time.
    /// Discrete (bool/int) results are checked bit-exact; transcendental activations within allclose.
    /// </summary>
    [TestClass]
    public class NDExprCombinatorTests
    {
        // ---- helpers ----------------------------------------------------------

        /// <summary>Assert a numeric result matches <paramref name="want"/> within allclose (upcasts int/float alike).</summary>
        private static void V(NDArray got, params double[] want)
        {
            var w = np.array(want);
            Assert.AreEqual(w.size, got.size, $"size: got {got.size} want {w.size} [{got}]");
            Assert.IsTrue(np.allclose(got, w), $"got {got} want {w}");
        }

        /// <summary>Assert a boolean result matches <paramref name="want"/> bit-exact.</summary>
        private static void B(NDArray got, params bool[] want)
            => Assert.IsTrue(np.array_equal(got, np.array(want)), $"got {got} want [{string.Join(",", want)}]");

        /// <summary>Assert a numeric result equals a reference array bit-exact (array_equal).</summary>
        private static void Exact(NDArray got, NDArray want)
            => Assert.IsTrue(np.array_equal(got, want), $"got {got} want {want}");

        /// <summary>Assert a numeric result is allclose to a reference array.</summary>
        private static void Close(NDArray got, NDArray want)
            => Assert.IsTrue(np.allclose(got, want), $"got {got} want {want}");

        // =====================================================================
        // Selection & masking
        // =====================================================================

        [TestMethod]
        public void If_SelectsByCondition()
        {
            var x = np.array(new double[] { -3, -1, 0, 2, 5 });
            var r = np.evaluate(NDExpr.If(NDExpr.GreaterEqual((NDExpr)x, 0.0), x, -(NDExpr)x));
            V(r, 3, 1, 0, 2, 5);                                       // == abs(x)
        }

        [TestMethod]
        public void IfNot_SwapsArms()
        {
            var x = np.array(new double[] { -3, -1, 0, 2, 5 });
            var r = np.evaluate(NDExpr.IfNot(NDExpr.GreaterEqual((NDExpr)x, 0.0), 1.0, -1.0));
            V(r, 1, 1, -1, -1, -1);                                    // 1 where x<0, else -1
        }

        [TestMethod]
        public void When_GatesToZero()
        {
            var x = np.array(new double[] { -3, -1, 0, 2, 5 });
            var r = np.evaluate(NDExpr.When(NDExpr.Greater((NDExpr)x, 0.0), x));
            V(r, 0, 0, 0, 2, 5);
        }

        [TestMethod]
        public void Unless_GatesNegated()
        {
            var x = np.array(new double[] { -3, -1, 0, 2, 5 });
            var r = np.evaluate(NDExpr.Unless(NDExpr.Greater((NDExpr)x, 0.0), x));
            V(r, -3, -1, 0, 0, 0);
        }

        [TestMethod]
        public void Switch_FirstMatchWins()
        {
            var s = np.array(new double[] { -5, 0.5, 3, 42 });
            var r = np.evaluate(NDExpr.Switch(0.0,
                (NDExpr.GreaterEqual((NDExpr)s, 10.0), 3.0),
                (NDExpr.GreaterEqual((NDExpr)s, 1.0), 2.0),
                (NDExpr.GreaterEqual((NDExpr)s, 0.0), 1.0)));
            V(r, 0, 1, 2, 3);
        }

        [TestMethod]
        public void Mux_SelectsByIndex()
        {
            var idx = np.array(new double[] { 0, 1, 2, 1 });
            var r = np.evaluate(NDExpr.Mux(idx, 10, 20, 30));
            V(r, 10, 20, 30, 20);
        }

        [TestMethod]
        public void Switch_NullDefault_Throws()
            => Assert.ThrowsException<ArgumentNullException>(() => NDExpr.Switch(null, (NDExpr.Const(true), NDExpr.Const(1))));

        [TestMethod]
        public void Mux_NullValue_Throws()
            => Assert.ThrowsException<ArgumentNullException>(() => NDExpr.Mux(NDExpr.Const(0), NDExpr.Const(1), null));

        // =====================================================================
        // Clamp & saturation
        // =====================================================================

        [TestMethod]
        public void ClampMin_ClampMax_Saturate()
        {
            var x = np.array(new double[] { -2, -0.5, 0.3, 1.5, 9 });
            V(np.evaluate(NDExpr.ClampMin(x, 0.0)), 0, 0, 0.3, 1.5, 9);
            V(np.evaluate(NDExpr.ClampMax(x, 1.0)), -2, -0.5, 0.3, 1, 1);
            V(np.evaluate(NDExpr.Saturate(x)), 0, 0, 0.3, 1, 1);
        }

        // =====================================================================
        // Robustness (NaN / finite)
        // =====================================================================

        [TestMethod]
        public void NanTo_ReplacesNaN()
        {
            double nan = double.NaN;
            var x = np.array(new double[] { 1, nan, 3, nan });
            V(np.evaluate(NDExpr.NanTo(x, -1.0)), 1, -1, 3, -1);
        }

        [TestMethod]
        public void Coalesce_TakesFirstFinite()
        {
            double nan = double.NaN, inf = double.PositiveInfinity;
            var f = np.array(new double[] { 1, nan, inf, 4 });
            var g = np.array(new double[] { 7, 7, 7, 7 });
            V(np.evaluate(NDExpr.Coalesce(f, g)), 1, 7, 7, 4);
        }

        // =====================================================================
        // Activations — compared to eager np.* references
        // =====================================================================

        [TestMethod]
        public void Relu_And_LeakyRelu()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 0.5, 2 });
            Close(np.evaluate(NDExpr.Relu(x)), np.maximum(x, 0.0));
            Close(np.evaluate(NDExpr.LeakyRelu(x, 0.1)), np.where(x > 0.0, x, 0.1 * x));
        }

        [TestMethod]
        public void Sigmoid_MatchesLogisticFormula()
        {
            var x = np.array(new double[] { -4, -1, 0, 1, 4 });
            Close(np.evaluate(NDExpr.Sigmoid(x)), 1.0 / (1.0 + np.exp(-x)));
        }

        [TestMethod]
        public void Swish_IsXTimesSigmoid()
        {
            var x = np.array(new double[] { -4, -1, 0, 1, 4 });
            Close(np.evaluate(NDExpr.Swish(x)), x * (1.0 / (1.0 + np.exp(-x))));
        }

        [TestMethod]
        public void Softplus_MatchesNaiveOnModerateRange_AndIsStableAtLargeX()
        {
            var x = np.array(new double[] { -5, -1, 0, 1, 5 });
            Close(np.evaluate(NDExpr.Softplus(x)), np.log(1.0 + np.exp(x)));   // independent (naive) reference

            // Stability: the naive form overflows near x≈710; the stable composition must not.
            var big = np.array(new double[] { 800, 1000 });
            var r = np.evaluate(NDExpr.Softplus(big));
            V(r, 800, 1000);                                                    // softplus(x) ≈ x for large x
        }

        [TestMethod]
        public void Elu_MatchesWhereForm()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 0.5, 2 });
            // Elu is defined over exp(x)-1 (vectorizable), not expm1; the eager reference matches that.
            Close(np.evaluate(NDExpr.Elu(x, 1.0)), np.where(x > 0.0, x, 1.0 * (np.exp(x) - 1.0)));
        }

        [TestMethod]
        public void Gelu_MatchesTanhApproximation()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 0.5, 2 });
            var reference = 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            Close(np.evaluate(NDExpr.Gelu(x)), reference);
        }

        [TestMethod]
        public void HardSigmoid_MatchesClippedAffine()
        {
            var x = np.array(new double[] { -9, -3, 0, 3, 9 });
            Close(np.evaluate(NDExpr.HardSigmoid(x)), np.clip(x / 6.0 + 0.5, 0.0, 1.0));
        }

        [TestMethod]
        public void Step_IsUnitStep()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 0.5, 2 });
            V(np.evaluate(NDExpr.Step(x)), 0, 0, 0, 1, 1);
        }

        /// <summary>
        /// The ML activations at float32 — the vectorizing hot path (min/max via hardware intrinsics,
        /// exp/log/tanh via the ported NDFloatMath SIMD kernels). Elu/Softplus MUST stay composed over
        /// Exp/Log (never Expm1/Log1p, which have no SIMD kernel and would scalarize the whole tree);
        /// this exercises that path and pins correctness against the eager float32 chain.
        /// </summary>
        [TestMethod]
        public void MlActivations_Float32_VectorPath_Correct()
        {
            var x = np.array(new float[] { -3f, -1f, -0.25f, 0f, 0.5f, 2f, 5f });
            Close(np.evaluate(NDExpr.Relu(x)), np.maximum(x, 0f));
            Close(np.evaluate(NDExpr.LeakyRelu(x, 0.1f)), np.where(x > 0f, x, 0.1f * x));
            Close(np.evaluate(NDExpr.Sigmoid(x)), 1f / (1f + np.exp(-x)));
            Close(np.evaluate(NDExpr.Swish(x)), x * (1f / (1f + np.exp(-x))));
            Close(np.evaluate(NDExpr.Elu(x, 1f)), np.where(x > 0f, x, 1f * (np.exp(x) - 1f)));
            Close(np.evaluate(NDExpr.Softplus(x)), np.maximum(x, 0f) + np.log(1f + np.exp(-np.abs(x))));
            Close(np.evaluate(NDExpr.HardSigmoid(x)), np.clip(x / 6f + 0.5f, 0f, 1f));
        }

        // =====================================================================
        // Boolean logic
        // =====================================================================

        [TestMethod]
        public void BooleanLogic_Family()
        {
            var A = np.array(new[] { true, true, false, false });
            var B_ = np.array(new[] { true, false, true, false });
            var C = np.array(new[] { true, false, false, true });
            B(np.evaluate(NDExpr.Nand(A, B_)), false, true, true, true);
            B(np.evaluate(NDExpr.Nor(A, B_)), false, false, false, true);
            B(np.evaluate(NDExpr.Xnor(A, B_)), true, false, false, true);
            B(np.evaluate(NDExpr.Implies(A, B_)), true, false, true, true);
            B(np.evaluate(NDExpr.Majority3(A, B_, C)), true, false, false, false);
        }

        // =====================================================================
        // Predicates
        // =====================================================================

        [TestMethod]
        public void Predicates_PositiveNegativeInteger()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 2, 3.5 });
            B(np.evaluate(NDExpr.IsPositive(x)), false, false, false, true, true);
            B(np.evaluate(NDExpr.IsNegative(x)), true, true, false, false, false);
            B(np.evaluate(NDExpr.IsInteger(x)), true, false, true, true, false);
        }

        [TestMethod]
        public void IsClose_UsesNumpyRelation()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });
            var b = np.array(new double[] { 1.0000001, 2.5, 3.0 });
            B(np.evaluate(NDExpr.IsClose(a, b, 1e-3, 0.0)), true, false, true);
        }

        [TestMethod]
        public void IsClose_InfEqualsInf_NaNCloseToNothing()
        {
            double inf = double.PositiveInfinity, nan = double.NaN;
            var a = np.array(new double[] { inf, inf, nan });
            var b = np.array(new double[] { inf, -inf, nan });
            B(np.evaluate(NDExpr.IsClose(a, b, 1e-5, 1e-8)), true, false, false);
        }

        [TestMethod]
        public void SameSign_GroupsByNegativity()
        {
            var p = np.array(new double[] { 3, -1, 4, -2 });
            var q = np.array(new double[] { 5, -8, -1, 0 });
            B(np.evaluate(NDExpr.SameSign(p, q)), true, true, false, false);
        }

        [TestMethod]
        public void Between_IsInclusive()
        {
            var x = np.array(new double[] { -2, -0.5, 0, 0.5, 2 });
            B(np.evaluate(NDExpr.Between(x, -1.0, 1.0)), false, true, true, true, false);
        }

        // =====================================================================
        // Directional / sign
        // =====================================================================

        [TestMethod]
        public void Cmp_ThreeWay()
        {
            var a = np.array(new double[] { 1, 2, 3, 4 });
            var b = np.array(new double[] { 3, 2, 1, 4 });
            V(np.evaluate(NDExpr.Cmp(a, b)), -1, 0, 1, 0);
        }

        [TestMethod]
        public void Heaviside_WithValueAtZero()
        {
            var x = np.array(new double[] { -2, 0, 3 });
            V(np.evaluate(NDExpr.Heaviside(x, 0.5)), 0, 0.5, 1);
        }

        [TestMethod]
        public void StepToward_ClampsTheApproach()
        {
            var pos = np.array(new double[] { 0, 0, 0, 0 });
            var tgt = np.array(new double[] { 10, -10, 1, -1 });
            V(np.evaluate(NDExpr.StepToward(pos, tgt, 3.0)), 3, -3, 1, -1);
        }

        [TestMethod]
        public void MaxMagnitude_KeepsStrongerSignal()
        {
            var u = np.array(new double[] { 1, -5, 3 });
            var v = np.array(new double[] { -2, 4, -1 });
            V(np.evaluate(NDExpr.MaxMagnitude(u, v)), -2, -5, 3);
        }

        // =====================================================================
        // Multi-way decision & interpolation
        // =====================================================================

        [TestMethod]
        public void Bucketize_CountsExceededEdges()
        {
            var x = np.array(new double[] { -1, 0.5, 3, 7 });
            V(np.evaluate(NDExpr.Bucketize(x, 0.0, 1.0, 5.0)), 0, 1, 2, 3);
        }

        [TestMethod]
        public void Bucketize_NullEdge_Throws()
            => Assert.ThrowsException<ArgumentNullException>(() => NDExpr.Bucketize(NDExpr.Const(1.0), NDExpr.Const(0.0), null));

        [TestMethod]
        public void Median3_MiddleValue()
        {
            var a = np.array(new double[] { 3, 1, 8 });
            var b = np.array(new double[] { 1, 5, 2 });
            var c = np.array(new double[] { 2, 9, 5 });
            V(np.evaluate(NDExpr.Median3(a, b, c)), 2, 5, 5);
        }

        [TestMethod]
        public void Threshold_SubstitutesBelow()
        {
            var t = np.array(new double[] { -2, -0.5, 0, 3, 7 });
            V(np.evaluate(NDExpr.Threshold(t, 0.0, -1.0)), -1, -1, -1, 3, 7);
        }

        [TestMethod]
        public void Lerp_Interpolates()
        {
            var a = np.array(new double[] { 0, 0, 0, 0 });
            var b = np.array(new double[] { 10, 20, 30, 40 });
            var t = np.array(new double[] { 0.0, 0.25, 0.5, 1.0 });
            V(np.evaluate(NDExpr.Lerp(a, b, t)), 0, 5, 15, 40);
        }

        // =====================================================================
        // Fusion parity + composition inside a bigger tree
        // =====================================================================

        [TestMethod]
        public void Combinator_FusesInsideChain_MatchesUnfusedNumpy()
        {
            var a = np.array(new double[] { -1, 2, -3, 4 });
            var b = np.array(new double[] { 0.5, 0.5, 0.5, 0.5 });
            var c = np.array(new double[] { 1, 1, 1, 1 });

            // relu(a*b + c), fused in one pass, must equal the eager np chain.
            var fused = np.evaluate(NDExpr.Relu((NDExpr)a * b + c));
            var reference = np.maximum(a * b + c, 0.0);
            Close(fused, reference);
        }

        [TestMethod]
        public void Combinators_ComposeWithEachOther()
        {
            var x = np.array(new double[] { -2, -0.5, 0.25, 3, 9 });
            // clamp then step: saturate, then classify > 0 — exercises two combinators in one tree.
            var fused = np.evaluate(NDExpr.Step(NDExpr.Saturate(x) - 0.5));
            var sat = np.clip(x, 0.0, 1.0);
            var reference = np.where(sat - 0.5 > 0.0, np.array(new[] { 1, 1, 1, 1, 1 }), np.array(new[] { 0, 0, 0, 0, 0 }));
            Exact(fused, reference);
        }
    }
}
