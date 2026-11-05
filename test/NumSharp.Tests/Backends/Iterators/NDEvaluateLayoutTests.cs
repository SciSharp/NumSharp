using System;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     np.evaluate's result LAYOUT is NumPy's: a fresh elementwise result of the default <c>order='K'</c> follows NpyIter's
    ///     K order over the operands (a 3-D transpose keeps its permutation, an F operand beside a broadcast row stays F, a
    ///     conflict is C), and an axis reduction finishes in NumPy's reduce layout (the child's K order with the axis removed,
    ///     or kept at its memory position, owning its data; C for argmax). Every expectation is NumPy 2.4.2's
    ///     <c>strides</c> / <c>flags</c> for the same statement. A stride on an extent-1 axis addresses no element, so NumPy's
    ///     flags ignore it and so do these checks (<see cref="Layout"/>).
    /// </summary>
    [TestClass]
    public class NDEvaluateLayoutTests
    {
        /// <summary>A (3,4,2) float64 transpose of arange(24).reshape(2,3,4): element strides (4,1,12).</summary>
        private static NDArray Permuted() => np.arange(24.0).reshape(2, 3, 4).transpose(new[] { 1, 2, 0 });

        /// <summary>
        ///     The layout facets NumPy's flags and the oracle compare: the byte strides of the axes of extent above 1, then
        ///     C-contiguity, F-contiguity and ownership.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <returns>The strides (extent-1 axes left out), then the three flags as 1 / 0.</returns>
        private static long[] Layout(NDArray a)
        {
            var strides = a.strides.Where((_, i) => a.shape[i] > 1).ToList();
            strides.Add(a.Shape.IsContiguous ? 1 : 0);
            strides.Add(a.Shape.IsFContiguous ? 1 : 0);
            strides.Add(a.Shape.OwnsData ? 1 : 0);
            return strides.ToArray();
        }

        /// <summary>The expected facets: byte strides of the extent-above-1 axes, then C, F, OWNDATA.</summary>
        /// <param name="facets">The facets in <see cref="Layout"/>'s order.</param>
        /// <returns>The facets as an array.</returns>
        private static long[] L(params long[] facets) => facets;

        /// <summary>Asserts a fresh result's <see cref="Layout"/> facets, then releases it.</summary>
        /// <param name="r">The result (disposed here).</param>
        /// <param name="expected">The facets in <see cref="Layout"/>'s order.</param>
        private static void Check(NDArray r, params long[] expected)
        {
            using (r)
                Layout(r).Should().Equal(expected);
        }

        /// <summary>
        ///     Elementwise trees over a permuted operand keep NumPy's K-order permutation (they were C-ordered): a plain
        ///     product, a two-node tree, a strided permuted operand, a where with a scalar branch and a bool result.
        /// </summary>
        [TestMethod]
        public void Elementwise_PermutedOperands_KeepTheKOrderPermutation()
        {
            using var x = Permuted();
            Check(np.evaluate((NDExpr)x * 2), 32, 8, 96, 0, 0, 1);
            Check(np.evaluate(NDExpr.Sqrt(x) + NDExpr.Arr(x)), 32, 8, 96, 0, 0, 1);
            Check(np.evaluate(NDExpr.Where(NDExpr.Greater(x, 5), x, 0.0)), 32, 8, 96, 0, 0, 1);
            // The bool result (itemsize 1) of two comparisons: (4, 1, 12).
            Check(np.evaluate(NDExpr.Greater(x, 5) & NDExpr.Less(x, 20)), 4, 1, 12, 0, 0, 1);

            // [::2] of a (4,3,4) transposed to (4,2,3): (8, 96, 32).
            using var s = np.arange(48.0).reshape(4, 3, 4)["::2"].transpose(new[] { 2, 0, 1 });
            Check(np.evaluate((NDExpr)s * 2), 8, 96, 32, 0, 0, 1);

            // The values are the statements' own (the layout changes where they sit, not what they are).
            using var fused = np.evaluate((NDExpr)x * 2 + 1);
            using var twice = x * 2;
            using var plain = twice + 1;
            np.array_equal(fused, plain).Should().BeTrue();
        }

        /// <summary>
        ///     NumPy's K order beside a broadcast operand and under a conflict: F operands with a broadcast row stay F (the
        ///     row's extent-1 axis abstains; the strict-F rule made it C), a permuted operand with a C one is C, and the explicit
        ///     orders force — <c>'C'</c> is C, <c>'A'</c> is F only for F operands.
        /// </summary>
        [TestMethod]
        public void Elementwise_BroadcastConflictAndExplicitOrders()
        {
            using var f = np.asfortranarray(np.arange(12.0).reshape(3, 4));
            using var row = np.arange(4.0).reshape(1, 4);
            Check(np.evaluate(NDExpr.Arr(f) + row), 8, 24, 0, 1, 1);
            Check(np.evaluate((NDExpr)f * 2), 8, 24, 0, 1, 1);

            using var x = Permuted();
            using var c = np.arange(24.0).reshape(3, 4, 2);
            Check(np.evaluate(NDExpr.Arr(x) + c), 64, 16, 8, 1, 0, 1);

            Check(np.evaluate((NDExpr)x * 2, order: 'C'), 64, 16, 8, 1, 0, 1);
            Check(np.evaluate((NDExpr)x * 2, order: 'A'), 64, 16, 8, 1, 0, 1);
            Check(np.evaluate((NDExpr)f * 2, order: 'A'), 8, 24, 0, 1, 1);
        }

        /// <summary>
        ///     Axis reductions finish in NumPy's reduce layout: over a permuted child the remaining axes keep the child's
        ///     memory order (they came back C-ordered), keepdims results OWN their data (they were reshaped views), argmax is
        ///     C, a flat keepdims result owns its one element. The values are NumPy's.
        /// </summary>
        [TestMethod]
        public void Reductions_FinishInNumPysReduceLayout()
        {
            using var x = Permuted();
            using var sum0 = np.evaluate(NDExpr.Sum(x, 0));
            Layout(sum0).Should().Equal(L(8, 32, 0, 1, 1));
            sum0.ravel().ToArray<double>().Should().Equal(12.0, 48.0, 15.0, 51.0, 18.0, 54.0, 21.0, 57.0);

            Check(np.evaluate(NDExpr.Sum(x, 0, true)), 8, 32, 0, 1, 1);
            Check(np.evaluate(NDExpr.Sum((NDExpr)x * 2, 1, true)), 8, 24, 0, 1, 1);
            Check(np.evaluate(NDExpr.Max(x, 2, true)), 32, 8, 1, 0, 1);
            Check(np.evaluate(NDExpr.Mean(x, 1)), 8, 24, 0, 1, 1);
            Check(np.evaluate(NDExpr.Var(x, 0, true)), 8, 32, 0, 1, 1);
            Check(np.evaluate(NDExpr.Any(NDExpr.Greater(x, 5), 0, true)), 1, 4, 0, 1, 1);
            Check(np.evaluate(NDExpr.CountNonzero(NDExpr.Greater(x, 5), 1)), 8, 24, 0, 1, 1);

            // argmax / argmin: NumPy computes over a C-ordered copy, so the result is C (int64).
            using var am = np.evaluate(NDExpr.ArgMax(x, 0, true));
            Layout(am).Should().Equal(L(16, 8, 1, 0, 1));
            am.ravel().ToArray<long>().Should().OnlyContain(v => v == 2);

            // C and F children: keepdims owns its data in C / F order (the C case was a reshaped view).
            using var c = np.arange(24.0).reshape(3, 4, 2);
            Check(np.evaluate(NDExpr.Sum(c, 1, true)), 16, 8, 1, 0, 1);
            using var f = np.asfortranarray(np.arange(12.0).reshape(3, 4));
            Check(np.evaluate(NDExpr.Sum(f, 1, true)), 8, 1, 1, 1);

            // A flat keepdims result: (1,1,1), owning its one element.
            using var flat = np.evaluate(NDExpr.Sum(x, keepdims: true));
            flat.shape.Should().Equal(1L, 1L, 1L);
            Layout(flat).Should().Equal(L(1, 1, 1));
            flat.GetDouble(0, 0, 0).Should().Be(276.0);
        }

        /// <summary>
        ///     A caller's <c>out=</c> keeps its own layout (the result is written into it), and a reduction's fresh result that
        ///     already has NumPy's layout is returned as it is (no relayout copy for a C child without keepdims).
        /// </summary>
        [TestMethod]
        public void Reductions_OutKeepsItsLayout_AndAMatchingResultIsNotCopied()
        {
            using var x = Permuted();
            using var o = np.zeros(new Shape(4, 2));
            var r = np.evaluate(NDExpr.Sum(x, 0), @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            Layout(o).Should().Equal(L(16, 8, 1, 0, 1));
            o.ravel().ToArray<double>().Should().Equal(12.0, 48.0, 15.0, 51.0, 18.0, 54.0, 21.0, 57.0);

            using var c = np.arange(24.0).reshape(3, 4, 2);
            using var s = np.evaluate(NDExpr.Sum(c, 1));
            Layout(s).Should().Equal(L(16, 8, 1, 0, 1));
        }
    }
}
