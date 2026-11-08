using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     Binary ufuncs over two F-ordered operands where one of them BROADCASTS — a <c>(2,1,2)</c> F array against a
    ///     <c>(2,3,2)</c> F array (plan <c>docs/plans/api-parameter-parity.md</c>, item X3). NumPy computes the logical
    ///     elementwise result and lays it out F (NpyIter's K order over two F operands).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The bug these tests pin (fixed 2026-10-05): the direct binary route allocated the result in F order whenever
    ///     every non-scalar operand was strictly F-contiguous by its OWN flags, and a <c>(2,1,2)</c> F array is. The
    ///     direct kernels write the result linearly in C-coordinate order over the coalesced dims, which is the F memory
    ///     order only when all dims coalesce into one run; the broadcast operand's stride-0 axis prevents that, so the
    ///     values landed at the wrong positions. It hit every SIMD dtype of <c>add</c> / <c>subtract</c> /
    ///     <c>multiply</c> / <c>true_divide</c> / <c>maximum</c> / <c>minimum</c> / <c>fmax</c> / <c>fmin</c> / the
    ///     bitwise ops (and the C# operators) under the small-broadcast cutoff, and float16 <c>add</c> / <c>subtract</c> /
    ///     <c>multiply</c> / <c>divide</c> at any size, because those always take the direct route. The fix allocates C
    ///     for such a set; the existing post-kernel branch relabels it F.
    ///     </para>
    ///     <para>
    ///     The oracle gate is the <c>pp_fbcast_*</c> pair layouts of the binary tiers (<c>layout_catalog.py</c>
    ///     <c>FBCAST_PAIR_LAYOUTS</c>): 3,074 cases, 388 of them diverge on the code before the fix.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class FOrderBroadcastBinaryTests
    {
        /// <summary>The F-ordered (2,3,2) array 0..11 that every literal case below starts from.</summary>
        /// <returns>A fresh F-contiguous float64 array holding <c>arange(12).reshape(2,3,2)</c>.</returns>
        private static NDArray A() => np.asfortranarray(np.arange(12.0).reshape(2, 3, 2));

        /// <summary>An F-ordered broadcast operand <c>arange(n) * 100 + 100</c> of the given shape.</summary>
        /// <param name="shape">The operand shape, with an extent-1 axis where it broadcasts against <see cref="A"/>.</param>
        /// <returns>A fresh F-contiguous float64 array.</returns>
        private static NDArray B(params long[] shape)
        {
            long n = shape.Aggregate(1L, (x, y) => x * y);
            return np.asfortranarray((np.arange((double)n) * 100.0 + 100.0).reshape(shape));
        }

        /// <summary>
        ///     <c>a - b</c> and <c>b - a</c> for a broadcast along each axis: NumPy 2.4.2's logical values (probed) and its
        ///     F-contiguous layout.
        /// </summary>
        [TestMethod]
        public void Subtract_FOrderedOperandBroadcastAlongEachAxis_MatchesNumPy()
        {
            var cases = new (long[] bshape, double[] aMinusB)[]
            {
                (new long[] { 2, 1, 2 }, new double[] { -100, -199, -98, -197, -96, -195, -294, -393, -292, -391, -290, -389 }),
                (new long[] { 1, 3, 2 }, new double[] { -100, -199, -298, -397, -496, -595, -94, -193, -292, -391, -490, -589 }),
                (new long[] { 2, 3, 1 }, new double[] { -100, -99, -198, -197, -296, -295, -394, -393, -492, -491, -590, -589 }),
            };
            foreach (var (bshape, aMinusB) in cases)
            {
                var a = A();
                var b = B(bshape);
                var r = np.subtract(a, b);
                r.shape.Should().Equal(2L, 3L, 2L);
                r.Shape.IsFContiguous.Should().BeTrue($"NumPy lays out F op F as F ({string.Join(",", bshape)})");
                r.flatten().ToArray<double>().Should().Equal(aMinusB, $"a - b, b of shape ({string.Join(",", bshape)})");
                var r2 = np.subtract(b, a);
                r2.Shape.IsFContiguous.Should().BeTrue();
                r2.flatten().ToArray<double>().Should().Equal(aMinusB.Select(v => -v), "b - a");
            }
        }

        /// <summary>
        ///     float16 add/subtract/multiply/divide always take the direct route (NDIter declines them), so the bug hit
        ///     them at any size: a 24,000-element product, NumPy 2.4.2's values at two positions and its float64 sum.
        /// </summary>
        [TestMethod]
        public void Float16_LargeFOrderBroadcast_MatchesNumPy()
        {
            var h = np.asfortranarray((np.arange(24000) % 7 + 1).astype(np.float16).reshape(30, 40, 20));
            var hb = np.asfortranarray((np.arange(600) % 5 + 2).astype(np.float16).reshape(30, 1, 20));
            var r = h * hb;
            r.Shape.IsFContiguous.Should().BeTrue();
            ((double)r.GetAtIndex<Half>(1 * 800 + 2 * 20 + 3)).Should().Be(20.0);
            ((double)r.GetAtIndex<Half>(29 * 800 + 39 * 20 + 19)).Should().Be(24.0);
            np.sum(r, dtype: np.float64).GetDouble().Should().Be(383977.0);
        }

        /// <summary>
        ///     Every binary function the route serves, on every dtype it serves, for each F broadcast shape pair and both
        ///     operand orders: the result equals the same call on C-contiguous copies (logical values do not depend on
        ///     the layout), and it is F-contiguous like NumPy's. Sizes are under the small-broadcast cutoff (the route
        ///     the bug lived in) except the last pair, which is above it.
        /// </summary>
        [TestMethod]
        public void EveryBinaryFunction_FOrderBroadcast_EqualsTheCOrderResult()
        {
            var fns = new (string name, Func<NDArray, NDArray, NDArray> f, bool intOnly)[]
            {
                ("add", (x, y) => np.add(x, y), false),
                ("subtract", (x, y) => np.subtract(x, y), false),
                ("multiply", (x, y) => np.multiply(x, y), false),
                ("true_divide", (x, y) => np.true_divide(x, y), false),
                ("maximum", (x, y) => np.maximum(x, y), false),
                ("minimum", (x, y) => np.minimum(x, y), false),
                ("fmax", (x, y) => np.fmax(x, y), false),
                ("fmin", (x, y) => np.fmin(x, y), false),
                ("bitwise_and", (x, y) => np.bitwise_and(x, y), true),
                ("bitwise_or", (x, y) => np.bitwise_or(x, y), true),
                ("bitwise_xor", (x, y) => np.bitwise_xor(x, y), true),
                ("operator +", (x, y) => x + y, false),
                ("operator *", (x, y) => x * y, false),
            };
            var dtypes = new[] { np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.uint8 };
            var pairs = new (long[] a, long[] b)[]
            {
                (new long[] { 2, 3, 2 }, new long[] { 2, 1, 2 }),
                (new long[] { 2, 3, 2 }, new long[] { 1, 3, 2 }),
                (new long[] { 2, 3, 2 }, new long[] { 2, 3, 1 }),
                (new long[] { 3, 4, 5, 2 }, new long[] { 3, 1, 5, 2 }),
                (new long[] { 30, 40, 20 }, new long[] { 30, 1, 20 }),
            };
            var wrong = new List<string>();
            foreach (var (ashape, bshape) in pairs)
            foreach (var dt in dtypes)
            foreach (var (name, f, intOnly) in fns)
            {
                bool isInt = dt != np.float64 && dt != np.float32 && dt != np.float16;
                if (intOnly && !isInt)
                    continue;
                long na = ashape.Aggregate(1L, (x, y) => x * y), nb = bshape.Aggregate(1L, (x, y) => x * y);
                var ac = (np.arange(na) % 7 + 1).astype(dt).reshape(ashape);
                var bc = (np.arange(nb) % 5 + 2).astype(dt).reshape(bshape);
                var af = np.asfortranarray(ac);
                var bf = np.asfortranarray(bc);
                for (int swap = 0; swap < 2; swap++)
                {
                    var r = swap == 0 ? f(af, bf) : f(bf, af);
                    var want = swap == 0 ? f(ac, bc) : f(bc, ac);
                    string label = $"{name} {dt.name} ({string.Join(",", ashape)}) with ({string.Join(",", bshape)}){(swap == 1 ? " swapped" : "")}";
                    if (!np.array_equal(r, want))
                        wrong.Add(label + ": values");
                    else if (!r.Shape.IsFContiguous)
                        wrong.Add(label + ": not F-contiguous");
                }
            }

            wrong.Should().BeEmpty("an F-ordered broadcast changes only the layout, never the logical values");
        }
    }
}
