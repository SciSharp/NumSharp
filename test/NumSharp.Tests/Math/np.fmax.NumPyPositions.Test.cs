using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     float32 / float64 <c>np.fmax</c> / <c>np.fmin</c> give a ±0 tie and a NaN pair NumPy 2.4.2's win-amd64 answer at
    ///     every position: inside whole AVX2 vectors of an inner-loop call (8 float32 / 4 float64 lanes) the second operand
    ///     on a tie and the FIRST operand of two NaNs; everywhere else — a call's tail, a call shorter than a vector, a call
    ///     whose operands overlap or whose float32 stride is negative — the UCRT's <c>fmax</c> / <c>fmin</c>: +0.0 / -0.0 on
    ///     a tie and the SECOND operand of two NaNs. Which elements are which follows NumPy's own calls (the trivial loop,
    ///     NpyIter's sort / coalescing / buffering, a where= mask's runs).
    /// </summary>
    /// <remarks>
    ///     Parity QA-20 (docs/bugs/parity-qa-2026-11-02.md), the owner's "NumPy's win-amd64 positions". Every expected
    ///     value was probed on NumPy 2.4.2 win-amd64 (AVX2). The model is pinned to that build on every host (the Windows
    ///     wheel has no AVX-512 target), so these tests hold on any CPU. A sign string reads one character per element in
    ///     C order: <c>-</c> for a negative sign bit, <c>+</c> otherwise.
    /// </remarks>
    [TestClass]
    public class FMaxFMinNumPyPositionsTests
    {
        /// <summary>The sign of every element of <paramref name="r"/>, in C order, as <c>+</c> / <c>-</c>.</summary>
        /// <param name="r">A float32 or float64 array.</param>
        /// <returns>One character per element.</returns>
        private static string Signs(NDArray r)
        {
            using var flat = r.flatten();
            var chars = new char[flat.size];
            for (int i = 0; i < chars.Length; i++)
                chars[i] = double.IsNegative(Convert.ToDouble(flat.GetValue(i))) ? '-' : '+';
            return new string(chars);
        }

        /// <summary>A float array of <paramref name="n"/> zeros with the given sign.</summary>
        /// <param name="n">The length.</param>
        /// <param name="dt">float32 or float64.</param>
        /// <param name="negative">True for -0.0.</param>
        /// <returns>The array.</returns>
        private static NDArray Zeros(int n, NPTypeCode dt, bool negative)
        {
            var z = np.zeros(new Shape(n), dt);
            if (!negative)
                return z;
            var nz = np.negative(z);
            z.Dispose();
            return nz;
        }

        /// <summary>
        ///     A float64 array of <paramref name="n"/> elements holding exactly <paramref name="bits"/>, built from an integer
        ///     array at run time: a NaN written as a float constant may reach the array quiet (RyuJIT folds float constants
        ///     through double, which quiets a signalling NaN).
        /// </summary>
        /// <param name="bits">The raw float64 bits.</param>
        /// <param name="n">The length.</param>
        /// <returns>The array (a float64 view of a fresh uint64 array).</returns>
        private static NDArray F64(ulong bits, int n) => np.array(Enumerable.Repeat(bits, n).ToArray()).view(np.float64);

        /// <summary>The float32 twin of <see cref="F64"/>.</summary>
        /// <param name="bits">The raw float32 bits.</param>
        /// <param name="n">The length.</param>
        /// <returns>The array (a float32 view of a fresh uint32 array).</returns>
        private static NDArray F32(uint bits, int n) => np.array(Enumerable.Repeat(bits, n).ToArray()).view(np.float32);

        /// <summary>
        ///     The finding's values: <c>np.fmax([0.0], [-0.0])</c> is <c>[0.0]</c> and <c>np.fmin([-0.0], [0.0])</c> is
        ///     <c>[-0.0]</c> (one element, no whole vector); a 7-element float32 fmax of +0 and -0 is +0 everywhere; a 9-element
        ///     float64 one is -0 in its two vectors and +0 at the tail.
        /// </summary>
        [TestMethod]
        public void FindingValues_AreNumPys()
        {
            using var p1 = Zeros(1, NPTypeCode.Double, false);
            using var n1 = Zeros(1, NPTypeCode.Double, true);
            using var fmax1 = np.fmax(p1, n1);
            using var fmin1 = np.fmin(n1, p1);
            Signs(fmax1).Should().Be("+");
            Signs(fmin1).Should().Be("-");

            using var p7 = Zeros(7, NPTypeCode.Single, false);
            using var n7 = Zeros(7, NPTypeCode.Single, true);
            using var fmax7 = np.fmax(p7, n7);
            Signs(fmax7).Should().Be("+++++++");

            using var p9 = Zeros(9, NPTypeCode.Double, false);
            using var n9 = Zeros(9, NPTypeCode.Double, true);
            using var fmax9 = np.fmax(p9, n9);
            Signs(fmax9).Should().Be("--------+");
        }

        /// <summary>
        ///     Contiguous calls of every length class: the whole vectors (4 float64 / 8 float32 lanes) keep the second
        ///     operand, the tail takes the C library's sign — for fmax(+0, -0) and fmin(-0, +0) alike.
        /// </summary>
        [TestMethod]
        public void ContiguousLengths_SplitAtNumPysVectorWidth()
        {
            var f64 = new (int n, string fmax, string fmin)[]
            {
                (1, "+", "-"), (3, "+++", "---"), (4, "----", "++++"), (5, "----+", "++++-"), (7, "----+++", "++++---"),
                (8, "--------", "++++++++"), (9, "--------+", "++++++++-"), (15, "------------+++", "++++++++++++---"),
                (17, "----------------+", "++++++++++++++++-"),
            };
            var f32 = new (int n, string fmax, string fmin)[]
            {
                (1, "+", "-"), (4, "++++", "----"), (7, "+++++++", "-------"), (8, "--------", "++++++++"),
                (9, "--------+", "++++++++-"), (12, "--------++++", "++++++++----"), (15, "--------+++++++", "++++++++-------"),
                (16, "----------------", "++++++++++++++++"), (20, "----------------++++", "++++++++++++++++----"),
            };
            foreach (var (dt, table) in new[] { (NPTypeCode.Double, f64), (NPTypeCode.Single, f32) })
            {
                foreach (var (n, fmax, fmin) in table)
                {
                    using var p = Zeros(n, dt, false);
                    using var m = Zeros(n, dt, true);
                    using var rmax = np.fmax(p, m);
                    using var rmin = np.fmin(m, p);
                    Signs(rmax).Should().Be(fmax, $"{dt} fmax(+0, -0) of length {n}");
                    Signs(rmin).Should().Be(fmin, $"{dt} fmin(-0, +0) of length {n}");
                }
            }
        }

        /// <summary>
        ///     A pair of NaNs: the vector rule keeps the FIRST operand's NaN, the C library the SECOND one's — both verbatim,
        ///     a signalling NaN unquieted; a lone NaN is skipped by both.
        /// </summary>
        [TestMethod]
        public void NaNPairs_TakeEachRulesOperand()
        {
            const ulong qa = 0x7FF8000000000001, sb = 0xFFF0000000000002;
            using var a1 = F64(qa, 1);
            using var b1 = F64(sb, 1);
            using var crt = np.fmax(a1, b1);
            BitConverter.DoubleToUInt64Bits(crt.GetAtIndex<double>(0)).Should().Be(sb, "one element: the C library's second NaN");

            using var a4 = F64(qa, 4);
            using var b4 = F64(sb, 4);
            using var vec = np.fmin(a4, b4);
            BitConverter.DoubleToUInt64Bits(vec.GetAtIndex<double>(0)).Should().Be(qa, "a whole vector: the first NaN");

            const uint fa = 0x7FC00001, fb = 0xFF800002;
            using var x8 = F32(fa, 9);
            using var y8 = F32(fb, 9);
            using var r8 = np.fmax(x8, y8);
            BitConverter.SingleToUInt32Bits(r8.GetAtIndex<float>(0)).Should().Be(fa, "float32 vector position");
            BitConverter.SingleToUInt32Bits(r8.GetAtIndex<float>(8)).Should().Be(fb, "float32 tail: the signalling NaN kept");

            using var one = np.full(new Shape(5), 1.0);
            using var nan5 = F64(qa, 5);
            using var skip = np.fmax(nan5, one);
            skip.GetAtIndex<double>(4).Should().Be(1.0, "a lone NaN is skipped at a scalar position too");
        }

        /// <summary>
        ///     The layouts NumPy's loop takes differently: a strided call vectorizes (simd_binary), a reversed float32
        ///     operand runs every element through the C library (its negative stride fails AVX2's float32 load limit) while
        ///     float64 keeps its vectors, a 0-d operand broadcasts with step 0, a 2-D array is one call over its 15 elements
        ///     (C-contiguous: the trivial loop; row slices: NpyIter buffers both inputs into one call).
        /// </summary>
        [TestMethod]
        public void Layouts_FollowNumPysCalls()
        {
            foreach (var dt in new[] { NPTypeCode.Double, NPTypeCode.Single })
            {
                bool f64 = dt == NPTypeCode.Double;
                using var p20 = Zeros(20, dt, false);
                using var n20 = Zeros(20, dt, true);
                using var sp = p20["::2"];
                using var sn = n20["::2"];
                using var strided = np.fmax(sp, sn);
                Signs(strided).Should().Be("--------++", $"{dt} strided");

                using var rp = p20[":9"]["::-1"];
                using var rn = n20[":9"]["::-1"];
                using var reversed = np.fmax(rp, rn);
                Signs(reversed).Should().Be(f64 ? "--------+" : "+++++++++", $"{dt} reversed");

                using var p9 = p20[":9"];
                using var n9 = n20[":9"];
                using var nz0 = np.negative(np.zeros(new Shape(), dt));
                using var scalar = np.fmax(p9, nz0);
                Signs(scalar).Should().Be("--------+", $"{dt} 0-d second operand");

                using var a2 = np.zeros(new Shape(3, 5), dt);
                using var b2 = np.negative(a2);
                using var c2d = np.fmax(a2, b2);
                Signs(c2d).Should().Be(f64 ? "------------+++" : "--------+++++++", $"{dt} 2-D C");

                using var ab = np.zeros(new Shape(3, 10), dt);
                using var bb = np.negative(ab);
                using var abv = ab[":, :5"];
                using var bbv = bb[":, :5"];
                using var rows = np.fmax(abv, bbv);
                Signs(rows).Should().Be(f64 ? "------------+++" : "--------+++++++", $"{dt} 2-D row slices, buffered");
            }
        }

        /// <summary>
        ///     out= and where=: a given out keeps the trivial loop's positions; a mask splits each call into its true runs,
        ///     each a call of its own (runs of 2 and 6 hold no whole float32 vector); an in-place call is one contiguous call.
        /// </summary>
        [TestMethod]
        public void OutWhereAndInPlace_FollowNumPysCalls()
        {
            foreach (var dt in new[] { NPTypeCode.Double, NPTypeCode.Single })
            {
                bool f64 = dt == NPTypeCode.Double;
                using var p9 = Zeros(9, dt, false);
                using var n9 = Zeros(9, dt, true);
                using var o1 = np.full(new Shape(9), 7, dt);
                np.fmax(p9, n9, o1);
                Signs(o1).Should().Be("--------+", $"{dt} out=");

                using var o2 = np.full(new Shape(9), 7, dt);
                using var gap = np.array(new[] { true, true, false, true, true, true, true, true, true });
                np.fmax(p9, n9, o2, gap);
                Signs(o2).Should().Be(f64 ? "+++----++" : "+++++++++", $"{dt} where= with a gap");

                using var x = Zeros(12, dt, false);
                using var y = Zeros(12, dt, true);
                np.fmax(x, y, x);
                Signs(x).Should().Be(f64 ? "------------" : "--------++++", $"{dt} in place");
            }
        }

        /// <summary>
        ///     An out= that overlaps an input: behind it (each element read before it is written) NumPy keeps the trivial loop,
        ///     whose overlapping ranges send the whole call through the C library; ahead of it, NpyIter computes into a copy
        ///     with the ordinary positions.
        /// </summary>
        [TestMethod]
        public void OverlappingOut_FollowsNumPysOverlapTest()
        {
            foreach (var (shift, expected) in new[] { (-1, "+++++++++"), (1, "--------+") })
            {
                using var buffer = np.zeros(new Shape(15));
                using var x = buffer["3:12"];
                using var y = Zeros(9, NPTypeCode.Double, true);
                using var o = buffer[$"{3 + shift}:{12 + shift}"];
                np.fmax(x, y, o);
                Signs(o).Should().Be(expected, $"out shifted by {shift}");
            }
        }

        /// <summary>
        ///     NpyIter's buffering decides the calls: (5000, 3) float64 against a (3,) row buffers the row into calls of
        ///     8190 elements (2730 rows, the largest multiple of the 3-element core within the 8192 buffer) and 6810, so
        ///     only each call's last two elements are scalar positions — 8188, 8189, 14998 and 14999.
        /// </summary>
        [TestMethod]
        public void BufferedCalls_EndWhereNumPysBuffersEnd()
        {
            using var a = np.zeros(new Shape(5000, 3));
            using var row = np.negative(np.zeros(new Shape(3)));
            using var r = np.fmax(a, row);
            using var flat = r.flatten();
            var plus = Enumerable.Range(0, (int)flat.size).Where(i => !double.IsNegative(flat.GetAtIndex<double>(i))).ToArray();
            plus.Should().Equal(8188, 8189, 14998, 14999);
        }

        /// <summary>
        ///     Cast operands and a cast output: an int32 zero is +0.0 in the float64 loop (the tail's tie still takes the C
        ///     library's +0), and a float32 call into a float64 out delivers each answer through the cast — the vector
        ///     position's first NaN and the tail's second NaN quieted as the float32 → float64 conversion does.
        /// </summary>
        [TestMethod]
        public void CastOperandsAndOut_DeliverNumPysAnswers()
        {
            using var ints = np.zeros(new Shape(9), NPTypeCode.Int32);
            using var n9 = Zeros(9, NPTypeCode.Double, true);
            using var mixed = np.fmax(ints, n9);
            Signs(mixed).Should().Be("--------+");

            using var p32 = Zeros(9, NPTypeCode.Single, false);
            using var n32 = Zeros(9, NPTypeCode.Single, true);
            using var o = np.full(new Shape(9), 7.0);
            np.fmax(p32, n32, o);
            Signs(o).Should().Be("--------+", "float32 loop into a float64 out");

            using var na = F32(0x7FC00001, 9);
            using var nb = F32(0xFFC00002, 9);
            using var o2 = np.zeros(new Shape(9));
            np.fmax(na, nb, o2);
            BitConverter.DoubleToUInt64Bits(o2.GetAtIndex<double>(0)).Should().Be(0x7FF8000020000000UL);
            BitConverter.DoubleToUInt64Bits(o2.GetAtIndex<double>(8)).Should().Be(0xFFF8000040000000UL);
        }

        /// <summary>
        ///     np.evaluate runs each float fmax / fmin node as NumPy's own call over its children (the unfused sequence):
        ///     at the root (with where=), over a computed child, under another node, nested, against a literal, and inside a
        ///     reduction — whose sum of <c>1 / fmax</c> shows each element's sign (-0 → -inf, +0 → +inf, both → NaN).
        /// </summary>
        [TestMethod]
        public void Evaluate_FollowsTheUnfusedCalls()
        {
            using var p9 = Zeros(9, NPTypeCode.Double, false);
            using var n9 = Zeros(9, NPTypeCode.Double, true);
            NDExpr P = p9, N = n9;

            using var o = np.full(new Shape(9), 7.0);
            using var gap = np.array(new[] { true, true, false, true, true, true, true, true, true });
            np.evaluate(NDExpr.FMax(P, N), null, o, gap);
            Signs(o).Should().Be("+++----++", "root with where=");

            using var child = np.evaluate(NDExpr.FMax(-N, N));
            Signs(child).Should().Be("--------+", "computed child");
            using var under = np.evaluate(-NDExpr.FMax(P, N));
            Signs(under).Should().Be("++++++++-", "under a negation");
            using var nested = np.evaluate(NDExpr.FMax(NDExpr.FMin(N, P), P));
            Signs(nested).Should().Be("+++++++++", "nested");
            using var literal = np.evaluate(NDExpr.FMax(P, -0.0));
            Signs(literal).Should().Be("--------+", "a -0.0 literal");

            foreach (var (n, f64, f32) in new[] { (3, double.PositiveInfinity, double.PositiveInfinity),
                                                   (8, double.NegativeInfinity, double.NegativeInfinity),
                                                   (9, double.NaN, double.NaN), (12, double.NegativeInfinity, double.NaN) })
            {
                foreach (var (dt, expected) in new[] { (NPTypeCode.Double, f64), (NPTypeCode.Single, f32) })
                {
                    using var x = Zeros(n, dt, false);
                    using var y = Zeros(n, dt, true);
                    using var s = np.evaluate(NDExpr.Sum(1.0 / NDExpr.FMax(x, y)));
                    double v = Convert.ToDouble(s.GetValue(0));
                    if (double.IsNaN(expected))
                        double.IsNaN(v).Should().BeTrue($"{dt} n={n}");
                    else
                        v.Should().Be(expected, $"{dt} n={n}");
                }
            }
        }
    }
}
