using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     complex128 <c>np.maximum</c> / <c>np.fmax</c> / <c>np.minimum</c> / <c>np.fmin</c> select like NumPy's
    ///     <c>CDOUBLE_maximum</c> family (loops.c.src): <c>CGE</c> / <c>CLE</c> are NON-strict, so two operands that compare
    ///     equal return the FIRST operand's bits — <c>np.maximum(0j, complex(-0.0, -0.0))</c> is <c>0j</c> — and the NaN
    ///     rules are NumPy's (maximum / minimum propagate the first operand's NaN, then the second's; fmax / fmin skip a NaN
    ///     operand, the first one when both carry one). Every literal was probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Parity QA-18 (docs/bugs/parity-qa-2026-11-02.md): maximum / fmax reused np.clip's helper, whose strict last step
    ///     returned the SECOND operand on a tie. The ufunc helpers are now <c>CGE</c> / <c>CLE</c> ports
    ///     (<c>DirectILKernelGenerator.ComplexMaxNaN</c> and siblings) and np.clip's two-bound loop has its own
    ///     (<c>ComplexClipMax</c> / <c>ComplexClipMin</c>, <see cref="np_clip_complex_tests"/>).
    /// </remarks>
    [TestClass]
    public class np_maximum_complex_tie_tests
    {
        /// <summary>The bits of +0.0.</summary>
        private const long P0 = 0L;

        /// <summary>The bits of -0.0.</summary>
        private const long N0 = unchecked((long)0x8000000000000000UL);

        /// <summary>Asserts the raw bits of both parts of <paramref name="z"/> (a value comparison cannot see a zero's sign).</summary>
        /// <param name="z">The value read back.</param>
        /// <param name="re">The expected bits of the real part.</param>
        /// <param name="im">The expected bits of the imaginary part.</param>
        /// <param name="because">What the value is, for the failure text.</param>
        private static void ShouldHaveBits(Complex z, long re, long im, string because)
        {
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(re, because);
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(im, because);
        }

        /// <summary>Asserts the raw bits of both parts of <paramref name="z"/> against a hex pair as NumPy prints it.</summary>
        /// <param name="z">The value read back.</param>
        /// <param name="re">The expected real part's bits.</param>
        /// <param name="im">The expected imaginary part's bits.</param>
        /// <param name="because">What the value is, for the failure text.</param>
        private static void ShouldHaveBits(Complex z, ulong re, ulong im, string because)
            => ShouldHaveBits(z, unchecked((long)re), unchecked((long)im), because);

        /// <summary>
        ///     The four ufuncs on equal operands that differ only in zero signs: each keeps the first operand's bits, in either
        ///     order (NumPy: <c>[0j, 0-0j]</c> for <c>(a, b)</c>, <c>[-0-0j, -0+0j]</c> for <c>(b, a)</c>).
        /// </summary>
        [TestMethod]
        public void MaximumFmaxMinimumFmin_Tie_KeepTheFirstOperand()
        {
            using var a = np.array(new[] { new Complex(0.0, 0.0), new Complex(0.0, -0.0) });
            using var b = np.array(new[] { new Complex(-0.0, -0.0), new Complex(-0.0, 0.0) });

            foreach (var (name, f) in new (string, Func<NDArray, NDArray, NDArray>)[]
                     {
                         ("maximum", (x, y) => np.maximum(x, y)), ("fmax", (x, y) => np.fmax(x, y)),
                         ("minimum", (x, y) => np.minimum(x, y)), ("fmin", (x, y) => np.fmin(x, y)),
                     })
            {
                using var ab = f(a, b);
                ShouldHaveBits(ab.GetAtIndex<Complex>(0), P0, P0, $"{name}(0j, -0-0j) keeps 0j");
                ShouldHaveBits(ab.GetAtIndex<Complex>(1), P0, N0, $"{name}(0-0j, -0+0j) keeps 0-0j");

                using var ba = f(b, a);
                ShouldHaveBits(ba.GetAtIndex<Complex>(0), N0, N0, $"{name}(-0-0j, 0j) keeps -0-0j");
                ShouldHaveBits(ba.GetAtIndex<Complex>(1), N0, P0, $"{name}(-0+0j, 0-0j) keeps -0+0j");
            }
        }

        /// <summary>
        ///     The tie rule on every route the elementwise kernel runs: <c>out=</c> (the iterator), <c>where=</c> (a masked-off
        ///     slot keeps its content), strided views that end at their buffer's last element, a 0-d operand on either side
        ///     (NumPy: a 0-d first operand wins every tie) and 0-d against 0-d.
        /// </summary>
        [TestMethod]
        public void Maximum_Tie_KeepsTheFirstOperand_OnEveryRoute()
        {
            var p = new Complex(0.0, 0.0);
            var pn = new Complex(0.0, -0.0);
            var nn = new Complex(-0.0, -0.0);
            var np_ = new Complex(-0.0, 0.0);
            using var a = np.array(new[] { p, pn });
            using var b = np.array(new[] { nn, np_ });

            using var o = np.full(new Shape(2), new Complex(9, 9));
            np.maximum(a, b, o);
            ShouldHaveBits(o.GetAtIndex<Complex>(0), P0, P0, "out=: 0j");
            ShouldHaveBits(o.GetAtIndex<Complex>(1), P0, N0, "out=: 0-0j");

            using var masked = np.full(new Shape(2), new Complex(9, 9));
            using var mask = np.array(new[] { true, false });
            np.maximum(a, b, masked, where: mask);
            ShouldHaveBits(masked.GetAtIndex<Complex>(0), P0, P0, "where=: the selected slot keeps 0j");
            ShouldHaveBits(masked.GetAtIndex<Complex>(1), 0x4022000000000000UL, 0x4022000000000000UL, "where=: the masked-off slot keeps 9+9j");

            using var aBuf = np.array(new[] { p, new Complex(7, 7), pn, new Complex(7, 7), p });
            using var bBuf = np.array(new[] { nn, new Complex(7, 7), np_, new Complex(7, 7), nn });
            using var strided = np.maximum(aBuf["::2"], bBuf["::2"]);
            ShouldHaveBits(strided.GetAtIndex<Complex>(0), P0, P0, "strided: 0j");
            ShouldHaveBits(strided.GetAtIndex<Complex>(1), P0, N0, "strided: 0-0j");
            ShouldHaveBits(strided.GetAtIndex<Complex>(2), P0, P0, "strided: the last element");

            using var row = np.array(new[] { p, pn, p });
            using var zeroD = NDArray.Scalar(nn);
            using var right = np.maximum(row, zeroD);
            ShouldHaveBits(right.GetAtIndex<Complex>(1), P0, N0, "a 0-d second operand loses the tie");
            using var left = np.maximum(zeroD, row);
            for (int i = 0; i < 3; i++)
                ShouldHaveBits(left.GetAtIndex<Complex>(i), N0, N0, "a 0-d first operand wins every tie");

            using var p0d = NDArray.Scalar(p);
            using var both0d = np.maximum(p0d, zeroD);
            ShouldHaveBits(both0d.GetAtIndex<Complex>(0), P0, P0, "0-d x 0-d: the first operand");
            using var pn0d = NDArray.Scalar(pn);
            using var np0d = NDArray.Scalar(np_);
            using var fmax0d = np.fmax(pn0d, np0d);
            ShouldHaveBits(fmax0d.GetAtIndex<Complex>(0), P0, N0, "fmax 0-d x 0-d: the first operand");
        }

        /// <summary>
        ///     np.evaluate's <see cref="NDExpr.Max"/> / <see cref="NDExpr.FMax"/> nodes run the engine's own scalar body, so a
        ///     fused tie keeps the first operand like the unfused ufunc.
        /// </summary>
        [TestMethod]
        public void Evaluate_MaxFMax_Tie_KeepTheFirstOperand()
        {
            using var a = np.array(new[] { new Complex(0.0, 0.0), new Complex(0.0, -0.0) });
            using var b = np.array(new[] { new Complex(-0.0, -0.0), new Complex(-0.0, 0.0) });

            using var max = np.evaluate(NDExpr.Max((NDExpr)a, (NDExpr)b));
            ShouldHaveBits(max.GetAtIndex<Complex>(0), P0, P0, "Max: 0j");
            ShouldHaveBits(max.GetAtIndex<Complex>(1), P0, N0, "Max: 0-0j");

            using var fmax = np.evaluate(NDExpr.FMax((NDExpr)a, (NDExpr)b));
            ShouldHaveBits(fmax.GetAtIndex<Complex>(0), P0, P0, "FMax: 0j");
            ShouldHaveBits(fmax.GetAtIndex<Complex>(1), P0, N0, "FMax: 0-0j");
        }

        /// <summary>
        ///     The NaN rules are unchanged by the tie fix and remain NumPy's: maximum / minimum return the first operand when it
        ///     holds a NaN, else the second when IT holds one (a NaN imaginary part counts: <c>CGE</c>'s guard fails); fmax /
        ///     fmin skip the NaN operand, and keep the first when both hold one. The plain order is lexicographic.
        /// </summary>
        [TestMethod]
        public void MaximumFmaxMinimumFmin_NaNAndOrder_AreNumPys()
        {
            double nan = BitConverter.Int64BitsToDouble(0x7ff8000000000000);   // Python's float('nan'), positive
            Complex Max(Complex x, Complex y) { using var r = np.maximum(np.array(new[] { x }), np.array(new[] { y })); return r.GetAtIndex<Complex>(0); }
            Complex FMax(Complex x, Complex y) { using var r = np.fmax(np.array(new[] { x }), np.array(new[] { y })); return r.GetAtIndex<Complex>(0); }
            Complex Min(Complex x, Complex y) { using var r = np.minimum(np.array(new[] { x }), np.array(new[] { y })); return r.GetAtIndex<Complex>(0); }
            Complex FMin(Complex x, Complex y) { using var r = np.fmin(np.array(new[] { x }), np.array(new[] { y })); return r.GetAtIndex<Complex>(0); }

            ShouldHaveBits(Max(new Complex(nan, 1), new Complex(2, 0)), 0x7ff8000000000000UL, 0x3ff0000000000000UL, "maximum: the first operand's NaN");
            ShouldHaveBits(Max(new Complex(2, 0), new Complex(nan, 1)), 0x7ff8000000000000UL, 0x3ff0000000000000UL, "maximum: the second operand's NaN");
            ShouldHaveBits(Max(new Complex(1, nan), new Complex(nan, 2)), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "maximum: both NaN -> the first");
            ShouldHaveBits(Max(new Complex(5, 0), new Complex(1, nan)), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "maximum: a NaN imaginary part propagates");
            ShouldHaveBits(FMax(new Complex(nan, 1), new Complex(2, 0)), 0x4000000000000000UL, 0x0UL, "fmax: the NaN first operand is skipped");
            ShouldHaveBits(FMax(new Complex(1, nan), new Complex(nan, 2)), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "fmax: both NaN -> the first");
            ShouldHaveBits(FMax(new Complex(5, 0), new Complex(1, nan)), 0x4014000000000000UL, 0x0UL, "fmax: a NaN imaginary part is skipped");
            ShouldHaveBits(Min(new Complex(2, 0), new Complex(nan, 1)), 0x7ff8000000000000UL, 0x3ff0000000000000UL, "minimum: the second operand's NaN");
            ShouldHaveBits(FMin(new Complex(nan, 1), new Complex(2, 0)), 0x4000000000000000UL, 0x0UL, "fmin: the NaN first operand is skipped");

            using var x = np.array(new[] { new Complex(1, 5), new Complex(2, 0), new Complex(2, 3) });
            using var y = np.array(new[] { new Complex(2, -9), new Complex(1, 9), new Complex(2, 4) });
            using var order = np.maximum(x, y);
            order.GetAtIndex<Complex>(0).Should().Be(new Complex(2, -9), "the larger real part wins");
            order.GetAtIndex<Complex>(1).Should().Be(new Complex(2, 0), "the larger real part wins whatever the imaginary parts");
            order.GetAtIndex<Complex>(2).Should().Be(new Complex(2, 4), "equal real parts: the larger imaginary part");
        }
    }
}
