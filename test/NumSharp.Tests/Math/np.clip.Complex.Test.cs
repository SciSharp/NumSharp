using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     complex128 <c>np.clip</c> selects like NumPy 2.4.2: two bounds run clip.cpp's loop,
    ///     <c>_NPY_MIN(_NPY_MAX(x, lo), hi)</c> with <c>_NPY_MAX(a, b) = isnan(a) || CGT(a, b) ? a : b</c> and the STRICT
    ///     <c>CGT(p, q) = p.re == q.re ? p.im &gt; q.im : p.re &gt; q.re</c> (<c>_NPY_MIN</c> the mirror with <c>CLT</c>): only
    ///     the running value is tested for NaN, so a bound's NaN wins only where the comparison picks the bound, and a tie
    ///     takes the bound. One bound runs <c>um.maximum</c> / <c>um.minimum</c> (NumPy's <c>_clip</c> routes it there),
    ///     whose non-strict <c>CGE</c> / <c>CLE</c> keep x on a tie and propagate any NaN. Every literal was probed against
    ///     NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Parity QA-19 (docs/bugs/parity-qa-2026-11-02.md): the clip kernels shared the ufuncs' helpers, which returned a
    ///     NaN-bearing bound first (<c>np.clip([5, 20], complex(1, nan), 10)</c> was <c>[1+nanj, 1+nanj]</c>) and kept x on an
    ///     upper-bound tie. Both clip routes (the flat kernel and the NDIter route) and <c>NDExpr.Clip</c> now call
    ///     <c>ComplexClipMax</c> / <c>ComplexClipMin</c> for two bounds.
    /// </remarks>
    [TestClass]
    public class np_clip_complex_tests
    {
        /// <summary>Python's <c>float('nan')</c>: the positive quiet NaN (C#'s <see cref="double.NaN"/> is negative).</summary>
        private static readonly double Nan = BitConverter.Int64BitsToDouble(0x7ff8000000000000);

        /// <summary>Asserts the raw bits of both parts of <paramref name="z"/> (a value comparison cannot see a zero's sign).</summary>
        /// <param name="z">The value read back.</param>
        /// <param name="re">The expected bits of the real part, as NumPy prints them.</param>
        /// <param name="im">The expected bits of the imaginary part.</param>
        /// <param name="because">What the value is, for the failure text.</param>
        private static void ShouldHaveBits(Complex z, ulong re, ulong im, string because)
        {
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(unchecked((long)re), because);
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(unchecked((long)im), because);
        }

        /// <summary>
        ///     A bound whose IMAGINARY part is NaN wins only where the real parts tie or the comparison picks it: NumPy's
        ///     <c>clip([5, 20], complex(1, nan), 10)</c> is <c>[5, 10]</c>, <c>clip([5], 1, complex(10, nan))</c> is <c>[5]</c>,
        ///     while a tie on the real part takes the NaN bound and a NaN REAL part fails every comparison (the bound
        ///     returned); an element's own NaN passes through.
        /// </summary>
        [TestMethod]
        public void TwoBounds_BoundNaN_CountsOnlyThroughTheComparison()
        {
            using var x = np.array(new[] { new Complex(5, 0), new Complex(20, 0) });

            using var nanLower = np.clip(x, np.array(new Complex(1, Nan)), np.array(new Complex(10, 0)));
            ShouldHaveBits(nanLower.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "5 > 1 keeps the element");
            ShouldHaveBits(nanLower.GetAtIndex<Complex>(1), 0x4024000000000000UL, 0UL, "20 clips to the upper bound");

            using var nanUpper = np.clip(np.array(new[] { new Complex(5, 0) }), np.array(new Complex(1, 0)), np.array(new Complex(10, Nan)));
            ShouldHaveBits(nanUpper.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "5 < 10 keeps the element");

            using var tieOnReal = np.clip(np.array(new[] { new Complex(1, 5) }), np.array(new Complex(1, Nan)), np.array(new Complex(10, 0)));
            ShouldHaveBits(tieOnReal.GetAtIndex<Complex>(0), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "equal real parts compare the NaN imaginary part: the bound");

            using var nanRealLower = np.clip(x, np.array(new Complex(Nan, 0)), np.array(new Complex(10, 0)));
            for (int i = 0; i < 2; i++)
                ShouldHaveBits(nanRealLower.GetAtIndex<Complex>(i), 0x7ff8000000000000UL, 0UL, "a NaN real part fails every comparison");

            using var nanElements = np.clip(np.array(new[] { new Complex(Nan, 1), new Complex(3, Nan) }), np.array(new Complex(1, 0)), np.array(new Complex(10, 0)));
            ShouldHaveBits(nanElements.GetAtIndex<Complex>(0), 0x7ff8000000000000UL, 0x3ff0000000000000UL, "the element's NaN passes through");
            ShouldHaveBits(nanElements.GetAtIndex<Complex>(1), 0x4008000000000000UL, 0x7ff8000000000000UL, "the element's NaN imaginary part passes through");
        }

        /// <summary>
        ///     A tie takes the BOUND at either end: <c>clip(complex(0, -0.0), -5-5j, complex(-0.0, 0))</c> is the upper bound's
        ///     <c>(-0, +0)</c>, <c>clip(0j, complex(-0.0, -0.0), 5+5j)</c> the lower bound's <c>(-0, -0)</c>, and a scalar-bound
        ///     row through both (NumPy's bits for every element).
        /// </summary>
        [TestMethod]
        public void TwoBounds_Tie_TakesTheBound()
        {
            using var upper = np.clip(np.array(new[] { new Complex(0.0, -0.0) }), np.array(new[] { new Complex(-5, -5) }), np.array(new[] { new Complex(-0.0, 0.0) }));
            ShouldHaveBits(upper.GetAtIndex<Complex>(0), 0x8000000000000000UL, 0UL, "the upper bound's zero signs");

            using var lower = np.clip(np.array(new[] { new Complex(0.0, 0.0) }), np.array(new[] { new Complex(-0.0, -0.0) }), np.array(new[] { new Complex(5, 5) }));
            ShouldHaveBits(lower.GetAtIndex<Complex>(0), 0x8000000000000000UL, 0x8000000000000000UL, "the lower bound's zero signs");

            using var row = np.array(new[] { new Complex(0.0, -0.0), new Complex(3, 0), new Complex(-0.0, 0.0), new Complex(12, 0) });
            using var scalarBounds = np.clip(row, np.array(new Complex(-0.0, 0.0)), np.array(new Complex(3, 0)));
            ShouldHaveBits(scalarBounds.GetAtIndex<Complex>(0), 0x8000000000000000UL, 0UL, "0-0j ties the lower bound -0+0j");
            ShouldHaveBits(scalarBounds.GetAtIndex<Complex>(1), 0x4008000000000000UL, 0UL, "3 ties the upper bound");
            ShouldHaveBits(scalarBounds.GetAtIndex<Complex>(2), 0x8000000000000000UL, 0UL, "-0+0j is the lower bound");
            ShouldHaveBits(scalarBounds.GetAtIndex<Complex>(3), 0x4008000000000000UL, 0UL, "12 clips to 3");

            using var both = np.clip(np.array(new[] { new Complex(2, -0.0), new Complex(2, 0.0) }), np.array(new Complex(2, 0.0)), np.array(new Complex(2, -0.0)));
            for (int i = 0; i < 2; i++)
                ShouldHaveBits(both.GetAtIndex<Complex>(i), 0x4000000000000000UL, 0x8000000000000000UL, "lo, then hi, on a double tie");

            using var crossed = np.clip(np.array(new[] { Complex.Zero, new Complex(5, 0), new Complex(9, 0) }), np.array(new Complex(6, 0)), np.array(new Complex(4, 0)));
            for (int i = 0; i < 3; i++)
                crossed.GetAtIndex<Complex>(i).Should().Be(new Complex(4, 0), "lo > hi gives hi");

            using var infinite = np.clip(np.array(new[] { new Complex(double.PositiveInfinity, 0), new Complex(double.NegativeInfinity, 1) }), np.array(new Complex(-5, 0)), np.array(new Complex(5, 0)));
            infinite.GetAtIndex<Complex>(0).Should().Be(new Complex(5, 0));
            infinite.GetAtIndex<Complex>(1).Should().Be(new Complex(-5, 0));
        }

        /// <summary>
        ///     One bound is <c>um.maximum</c> / <c>um.minimum</c>, not clip's loop: x wins a tie and a NaN-bearing bound
        ///     propagates (NumPy: <c>clip(x, complex(1, nan), None)</c> is <c>[1+nanj, 1+nanj]</c>).
        /// </summary>
        [TestMethod]
        public void OneBound_IsTheMaximumOrMinimumUfunc()
        {
            using var tieLow = np.clip(np.array(new[] { new Complex(0.0, 0.0) }), np.array(new[] { new Complex(-0.0, -0.0) }), null);
            ShouldHaveBits(tieLow.GetAtIndex<Complex>(0), 0UL, 0UL, "maximum keeps x on a tie");

            using var tieHigh = np.clip(np.array(new[] { new Complex(0.0, -0.0) }), null, np.array(new[] { new Complex(-0.0, 0.0) }));
            ShouldHaveBits(tieHigh.GetAtIndex<Complex>(0), 0UL, 0x8000000000000000UL, "minimum keeps x on a tie");

            using var x = np.array(new[] { new Complex(5, 0), new Complex(20, 0) });
            using var nanLow = np.clip(x, np.array(new Complex(1, Nan)), null);
            using var nanHigh = np.clip(x, null, np.array(new Complex(10, Nan)));
            for (int i = 0; i < 2; i++)
            {
                ShouldHaveBits(nanLow.GetAtIndex<Complex>(i), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "maximum propagates the bound's NaN");
                ShouldHaveBits(nanHigh.GetAtIndex<Complex>(i), 0x4024000000000000UL, 0x7ff8000000000000UL, "minimum propagates the bound's NaN");
            }
        }

        /// <summary>
        ///     The NDIter route (array bounds, <c>where=</c> with an out, a strided view ending at its buffer's last element,
        ///     broadcast 2-D bounds) selects exactly as the flat kernel does.
        /// </summary>
        [TestMethod]
        public void TwoBounds_IteratorRoute_MatchesNumPy()
        {
            using var x = np.array(new[] { new Complex(5, 0), new Complex(20, 0), new Complex(0.0, -0.0), new Complex(1, 5) });
            using var lo = np.array(new[] { new Complex(1, Nan), new Complex(1, 0), new Complex(-0.0, 0.0), new Complex(1, Nan) });
            using var hi = np.array(new[] { new Complex(10, 0), new Complex(10, Nan), new Complex(3, 0), new Complex(10, 0) });

            using var arrayBounds = np.clip(x, lo, hi);
            ShouldHaveBits(arrayBounds.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "array bounds [0]");
            ShouldHaveBits(arrayBounds.GetAtIndex<Complex>(1), 0x4024000000000000UL, 0x7ff8000000000000UL, "array bounds [1]: 20 > 10, the NaN bound");
            ShouldHaveBits(arrayBounds.GetAtIndex<Complex>(2), 0x8000000000000000UL, 0UL, "array bounds [2]: the lower bound's zeros");
            ShouldHaveBits(arrayBounds.GetAtIndex<Complex>(3), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "array bounds [3]: a real tie takes the NaN bound");

            using var o = np.full(new Shape(4), new Complex(9, 9));
            using var mask = np.array(new[] { true, true, false, true });
            np.clip(x, lo, hi, o, where: mask);
            ShouldHaveBits(o.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "where= [0]");
            ShouldHaveBits(o.GetAtIndex<Complex>(1), 0x4024000000000000UL, 0x7ff8000000000000UL, "where= [1]");
            ShouldHaveBits(o.GetAtIndex<Complex>(2), 0x4022000000000000UL, 0x4022000000000000UL, "where=: the masked-off slot keeps 9+9j");
            ShouldHaveBits(o.GetAtIndex<Complex>(3), 0x3ff0000000000000UL, 0x7ff8000000000000UL, "where= [3]");

            using var buffer = np.array(new[] { new Complex(5, 0), Complex.Zero, new Complex(20, 0), Complex.Zero, new Complex(0.0, -0.0) });
            using var strided = np.clip(buffer["::2"], np.array(new Complex(-0.0, 0.0)), np.array(new Complex(10, Nan)));
            ShouldHaveBits(strided.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "strided [0]");
            ShouldHaveBits(strided.GetAtIndex<Complex>(1), 0x4024000000000000UL, 0x7ff8000000000000UL, "strided [1]");
            ShouldHaveBits(strided.GetAtIndex<Complex>(2), 0x8000000000000000UL, 0UL, "strided [2]: the buffer's last element ties the lower bound");

            using var grid = np.array(new[,] { { new Complex(5, 0), new Complex(20, 0) }, { new Complex(5, 0), new Complex(20, 0) } });
            using var column = np.array(new[,] { { new Complex(1, Nan) }, { new Complex(1, 0) } });
            using var broadcast = np.clip(grid, column, np.array(new Complex(10, Nan)));
            for (int r = 0; r < 2; r++)
            {
                ShouldHaveBits(broadcast.GetAtIndex<Complex>(r * 2), 0x4014000000000000UL, 0UL, "broadcast bounds: 5 stays");
                ShouldHaveBits(broadcast.GetAtIndex<Complex>(r * 2 + 1), 0x4024000000000000UL, 0x7ff8000000000000UL, "broadcast bounds: 20 takes the NaN upper bound");
            }
        }

        /// <summary>
        ///     np.evaluate's <see cref="NDExpr.Clip"/> with both bounds is clip's loop (bound NaN through the comparison, the
        ///     bound on a tie), while <see cref="NDExpr.Clamp"/> stays the composition of np.maximum and np.minimum (x on a tie),
        ///     and the two never share a compiled program.
        /// </summary>
        [TestMethod]
        public void Evaluate_Clip_IsClipsLoop_AndClampIsTheUfuncComposition()
        {
            using var x = np.array(new[] { new Complex(5, 0), new Complex(20, 0) });
            using var nanLower = np.evaluate(NDExpr.Clip((NDExpr)x, NDExpr.Const(new Complex(1, Nan)), NDExpr.Const(new Complex(10, 0))));
            ShouldHaveBits(nanLower.GetAtIndex<Complex>(0), 0x4014000000000000UL, 0UL, "Clip: 5 stays");
            ShouldHaveBits(nanLower.GetAtIndex<Complex>(1), 0x4024000000000000UL, 0UL, "Clip: 20 clips to 10");

            using var t = np.array(new[] { new Complex(0.0, -0.0) });
            using var lo = np.array(new[] { new Complex(-5, -5) });
            using var hi = np.array(new[] { new Complex(-0.0, 0.0) });
            using var clipTie = np.evaluate(NDExpr.Clip((NDExpr)t, (NDExpr)lo, (NDExpr)hi));
            ShouldHaveBits(clipTie.GetAtIndex<Complex>(0), 0x8000000000000000UL, 0UL, "Clip: a tie takes the upper bound");

            using var clampTie = np.evaluate(NDExpr.Clamp((NDExpr)t, (NDExpr)lo, (NDExpr)hi));
            ShouldHaveBits(clampTie.GetAtIndex<Complex>(0), 0UL, 0x8000000000000000UL, "Clamp is minimum(maximum(...)): x on a tie");

            // The same structure evaluated again after the other spelling: the cache keeps the two programs apart.
            using var clipAgain = np.evaluate(NDExpr.Clip((NDExpr)t, (NDExpr)lo, (NDExpr)hi));
            ShouldHaveBits(clipAgain.GetAtIndex<Complex>(0), 0x8000000000000000UL, 0UL, "Clip after Clamp: still clip's loop");
        }
    }
}
