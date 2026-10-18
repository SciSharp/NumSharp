using System;
using System.Globalization;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.IO;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     np.evaluate bakes each literal into its kernel's IL and caches the kernel by a text key, so the key must tell apart
    ///     every two literals the IL tells apart. It wrote floats with .NET's <c>ToString("R")</c>, which does not: every NaN
    ///     prints alike, the .NET 10 runtime (10.0.1) prints 2^-25 and the double below it as the same 16 digits, and a
    ///     complex literal was printed in the CURRENT culture, so <c>Complex(1.5, 2)</c> and <c>Complex(1, 5.2)</c> were both
    ///     <c>"1,5,2"</c> under a comma decimal separator. A collision handed the second tree the first literal's kernel: a
    ///     wrong result, silently. The key now holds the literal's bits (<see cref="ConstNode.AppendSignature"/>).
    ///     <see cref="PyLiteral"/>'s float repr (the <c>.npy</c> header writer's Python repr) left .NET's "R" in the same
    ///     change.
    /// </summary>
    /// <remarks>
    ///     Each test evaluates the colliding pair one after the other in one process and requires each result's own bits; the
    ///     kernel cache is process-wide, so whichever literal compiled first, a collision makes the other one wrong. The
    ///     culture test sets a culture on its own thread only (<see cref="CultureInfo.CurrentCulture"/> is per thread), so the
    ///     class runs in parallel with the suite.
    /// </remarks>
    [TestClass]
    public class NDEvaluateLiteralKeyTests
    {
        /// <summary>The bits of element 0 of a float64 result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its first element's bits.</returns>
        private static long Bits0(NDArray r) => BitConverter.DoubleToInt64Bits(r.GetDouble(0));

        /// <summary>
        ///     Literals whose .NET shortest text coincides on the .NET 10 runtime (2^-25 and 2^-958 with the double below
        ///     each), and both zeros: each tree computes with its own literal.
        /// </summary>
        [TestMethod]
        public void FloatLiterals_WithTheSameRuntimeText_CompileTheirOwnKernels()
        {
            using var a = np.array(new[] { 1.0, 2.0 });
            foreach (double hi in new[] { System.Math.ScaleB(1.0, -25), System.Math.ScaleB(1.0, -958), 0.0 })
            {
                double lo = hi == 0 ? -0.0 : System.Math.BitDecrement(hi);
                using var r1 = np.evaluate((NDExpr)a * hi);
                using var r2 = np.evaluate((NDExpr)a * lo);
                Bits0(r1).Should().Be(BitConverter.DoubleToInt64Bits(hi), $"a * {hi:R}");
                Bits0(r2).Should().Be(BitConverter.DoubleToInt64Bits(lo), $"a * the double below {hi:R}");
                r2.GetDouble(1).Should().Be(2 * lo);
            }
        }

        /// <summary>
        ///     Two NaN literals with different payloads (read from an array at run time, not written as constants): each
        ///     tree's result carries its own literal's NaN (the only NaN operand of the multiply).
        /// </summary>
        [TestMethod]
        public void NaNLiterals_WithDifferentPayloads_CompileTheirOwnKernels()
        {
            long[] payloads = { 0x7FF8000000000001, 0x7FF8000000000002, unchecked((long)0xFFF8000000000003) };
            using var a = np.array(new[] { 1.0, -3.0 });
            foreach (long bits in payloads)
            {
                double nan = BitConverter.Int64BitsToDouble(bits);
                using var r = np.evaluate((NDExpr)a * nan);
                Bits0(r).Should().Be(bits, $"a * NaN 0x{bits:X16}");
            }
        }

        /// <summary>
        ///     Complex literals under a culture whose decimal separator is a comma (built from the invariant culture, so the
        ///     test needs no ICU data): <c>c * (1 + 5.2j)</c> after <c>c * (1.5 + 2j)</c> is <c>(1, 5.2)</c>.
        /// </summary>
        [TestMethod]
        public void ComplexLiterals_UnderACommaDecimalSeparator_CompileTheirOwnKernels()
        {
            var comma = (CultureInfo)CultureInfo.InvariantCulture.Clone();
            comma.NumberFormat.NumberDecimalSeparator = ",";
            var previous = CultureInfo.CurrentCulture;
            CultureInfo.CurrentCulture = comma;
            try
            {
                using var c = np.array(new[] { new Complex(1, 0) });
                using var r1 = np.evaluate((NDExpr)c * new Complex(1.5, 2));
                using var r2 = np.evaluate((NDExpr)c * new Complex(1, 5.2));
                r1.GetValue<Complex>(0).Should().Be(new Complex(1.5, 2));
                r2.GetValue<Complex>(0).Should().Be(new Complex(1, 5.2));
            }
            finally
            {
                CultureInfo.CurrentCulture = previous;
            }
        }

        /// <summary>
        ///     <see cref="PyLiteral.Repr"/> of a float is Python's <c>repr()</c> (probed: CPython 3.12) on every runtime: the
        ///     shortest digits that read back (17 for 2^-25 and 2^-958, which .NET 10's "R" writes in 16), a lower-case
        ///     exponent with at least two digits, <c>.0</c> on an integral positional value.
        /// </summary>
        [TestMethod]
        public void PyLiteralRepr_OfAFloat_IsPythons()
        {
            (double Value, string Repr)[] cases =
            {
                (System.Math.ScaleB(1.0, -25), "2.9802322387695312e-08"), (System.Math.ScaleB(1.0, -958), "4.1045368012983762e-289"),
                (1e16, "1e+16"), (9999999999999998.0, "9999999999999998.0"), (1.0, "1.0"), (-0.0, "-0.0"),
                (0.0001, "0.0001"), (1e-05, "1e-05"), (123456789012345678.0, "1.2345678901234568e+17"), (0.1, "0.1"),
                (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"), (-2.5, "-2.5"),
                (double.NaN, "nan"), (double.PositiveInfinity, "inf"), (double.NegativeInfinity, "-inf"),
            };
            foreach (var (value, repr) in cases)
                PyLiteral.Repr(value).Should().Be(repr, $"repr({value:R})");
        }
    }
}
