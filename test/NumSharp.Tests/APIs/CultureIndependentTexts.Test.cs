using System;
using System.Globalization;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     Parity QA-41: NumPy's texts are culture-independent ASCII, so NumSharp's are too. <c>np.finfo</c> /
    ///     <c>np.iinfo</c> print NumPy's reprs byte for byte, and every exception text formats its numbers with the
    ///     invariant culture (the build enforces it on NumSharp.Core with NDW020) under cultures that change a number's
    ///     text: de-DE (decimal comma), sv-SE (U+2212 minus), ar-SA (U+061C before the minus, U+066B decimal separator),
    ///     fa-IR (U+200E + U+2212), en-US (<c>∞</c> for an infinity). Every expected text is NumPy 2.4.2's
    ///     (scratchpad probes p176 / p176_numpy.py, win-amd64).
    /// </summary>
    [TestClass]
    public class CultureIndependentTextsTests
    {
        private static readonly string[] Cultures = { "", "en-US", "de-DE", "sv-SE", "ar-SA", "fa-IR" };

        /// <summary>
        ///     Runs <paramref name="check"/> once under each culture of <see cref="Cultures"/> on this thread, restoring the
        ///     thread's culture afterwards (the setting is per thread, so parallel tests are not affected).
        /// </summary>
        /// <param name="check">The assertions; receives the culture's name for the failure message.</param>
        /// <exception cref="AssertFailedException">Whatever <paramref name="check"/> raises, under the first culture that fails.</exception>
        private static void UnderEachCulture(Action<string> check)
        {
            var saved = CultureInfo.CurrentCulture;
            try
            {
                foreach (var name in Cultures)
                {
                    CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(name);
                    check(name.Length == 0 ? "the invariant culture" : name);
                }
            }
            finally
            {
                CultureInfo.CurrentCulture = saved;
            }
        }

        /// <summary>
        ///     <c>repr(np.finfo(t))</c>: resolution through the dtype's own scalar <c>str()</c>, min / max through its
        ///     <c>_MACHAR_PARAMS</c> printf format (<c>%24.16e</c> / <c>%15.7e</c> / <c>%12.5e</c>), the long double's
        ///     <c>str()</c>, a complex dtype's real part. Before QA-41 it was <c>finfo(resolution=1E-06,
        ///     min=-3.402823E+038, ...)</c> with a decimal comma under de-DE.
        /// </summary>
        /// <exception cref="AssertFailedException">A repr differs from NumPy's under some culture.</exception>
        [TestMethod]
        public void Finfo_Repr_IsNumPys_UnderEveryCulture()
        {
            UnderEachCulture(culture =>
            {
                np.finfo(np.float16).ToString().Should().Be("finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)", culture);
                np.finfo(np.float32).ToString().Should().Be("finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)", culture);
                np.finfo(np.float64).ToString().Should().Be("finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)", culture);
                np.finfo(np.longdouble).ToString().Should().Be("finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)", culture);
                np.finfo(np.complex128).ToString().Should().Be("finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)", culture);
            });
        }

        /// <summary>
        ///     <c>repr(np.iinfo(t))</c> for every integer width (the C classes included): Python ints, no culture, and an
        ///     unsigned type's true maximum (uint64 prints 18446744073709551615, not the clamped <c>max</c> property).
        /// </summary>
        /// <exception cref="AssertFailedException">A repr differs from NumPy's under some culture.</exception>
        [TestMethod]
        public void Iinfo_Repr_IsNumPys_UnderEveryCulture()
        {
            UnderEachCulture(culture =>
            {
                np.iinfo(np.int8).ToString().Should().Be("iinfo(min=-128, max=127, dtype=int8)", culture);
                np.iinfo(np.uint8).ToString().Should().Be("iinfo(min=0, max=255, dtype=uint8)", culture);
                np.iinfo(np.int16).ToString().Should().Be("iinfo(min=-32768, max=32767, dtype=int16)", culture);
                np.iinfo(np.uint16).ToString().Should().Be("iinfo(min=0, max=65535, dtype=uint16)", culture);
                np.iinfo(np.int32).ToString().Should().Be("iinfo(min=-2147483648, max=2147483647, dtype=int32)", culture);
                np.iinfo(np.uint32).ToString().Should().Be("iinfo(min=0, max=4294967295, dtype=uint32)", culture);
                np.iinfo(np.int64).ToString().Should().Be("iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)", culture);
                np.iinfo(np.uint64).ToString().Should().Be("iinfo(min=0, max=18446744073709551615, dtype=uint64)", culture);
                np.iinfo(np.intc).ToString().Should().Be("iinfo(min=-2147483648, max=2147483647, dtype=int32)", culture);
                np.iinfo(np.uintc).ToString().Should().Be("iinfo(min=0, max=4294967295, dtype=uint32)", culture);
            });
        }

        /// <summary>
        ///     Error texts that interpolate a negative number keep NumPy's ASCII hyphen-minus under every culture: an index
        ///     out of bounds (QA-41's <c>index ؜-6</c> under ar-SA), np.take's index, np.linspace's sample count,
        ///     np.partition's kth, and the AxisError, whose .NET base appends <c>Actual value was …</c> (that line formats
        ///     the axis invariantly too, the value handed over as text).
        /// </summary>
        /// <exception cref="AssertFailedException">A text differs from NumPy's under some culture.</exception>
        [TestMethod]
        public void ErrorTexts_FormatNegativeNumbersInvariantly_UnderEveryCulture()
        {
            using var a = np.arange(5.0);
            using var ones = np.ones(new Shape(2, 2));
            using var idx = np.array(new long[] { -9 });
            UnderEachCulture(culture =>
            {
                new Action(() => a[-6].Dispose()).Should().Throw<IndexError>()
                    .WithMessage("index -6 is out of bounds for axis 0 with size 5", culture);
                new Action(() => np.take(a, idx).Dispose()).Should().Throw<IndexError>()
                    .WithMessage("index -9 is out of bounds for axis 0 with size 5", culture);
                new Action(() => np.linspace(0, 1, -1).Dispose()).Should().Throw<ValueError>()
                    .WithMessage("Number of samples, -1, must be non-negative.", culture);
                new Action(() => np.partition(a, -9).Dispose()).Should().Throw<ValueError>()
                    .WithMessage("kth(=-4) out of bounds (5)", culture);

                var axis = new Action(() => np.sum(ones, -3).Dispose()).Should().Throw<AxisError>().Which;
                axis.Message.Should().StartWith("axis -3 is out of bounds for array of dimension 2", culture);
                axis.Message.Should().EndWith("Actual value was -3.", culture);
                axis.Axis.Should().Be(-3);
                axis.ActualValue.Should().Be("-3", culture);
            });
        }

        /// <summary>
        ///     A float in an error text keeps a decimal point under a decimal-comma culture: the power-series guard for a
        ///     power beyond int64 (a NumSharp text: NumPy would run out of memory) prints <c>1.5E+19</c>, which read
        ///     <c>1,5E+19</c> under de-DE. Since API-parity batch 11 the public <c>{p}pow</c> takes NumPy's integer power
        ///     only, so the float-power guard is reached through the computation it lives in.
        /// </summary>
        /// <exception cref="AssertFailedException">The text changed with the culture.</exception>
        [TestMethod]
        public void ErrorTexts_FormatFloatsInvariantly_UnderEveryCulture()
        {
            UnderEachCulture(culture =>
                new Action(() => NumSharp.NDPolyAlgebra.Pow(NumSharp.Backends.Kernels.PolyBasis.Power, new[] { 1.0 }, 1.5e19, null).Dispose())
                    .Should().Throw<OverflowException>()
                    .WithMessage("power 1.5E+19 is beyond int64: NumPy would run out of memory computing it", culture));
        }

        /// <summary>
        ///     The shape texts error messages are built from print the same digits under every culture
        ///     (<c>Shape.ToString</c> and the Python-tuple forms NumPy's texts use).
        /// </summary>
        /// <exception cref="AssertFailedException">A shape text changed with the culture.</exception>
        [TestMethod]
        public void ShapeTexts_AreCultureIndependent()
        {
            UnderEachCulture(culture =>
            {
                new Shape(2, 3).ToString().Should().Be("(2, 3)", culture);
                new Shape(2, 3).ToPythonTuple().Should().Be("(2,3)", culture);
                new Shape(7).ToPythonTuple().Should().Be("(7,)", culture);
            });
        }
    }
}
