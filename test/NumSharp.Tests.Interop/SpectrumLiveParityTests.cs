using System;
using System.Numerics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Live byte-parity for the two kernels every NumSharp spectrum pipeline is built from — the
    ///     complex magnitude (<c>np.abs</c> on complex128) and the real FFT (<c>np.fft.rfft</c>) — each
    ///     computed by NumSharp and by live NumPy over the SAME exported buffer, so a divergence is
    ///     pinned to its stage instead of surfacing, amplified, at the end of a pipeline.
    /// </summary>
    /// <remarks>
    ///     <para>Why they exist: the Gist frequency estimators (<c>GistSignal*LiveParityTests</c>) compare
    ///     only final estimates, and on the first CI run that installed SciPy on macOS the
    ///     harmonic-product-spectrum estimate — a product of seven FFT magnitudes, then a log and a
    ///     parabolic vertex — came out 20 ULP off. The cause was a first-stage bug no end-to-end test
    ///     could name: NumSharp's complex <c>abs</c> was .NET's UNFUSED <c>Complex.Abs</c>, while
    ///     NumPy's SIMD <c>simd_cabsolute</c> FUSES its multiply-add on every current dispatch target
    ///     (35.5% of random complex values differed, on x64 too — the x64 runs had passed on lucky
    ///     inputs). These tests hold each stage to NumPy on their own, with inputs chosen to expose
    ///     exactly that kind of last-bit difference rather than avoid it.</para>
    ///     <para>Both run in the numpy-only <c>parity</c> environment (no <c>[PythonEcosystem]</c>): they
    ///     need nothing beyond NumPy, and they belong beside the oracle they protect.</para>
    /// </remarks>
    [TestClass]
    public class SpectrumLiveParityTests : InteropTestBase
    {
        /// <summary>How many seeded random complex values the magnitude test compares.</summary>
        private const int RandomCount = 100_000;

        /// <summary>
        ///     Seeded random complex values with magnitudes spread over 13 decades per component, so the
        ///     <c>ratio = smaller / larger</c> that NumPy's kernel squares takes every size from ~1e-12 to
        ///     1 — the regime where a fused and an unfused <c>ratio·ratio + 1</c> round differently.
        /// </summary>
        [TestMethod]
        public void ComplexAbs_SeededRandomValues_ByteExactVsLiveNumpy()
        {
            using var z = RandomComplex(RandomCount, seed: 20260923);
            using var magnitude = np.abs(z);
            using (Gil())
            {
                using PyObject exported = z.ToNumpy();
                using PyObject expected = Python.np.with("np.abs(z)", ("z", exported));
                AssertSameDoubles(magnitude, expected, z, "np.abs over 100K seeded random complex128 values");
            }
        }

        /// <summary>
        ///     The kernel's branches and IEEE edges — ±0 parts (the <c>larger == 0</c> mask), subnormals,
        ///     a <c>ratio²</c> that underflows, overflow to <c>+inf</c>, infinities beside NaN (C99's
        ///     inf-beats-NaN rule), NaN (NumPy's positive <c>NPY_NAN</c>) — through contiguous, strided,
        ///     reversed and transposed exports, since NumSharp's kernel walks each layout differently.
        /// </summary>
        [TestMethod]
        public void ComplexAbs_EdgeValuesAndLayouts_ByteExactVsLiveNumpy()
        {
            double inf = double.PositiveInfinity, nan = double.NaN, sub = 4.9e-322, max = double.MaxValue;
            var values = new Complex[]
            {
                new(0, 0), new(-0.0, -0.0), new(0, -0.0), new(-0.0, 0),
                new(3, 4), new(-3, 4), new(1, 1), new(1, 1e-200), new(1e-200, 1),
                new(sub, sub), new(sub, 1), new(1e-310, 3e-310),
                new(1e308, 1e308), new(max, 0), new(max, max), new(-max, 1),
                new(inf, 1), new(1, -inf), new(inf, nan), new(nan, -inf),
                new(nan, 1), new(1, nan), new(nan, nan), new(0, nan),
                new(0.1, 0.7), new(123456.789, 0.001), new(1e-5, 1e5), new(-2.5e-3, 7.75e-3),
            };

            // Tile to 3 rows so the 2-D layouts are genuine (non-degenerate) strided walks.
            var tiled = new Complex[values.Length * 3];
            for (int i = 0; i < tiled.Length; i++) tiled[i] = values[i % values.Length] * (1 + i / values.Length);
            using var baseArray = np.array(tiled).reshape(3, values.Length);

            (NDArray view, string label)[] layouts =
            {
                (baseArray, "contiguous (3,28)"),
                (baseArray[":, ::3"], "strided columns (3,10)"),
                (baseArray[":, ::-1"], "reversed columns (3,28)"),
                (baseArray.T, "transposed (28,3)"),
            };

            foreach (var (view, label) in layouts)
            {
                using var magnitude = np.abs(view);
                using (Gil())
                {
                    using PyObject exported = view.ToNumpy();
                    using PyObject expected = Python.np.with("np.abs(z)", ("z", exported));
                    AssertSameDoubles(magnitude, expected, view, $"np.abs edge values, {label}");
                }
            }
        }

        /// <summary>
        ///     <c>np.fft.rfft</c> of a seven-harmonic signal (the Gist demo's) at three sizes chosen to
        ///     drive each pocketfft route NumSharp ports: 1024 (radix-4/2 passes), 1000 = 2³·5³ (mixed
        ///     radix) and 1021 (a prime whose square exceeds it — the Bluestein chirp-z path).
        /// </summary>
        [TestMethod]
        public void Rfft_HarmonicSignal_PowerOfTwoMixedRadixAndBluestein_ByteExactVsLiveNumpy()
        {
            foreach (int n in new[] { 1024, 1000, 1021 })
            {
                using var scope = NDScope.Open();
                var time = np.arange(n).astype(NPTypeCode.Double) / 8192.0;
                var signal = np.zeros(new Shape(n), NPTypeCode.Double);
                for (int h = 1; h <= 7; h++) signal = signal + np.sin(2 * Math.PI * 384 * h * time) / h;
                var spectrum = np.fft.rfft(signal);
                using (Gil())
                {
                    using PyObject exported = signal.ToNumpy();
                    using PyObject expected = Python.np.with("np.fft.rfft(s)", ("s", exported));
                    AssertSameDoubles(spectrum, expected, signal, $"np.fft.rfft, n={n}");
                }
            }
        }

        /// <summary>
        ///     <paramref name="count"/> complex values whose real and imaginary parts are drawn
        ///     independently as <c>u · 10^k</c>, <c>u</c> uniform in (-1, 1), <c>k</c> in [-6, 6].
        /// </summary>
        /// <param name="count">How many values to draw.</param>
        /// <param name="seed">The <see cref="Random"/> seed, so a failure reproduces.</param>
        /// <returns>A contiguous complex128 array; the caller owns it.</returns>
        private static NDArray RandomComplex(int count, int seed)
        {
            var rng = new Random(seed);
            var values = new Complex[count];
            for (int i = 0; i < count; i++)
                values[i] = new Complex((rng.NextDouble() * 2 - 1) * Math.Pow(10, rng.Next(-6, 7)),
                                        (rng.NextDouble() * 2 - 1) * Math.Pow(10, rng.Next(-6, 7)));
            return np.array(values);
        }

        /// <summary>
        ///     Asserts NumSharp's float64 result is byte-identical to NumPy's, reporting HOW MANY 8-byte
        ///     lanes differ and the first one (both values' bit patterns and the input element) instead
        ///     of dumping two hundred-kilobyte byte arrays. Call under the GIL.
        /// </summary>
        /// <param name="numsharp">NumSharp's result (float64, or complex128 read as float64 pairs).</param>
        /// <param name="numpy">NumPy's result over the same exported input.</param>
        /// <param name="input">The input, printed at the first difference when it is complex128.</param>
        /// <param name="because">What was computed, for the failure message.</param>
        /// <exception cref="AssertFailedException">The byte lengths or any lane differ.</exception>
        private static void AssertSameDoubles(NDArray numsharp, PyObject numpy, NDArray input, string because)
        {
            byte[] ours = ByteContract.NsBytes(numsharp);
            byte[] theirs = numpy.bytes_c();
            ours.Length.Should().Be(theirs.Length, $"{because}: result byte lengths");

            ReadOnlySpan<long> a = MemoryMarshal.Cast<byte, long>(ours);
            ReadOnlySpan<long> b = MemoryMarshal.Cast<byte, long>(theirs);
            int differing = 0, first = -1;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] == b[i]) continue;
                differing++;
                if (first < 0) first = i;
            }

            if (differing == 0)
                return;

            string where = "";
            if (input.typecode == NPTypeCode.Complex && numsharp.typecode == NPTypeCode.Double)
            {
                // A magnitude lane i came from input element i (in C order).
                using var flat = np.ascontiguousarray(input).ravel();
                var z = (Complex)flat.GetAtIndex(first);
                where = $" (input {z.Real:R} + {z.Imaginary:R}j)";
            }

            Assert.Fail($"{because}: {differing} of {a.Length} float64 lanes differ; first at lane {first}{where}: " +
                        $"NumSharp=0x{a[first]:X16} ({BitConverter.Int64BitsToDouble(a[first]):R}), " +
                        $"NumPy=0x{b[first]:X16} ({BitConverter.Int64BitsToDouble(b[first]):R}).");
        }
    }
}
