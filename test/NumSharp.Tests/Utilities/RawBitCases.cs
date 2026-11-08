using System;
using System.Collections.Generic;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Utilities
{
    /// <summary>
    ///     Raw-bit case runner for reduction tests that must match NumPy to the bit: each case renders a result as
    ///     <c>dtype (shape) bits…</c> and compares the string with NumPy 2.4.2's output for the same call. A value
    ///     comparison treats <c>-0.0</c> and <c>+0.0</c> as equal (and every NaN as equal to every other), which is how
    ///     the signed-zero reduction bugs this runner pins went unnoticed — so it compares raw bits instead.
    /// </summary>
    /// <remarks>
    ///     Cases are collected, not asserted one by one: <see cref="Check"/> records every mismatch (a throwing case
    ///     included) and <see cref="AssertNoMismatch"/> fails once with the full list, so a partial fix shows exactly
    ///     which cases still differ. Import with <c>using static NumSharp.Tests.Utilities.RawBitCases;</c>.
    /// </remarks>
    internal static class RawBitCases
    {
        /// <summary>Builds a 1-D float64 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float64 array holding <c>-0.0</c> in every slot.</returns>
        internal static NDArray F64(int n)
        {
            var v = new double[n];
            Array.Fill(v, -0.0);
            return np.array(v);
        }

        /// <summary>Builds a 1-D float32 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float32 array holding <c>-0.0f</c> in every slot.</returns>
        internal static NDArray F32(int n)
        {
            var v = new float[n];
            Array.Fill(v, -0.0f);
            return np.array(v);
        }

        /// <summary>Builds a 1-D float16 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float16 array holding <see cref="Half.NegativeZero"/> in every slot.</returns>
        internal static NDArray F16(int n)
        {
            var v = new Half[n];
            Array.Fill(v, Half.NegativeZero);
            return np.array(v);
        }

        /// <summary>Builds a 1-D complex128 array of <paramref name="n"/> values <c>-0.0 - 0.0j</c>.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous complex128 array whose real and imaginary parts are both <c>-0.0</c>.</returns>
        internal static NDArray C128(int n)
        {
            var v = new Complex[n];
            Array.Fill(v, new Complex(-0.0, -0.0));
            return np.array(v);
        }

        /// <summary>
        ///     Runs one case and records a line when its <see cref="Describe"/> string differs from NumPy's. An
        ///     exception is recorded as a mismatch too, so one throwing case cannot hide the state of the others in the
        ///     same test.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test.</param>
        /// <param name="expected">NumPy 2.4.2's result in <see cref="Describe"/>'s format.</param>
        /// <param name="rawNaN">When true a float NaN is compared by its raw bits (sign, quiet bit and payload); when
        /// false it is the token <c>nan</c>, for cases whose NaN bits are not part of NumPy's contract (a 0/0 NaN's
        /// sign).</param>
        internal static void Check(List<string> mismatches, string name, Func<NDArray> call, string expected, bool rawNaN = false)
        {
            string actual;
            try
            {
                actual = Describe(call(), rawNaN);
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            if (actual != expected)
                mismatches.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>
        ///     Runs one case that NumPy rejects and records a line unless it throws <typeparamref name="TException"/>
        ///     whose message starts with NumPy's text. The prefix comparison is deliberate: NumSharp's
        ///     <see cref="AxisError"/> derives from <see cref="ArgumentOutOfRangeException"/>, so .NET appends
        ///     <c>(Parameter 'axis')</c> to the message NumPy prints.
        /// </summary>
        /// <typeparam name="TException">The exception type NumPy's error maps to.</typeparam>
        /// <param name="mismatches">Collects one line per failing case.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test.</param>
        /// <param name="numpyMessage">NumPy 2.4.2's exception message, verbatim.</param>
        internal static void CheckThrows<TException>(List<string> mismatches, string name, Func<NDArray> call, string numpyMessage)
            where TException : Exception
        {
            try
            {
                var r = call();
                mismatches.Add($"{name}: expected {typeof(TException).Name} [{numpyMessage}] but NumSharp returned [{Describe(r, rawNaN: false)}]");
            }
            catch (TException e) when (e.Message.StartsWith(numpyMessage, StringComparison.Ordinal))
            {
                // The expected outcome: NumPy's exception type with NumPy's text.
            }
            catch (Exception e)
            {
                mismatches.Add($"{name}: expected {typeof(TException).Name} [{numpyMessage}] but NumSharp threw {e.GetType().Name} [{e.Message}]");
            }
        }

        /// <summary>
        ///     Fails the test when a check recorded any mismatch, listing EVERY recorded line. A plain <c>BeEmpty()</c>
        ///     on the list prints only its first item, which hides how much of a path a partial fix has repaired.
        /// </summary>
        /// <param name="mismatches">The collected lines; empty means every case matched NumPy.</param>
        /// <param name="because">The NumPy rule the cases check, shown in the failure message.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, when <paramref name="mismatches"/>
        /// is not empty.</exception>
        internal static void AssertNoMismatch(List<string> mismatches, string because)
        {
            string.Join(Environment.NewLine, mismatches).Should().BeEmpty(because);
        }

        /// <summary>
        ///     Renders a result as <c>dtype (shape) bits…</c>: the NumPy dtype name, the shape in tuple form
        ///     (<c>()</c>, <c>(3,)</c>, <c>(1,3)</c>), then every element in logical C order — float64 as 16 hex digits,
        ///     float32 as 8, float16 as 4, complex128 as its real then imaginary word, any other dtype as its value
        ///     (<c>True</c>/<c>False</c> for bool).
        /// </summary>
        /// <param name="r">The result to render.</param>
        /// <param name="rawNaN">When false a float16/32/64 NaN element is the token <c>nan</c>; when true it is its raw
        /// bits like any other value. A complex128 element is always raw bits (both parts).</param>
        /// <returns>The rendering; an empty result ends with the space after its shape.</returns>
        internal static string Describe(NDArray r, bool rawNaN = false)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // flatten() copies in logical C order, the order NumPy's reshape(-1) printed the expected bits in; a 0-d
            // result has one element, so a reshape view of it is enough.
            var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            var parts = new List<string>();
            for (long i = 0; i < flat.size; i++)
            {
                switch (flat.typecode)
                {
                    case NPTypeCode.Double:
                    {
                        double v = flat.GetAtIndex<double>(i);
                        parts.Add(double.IsNaN(v) && !rawNaN ? "nan" : BitConverter.DoubleToUInt64Bits(v).ToString("x16"));
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        float v = flat.GetAtIndex<float>(i);
                        parts.Add(float.IsNaN(v) && !rawNaN ? "nan" : BitConverter.SingleToUInt32Bits(v).ToString("x8"));
                        break;
                    }
                    case NPTypeCode.Half:
                    {
                        Half v = flat.GetAtIndex<Half>(i);
                        parts.Add(Half.IsNaN(v) && !rawNaN ? "nan" : BitConverter.HalfToUInt16Bits(v).ToString("x4"));
                        break;
                    }
                    case NPTypeCode.Complex:
                    {
                        Complex c = flat.GetAtIndex<Complex>(i);
                        parts.Add(BitConverter.DoubleToUInt64Bits(c.Real).ToString("x16"));
                        parts.Add(BitConverter.DoubleToUInt64Bits(c.Imaginary).ToString("x16"));
                        break;
                    }
                    default:
                        parts.Add(flat.GetAtIndex(i).ToString());
                        break;
                }
            }

            return $"{r.dtype.name} {shape} {string.Join(" ", parts)}";
        }
    }
}
