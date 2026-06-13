using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Complex-math helpers matching NumPy's complex ufunc edge semantics where the
    /// .NET BCL diverges.
    ///
    /// <para><see cref="Abs"/> replicates NumPy's <c>npy_cabs</c>
    /// (<c>= npy_hypot(creal(z), cimag(z))</c>, <c>npy_math_complex.c.src</c>): under C99
    /// <c>hypot</c>, an infinite component yields <c>+inf</c> <em>even when the other component
    /// is NaN</em> (the infinity test precedes the NaN test). <see cref="Complex.Abs"/> routes
    /// through a private <c>Hypot</c> that orders its operands with a NaN-unaware comparison, so on
    /// <c>net8.0</c> it returns <c>NaN</c> for <c>abs(NaN + inf*i)</c> instead of <c>+inf</c>
    /// (fixed in the .NET 9+ BCL). Guarding the infinity case explicitly makes every target
    /// framework agree with NumPy while leaving the finite/NaN-only magnitudes — which already
    /// match NumPy bit-for-bit — to <see cref="Complex.Abs"/>.</para>
    /// </summary>
    public static class NpyComplexMath
    {
        /// <summary>
        /// <c>|z| = hypot(re, im)</c> with NumPy/C99 infinity semantics: a ±infinite real or
        /// imaginary part returns <c>+inf</c> regardless of the other part (including NaN).
        /// All other inputs defer to <see cref="Complex.Abs"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Abs(Complex z)
        {
            if (double.IsInfinity(z.Real) || double.IsInfinity(z.Imaginary))
                return double.PositiveInfinity;
            return Complex.Abs(z);
        }
    }
}
