using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evenly round to the given number of decimals (NumPy's <c>np.round</c>, the NumPy 2.x name of the function;
        ///     <see cref="around"/> is its documented alias).
        /// </summary>
        /// <param name="a">Input array (NumPy's keyword <c>a</c>; a C# scalar converts to a 0-d array).</param>
        /// <param name="decimals">Number of decimal places to round to (default 0). Half is rounded to even; a negative
        ///     value rounds to positions left of the decimal point (<c>round([15, 25, -35], -1)</c> is
        ///     <c>[20, 20, -40]</c>).</param>
        /// <param name="out">A location into which the result is stored; must be the correct shape, returned as-is.
        ///     np.round is a FUNCTION, not a ufunc, so it takes <c>out=</c> only, no <c>where=</c> / <c>dtype=</c> (probed
        ///     2.4.2).</param>
        /// <returns>An array of the same dtype as <paramref name="a"/>, containing the rounded values (an integer input is
        ///     an identity copy for <c>decimals &gt;= 0</c>). Unless <paramref name="out"/> was specified, a new array is
        ///     created; with it, <paramref name="out"/> itself is returned. The real and imaginary parts of complex numbers
        ///     are rounded separately.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.round.html
        ///     <para>NumPy's signature <c>round(a, decimals=0, out=None)</c>. NumPy 2.0 removed <c>np.round_</c> ("Use
        ///     <c>np.round</c> instead"), and NumSharp's <c>np.round_</c> went with it on 2026-11-11, together with the
        ///     <c>dtype</c> parameter NumPy's round never had: <c>np.round(a).astype(dtype)</c> is the spelling of a
        ///     rounded result in another dtype.</para>
        /// </remarks>
        public static NDArray round(NDArray a, int decimals = 0, NDArray @out = null)
            // ndarray.round is the engine's Round; a numpy.matrix operand's class follows decimals (the rint ufunc for 0,
            // a plain result otherwise: the computation's rule).
            => a.TensorEngine.Round(a, decimals, dtype: null, @out: @out);

        /// <summary>
        ///     Evenly round to the given number of decimals (NumPy's <c>np.around</c>, the documented alias of
        ///     <see cref="round"/>: the same signature and the same result).
        /// </summary>
        /// <param name="a">Input array (NumPy's keyword <c>a</c>; a C# scalar converts to a 0-d array).</param>
        /// <param name="decimals">Number of decimal places to round to (default 0). Half is rounded to even; a negative
        ///     value rounds to positions left of the decimal point.</param>
        /// <param name="out">A location into which the result is stored; must be the correct shape, returned as-is (out=
        ///     only — see <see cref="round"/>).</param>
        /// <returns>An array of the same dtype as <paramref name="a"/>, containing the rounded values; <paramref name="out"/>
        ///     itself when it was given. The real and imaginary parts of complex numbers are rounded separately.</returns>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.around.html
        ///     <para>NumPy's signature <c>around(a, decimals=0, out=None)</c>, keyword for keyword: the input was <c>x</c>
        ///     and a NumSharp-only <c>dtype</c> parameter followed <c>out</c> until 2026-11-11.</para>
        /// </remarks>
        public static NDArray around(NDArray a, int decimals = 0, NDArray @out = null)
            // The same engine call as np.round: NumPy's around and round are two functions with one body.
            => a.TensorEngine.Round(a, decimals, dtype: null, @out: @out);
    }
}
