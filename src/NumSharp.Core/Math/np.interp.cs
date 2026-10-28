using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     One-dimensional linear interpolation for monotonically increasing sample points. <br></br>
        ///     Returns the piecewise-linear interpolant to the discrete data points (xp, fp), evaluated
        ///     at each coordinate in x. Port of NumPy's <c>numpy.interp</c>
        ///     (<c>compiled_base.c::arr_interp</c> + the <c>_function_base_impl.py</c> wrapper).
        /// </summary>
        /// <param name="x">The x-coordinates at which to evaluate the interpolated values (any shape; the result has the same shape).</param>
        /// <param name="xp">The x-coordinates of the data points — 1-D, must be increasing unless <paramref name="period"/> is given.</param>
        /// <param name="fp">The y-coordinates of the data points, same length as xp (float or complex).</param>
        /// <param name="left">Value returned for <c>x &lt; xp[0]</c>; default is <c>fp[0]</c>. Ignored when <paramref name="period"/> is given.</param>
        /// <param name="right">Value returned for <c>x &gt; xp[-1]</c>; default is <c>fp[-1]</c>. Ignored when <paramref name="period"/> is given.</param>
        /// <param name="period">A period for the x-coordinates, allowing proper interpolation of angular x-coordinates. Must be non-zero.</param>
        /// <returns>The interpolated values, same shape as x (float64, or complex128 when fp is complex). A scalar (0-d) if x is a scalar.</returns>
        /// <exception cref="ValueError"><paramref name="fp"/> or <paramref name="xp"/> is not 1-D (<c>object too deep for desired
        ///     array</c> / <c>object of too small depth for desired array</c>), xp is empty, the lengths differ, or
        ///     <paramref name="period"/> is zero (and, with a period, <c>Data points must be 1-D sequences</c>).</exception>
        /// <exception cref="InvalidCastException">Without a period, an operand does not cast 'safely' to the loop's dtype
        ///     (text, or a complex x / xp — NumPy's TypeError <c>Cannot cast array data from dtype('&lt;U2') to dtype('float64')
        ///     according to the rule 'safe'</c>, <c>Cannot cast scalar …</c> for a 0-d x).</exception>
        /// <exception cref="ValueError">With a period, a text operand that does not parse as float64: NumPy parses the text
        ///     first (<c>np.asarray(…, dtype=float64)</c>: <c>could not convert string to float</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        ///     <para>
        ///     <b>Check order.</b> NumPy picks the loop from <paramref name="fp"/> alone (<c>np.iscomplexobj(fp)</c>). Without a
        ///     period, the compiled loop converts fp, then xp, then x (<c>PyArray_ContiguousFromAny</c>), each by checking the
        ///     operand's rank FIRST (fp and xp must be 1-D; x may have any rank) and then its 'safe' cast — so a 2-D text xp is
        ///     the rank's ValueError, not the cast's TypeError, and fp's error wins over xp's. With a period, NumPy instead
        ///     converts x, xp and fp with <c>np.asarray(…, dtype=…)</c>, which PARSES text (an empty text array converts to an
        ///     empty float array without parsing).
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            double? left = null, double? right = null, double? period = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, xp, fp);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return np.EngineOf(x).Interp(x, xp, fp, left, right, period);
        }

        /// <summary>
        ///     Complex-fill overload of <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>: complex
        ///     <paramref name="left"/> / <paramref name="right"/> fill values for a complex <paramref name="fp"/>.
        /// </summary>
        /// <param name="x">The x-coordinates at which to evaluate the interpolated values.</param>
        /// <param name="xp">The x-coordinates of the data points (1-D).</param>
        /// <param name="fp">The y-coordinates of the data points (1-D, same length as xp).</param>
        /// <param name="left">Value returned for <c>x &lt; xp[0]</c>; null means <c>fp[0]</c>.</param>
        /// <param name="right">Value returned for <c>x &gt; xp[-1]</c>; null means <c>fp[-1]</c>.</param>
        /// <param name="period">A period for the x-coordinates; when given, <paramref name="left"/> and <paramref name="right"/>
        ///     are ignored.</param>
        /// <returns>The interpolated values, same shape as x: complex128 for a complex fp, else float64.</returns>
        /// <exception cref="ValueError">As <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="InvalidCastException">As <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="ValueError">As <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="fp"/> is not complex, no period is given, and <paramref name="left"/> or
        ///     <paramref name="right"/> is given: <c>must be real number, not complex</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        ///     <para>
        ///     NumPy picks the loop from <paramref name="fp"/> only. A real fp runs the real loop, which converts the fill values
        ///     with <c>PyFloat_AsDouble</c> AFTER its operand and length checks — and that refuses a Python complex even with a
        ///     zero imaginary part. (This overload used to compute in complex for a real fp, and so returned complex128 where
        ///     NumPy raises or returns float64.)
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            Complex? left, Complex? right = null, double? period = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, xp, fp);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return np.EngineOf(x).Interp(x, xp, fp, left, right, period);
        }
    }
}
