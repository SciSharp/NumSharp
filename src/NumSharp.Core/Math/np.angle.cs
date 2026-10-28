using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the angle (phase) of the complex argument, element-wise — the counterclockwise angle
        ///     from the positive real axis, in the range (-pi, pi]. One of the four basic complex-number
        ///     accessors (with <see cref="real"/>, <see cref="imag"/> and <see cref="conjugate"/>) — the
        ///     standard post-FFT PHASE spectrum: for <c>A = np.fft.fft(a)</c>, <c>np.angle(A)</c> is the
        ///     phase spectrum.
        /// </summary>
        /// <param name="z">A complex number or sequence of complex numbers (any real dtype is also accepted).</param>
        /// <param name="deg">Return the angle in degrees if <c>true</c>, radians (default) if <c>false</c>.</param>
        /// <returns>
        ///     The phase angle, computed as <c>arctan2(imag, real)</c> element-wise (so it follows the
        ///     <see cref="arctan2(NDArray, NDArray, NDArray, NDArray, System.Nullable{NPTypeCode})"/> IEEE
        ///     conventions at magnitude zero and at the infinities). complex128 -&gt; float64. A REAL input
        ///     yields <c>0</c> for a positive value and <c>pi</c> for a negative one (<c>arctan2(0, z)</c>),
        ///     with NumPy's per-dtype float tier: bool/int32+/float64/complex -&gt; float64, int8/uint8/float16
        ///     -&gt; float16, int16/uint16/float32 -&gt; float32.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.angle.html. The facade holds only the matrix boundary; NumPy's body is the engine's <see cref="TensorEngine.Angle(NDArray, bool)"/>.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">Text: <c>arctan2</c> has no string loop (NumPy's no-loop text for <c>StringDType</c>, the legacy resolver's text for <c>S</c> / <c>U</c>).</exception>
        [NDScoped]
        public static NDArray angle(NDArray z, bool deg = false)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(z))
                return NDMatrixBoundary.Wrapped((z, deg), static s => angle(s.z, s.deg));
            // NumPy's angle body (the complex test, arctan2 and the degree scaling, the text refusals) is the
            // engine member's: the facade holds only the class rule above.
            return z.TensorEngine.Angle(z, deg);
        }
    }
}
