using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise — NumPy's two-output
        ///     <c>modf</c> ufunc. Both parts carry the sign of the input value.
        /// </summary>
        /// <param name="x">Input array. Integer/bool/char promote to the narrowest float that fits the
        ///     width (bool/int8/uint8 to float16, int16/uint16/char to float32, int32+ to float64);
        ///     float16/float32/float64 are preserved; Complex has no loop (TypeError); Decimal is a
        ///     NumSharp extension.</param>
        /// <param name="outFrac">Optional provided output for the fractional part (NumPy's <c>out[0]</c>,
        ///     i.e. the first positional out — <c>np.modf(x, out1)</c>). Must be same_kind-castable from
        ///     the loop dtype; returned as-is (reference identity). Null auto-allocates.</param>
        /// <param name="outIntegral">Optional provided output for the integral part (NumPy's <c>out[1]</c>).
        ///     Null auto-allocates.</param>
        /// <param name="where">Optional boolean write mask; masked-off slots of BOTH provided outputs keep
        ///     their prior contents.</param>
        /// <param name="dtype">Optional loop dtype override; only the float loops (Half/Single/Double/Decimal)
        ///     are valid — a non-float request raises the no-loop TypeError.</param>
        /// <returns>The (fractional, integral) pair. This is a scalar-shaped 0-d pair if <paramref name="x"/>
        ///     is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.modf.html
        ///
        ///   modf(1.5) = (0.5, 1.0);  modf(-2.7) = (-0.7, -2.0);  modf(inf) = (+0.0, +inf);
        ///   modf(-inf) = (-0.0, -inf);  modf(nan) = (nan, nan);  modf(-0.0) = (-0.0, -0.0).
        ///
        /// Fresh (auto-allocated) outputs take NumPy's ufunc output layout, the K order of the input: an
        /// F-contiguous input yields F-contiguous outputs and a permuted one keeps its memory order, matching
        /// NumPy (parity QA-36).</remarks>
        [NDScoped]
        public static (NDArray Fractional, NDArray Integral) modf(
            NDArray x, NDArray outFrac = null, NDArray outIntegral = null, NDArray where = null, DType dtype = null)
            // The engine lays fresh outputs out in NumPy's K order (DefaultEngine.ModF, as np.frexp's); a provided
            // out owns its own layout, so it is never relabelled (its reference identity is returned untouched).
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: both outputs come back as matrices, as from NumPy
                ? NDMatrixBoundary.WrappedPair((x, outFrac, outIntegral, where, dtype),
                    static s => modf(s.x, s.outFrac, s.outIntegral, s.where, s.dtype), outFrac, outIntegral)
                : x.TensorEngine.ModF(x, dtype, outFrac, outIntegral, where);
    }
}
