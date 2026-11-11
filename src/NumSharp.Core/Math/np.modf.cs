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
        /// <param name="out">NumPy's <c>out=(o1, o2)</c>: the fractional output and the integral output, each
        ///     same_kind-castable from the loop dtype and returned as-is (reference identity); either slot may be null
        ///     (auto-allocated), <c>default</c> allocates both.</param>
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
        /// NumPy (parity QA-36).
        /// <para>NumPy's signature <c>modf(x, /, out=(None, None), *, where=True, ..., dtype=None)</c>: the outputs are ONE
        /// tuple parameter, as <see cref="divmod"/>'s (the former <c>outFrac</c> / <c>outIntegral</c> went on
        /// 2026-11-11).</para></remarks>
        /// <exception cref="IncorrectTypeException">A complex input, or a non-float <paramref name="dtype"/> (no loop).</exception>
        [NDScoped]
        public static (NDArray Fractional, NDArray Integral) modf(
            NDArray x, (NDArray, NDArray) @out = default, NDArray where = null, DType dtype = null)
            // The engine lays fresh outputs out in NumPy's K order (DefaultEngine.ModF, as np.frexp's); a provided
            // out owns its own layout, so it is never relabelled (its reference identity is returned untouched).
            => x.TensorEngine.ModF(x, dtype, @out.Item1, @out.Item2, where);
    }
}
