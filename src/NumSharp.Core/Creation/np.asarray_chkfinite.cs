using System;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">Input data. No copy is performed if the input is already an ndarray matching the requested dtype/order.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <param name="order">'C' (row-major), 'F' (column-major), 'A' (any), 'K' (keep — default, equivalent to NumPy's <c>order=None</c>).</param>
        /// <returns>Array interpretation of <paramref name="a"/>.</returns>
        /// <exception cref="ValueError">If <paramref name="a"/> contains NaN (Not a Number) or Inf (Infinity).</exception>
        /// <remarks>
        ///     Mirrors NumPy 2.x <c>numpy.asarray_chkfinite(a, dtype=None, order=None)</c>: the finiteness
        ///     check runs ONLY for float-family dtypes (Half/Single/Double/Complex — NumPy's
        ///     <c>typecodes['AllFloat']</c>), since integer, boolean, char and decimal arrays can never
        ///     hold inf/NaN. Complex is finite iff both its real and imaginary parts are finite.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html
        /// </remarks>
        public static NDArray asarray_chkfinite(NDArray a, Type dtype = null, char order = 'K')
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            var arr = asarray(a, dtype, order);

            if (IsFloatFamily(arr.typecode) && !FiniteScan.IsAllFinite(arr))
                throw new ValueError("array must not contain infs or NaNs");

            return arr;
        }

        /// <summary>
        ///     Convert the input to an array, checking for NaNs or Infs. Convenience overload taking a
        ///     NumPy-style dtype string (e.g. <c>"float32"</c>, <c>"&lt;c16"</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html</remarks>
        public static NDArray asarray_chkfinite(NDArray a, string dtype, char order = 'K')
            => asarray_chkfinite(a, dtype == null ? null : np.dtype(dtype).type, order);

        /// <summary>
        ///     Convert the input to an array, checking for NaNs or Infs. Convenience overload taking <see cref="NPTypeCode"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html</remarks>
        public static NDArray asarray_chkfinite(NDArray a, NPTypeCode dtype, char order = 'K')
            => asarray_chkfinite(a, dtype == NPTypeCode.Empty ? null : dtype.AsType(), order);

        /// <summary>
        ///     True for the dtypes NumPy checks for finiteness (its <c>typecodes['AllFloat']</c>: the real
        ///     and complex floating kinds). Decimal is excluded — it has no NumPy analog and, being exact,
        ///     can never represent inf/NaN.
        /// </summary>
        private static bool IsFloatFamily(NPTypeCode tc)
            => tc == NPTypeCode.Half || tc == NPTypeCode.Single || tc == NPTypeCode.Double || tc == NPTypeCode.Complex;
    }
}
