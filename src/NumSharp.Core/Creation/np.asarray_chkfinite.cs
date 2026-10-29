using System;
using System.Numerics;

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
        /// <returns>Array interpretation of <paramref name="a"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <exception cref="ValueError">If <paramref name="a"/> contains NaN (Not a Number) or Inf (Infinity).</exception>
        /// <remarks>
        ///     Mirrors NumPy 2.x <c>numpy.asarray_chkfinite(a, dtype=None, order=None)</c>: the finiteness
        ///     check runs ONLY for float-family dtypes (Half/Single/Double/Complex — NumPy's
        ///     <c>typecodes['AllFloat']</c>), since integer, boolean, char and decimal arrays can never
        ///     hold inf/NaN. Complex is finite iff both its real and imaginary parts are finite.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html
        /// </remarks>
        public static NDArray asarray_chkfinite(NDArray a, DType dtype = null, char order = 'K')
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Plain((a, dtype, order), static s => asarray_chkfinite(s.a, s.dtype, s.order));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Plain((a, dtype, order), static s => asarray_chkfinite(s.a, s.dtype, s.order));
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy's numpy.asarray_chkfinite: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.AsarrayChkfinite(a, dtype, order);
        }
    }
}
