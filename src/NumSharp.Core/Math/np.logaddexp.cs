using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Logarithm of the sum of exponentiations of the inputs. <br></br>
        ///     Calculates <c>log(exp(x1) + exp(x2))</c>, element-wise, without overflow/underflow —
        ///     the stable way to add probabilities held as logarithms.
        ///     Mirrors NumPy's ufunc signature: <c>logaddexp(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First input array (a log-domain value).</param>
        /// <param name="x2">Second input array. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (float-family only; int/bool/complex raise NumPy's "No loop matching" error).</param>
        /// <returns><c>log(exp(x1) + exp(x2))</c>. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html</remarks>
        public static NDArray logaddexp(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.LogAddExp(x1, x2, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload
        ///     (NumPy accepts dtype only as a keyword).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html</remarks>
        public static NDArray logaddexp(NDArray x1, NDArray x2, NPTypeCode dtype)
            => x1.TensorEngine.LogAddExp(x1, x2, dtype);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html</remarks>
        public static NDArray logaddexp(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.LogAddExp(x1, x2, dtype);

        /// <summary>
        ///     Logarithm of the sum of exponentiations of the inputs in base-2. <br></br>
        ///     Calculates <c>log2(2**x1 + 2**x2)</c>, element-wise — the base-2 analogue of
        ///     <see cref="logaddexp(NDArray,NDArray,NDArray,NDArray,NPTypeCode?)"/>, useful in machine
        ///     learning when the calculated probabilities are expressed in base-2.
        ///     Mirrors NumPy's ufunc signature: <c>logaddexp2(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First input array (a log2-domain value).</param>
        /// <param name="x2">Second input array. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc out=).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (float-family only).</param>
        /// <returns><c>log2(2**x1 + 2**x2)</c>. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html</remarks>
        public static NDArray logaddexp2(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.LogAddExp2(x1, x2, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html</remarks>
        public static NDArray logaddexp2(NDArray x1, NDArray x2, NPTypeCode dtype)
            => x1.TensorEngine.LogAddExp2(x1, x2, dtype);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html</remarks>
        public static NDArray logaddexp2(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.LogAddExp2(x1, x2, dtype);

        /// <summary>
        ///     Return the next floating-point value after x1 towards x2, element-wise. <br></br>
        ///     Mirrors NumPy's ufunc signature: <c>nextafter(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Values to find the next representable value of.</param>
        /// <param name="x2">The direction where to look for the next representable value of x1. If shapes differ they must broadcast to a common shape.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc out=).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (float-family only).</param>
        /// <returns>The next representable values of x1 in the direction of x2. This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html</remarks>
        public static NDArray nextafter(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
            => x1.TensorEngine.NextAfter(x1, x2, dtype, @out, where);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html</remarks>
        public static NDArray nextafter(NDArray x1, NDArray x2, NPTypeCode dtype)
            => x1.TensorEngine.NextAfter(x1, x2, dtype);

        /// <summary>
        ///     Computed in <paramref name="dtype"/> — positional-dtype convenience overload.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html</remarks>
        public static NDArray nextafter(NDArray x1, NDArray x2, Type dtype)
            => x1.TensorEngine.NextAfter(x1, x2, dtype);
    }
}
