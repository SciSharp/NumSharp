using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="decimals">Number of decimal places to round to (default 0). Half is rounded to even.</param>
        /// <param name="out">A location into which the result is stored; must be the correct shape, returned as-is. np.round/np.around are FUNCTIONS, not ufuncs, so they accept <c>out=</c> only — there is no <c>where=</c>/<c>dtype=</c> ufunc kwarg (probed 2.4.2). The <paramref name="dtype"/> here is NumSharp's dtype-target convenience, taken as a keyword.</param>
        /// <param name="dtype">The <see cref="DType"/> the returned ndarray should be of (a C# <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or a NumPy dtype string all convert implicitly). <see langword="null"/> preserves NumPy's round dtype rules (integer inputs are an identity copy).</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned. The real and imaginary parts of complex numbers are rounded separately. The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray round_(NDArray x, int decimals = 0, NDArray @out = null, DType dtype = null)
            => x.TensorEngine.Round(x, decimals, dtype, @out);

        /// <summary>
        ///     Evenly round to the given number of decimals (alias of <see cref="round_"/>; NumPy: <c>np.around is np.round</c>).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="decimals">Number of decimal places to round to (default 0). Half is rounded to even.</param>
        /// <param name="out">A location into which the result is stored; must be the correct shape, returned as-is (out= only — see <see cref="round_"/>).</param>
        /// <param name="dtype">The <see cref="DType"/> the returned ndarray should be of (implicit from <see cref="System.Type"/> / <see cref="NPTypeCode"/> / NumPy dtype string).</param>
        /// <returns>An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned. The real and imaginary parts of complex numbers are rounded separately. The result of rounding a float is a float.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.around.html</remarks>
        public static NDArray around(NDArray x, int decimals = 0, NDArray @out = null, DType dtype = null)
            => x.TensorEngine.Round(x, decimals, dtype, @out);
    }
}
