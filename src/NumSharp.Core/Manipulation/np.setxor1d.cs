using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the set exclusive-or of two arrays.<br></br>
        ///     Return the sorted, unique values that are in only one (not both) of the input arrays.
        /// </summary>
        /// <param name="ar1">Input array.</param>
        /// <param name="ar2">Input array.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>Sorted 1-D array of unique values that are in only one of the input arrays.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.setxor1d.html</remarks>
        /// <exception cref="UnicodeDecodeError">A <c>U</c> result over <c>S</c> text holding a non-ASCII byte.</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> operand beside a class the concatenation finds no common
        ///     DType with (a number, bool, <c>S</c> bytes): NumPy's verbatim promotion text.</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> array whose sort compares a None-like NA (an NA object
        ///     that is neither a string nor NaN-like) with another entry: NumPy's <c>Cannot compare null that is not a
        ///     nan-like value</c>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="ar1"/> is null.</exception>
        [NDScoped]
        public static NDArray setxor1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            if (ar1 is null) throw new ArgumentNullException(nameof(ar1));
            using var plain = NDMatrixBoundary.PlainFor(ar1, ar2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return ar1.TensorEngine.Setxor1d(ar1, ar2, assume_unique);
        }
    }
}
