using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the set difference of two arrays.<br></br>
        ///     Return the unique values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>.
        /// </summary>
        /// <param name="ar1">Input array.</param>
        /// <param name="ar2">Input comparison array.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>1-D array of values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>. Sorted
        ///     when <paramref name="assume_unique"/> is False.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html</remarks>
        /// <exception cref="UnicodeDecodeError">The membership test's sort path decodes <c>S</c> text holding a byte that is
        ///     not ASCII (see <c>np.isin</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> array whose sort compares a None-like NA (an NA object
        ///     that is neither a string nor NaN-like) with another entry: NumPy's <c>Cannot compare null that is not a
        ///     nan-like value</c>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="ar1"/> is null.</exception>
        [NDScoped]
        public static NDArray setdiff1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            if (ar1 is null) throw new ArgumentNullException(nameof(ar1));
            // a numpy.char.chararray ar1: np.unique keeps the class and its boolean index is a chararray VIEW (with assume_unique NumPy
            // converts with np.asarray: a plain array) (NDCharArrayBoundary)
            if (!assume_unique && NDCharArrayBoundary.Sees(ar1))
                return NDCharArrayBoundary.Wrapped((ar1, ar2, assume_unique), static s => setdiff1d(s.ar1, s.ar2, s.assume_unique));
            using var plain = NDMatrixBoundary.PlainFor(ar1, ar2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return ar1.TensorEngine.Setdiff1d(ar1, ar2, assume_unique);
        }
    }
}
