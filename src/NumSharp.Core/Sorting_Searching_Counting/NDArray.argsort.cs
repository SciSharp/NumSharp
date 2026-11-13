using System;
using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns the indices that would sort this array along <paramref name="axis"/> (null flattens). NumPy's
        ///     <c>ndarray.argsort(axis=-1, kind=None, order=None, *, stable=None, descending=False)</c>.
        /// </summary>
        /// <param name="axis">Axis to sort along; null sorts the flattened array.</param>
        /// <param name="kind">
        ///     The sort algorithm, as NumPy parses it: the first letter decides, case-insensitively — <c>'q'</c>uicksort
        ///     (the default), <c>'h'</c>eapsort (the default kind under another name), <c>'m'</c>ergesort or
        ///     <c>'s'</c>table. Over <c>S</c> / <c>U</c> text the default kind gives NumPy's introsort permutation (equal
        ///     strings in its order, which is not their input order once a line has more than 16 elements) and the stable
        ///     kinds the stable permutation; NumSharp's numeric ascending argsort gives the stable permutation for every kind,
        ///     its descending default-kind argsort NumPy's introsort permutation.
        /// </param>
        /// <param name="order">Field names; any value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">NumPy's keyword-only <c>stable</c>: true asks for the stable kind. It cannot be combined
        ///     with <paramref name="kind"/> (false included).</param>
        /// <param name="descending">NumPy 2.5's keyword-only <c>descending</c>: true orders the largest value first, NaN
        ///     still last. It cannot be combined with <paramref name="kind"/> (false included).</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/> or <paramref name="descending"/>, or an <paramref name="order"/>.</exception>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d array ravels first, so it accepts 0 and -1).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array sorted descending (<c>unsupported sort kind</c>), before
        ///     the axis check.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>, raised after the lines before it were sorted).</exception>
        /// <exception cref="MemoryError">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        [NewSubtype("this")]
        [AsArrayFor(typeof(NDCharArray))]
        public NDArray argsort(int? axis = -1, string kind = null, object order = null, bool? stable = null,
            bool? descending = null)
            => TensorEngine.ArgSort(this, axis, kind, order, stable, descending);
    }
}
