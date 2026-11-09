using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a sorted copy of an array, sorted along <paramref name="axis"/> (default last; <c>null</c> flattens
        ///     first). NumPy's <c>np.sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False)</c>.
        /// </summary>
        /// <param name="a">Array to sort.</param>
        /// <param name="axis">Axis to sort along. -1 = last axis. null = sort the flattened array.</param>
        /// <param name="kind">
        ///     The sort algorithm, as NumPy parses it: the first letter decides, case-insensitively — <c>'q'</c>uicksort
        ///     (the default), <c>'h'</c>eapsort (the default kind under another name), <c>'m'</c>ergesort or
        ///     <c>'s'</c>table. A value sort gives one answer for every kind, except where equal values have different bits:
        ///     the default kind leaves the float NaNs in the order NumPy's x86-simd-sort moves them to the end and the stable
        ///     kind in input order (every NaN's bits kept), and a descending default-kind sort places equal values as
        ///     NumPy's introsort does.
        /// </param>
        /// <param name="order">Field names to sort by. No NumSharp dtype has fields, so any value is refused, as NumPy
        ///     refuses it for an array without fields.</param>
        /// <param name="stable">NumPy's keyword-only <c>stable</c>: true asks for the stable kind. It cannot be combined
        ///     with <paramref name="kind"/> (false included).</param>
        /// <param name="descending">NumPy 2.5's keyword-only <c>descending</c>: true sorts the largest value first, NaN
        ///     still last (complex: the NaN classes last, each descending). It cannot be combined with
        ///     <paramref name="kind"/> (false included); null and false sort ascending.</param>
        /// <returns>The sorted copy, laid out as <c>a.copy(order='K')</c> (a 1-D C array for <c>axis=null</c>).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     NaN floats sort to the end; complex sorts lexicographically (real then imaginary), any-NaN-part last; text
        ///     (<c>S</c> / <c>U</c>) sorts by code points, compared unsigned — matching NumPy 2.5.4.
        /// </remarks>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/> or <paramref name="descending"/>, or an <paramref name="order"/>; or a
        ///     <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too
        ///     large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d input refuses every axis).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array sorted descending (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>, raised after the lines before it were sorted).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public static NDArray sort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null,
            bool? descending = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Sort(a, axis, kind, order, stable, descending);
        }
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Sort this array in place along <paramref name="axis"/> (default last; <c>null</c> flattens, a NumSharp
        ///     extension). NumPy's <c>ndarray.sort(axis=-1, kind=None, order=None, *, stable=None, descending=False)</c>.
        /// </summary>
        /// <param name="axis">Axis to sort along; null flattens (NumSharp's extension — NumPy's method takes no None).</param>
        /// <param name="kind">The sort algorithm (see <see cref="np.sort"/>).</param>
        /// <param name="order">Field names; any value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">True asks for the stable kind; cannot be combined with <paramref name="kind"/>.</param>
        /// <param name="descending">True sorts the largest value first, NaN still last (NumPy 2.5); cannot be combined with
        ///     <paramref name="kind"/>.</param>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/> or <paramref name="descending"/>, an <paramref name="order"/>, or a read-only array
        ///     (<c>sort array is read-only</c>).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array sorted descending (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>, raised after the lines before it were sorted).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        public void sort(int? axis = -1, string kind = null, object order = null, bool? stable = null, bool? descending = null)
            => TensorEngine.SortInPlace(this, axis, kind, order, stable, descending);
    }
}
