using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a sorted copy of an array, sorted along <paramref name="axis"/> (default last; <c>null</c> flattens
        ///     first). NumPy's <c>np.sort(a, axis=-1, kind=None, order=None, *, stable=None)</c>.
        /// </summary>
        /// <param name="a">Array to sort.</param>
        /// <param name="axis">Axis to sort along. -1 = last axis. null = sort the flattened array.</param>
        /// <param name="kind">
        ///     The sort algorithm, as NumPy parses it: the first letter decides, case-insensitively — <c>'q'</c>uicksort
        ///     (the default), <c>'h'</c>eapsort, <c>'m'</c>ergesort or <c>'s'</c>table. A value sort gives one answer for
        ///     every kind, except that the stable kind keeps every float NaN's bits where the default writes NumPy's
        ///     canonical quiet NaN, and a <c>StringDType</c> array refuses the heapsort kind.
        /// </param>
        /// <param name="order">Field names to sort by. No NumSharp dtype has fields, so any value is refused, as NumPy
        ///     refuses it for an array without fields.</param>
        /// <param name="stable">NumPy's keyword-only <c>stable</c>: true asks for the stable kind. It cannot be combined
        ///     with <paramref name="kind"/> (false included).</param>
        /// <returns>The sorted copy, laid out as <c>a.copy(order='K')</c> (a 1-D C array for <c>axis=null</c>).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     NaN floats sort to the end; complex sorts lexicographically (real then imaginary), any-NaN-part last; text
        ///     (<c>S</c> / <c>U</c>) sorts by code points, compared unsigned — matching NumPy 2.4.2.
        /// </remarks>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/>, or an <paramref name="order"/>; or a <c>numpy.matrix</c> operand whose result
        ///     cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d input refuses every axis).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>, raised after the lines before it were sorted).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        public static NDArray sort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, axis, kind, order, stable), static s => sort(s.a, s.axis, s.kind, s.order, s.stable))
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((a, axis, kind, order, stable), static s => sort(s.a, s.axis, s.kind, s.order, s.stable))
            : AxisSort.Sort(a, axis, SortKindArguments.Resolve(kind, order, stable));
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Sort this array in place along <paramref name="axis"/> (default last; <c>null</c> flattens, a NumSharp
        ///     extension). NumPy's <c>ndarray.sort(axis=-1, kind=None, order=None, *, stable=None)</c>.
        /// </summary>
        /// <param name="axis">Axis to sort along; null flattens (NumSharp's extension — NumPy's method takes no None).</param>
        /// <param name="kind">The sort algorithm (see <see cref="np.sort"/>).</param>
        /// <param name="order">Field names; any value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">True asks for the stable kind; cannot be combined with <paramref name="kind"/>.</param>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, <paramref name="kind"/> given with
        ///     <paramref name="stable"/>, an <paramref name="order"/>, or a read-only array (<c>sort array is read-only</c>).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements that holds an NA of a descriptor
        ///     whose NA object is neither a string nor NaN-like (NumPy's <c>Cannot compare null that is not a nan-like
        ///     value</c>, raised after the lines before it were sorted).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        public void sort(int? axis = -1, string kind = null, object order = null, bool? stable = null)
            => AxisSort.SortInPlace(this, axis, SortKindArguments.Resolve(kind, order, stable));
    }
}
