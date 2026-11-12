using System;
using System.Collections;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Creates a new 1-D array from an iterable — NumPy's <c>np.fromiter(iter, dtype, count=-1)</c>: every item is
        ///     converted into <paramref name="dtype"/> the way NumPy's per-item store (<c>PyArray_Pack</c>, the dtype's
        ///     setitem) converts it, and the result holds exactly the items read.
        /// </summary>
        /// <param name="iter">Any C# sequence (an array, a list, a LINQ query, an iterator method, an
        ///     <see cref="NDArray"/>). Items are Python values under NumSharp's boundary map: numbers by value (a float into
        ///     an integer dtype truncates; an integer the dtype cannot hold is NumPy's OverflowError), text by parsing
        ///     (<c>"1.5"</c> into float64), <see langword="null"/> as None (NaN for a float, TypeError for an integer), a
        ///     <see cref="char"/> as a one-character str (a <see cref="string"/> iterates its characters, as a Python str
        ///     does), a 0-d <see cref="NDArray"/> as its value. Iterating an N-D <see cref="NDArray"/> yields its rows, as in
        ///     NumPy.</param>
        /// <param name="dtype">The result's dtype — NumPy's required argument; null is NumPy's None, the default float64
        ///     (not a dtype inferred from the items).</param>
        /// <param name="count">How many items to read: -1 (the default) reads them all; a non-negative count reads at most
        ///     that many (an infinite sequence is fine) and requires that many; any other negative count reads nothing
        ///     (NumPy's loop, <c>i &lt; count || count == -1</c>, gives an empty array).</param>
        /// <returns>A new, owning, C-contiguous 1-D array of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="iter"/> is null (<c>'NoneType' object is not iterable</c>) or a 0-d
        ///     array (<c>iteration over a 0-d array</c>) — checked first, as NumPy gets the iterator first; an item the dtype's
        ///     conversion refuses with Python's TypeError (<c>int() argument must be a string, a bytes-like object or a real
        ///     number, not 'NoneType'</c>, <c>float() argument must be a string or a real number, not 'complex'</c>; for a
        ///     sequence item <c>int() ... not 'list'</c> into an integer dtype, <c>must be real number, not list</c> /
        ///     <c>only 0-dimensional arrays can be converted to Python scalars</c> into complex).</exception>
        /// <exception cref="ValueError"><paramref name="dtype"/> is an unsized text dtype (<c>"U"</c>, <c>"S"</c>: <c>Must
        ///     specify length when using variable-size data-type.</c>); fewer items than a non-negative
        ///     <paramref name="count"/> (<c>iterator too short: Expected 7 but iterator had only 5 items.</c>, after the
        ///     items read converted); an item that does not parse (<c>invalid literal for int() with base 10: 'a'</c>), a NaN
        ///     into an integer dtype, a sequence item into a float / text dtype (<c>setting an array element with a
        ///     sequence.</c>; text's has no period).</exception>
        /// <exception cref="OverflowError">An integer the dtype cannot hold (<c>Python integer 300 out of bounds for
        ///     int8</c>), an infinity into an integer dtype.</exception>
        /// <exception cref="NotSupportedException">A dtype NumSharp has no storage for (object, void, a datetime before its
        ///     storage stage).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html — NumPy's <c>PyArray_FromIter</c>, step
        ///     for step: the iterator, the unsized-dtype check, one item at a time until <paramref name="count"/>, each
        ///     item packed (its error raised at once, so a bad item beats a short iterator), then the short-iterator check.
        ///     <para>The packing of plain items is <see cref="array(Array, DType, int, bool, char)"/> over the items read,
        ///     whose list conversion is NumPy's setitem per element. A SEQUENCE item (a C# array or list is a Python list, a
        ///     tuple a Python tuple, an array of rank 1 or more an ndarray) is answered here, because NumPy's per-item
        ///     store does not nest it the way <c>np.array</c> does: each dtype's setitem answers in its own words (an
        ///     integer's <c>int()</c> TypeError, a float's and text's "sequence" ValueError, complex's TypeErrors), a bool
        ///     dtype stores its truth value (a non-empty list is True; an array only when it has one element), and a
        ///     <c>StringDType</c> stores its <c>str()</c>. The items before it convert first, so their errors come
        ///     first.</para>
        ///     <para><c>like=</c> is not modelled (the parity plan's exclusion). Added in API-parity batch 7 (the gap
        ///     document's OD10 row for fromiter, decided by the owner's approval).</para>
        /// </remarks>
        public static NDArray fromiter(IEnumerable iter, DType dtype, long count = -1)
            => NDArray.fromiter(iter, dtype, count);
    }
}
