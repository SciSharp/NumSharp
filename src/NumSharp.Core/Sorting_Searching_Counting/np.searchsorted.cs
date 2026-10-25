using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <exception cref="ArgumentException"><paramref name="side"/> is not <c>"left"</c> / <c>"right"</c> (NumPy's two
        ///     ValueError texts).</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is not 1-D, a sorter of uint64, a sorter whose size differs
        ///     from <paramref name="a"/>'s, or a sorter index outside <paramref name="a"/> that the search reaches.</exception>
        /// <exception cref="TypeError">A sorter that is not 1-D (<c>could not parse sorter argument</c>) or not integer
        ///     (<c>sorter must only contain integers</c>), or a StringDType <paramref name="a"/> whose first comparison with
        ///     the key Python refuses (<c>'&lt;' not supported between instances of 'str' and 'int'</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        ///     <para>A text <paramref name="a"/> is searched as text: the key is formatted into the common string dtype, as
        ///     NumPy converts it.</para>
        ///     <para>The whole call is the engine's (<see cref="TensorEngine.SearchSorted(NDArray, int, string, NDArray)"/>);
        ///     this function applies the matrix boundary.</para>
        /// </remarks>
        public static long searchsorted(NDArray a, int v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's searchsorted IS PyArray_SearchSorted (the descriptor resolution, the checks, the binary search), and
            // the whole call is the engine's: DefaultEngine.SearchSorted.
            return a.TensorEngine.SearchSorted(a, v, side, sorter);
        }

        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <exception cref="ArgumentException">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="TypeError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/> (the object path's text
        ///     names <c>'float'</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>The whole call is the engine's (<see cref="TensorEngine.SearchSorted(NDArray, double, string, NDArray)"/>);
        ///     this function applies the matrix boundary.</remarks>
        public static long searchsorted(NDArray a, double v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a.TensorEngine.SearchSorted(a, v, side, sorter);
        }

        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array <paramref name="a"/> such that, if the corresponding elements
        /// in <paramref name="v"/> were inserted before the indices, the order of <paramref name="a"/> would be preserved.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Values to insert into <paramref name="a"/>. May be a scalar or any shape.</param>
        /// <param name="side">If "left" (default), the index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Array of insertion points with the same shape as <paramref name="v"/>, or a scalar if <paramref name="v"/> is a scalar.</returns>
        /// <exception cref="ArgumentException">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>; or a StringDType
        ///     comparison that touched a None-like NA.</exception>
        /// <exception cref="TypeError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>; or a U array converted to
        ///     StringDType (NumPy's 'safe' cast refusal).</exception>
        /// <exception cref="UnicodeDecodeError">A text search converts bytes that are not ASCII to the common <c>U</c> dtype.</exception>
        /// <exception cref="DTypePromotionError">Text with an operand that has no common dtype with it.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> or <paramref name="v"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        ///     <para>Text in <paramref name="a"/> or <paramref name="v"/> is searched as NumPy searches it: both converted to
        ///     their common string dtype (a number formatted, bytes decoded), then NumPy's generic binary search step for step
        ///     — its monotonic carry and its sorter checks included, so an unsorted haystack gives NumPy's answer too
        ///     (<see cref="Backends.Sorting.TextSearch"/>).</para>
        ///     <para>The whole call is the engine's (<see cref="TensorEngine.SearchSorted(NDArray, NDArray, string, NDArray)"/>);
        ///     this function applies the matrix boundary.</para>
        /// </remarks>
        [NDScoped] // the boundary of the plain region and the engine call (the engine reclaims its own temporaries)
        public static NDArray searchsorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, v, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a.TensorEngine.SearchSorted(a, v, side, sorter);
        }
    }
}
