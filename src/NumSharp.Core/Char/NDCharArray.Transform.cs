// =============================================================================
// NDCharArray.Transform.cs — chararray.center / ljust / rjust / zfill / expandtabs (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are one line each over the numpy.char functions (numpy.strings' own), with the chararray as the text:
//     center(self, width, fillchar=' ')  -> asarray(center(self, width, fillchar))
//     ljust(self, width, fillchar=' ')   -> asarray(ljust(self, width, fillchar))
//     rjust(self, width, fillchar=' ')   -> asarray(rjust(self, width, fillchar))
//     zfill(self, width)                 -> asarray(zfill(self, width))
//     expandtabs(self, tabsize=8)        -> asarray(expandtabs(self, tabsize))
// The function reads the stored text UNSTRIPPED (a ufunc reads the array, never chararray.__getitem__) and returns a
// chararray view of its result (chararray.__array_wrap__ over its np.empty_like out); np.char.asarray then COPIES a str
// result (it compares the width in characters with the itemsize in bytes) and VIEWS a bytes one. So a U member result
// owns its data and an S member result is a view of the computed array.
//
// The members compute the function's array once and give it NumPy's final form directly (AsArrayOfFresh): the copy of a
// str result is the array itself taken over (the computed array is private to the call, so no caller can tell a copy from
// it: the same values, OWNDATA, strides), and the view of a bytes result is a view of it. NumPy's intermediate chararray
// view is never made. At 1M U elements the copy was ~40 % of the call.
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.center(width, fillchar=' ')</c>: every element centered in a string of
        ///     <paramref name="width"/> characters, padded with <paramref name="fillchar"/> — the stored text, unstripped, so
        ///     trailing whitespace is part of what is centered.
        /// </summary>
        /// <param name="width">The widths: an integer array or a C# integer, broadcast against the elements.</param>
        /// <param name="fillchar">The one-character fills, broadcast; null is NumPy's default <c>' '</c> (a <c>str</c> fill
        ///     is cast to bytes for an <c>S</c> chararray, a <c>bytes</c> fill to text for a <c>U</c> one).</param>
        /// <returns>A chararray at the widest result width, laid out like this array: owning its data for <c>U</c>, a view of
        ///     the computed array for <c>S</c> (NumPy's <c>np.char.asarray</c> rule).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">A <paramref name="width"/> that is not an integer array (<c>unsupported type float64
        ///     for operand 'width'</c>); a fill that is not one character (<c>The fill character must be exactly one
        ///     character long</c>); a uint64 width.</exception>
        /// <exception cref="ValueError">An empty array (<c>zero-size array to reduction operation maximum which has no
        ///     identity</c>, NumPy's sizing).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <remarks>NumPy's member is <c>asarray(center(self, width, fillchar))</c>; see
        ///     <see cref="CharModule.center"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray center(object width, object fillchar = null)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.center(this, width, fillchar);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.ljust(width, fillchar=' ')</c>: every element left-justified in a string of
        ///     <paramref name="width"/> characters, padded on the right with <paramref name="fillchar"/> (the stored text,
        ///     unstripped).
        /// </summary>
        /// <param name="width">The integer widths, broadcast against the elements.</param>
        /// <param name="fillchar">The one-character fills, broadcast; null is NumPy's default <c>' '</c>.</param>
        /// <returns>A chararray at the widest result width: owning its data for <c>U</c>, a view of the computed array for
        ///     <c>S</c>.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty array.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        [NDScoped]
        public NDCharArray ljust(object width, object fillchar = null)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.ljust(this, width, fillchar);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.rjust(width, fillchar=' ')</c>: every element right-justified in a string of
        ///     <paramref name="width"/> characters, padded on the left with <paramref name="fillchar"/> (the stored text,
        ///     unstripped, so trailing whitespace stays at the right end).
        /// </summary>
        /// <param name="width">The integer widths, broadcast against the elements.</param>
        /// <param name="fillchar">The one-character fills, broadcast; null is NumPy's default <c>' '</c>.</param>
        /// <returns>A chararray at the widest result width: owning its data for <c>U</c>, a view of the computed array for
        ///     <c>S</c>.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty array.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        [NDScoped]
        public NDCharArray rjust(object width, object fillchar = null)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.rjust(this, width, fillchar);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.zfill(width)</c>: every element left-filled with <c>'0'</c> to <paramref name="width"/>
        ///     characters, a leading <c>'+'</c> / <c>'-'</c> moved in front of the zeros.
        /// </summary>
        /// <param name="width">The integer widths, broadcast against the elements.</param>
        /// <returns>A chararray at the widest result width: owning its data for <c>U</c>, a view of the computed array for
        ///     <c>S</c>.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">A <paramref name="width"/> that is not an integer array (<c>unsupported type &lt;U1 for
        ///     operand 'width'</c>); a uint64 width.</exception>
        /// <exception cref="ValueError">An empty array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="width"/> does not broadcast.</exception>
        [NDScoped]
        public NDCharArray zfill(object width)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.zfill(this, width);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.expandtabs(tabsize=8)</c>: every tab replaced by spaces up to the next multiple of
        ///     <paramref name="tabsize"/> columns (the column restarts after <c>'\n'</c> and <c>'\r'</c>).
        /// </summary>
        /// <param name="tabsize">The tab sizes (any integer dtype), broadcast; null is NumPy's default 8.</param>
        /// <returns>A chararray at NumPy's result width (a tab's columns plus every other character's BYTES, so a <c>U</c>
        ///     result is wider than its text): owning its data for <c>U</c>, a view of the computed array for
        ///     <c>S</c>.</returns>
        /// <exception cref="System.ArgumentException">A tab size that cannot reach int64 by a same_kind cast (a float).</exception>
        /// <exception cref="ValueError">An empty array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="tabsize"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException">An expanded string longer than 2^31 - 1.</exception>
        [NDScoped]
        public NDCharArray expandtabs(object tabsize = null)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.expandtabs(this, tabsize);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>np.char.asarray(result)</c> for an array a member computed itself (a <c>numpy.char</c> function's
        ///     fresh result, never seen by the caller): a <c>U</c> result is the owning chararray NumPy's copy would be — this
        ///     array's storage, taken over, so no copy is made — and an <c>S</c> result is a chararray VIEW of it, as NumPy's
        ///     <c>asarray</c> returns.
        /// </summary>
        /// <param name="fresh">The member's result: a fresh, dense <c>S</c> / <c>U</c> array no other array refers to. The
        ///     calling member's scope reclaims its wrapper.</param>
        /// <returns>The chararray: owning its data (<c>U</c>) or a view (<c>S</c>), with <paramref name="fresh"/>'s values,
        ///     shape and strides.</returns>
        /// <exception cref="ValueError"><paramref name="fresh"/> holds data that is not string data (never, for the
        ///     members' results).</exception>
        /// <remarks>
        ///     NumPy's <c>asarray</c> compares a <c>U</c> array's width in CHARACTERS with its itemsize in BYTES and casts
        ///     it — <c>astype(..., copy=True)</c>, order K, so the copy has the computed array's strides and owns its data.
        ///     Taking over a dense array that only this call holds is the same object to every observer (values, OWNDATA,
        ///     strides, memory sharing with the operands: none), without the copy. Only a fresh result qualifies: an array a
        ///     caller holds would end up shared.
        /// </remarks>
        internal static NDCharArray AsArrayOfFresh(NDArray fresh)
            => fresh.dtype.@char == 'U' ? TakeOver(fresh, null) : ViewOf(fresh);
    }
}
