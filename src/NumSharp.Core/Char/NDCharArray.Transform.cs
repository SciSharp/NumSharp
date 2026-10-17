// =============================================================================
// NDCharArray.Transform.cs — chararray.center / ljust / rjust / zfill / expandtabs and strip / lstrip / rstrip / replace
// (numpy/_core/defchararray.py)
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
//
// strip / lstrip / rstrip / replace are NumPy's rule 2: the member returns the function's result itself,
//     strip(self, chars=None)            -> strip(self, chars)
//     replace(self, old, new, count=None) -> replace(self, old, new, count if count is not None else -1)
// which the function's chararray rule already makes the chararray view NumPy returns (OWNDATA false). The members forward
// to it: there is no copy to take over and nothing to wrap.
//
// A StringDType ARGUMENT (a set, an old / new, a fill) makes the function's loop StringDType. NumPy's trivial loop then
// copies a 0-d or short 1-d chararray (at most np.getbufsize() elements) into StringDType() AS A CHARARRAY, which
// __array_finalize__ refuses ("Can only create a chararray from string data."); a larger chararray is not copied and the
// function computes a plain StringDType array. A rule-2 member returns that array in NumPy, which its NDCharArray type
// cannot hold: it raises NotSupportedException naming the numpy.char function (the operator +'s precedent). A rule-1 pad
// member converts it with np.char.asarray, an S chararray of its ASCII text, as NumPy's asarray does.
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
        ///     the computed array for <c>S</c> (NumPy's <c>np.char.asarray</c> rule). With a <c>StringDType</c> fill beside a
        ///     chararray NumPy's trivial loop does not copy (larger than 1-d, or 1-d past <c>np.getbufsize()</c>
        ///     elements), the function's <c>StringDType</c> result as <c>np.char.asarray</c> makes it: an <c>S</c> chararray
        ///     (its text encoded as ASCII).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">A <paramref name="width"/> that is not an integer array (<c>unsupported type float64
        ///     for operand 'width'</c>); a fill that is not one character (<c>The fill character must be exactly one
        ///     character long</c>); a uint64 width.</exception>
        /// <exception cref="ValueError">An empty array (<c>zero-size array to reduction operation maximum which has no
        ///     identity</c>, NumPy's sizing); a <c>StringDType</c> fill when this chararray (or a chararray fill) is 0-d, or
        ///     1-d with at most <c>np.getbufsize()</c> elements: NumPy's trivial loop copies it into <c>StringDType()</c> as
        ///     a chararray (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray; non-ASCII text in the
        ///     <c>StringDType</c> result <c>np.char.asarray</c> encodes.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <remarks>NumPy's member is <c>asarray(center(self, width, fillchar))</c>; see
        ///     <see cref="CharModule.center"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray center(object width, object fillchar = null)
        {
            NDArray fresh;
            // NumPy's member calls center(self, width, fillchar): this chararray and a chararray fill are inputs of that
            // ufunc call, so its trivial loop copies them into StringDType() beside a StringDType fill (the finalize error).
            using (NDCharArrayBoundary.InsideUfunc(this, fillchar as NDArray))
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
        ///     <c>S</c> (with a <c>StringDType</c> fill NumPy's trivial loop does not copy, see <see cref="center"/>).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty array; a <c>StringDType</c> fill beside a chararray NumPy's trivial loop
        ///     copies (see <see cref="center"/>).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        [NDScoped]
        public NDCharArray ljust(object width, object fillchar = null)
        {
            NDArray fresh;
            // NumPy's member calls ljust(self, width, fillchar): this chararray and a chararray fill are inputs of that
            // ufunc call, so its trivial loop copies them into StringDType() beside a StringDType fill (the finalize error).
            using (NDCharArrayBoundary.InsideUfunc(this, fillchar as NDArray))
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
        ///     <c>S</c> (with a <c>StringDType</c> fill NumPy's trivial loop does not copy, see <see cref="center"/>).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="width"/> is null.</exception>
        /// <exception cref="TypeError">See <see cref="center"/>.</exception>
        /// <exception cref="ValueError">An empty array; a <c>StringDType</c> fill beside a chararray NumPy's trivial loop
        ///     copies (see <see cref="center"/>).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        [NDScoped]
        public NDCharArray rjust(object width, object fillchar = null)
        {
            NDArray fresh;
            // NumPy's member calls rjust(self, width, fillchar): this chararray and a chararray fill are inputs of that
            // ufunc call, so its trivial loop copies them into StringDType() beside a StringDType fill (the finalize error).
            using (NDCharArrayBoundary.InsideUfunc(this, fillchar as NDArray))
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
            => fresh.dtype.@char switch
            {
                'U' => TakeOver(fresh, null),
                'S' => ViewOf(fresh),
                // A StringDType result (a StringDType fill beside a chararray NumPy's trivial loop did not copy): NumPy's
                // asarray converts its tolist() with the default bytes_ class, an S chararray view (UnicodeEncodeError for
                // non-ASCII text).
                _ => np.@char.asarray(fresh),
            };

        // =============================================================================================================
        // strip / lstrip / rstrip / replace: NumPy's rule 2, the function's own chararray view
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>chararray.strip(chars=None)</c>: every element without its leading and trailing characters found in
        ///     <paramref name="chars"/> (whitespace for null) — the stored text, so the trailing whitespace element access
        ///     hides is stripped too.
        /// </summary>
        /// <param name="chars">The characters to remove, as a set, broadcast against the elements (this array's class); null
        ///     strips whitespace (ASCII whitespace for <c>S</c>, Python's <c>str.isspace</c> for <c>U</c>).</param>
        /// <returns>A chararray VIEW of the stripped array, of this array's dtype: NumPy's member returns the function's
        ///     result, which <c>chararray.__array_wrap__</c> makes a view (it does not own its data).</returns>
        /// <exception cref="TypeError">No loop for the classes (a <c>bytes</c> set for a <c>U</c> chararray, a
        ///     number).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> set when this chararray is 0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements: NumPy's trivial loop copies it into <c>StringDType()</c> as a chararray
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> set beside a larger chararray: NumPy's member then
        ///     returns the function's PLAIN <c>StringDType</c> array, which this member's <see cref="NDCharArray"/> type
        ///     cannot hold (<c>np.char.strip</c> returns it).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="chars"/> does not broadcast.</exception>
        /// <remarks>NumPy's member is <c>strip(self, chars)</c>; see <see cref="CharModule.strip"/> for the function. The
        ///     member forwards without a scope or a region: the function's chararray rule makes the view NumPy returns, and
        ///     it allocates nothing the member must reclaim.</remarks>
        public NDCharArray strip(object chars = null) => FunctionResult(np.@char.strip(this, chars), "strip");

        /// <summary>
        ///     NumPy's <c>chararray.lstrip(chars=None)</c>: every element without its LEADING characters found in
        ///     <paramref name="chars"/> (whitespace for null), over the stored text.
        /// </summary>
        /// <param name="chars">The characters to remove, as a set, broadcast (see <see cref="strip"/>); null strips
        ///     whitespace.</param>
        /// <returns>A chararray view of the stripped array, of this array's dtype.</returns>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> set when this chararray is 0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements (NumPy's trivial-loop copy: <c>Can only create a chararray from string
        ///     data.</c>).</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> set beside a larger chararray (NumPy's plain
        ///     <c>StringDType</c> result; <c>np.char.lstrip</c> returns it).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="chars"/> does not broadcast.</exception>
        public NDCharArray lstrip(object chars = null) => FunctionResult(np.@char.lstrip(this, chars), "lstrip");

        /// <summary>
        ///     NumPy's <c>chararray.rstrip(chars=None)</c>: every element without its TRAILING characters found in
        ///     <paramref name="chars"/> (whitespace for null), over the stored text: <c>rstrip('x')</c> of a stored
        ///     <c>'ax  '</c> keeps <c>'ax  '</c> (the spaces end it), where its stripped element reads <c>'ax'</c>.
        /// </summary>
        /// <param name="chars">The characters to remove, as a set, broadcast (see <see cref="strip"/>); null strips
        ///     whitespace.</param>
        /// <returns>A chararray view of the stripped array, of this array's dtype.</returns>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> set when this chararray is 0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements (NumPy's trivial-loop copy: <c>Can only create a chararray from string
        ///     data.</c>).</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> set beside a larger chararray (NumPy's plain
        ///     <c>StringDType</c> result; <c>np.char.rstrip</c> returns it).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="chars"/> does not broadcast.</exception>
        public NDCharArray rstrip(object chars = null) => FunctionResult(np.@char.rstrip(this, chars), "rstrip");

        /// <summary>
        ///     NumPy's <c>chararray.replace(old, new, count=None)</c>: every element with its occurrences of
        ///     <paramref name="old"/> replaced by <paramref name="new"/>, left to right, at most <paramref name="count"/> of
        ///     them, over the stored text.
        /// </summary>
        /// <param name="old">The searched strings, broadcast: a C# <c>string</c> / <c>byte[]</c> (Python's <c>str</c> /
        ///     <c>bytes</c>) is cast to this array's dtype first; an array keeps its own dtype.</param>
        /// <param name="new">The replacements, broadcast (the rule of <paramref name="old"/>).</param>
        /// <param name="count">The most replacements per element: an integer array or literal, broadcast; null is NumPy's
        ///     <c>None</c>, which the member passes as -1 (every occurrence).</param>
        /// <returns>A chararray view of the replaced array: this array's class at the widest result width.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="old"/> or <paramref name="new"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="count"/> is not an integer array (<c>unsupported type float64 for
        ///     operand 'count'</c>); no loop for the classes.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> <paramref name="old"/> / <paramref name="new"/> when this
        ///     chararray is 0-d, or 1-d with at most <c>np.getbufsize()</c> elements (NumPy's trivial-loop copy: <c>Can only
        ///     create a chararray from string data.</c>); a number cast to text.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> <paramref name="old"/> / <paramref name="new"/>
        ///     beside a larger chararray: NumPy's member returns the function's PLAIN <c>StringDType</c> array, which this
        ///     member's <see cref="NDCharArray"/> type cannot hold (<c>np.char.replace</c> returns it).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII <paramref name="old"/> / <paramref name="new"/> for an <c>S</c>
        ///     chararray.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <remarks>NumPy's member is <c>replace(self, old, new, count if count is not None else -1)</c>: the module
        ///     function's null <paramref name="count"/> is that same -1, so the member forwards it.</remarks>
        public NDCharArray replace(object old, object @new, object count = null)
            => FunctionResult(np.@char.replace(this, old, @new, count), "replace");

        /// <summary>
        ///     A rule-2 member's answer: the function's result itself, which its chararray rule made a chararray view — or,
        ///     when NumPy's function returns a PLAIN array (a <c>StringDType</c> argument beside a chararray NumPy's trivial
        ///     loop did not copy: larger than 1-d, or 1-d past <c>np.getbufsize()</c> elements; <c>__array_wrap__</c> keeps
        ///     a <c>StringDType</c> result an ndarray), the refusal: the member's <see cref="NDCharArray"/> type cannot hold
        ///     it, and the <c>numpy.char</c> function returns it.
        /// </summary>
        /// <param name="result">The function's result (a chararray view, or the fresh plain array).</param>
        /// <param name="function">NumPy's member / function name, for the message.</param>
        /// <returns>The chararray view.</returns>
        /// <exception cref="System.NotSupportedException">The result is not a chararray (the operator <c>+</c>'s precedent for the
        ///     same NumPy answer).</exception>
        private static NDCharArray FunctionResult(NDArray result, string function)
        {
            if (result is NDCharArray chars)
                return chars;
            string dtype = result.dtype.ToString(true);
            // The plain result is this call's own allocation and nobody else holds it: release it before refusing.
            result.Dispose();
            throw new System.NotSupportedException(
                $"chararray.{function}: NumPy returns a plain {dtype} array here, which the NDCharArray result of this member " +
                $"cannot hold; call np.char.{function} for it.");
        }
    }
}
