// =============================================================================
// NDCharArray.Case.cs — chararray.capitalize / lower / swapcase / title / upper / translate (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are NumPy's rule 1, one line each over the numpy.char functions (numpy.strings' own):
//     upper(self)                              -> asarray(upper(self))   (capitalize, lower, swapcase, title alike)
//     translate(self, table, deletechars=None) -> asarray(translate(self, table, deletechars))
// The function reads the STORED text (np.asarray of the chararray, never __getitem__'s stripped element) and returns a
// PLAIN array; np.char.asarray then copies a str result (an owning chararray) and views a bytes one (OWNDATA false).
// The members compute the plain result inside the boundary (the function's own chararray rule would otherwise be applied
// to a call the member makes) and give it that final form directly (AsArrayOfFresh, NDCharArray.Transform.cs): a U result
// is taken over instead of copied, an S result is viewed.
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.capitalize()</c>: <c>asarray(capitalize(self))</c>, each element's first character in
        ///     titlecase and the rest lowered, over the stored text (a chararray's trailing whitespace included).
        /// </summary>
        /// <returns>A chararray of this array's dtype and shape: owning its data for <c>U</c>, a view of the computed array
        ///     for <c>S</c> (NumPy's <c>np.char.asarray</c> rule); 0-d for a 0-d chararray.</returns>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <remarks>See <see cref="CharModule.capitalize"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray capitalize()
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.capitalize(this);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.lower()</c>: <c>asarray(lower(self))</c>, each element's text lowercased (CPython's
        ///     <c>str.lower</c> for <c>U</c>, ASCII letters for <c>S</c>) over the stored text.
        /// </summary>
        /// <returns>A chararray of this array's dtype and shape: owning its data for <c>U</c>, a view of the computed array
        ///     for <c>S</c>; 0-d for a 0-d chararray.</returns>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <remarks>See <see cref="CharModule.lower"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray lower()
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.lower(this);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.swapcase()</c>: <c>asarray(swapcase(self))</c>, each element's uppercase characters
        ///     lowered and lowercase characters uppercased, over the stored text.
        /// </summary>
        /// <returns>A chararray of this array's dtype and shape: owning its data for <c>U</c>, a view of the computed array
        ///     for <c>S</c>; 0-d for a 0-d chararray.</returns>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <remarks>See <see cref="CharModule.swapcase"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray swapcase()
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.swapcase(this);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.title()</c>: <c>asarray(title(self))</c>, each element in titlecase (a character after
        ///     a cased character lowered, any other titlecased), over the stored text.
        /// </summary>
        /// <returns>A chararray of this array's dtype and shape: owning its data for <c>U</c>, a view of the computed array
        ///     for <c>S</c>; 0-d for a 0-d chararray.</returns>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <remarks>See <see cref="CharModule.title"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray title()
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.title(this);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.upper()</c>: <c>asarray(upper(self))</c>, each element's text uppercased (CPython's
        ///     <c>str.upper</c> for <c>U</c>, a growing mapping cut to the element width; ASCII letters for <c>S</c>), over
        ///     the stored text.
        /// </summary>
        /// <returns>A chararray of this array's dtype and shape: owning its data for <c>U</c>, a view of the computed array
        ///     for <c>S</c>; 0-d for a 0-d chararray.</returns>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <remarks>See <see cref="CharModule.upper"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray upper()
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.upper(this);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     NumPy's <c>chararray.translate(table, deletechars=None)</c>: <c>asarray(translate(self, table,
        ///     deletechars))</c>, every character of each element mapped through <paramref name="table"/>, over the stored
        ///     text.
        /// </summary>
        /// <param name="table">The table (see <see cref="np.strings.translate"/>): a dict keyed by code point for a <c>U</c>
        ///     chararray, a 256-byte table or null for an <c>S</c> one; broadcast against the elements.</param>
        /// <param name="deletechars">The bytes to delete (an <c>S</c> chararray only; NumPy ignores it for <c>U</c>); null
        ///     passes nothing.</param>
        /// <returns>A chararray of this array's dtype over the broadcast shape: owning its data for <c>U</c>, a view of the
        ///     computed array for <c>S</c>; 0-d for a 0-d chararray.</returns>
        /// <exception cref="TypeError">A table or delete set of the wrong kind (<c>a bytes-like object is required, not
        ///     'dict'</c> for <c>S</c>; a dict value that is not an int, None or str for <c>U</c>).</exception>
        /// <exception cref="ValueError">A mapping outside <c>range(0x110000)</c>; an <c>S</c> table that is not 256 bytes
        ///     long.</exception>
        /// <exception cref="SystemError">A <c>U</c> unit past U+10FFFF that the call would keep.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="table"/> does not broadcast.</exception>
        /// <exception cref="System.NotSupportedException">A table with no NumSharp form (an object array of
        ///     lists).</exception>
        /// <remarks>See <see cref="CharModule.translate"/> for the function.</remarks>
        [NDScoped]
        public NDCharArray translate(object table, object deletechars = null)
        {
            NDArray fresh;
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.translate(this, table, deletechars);
            return AsArrayOfFresh(fresh);
        }
    }
}
