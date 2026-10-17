// =============================================================================
// NDCharArray.Predicates.cs — chararray.isalnum / isalpha / isdecimal / isdigit / islower / isnumeric / isspace /
// istitle / isupper (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are one line each over the numpy.char predicate, which IS numpy.strings' ufunc:
//     isalpha(self) -> isalpha(self)
// The ufunc reads the STORED text (never chararray.__getitem__'s stripped element), so trailing whitespace counts: a
// stored 'ab  ' is not alpha, a stored 'AB  ' is upper. Its bool result stays a PLAIN array (chararray.__array_wrap__
// keeps the class only for text), and a 0-d chararray gives a 0-d array, not a scalar (the wrap ignores
// return_scalar). The members forward without a scope: the function makes the plain result and allocates nothing the
// member must reclaim. isdecimal / isnumeric have no bytes loop, so an S chararray refuses them with NumPy's no-loop
// TypeError.
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.isalnum()</c>: <c>isalnum(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one character and only alphanumeric characters (CPython's
        ///     <c>str.isalnum</c>: alphabetic, decimal, digit or numeric); <c>'a1'</c> is True; a stored <c>'ab  '</c>
        ///     is False (its trailing spaces are not alphanumeric).
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.isalnum"/> for the function.</remarks>
        public NDArray isalnum() => np.@char.isalnum(this);

        /// <summary>
        ///     NumPy's <c>chararray.isalpha()</c>: <c>isalpha(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one character and only alphabetic characters (CPython's
        ///     <c>str.isalpha</c>: the Unicode letter categories); <c>'abc'</c> is True; a stored <c>'ab  '</c> is
        ///     False, though its element <c>c[i]</c> reads <c>'ab'</c>.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.isalpha"/> for the function.</remarks>
        public NDArray isalpha() => np.@char.isalpha(this);

        /// <summary>
        ///     NumPy's <c>chararray.isdecimal()</c>: <c>isdecimal(self)</c>, the ufunc over the stored text. True for
        ///     each element whose text has at least one character and only decimal characters (CPython's
        ///     <c>str.isdecimal</c>: <c>'٣'</c> is one, <c>'²'</c> is not); <c>'123'</c> is True; a stored <c>'12 '</c>
        ///     is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="TypeError">An <c>S</c> chararray: NumPy's ufunc has no bytes loop (<c>ufunc 'isdecimal' did
        ///     not contain a loop with signature matching types &lt;class 'numpy.dtypes.BytesDType'&gt; -&gt;
        ///     None</c>).</exception>
        /// <remarks>See <see cref="CharModule.isdecimal"/> for the function.</remarks>
        public NDArray isdecimal() => np.@char.isdecimal(this);

        /// <summary>
        ///     NumPy's <c>chararray.isdigit()</c>: <c>isdigit(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one character and only digits (CPython's <c>str.isdigit</c>: the decimal
        ///     digits and the other digits, such as <c>'²'</c>; not <c>'½'</c>); <c>'123'</c> is True; <c>'½'</c> is
        ///     False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.isdigit"/> for the function.</remarks>
        public NDArray isdigit() => np.@char.isdigit(this);

        /// <summary>
        ///     NumPy's <c>chararray.islower()</c>: <c>islower(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one cased character and every cased character lowercase; a stored
        ///     <c>'ab  '</c> is True (spaces are uncased); <c>'aBc'</c> is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.islower"/> for the function.</remarks>
        public NDArray islower() => np.@char.islower(this);

        /// <summary>
        ///     NumPy's <c>chararray.isnumeric()</c>: <c>isnumeric(self)</c>, the ufunc over the stored text. True for
        ///     each element whose text has at least one character and only numeric characters (CPython's
        ///     <c>str.isnumeric</c>: digits, fractions such as <c>'½'</c>, Roman and CJK numerals); <c>'½'</c> is True;
        ///     <c>'a1'</c> is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="TypeError">An <c>S</c> chararray: NumPy's ufunc has no bytes loop (<c>ufunc 'isnumeric' did
        ///     not contain a loop with signature matching types &lt;class 'numpy.dtypes.BytesDType'&gt; -&gt;
        ///     None</c>).</exception>
        /// <remarks>See <see cref="CharModule.isnumeric"/> for the function.</remarks>
        public NDArray isnumeric() => np.@char.isnumeric(this);

        /// <summary>
        ///     NumPy's <c>chararray.isspace()</c>: <c>isspace(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one character and only whitespace (CPython's <c>str.isspace</c> for
        ///     <c>U</c> / <c>StringDType</c>; space and <c>\t</c>–<c>\r</c> for <c>S</c>); <c>'  '</c> is True; an
        ///     empty element is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.isspace"/> for the function.</remarks>
        public NDArray isspace() => np.@char.isspace(this);

        /// <summary>
        ///     NumPy's <c>chararray.istitle()</c>: <c>istitle(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text is titlecased: an uppercase or titlecase character never follows a cased character, a
        ///     lowercase one always does, and at least one character is cased; <c>'Abc Def'</c> is True; <c>'ABC'</c>
        ///     is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.istitle"/> for the function.</remarks>
        public NDArray istitle() => np.@char.istitle(this);

        /// <summary>
        ///     NumPy's <c>chararray.isupper()</c>: <c>isupper(self)</c>, the ufunc over the stored text. True for each
        ///     element whose text has at least one cased character and every cased character uppercase; a stored
        ///     <c>'AB  '</c> is True (spaces are uncased); <c>'Ab'</c> is False.
        /// </summary>
        /// <returns>A plain bool array of this array's shape (0-d for a 0-d chararray).</returns>
        /// <remarks>See <see cref="CharModule.isupper"/> for the function.</remarks>
        public NDArray isupper() => np.@char.isupper(this);
    }
}
