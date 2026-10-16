// =============================================================================
// NDCharArray.Split.cs — chararray.split / rsplit / splitlines / join (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are one line each over the module functions, with the chararray as the first argument:
//     split(self, sep=None, maxsplit=None)  -> split(self, sep, maxsplit)       (an object array of lists)
//     rsplit(self, sep=None, maxsplit=None) -> rsplit(self, sep, maxsplit)
//     splitlines(self, keepends=None)       -> splitlines(self, keepends)
//     join(self, seq)                       -> join(self, seq)                  (the chararray is the SEPARATOR)
// _vec_string reads each element through PyArray_ToScalar, never through chararray.__getitem__, so nothing is stripped:
// np.char.array(['a b ']).split(' ') is [['a', 'b', '']]. The results are not chararrays (an object array; join's text
// is np.asarray(result.tolist()), a plain ndarray).
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.split(sep=None, maxsplit=None)</c>: <c>np.char.split(self, sep, maxsplit)</c> — every
        ///     element's list of words, read UNSTRIPPED (the elements reach Python's <c>str.split</c> as NumPy scalars, not
        ///     through the chararray's stripping element access).
        /// </summary>
        /// <param name="sep">The separator (null for whitespace; see <see cref="CharModule.split"/>).</param>
        /// <param name="maxsplit">The most splits per element (null or negative for no limit).</param>
        /// <returns>NumPy's object array of lists.</returns>
        /// <exception cref="TypeError">See <see cref="CharModule.split"/>.</exception>
        /// <exception cref="OverflowException">A maxsplit past <c>Py_ssize_t</c>.</exception>
        /// <exception cref="ValueError">An empty separator.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> element, separator or word above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        public NDListArray split(object sep = null, object maxsplit = null) => np.@char.split(this, sep, maxsplit);

        /// <summary>
        ///     NumPy's <c>chararray.rsplit(sep=None, maxsplit=None)</c>: <c>np.char.rsplit(self, sep, maxsplit)</c>, the
        ///     elements unstripped.
        /// </summary>
        /// <param name="sep">The separator (null for whitespace).</param>
        /// <param name="maxsplit">The most splits per element, from the right.</param>
        /// <returns>NumPy's object array of lists.</returns>
        /// <exception cref="TypeError">See <see cref="CharModule.rsplit"/>.</exception>
        /// <exception cref="OverflowException">A maxsplit past <c>Py_ssize_t</c>.</exception>
        /// <exception cref="ValueError">An empty separator.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> element, separator or word above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        public NDListArray rsplit(object sep = null, object maxsplit = null) => np.@char.rsplit(this, sep, maxsplit);

        /// <summary>
        ///     NumPy's <c>chararray.splitlines(keepends=None)</c>: <c>np.char.splitlines(self, keepends)</c>, the elements
        ///     unstripped (a trailing line boundary is not lost to the chararray's strip).
        /// </summary>
        /// <param name="keepends">Whether each line keeps its boundary (Python's truth rule; null is false).</param>
        /// <returns>NumPy's object array of lists.</returns>
        /// <exception cref="TypeError">See <see cref="CharModule.splitlines"/>.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> element, keepends value or line above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        public NDListArray splitlines(object keepends = null) => np.@char.splitlines(this, keepends);

        /// <summary>
        ///     NumPy's <c>chararray.join(seq)</c>: <c>np.char.join(self, seq)</c> — the chararray's elements (unstripped) are
        ///     the SEPARATORS, each joining the characters of its <paramref name="seq"/> string. The result is a plain text
        ///     array, as NumPy's <c>np.asarray(result.tolist())</c> is.
        /// </summary>
        /// <param name="seq">The strings, broadcast against this array.</param>
        /// <returns>The joined text (of <paramref name="seq"/>'s class).</returns>
        /// <exception cref="TypeError">See <see cref="CharModule.join(object, object)"/>.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> separator or sequence above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        /// <exception cref="NotSupportedException"><paramref name="seq"/> is an empty <see cref="NDListArray"/> passed as an
        ///     object (call <see cref="join(NDListArray)"/>).</exception>
        public NDArray join(object seq) => np.@char.join(this, seq);

        /// <summary>
        ///     NumPy's <c>chararray.join(seq)</c> over an object array of lists: NumPy 2.4.2 joins every list, then raises
        ///     (see <see cref="CharModule.join(object, NDListArray)"/>); an empty broadcast is an empty list array.
        /// </summary>
        /// <param name="seq">The lists.</param>
        /// <returns>The empty list array of an empty broadcast.</returns>
        /// <exception cref="TypeError">A list item of the wrong class, else NumPy's <c>ObjectDType</c> error for a non-empty
        ///     broadcast.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        public NDListArray join(NDListArray seq) => np.@char.join(this, seq);
    }
}
