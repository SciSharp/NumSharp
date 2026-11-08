// =============================================================================
// NDCharArray.Codec.cs — chararray.encode / decode (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are one line each over the numpy.char functions (numpy.strings' own), with the chararray as the text:
//     encode(self, encoding=None, errors=None) -> encode(self, encoding, errors)
//     decode(self, encoding=None, errors=None) -> decode(self, encoding, errors)
// There is no asarray step: the function's S / U result is a new PLAIN array (_to_bytes_or_str_array builds it from the
// object array's tolist()), so the member returns it as it is — a plain array, a 0-d one for a 0-d chararray. The text
// read is the stored one, unstripped (_vec_string reads the buffer, never chararray.__getitem__):
// np.char.array(['ab  ']).encode() is b'ab  '. An S chararray has no encode and a U one no decode (CPython's
// AttributeError, raised by the function).
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.encode(encoding=None, errors=None)</c>: CPython's <c>str.encode</c> per element of this
        ///     <c>U</c> chararray's stored text (trailing whitespace included), the results as a plain <c>S</c> array.
        /// </summary>
        /// <param name="encoding">The codec name, or null for UTF-8; broadcast against this array (see
        ///     <see cref="CharModule.encode"/>).</param>
        /// <param name="errors">The error handler, or null for <c>strict</c>; ignored when <paramref name="encoding"/> is
        ///     null.</param>
        /// <returns>A fresh C-ordered PLAIN <c>S</c> array of the broadcast shape (0-d for a 0-d chararray), as wide as the
        ///     longest encoding (at least 1).</returns>
        /// <exception cref="AttributeError">An <c>S</c> chararray: <c>type object 'bytes' has no attribute 'encode'</c>.</exception>
        /// <exception cref="TypeError">An argument that is not a str.</exception>
        /// <exception cref="ValueError">An argument holding a NUL; a ragged argument list.</exception>
        /// <exception cref="UnicodeEncodeError">A character the codec cannot encode under <c>strict</c>, or a handler that
        ///     re-raises it.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec; an unknown handler name at the first error.</exception>
        /// <exception cref="SystemError">An element of one unit past U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <exception cref="System.NotSupportedException">A Python codec NumSharp does not implement.</exception>
        /// <remarks>NumPy's member is <c>encode(self, encoding, errors)</c>; see <see cref="CharModule.encode"/> for the
        ///     function.</remarks>
        public NDArray encode(object encoding = null, object errors = null) => np.@char.encode(this, encoding, errors);

        /// <summary>
        ///     NumPy's <c>chararray.decode(encoding=None, errors=None)</c>: CPython's <c>bytes.decode</c> per element of this
        ///     <c>S</c> chararray's stored bytes (trailing whitespace included, trailing NULs dropped), the results as a plain
        ///     <c>U</c> array.
        /// </summary>
        /// <param name="encoding">The codec name, or null for UTF-8; broadcast against this array (see
        ///     <see cref="CharModule.decode"/>).</param>
        /// <param name="errors">The error handler, or null for <c>strict</c>; ignored when <paramref name="encoding"/> is
        ///     null.</param>
        /// <returns>A fresh C-ordered PLAIN <c>U</c> array of the broadcast shape (0-d for a 0-d chararray), as wide as the
        ///     longest decoding (at least 1).</returns>
        /// <exception cref="AttributeError">A <c>U</c> chararray: <c>type object 'str' has no attribute 'decode'</c>.</exception>
        /// <exception cref="TypeError">An argument that is not a str; <c>xmlcharrefreplace</c> / <c>namereplace</c> at an
        ///     error.</exception>
        /// <exception cref="ValueError">An argument holding a NUL; a ragged argument list.</exception>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>, or a handler that re-raises
        ///     it.</exception>
        /// <exception cref="UnicodeEncodeError">A surrogate in an argument.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec over a non-empty element.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec; an unknown handler name at the first error.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <exception cref="System.NotSupportedException">A Python codec NumSharp does not implement.</exception>
        /// <remarks>NumPy's member is <c>decode(self, encoding, errors)</c>; see <see cref="CharModule.decode"/> for the
        ///     function.</remarks>
        public NDArray decode(object encoding = null, object errors = null) => np.@char.decode(this, encoding, errors);
    }
}
