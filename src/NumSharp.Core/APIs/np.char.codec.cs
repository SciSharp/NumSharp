using System;

// =============================================================================
// np.char.codec.cs — the encode / decode numpy.char re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports encode and decode from numpy.strings: the same function objects
// (np.char.encode is np.strings.encode). Each is _to_bytes_or_str_array(_vec_string(a, np.object_, 'encode' | 'decode',
// _clean_args(encoding, errors)), np.bytes_(b'') | np.str_('')): CPython's str.encode / bytes.decode per element, the
// results as an S / U array as wide as the longest one. They are not ufuncs (no out= / where= / dtype=). _vec_string writes
// a NEW object array and _to_bytes_or_str_array builds a NEW array from its tolist(), so a chararray operand gives a PLAIN
// array, computed from its STORED text (trailing whitespace included: _vec_string reads the buffer, never
// chararray.__getitem__'s stripped element). The re-exports convert their operand with the module's text rule
// (CharModule.TextOperand: a byte[] is Python bytes) and call the np.strings function, so every answer (the result, its
// layout, every error and its order) is np.strings'.
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        /// <summary>
        ///     NumPy's <c>np.char.encode(a, encoding=None, errors=None)</c>, which IS <see cref="np.strings.encode"/>:
        ///     CPython's <c>str.encode(encoding, errors)</c> per element of a <c>U</c> / <c>StringDType</c> array, the
        ///     results as an <c>S</c> array. The text is the stored one, a chararray's trailing whitespace included.
        /// </summary>
        /// <param name="a">The text: a <c>U</c> / <c>StringDType</c> array or a text array_like (see the class remarks; a
        ///     C# <c>string</c> is a 0-d <c>U</c> array).</param>
        /// <param name="encoding">The codec name, or null for UTF-8 (the codecs and aliases of
        ///     <see cref="np.strings.encode"/>), converted as NumPy's <c>np.asarray</c> converts it and broadcast against
        ///     <paramref name="a"/>.</param>
        /// <param name="errors">The error handler, or null for <c>strict</c> (<c>"ignore"</c>, <c>"replace"</c>,
        ///     <c>"backslashreplace"</c>, <c>"xmlcharrefreplace"</c>, <c>"namereplace"</c>, <c>"surrogateescape"</c>,
        ///     <c>"surrogatepass"</c>); ignored when <paramref name="encoding"/> is null (NumPy's <c>_clean_args</c> stops at
        ///     the first None). Broadcast like <paramref name="encoding"/>.</param>
        /// <returns>A fresh C-ordered PLAIN <c>S</c> array over the broadcast shape (a chararray operand too), as wide as the
        ///     longest encoding (at least 1), 0-d for a scalar; an empty <c>S1</c> array when the broadcast is
        ///     empty.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="AttributeError"><paramref name="a"/> is <c>S</c>: <c>type object 'bytes' has no attribute
        ///     'encode'</c>.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA object that is not a str; an argument that is not a str (<c>encode() argument
        ///     'encoding' must be str, not numpy.bytes_</c>).</exception>
        /// <exception cref="ValueError">An argument holding a NUL (<c>embedded null character</c>); a ragged argument
        ///     list.</exception>
        /// <exception cref="UnicodeEncodeError">A character the codec cannot encode under <c>strict</c>, a handler that
        ///     re-raises it, or a surrogate in an argument.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec (<c>unknown encoding: nope</c>); an unknown handler
        ///     name at the first error.</exception>
        /// <exception cref="SystemError">A <c>U</c> element of one unit past U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="NotSupportedException">A Python codec NumSharp does not implement; an argument with no NumSharp
        ///     array form.</exception>
        [NDScoped]
        public NDArray encode(object a, object encoding = null, object errors = null)
            => np.strings.encode(TextOperand(a, nameof(a)), encoding, errors);

        /// <summary>
        ///     NumPy's <c>np.char.decode(a, encoding=None, errors=None)</c>, which IS <see cref="np.strings.decode"/>:
        ///     CPython's <c>bytes.decode(encoding, errors)</c> per element of an <c>S</c> array, the results as a <c>U</c>
        ///     array. An element is its bytes without the trailing NULs, a chararray's trailing whitespace included.
        /// </summary>
        /// <param name="a">The bytes: an <c>S</c> array or a bytes array_like (see the class remarks: a <c>byte[]</c> is
        ///     Python bytes, a list of <c>byte[]</c> an <c>S</c> array).</param>
        /// <param name="encoding">The codec name, or null for UTF-8 (the codecs of <see cref="np.strings.decode"/>);
        ///     broadcast against <paramref name="a"/>.</param>
        /// <param name="errors">The error handler, or null for <c>strict</c> (<c>"ignore"</c>, <c>"replace"</c>,
        ///     <c>"backslashreplace"</c>, <c>"surrogateescape"</c>, <c>"surrogatepass"</c>; <c>"xmlcharrefreplace"</c> and
        ///     <c>"namereplace"</c> are CPython's TypeError at the first error); ignored when <paramref name="encoding"/> is
        ///     null. Broadcast like <paramref name="encoding"/>.</param>
        /// <returns>A fresh C-ordered PLAIN <c>U</c> array over the broadcast shape (a chararray operand too), as wide as the
        ///     longest decoding (at least 1), 0-d for a scalar; an empty <c>U1</c> array when the broadcast is
        ///     empty.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="AttributeError"><paramref name="a"/> is <c>U</c> or <c>StringDType</c>: <c>type object 'str' has
        ///     no attribute 'decode'</c>.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text; an argument that is not a str;
        ///     <c>xmlcharrefreplace</c> / <c>namereplace</c> at an error (<c>don't know how to handle UnicodeDecodeError in
        ///     error callback</c>).</exception>
        /// <exception cref="ValueError">An argument holding a NUL; a ragged argument list.</exception>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>, or a handler that
        ///     re-raises it.</exception>
        /// <exception cref="UnicodeEncodeError">A surrogate in an argument.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec over a non-empty element.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec; an unknown handler name at the first error.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="NotSupportedException">A Python codec NumSharp does not implement; an argument with no NumSharp
        ///     array form.</exception>
        [NDScoped]
        public NDArray decode(object a, object encoding = null, object errors = null)
            => np.strings.decode(TextOperand(a, nameof(a)), encoding, errors);
    }
}
