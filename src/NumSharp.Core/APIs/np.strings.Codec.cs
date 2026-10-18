using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.strings.Codec.cs — numpy.strings.encode / numpy.strings.decode (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// NumPy's encode(a, encoding=None, errors=None) is _to_bytes_or_str_array(_vec_string(a, np.object_, 'encode',
// _clean_args(encoding, errors)), np.bytes_(b'')): CPython's str.encode per element of a U / StringDType array, the results
// as an S array as wide as the longest one; decode is bytes.decode per element of an S array into a U array. Not ufuncs
// (no out= / where= / dtype=). _clean_args stops at the first None, so an errors argument without an encoding is ignored.
// np.asarray drops a chararray's class, so a chararray operand gives a plain array. The engine is NDCharObjectPath.Codec
// (the walk, the arguments, the rows) over NDCodec (CPython's codecs and error handlers).
// =============================================================================

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            /// <summary>
            ///     NumPy's <c>np.strings.encode</c>: CPython's <c>str.encode(encoding, errors)</c> per element of a <c>U</c> /
            ///     <c>StringDType</c> array, the results as an <c>S</c> array.
            /// </summary>
            /// <param name="a">The text: <c>U</c> or <c>StringDType</c> (a C# <c>string</c> converts to a 0-d <c>U</c> array).</param>
            /// <param name="encoding">
            ///     The codec name, or null for UTF-8. Any name and alias CPython's <c>codecs.lookup</c> accepts (<c>"utf-8"</c>,
            ///     <c>"UTF8"</c>, <c>"latin-1"</c>, <c>"cp1252"</c>, <c>"utf-16-le"</c>, <c>"ascii"</c> …) for the codecs NumSharp
            ///     implements: UTF-8, <c>utf-8-sig</c>, UTF-16 and UTF-32 (with a BOM, little- or big-endian), Latin-1, ASCII,
            ///     the single-byte code pages and <c>undefined</c>. Converted as NumPy's <c>np.asarray</c> converts it and
            ///     broadcast against <paramref name="a"/>: an array gives one encoding per element; an element that is not a
            ///     str (a <c>byte[]</c>, a number, null in an object list) is CPython's TypeError.
            /// </param>
            /// <param name="errors">
            ///     The error handler, or null for <c>strict</c>: <c>"strict"</c>, <c>"ignore"</c>, <c>"replace"</c> (<c>?</c>),
            ///     <c>"backslashreplace"</c> (<c>\xe9</c>, <c>€</c>, <c>\U0001f600</c>), <c>"xmlcharrefreplace"</c>
            ///     (<c>&amp;#233;</c>), <c>"namereplace"</c> (<c>\N{EURO SIGN}</c>, CPython 3.12's Unicode 15.0 names),
            ///     <c>"surrogateescape"</c> (U+DC80..U+DCFF back to their bytes) or <c>"surrogatepass"</c> (lone surrogates
            ///     encoded, UTF-8 / 16 / 32 only). An unknown name raises <see cref="LookupError"/> only at the first character
            ///     the codec cannot encode. Ignored when <paramref name="encoding"/> is null (NumPy's <c>_clean_args</c>).
            ///     Broadcast like <paramref name="encoding"/>.
            /// </param>
            /// <returns>A fresh C-ordered <c>S</c> array over the broadcast shape, as wide as the longest encoding (at least 1;
            ///     a trailing NUL byte counts: <c>encode(["a"], "utf-16-le")</c> is <c>S2</c> holding <c>b"a"</c>); an empty
            ///     <c>S1</c> array when the broadcast is empty.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="AttributeError"><paramref name="a"/> is <c>S</c>: <c>type object 'bytes' has no attribute
            ///     'encode'</c>.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
            ///     <c>StringDType</c> NA object that is not a str (<c>descriptor 'encode' for 'str' objects doesn't apply to a
            ///     'NoneType' object</c>); an argument that is not a str (<c>encode() argument 'encoding' must be str, not
            ///     numpy.bytes_</c>).</exception>
            /// <exception cref="ValueError">An argument holding a NUL (<c>embedded null character</c>); a ragged argument list.</exception>
            /// <exception cref="UnicodeEncodeError">A character the codec cannot encode under <c>strict</c> (<c>'ascii' codec can't
            ///     encode character '\xe9' in position 1: ordinal not in range(128)</c>; a run of them reads <c>characters in
            ///     position 1-3</c>; a code page's errors name <c>'charmap'</c>), a handler that re-raises it, or a surrogate in
            ///     an argument.</exception>
            /// <exception cref="UnicodeError">The <c>undefined</c> codec (<c>undefined encoding</c>).</exception>
            /// <exception cref="LookupError"><c>unknown encoding: nope</c>; <c>'base64' is not a text encoding; use
            ///     codecs.encode() to handle arbitrary codecs</c>; <c>unknown error handler name 'x'</c> at the first
            ///     error.</exception>
            /// <exception cref="SystemError">A <c>U</c> element of one unit past U+10FFFF.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's <c>shape mismatch</c> text).</exception>
            /// <exception cref="NotSupportedException">A Python codec NumSharp does not implement (the multi-byte CJK codecs,
            ///     <c>utf-7</c>, the escape codecs, <c>idna</c>, <c>mbcs</c> …); an object array of lists as an argument.</exception>
            /// <remarks>
            ///     <para>An element is the text without its trailing NULs (<c>U</c>, NumPy's np.str_) or the whole cell
            ///     (<c>StringDType</c>). The first element (C order) that raises ends the call, as in NumPy; an argument is
            ///     checked at every element it reaches, so an invalid encoding raises at the first element.</para>
            ///     <para><c>np.strings.encode(np.array(new[] { "a€", "b" }), "ascii", "xmlcharrefreplace")</c> is
            ///     <c>[b"a&amp;#8364;", b"b"]</c> (<c>S8</c>); <c>"utf-16"</c> writes a byte-order mark before every element.</para>
            /// </remarks>
            public static NDArray encode(NDArray a, object encoding = null, object errors = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain((a, encoding, errors), static s => encode(s.a, s.encoding, s.errors));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDCharObjectPath.Encode(a, encoding, errors);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.decode</c>: CPython's <c>bytes.decode(encoding, errors)</c> per element of an <c>S</c>
            ///     array, the results as a <c>U</c> array.
            /// </summary>
            /// <param name="a">The bytes: an <c>S</c> array (build a 0-d one with <see cref="NDArray.Scalar(object)"/> of a
            ///     <c>byte[]</c>).</param>
            /// <param name="encoding">The codec name, or null for UTF-8 (the codecs of <see cref="encode"/>); broadcast
            ///     against <paramref name="a"/>.</param>
            /// <param name="errors">
            ///     The error handler, or null for <c>strict</c>: <c>"strict"</c>, <c>"ignore"</c>, <c>"replace"</c> (one
            ///     U+FFFD per bad sequence), <c>"backslashreplace"</c> (<c>\xff</c> per byte), <c>"surrogateescape"</c> (each
            ///     byte 0x80..0xFF as U+DC80..U+DCFF) or <c>"surrogatepass"</c> (an encoded surrogate decoded, UTF-8 / 16 / 32
            ///     only); <c>"xmlcharrefreplace"</c> and <c>"namereplace"</c> raise CPython's TypeError at the first error. An
            ///     unknown name raises <see cref="LookupError"/> only at the first error. Ignored when
            ///     <paramref name="encoding"/> is null.
            /// </param>
            /// <returns>A fresh C-ordered <c>U</c> array over the broadcast shape, as wide as the longest decoding (at least 1);
            ///     an empty <c>U1</c> array when the broadcast is empty.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="AttributeError"><paramref name="a"/> is <c>U</c> or <c>StringDType</c>: <c>type object 'str' has
            ///     no attribute 'decode'</c>.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text; an argument that is not a str
            ///     (<c>decode() argument 'encoding' must be str, not numpy.int64</c>); <c>xmlcharrefreplace</c> /
            ///     <c>namereplace</c> at an error (<c>don't know how to handle UnicodeDecodeError in error callback</c>).</exception>
            /// <exception cref="ValueError">An argument holding a NUL; a ragged argument list.</exception>
            /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c> (<c>'utf-8' codec can't
            ///     decode byte 0xff in position 0: invalid start byte</c>), or a handler that re-raises it.</exception>
            /// <exception cref="UnicodeEncodeError">A surrogate in an argument.</exception>
            /// <exception cref="UnicodeError">The <c>undefined</c> codec over a non-empty element.</exception>
            /// <exception cref="LookupError">An unknown or non-text codec; an unknown handler name at the first error.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="NotSupportedException">A Python codec NumSharp does not implement; an object array of lists as
            ///     an argument.</exception>
            /// <remarks>
            ///     <para>An element is its bytes without the trailing NULs (NumPy's np.bytes_), so an empty element decodes to
            ///     <c>''</c> before the codec is even looked up — an unknown encoding over empty elements is no error, as in
            ///     NumPy — and UTF-16 / UTF-32 text whose last unit ends in a zero byte loses that byte (NumPy's
            ///     <c>truncated data</c> error, reproduced).</para>
            ///     <para><c>np.strings.decode(np.array(new[] { "a\xff"u8.ToArray() }), "utf-8", "replace")</c> is <c>["a�"]</c>
            ///     (<c>U2</c>).</para>
            /// </remarks>
            public static NDArray decode(NDArray a, object encoding = null, object errors = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain((a, encoding, errors), static s => decode(s.a, s.encoding, s.errors));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDCharObjectPath.Decode(a, encoding, errors);
            }
        }
    }
}
