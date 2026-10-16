using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public sealed partial class CharModule
    {
        // =============================================================================================================
        // The object path: split, rsplit, splitlines, join (docs/plans/np-char.md, C3)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.split(a, sep=None, maxsplit=None)</c>: every element's list of words, as Python's
        ///     <c>str.split</c> / <c>bytes.split</c> returns it. With <paramref name="sep"/> null the words are the runs of
        ///     non-whitespace (CPython's 29 whitespace code points for <c>U</c> / <c>StringDType</c>, 6 ASCII bytes for
        ///     <c>S</c>) and empty words never appear; with a separator every occurrence splits, so empty words do.
        /// </summary>
        /// <param name="a">The text: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the class
        ///     remarks).</param>
        /// <param name="sep">The separator, broadcast per element: null (NumPy's <c>None</c>) for whitespace; text; for an
        ///     <c>S</c> array also any NumPy number, whose raw bytes are the separator (NumPy's <c>bytes.split</c> reads a
        ///     NumPy scalar through the buffer protocol: <c>np.char.split(s, 5)</c> splits on the 8 bytes of
        ///     <c>np.int64(5)</c>, and a <c>str</c> separator on its UCS-4 bytes); a C# list holding null gives some elements
        ///     <c>None</c> (NumPy's object array).</param>
        /// <param name="maxsplit">The most splits per element, broadcast: null (NumPy's <c>None</c>) or a negative value for
        ///     no limit; any integer (NumSharp's <c>Char</c> is NumPy's <c>uint16</c>).</param>
        /// <returns>NumPy's object array of lists (<see cref="NDListArray"/>), C-ordered over the broadcast shape, 0-d for a
        ///     0-d broadcast.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>, also
        ///     for null: Python's <c>None</c> is an object array here, see the class remarks); per
        ///     element, in NumPy's order: a <c>StringDType</c> NA that is not a str (<c>descriptor 'split' for 'str' objects
        ///     doesn't apply to a 'NoneType' object</c>), a <paramref name="maxsplit"/> without <c>__index__</c>
        ///     (<c>'numpy.float64' object cannot be interpreted as an integer</c>; <c>np.bool</c> has none), a separator of
        ///     the wrong type (<c>must be str or None, not numpy.bytes_</c> for <c>U</c>; <c>a bytes-like object is required,
        ///     not 'str'</c> for <c>S</c> with a <c>StringDType</c> separator).</exception>
        /// <exception cref="OverflowException">A <paramref name="maxsplit"/> past <c>Py_ssize_t</c> (<c>Python int too large
        ///     to convert to C ssize_t</c>).</exception>
        /// <exception cref="ValueError">An empty separator (<c>empty separator</c>); a ragged list argument (NumPy's
        ///     inhomogeneous-shape text).</exception>
        /// <exception cref="SystemError">A <c>U</c> element, separator or word that is one code unit above U+10FFFF
        ///     (CPython's <c>PyUnicode_New</c> refusal).</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast (NumPy's <c>shape mismatch: objects
        ///     cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1 with shape
        ///     (3,).</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     The first element (C order) that raises ends the call, as NumPy's loop does; an empty array raises none of the
        ///     element errors (<c>np.char.split(empty, '')</c> is an empty result). Each element is the NumPy scalar
        ///     <c>_vec_string</c> makes of it: trailing NULs are gone, embedded ones stay.
        ///     </para>
        ///     <para>
        ///     A <c>U</c> element shorter than the separator, or whose characters are all of a narrower kind than the
        ///     separator's (<c>'abc'</c> split on <c>'€'</c>), is CPython's <c>[self]</c>: the list holds the element's own
        ///     <c>np.str_</c>, which <see cref="NDListArray"/>'s repr prints as <c>np.str_('abc')</c>, as NumPy's does.
        ///     </para>
        /// </remarks>
        [NDAnchored]
        public NDListArray split(object a, object sep = null, object maxsplit = null) => SplitCore(a, sep, maxsplit, reverse: false);

        /// <summary>
        ///     NumPy's <c>np.char.rsplit(a, sep=None, maxsplit=None)</c>: <see cref="split"/> from the right — with a
        ///     <paramref name="maxsplit"/>, the splits counted from the end (<c>'a,b,c'.rsplit(',', 1)</c> is
        ///     <c>['a,b', 'c']</c>). Without one, the same words as <see cref="split"/>.
        /// </summary>
        /// <param name="a">The text (see <see cref="split"/>).</param>
        /// <param name="sep">The separator (see <see cref="split"/>).</param>
        /// <param name="maxsplit">The most splits per element, from the right (see <see cref="split"/>).</param>
        /// <returns>NumPy's object array of lists (<see cref="NDListArray"/>).</returns>
        /// <exception cref="TypeError">As <see cref="split"/> (the descriptor names <c>'rsplit'</c>).</exception>
        /// <exception cref="OverflowException">A <paramref name="maxsplit"/> past <c>Py_ssize_t</c>.</exception>
        /// <exception cref="ValueError">An empty separator, or a ragged list argument.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> element, separator or word above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        [NDAnchored]
        public NDListArray rsplit(object a, object sep = null, object maxsplit = null) => SplitCore(a, sep, maxsplit, reverse: true);

        /// <summary>
        ///     NumPy's <c>np.char.splitlines(a, keepends=None)</c>: every element's list of lines. <c>U</c> /
        ///     <c>StringDType</c> lines end at <c>\n</c>, <c>\r</c>, <c>\r\n</c>, <c>\v</c>, <c>\f</c>, <c>\x1c</c>,
        ///     <c>\x1d</c>, <c>\x1e</c>, <c>\x85</c>, U+2028 and U+2029; <c>S</c> lines at <c>\n</c>, <c>\r</c> and
        ///     <c>\r\n</c> only. A trailing boundary ends the last line (no empty line after it); an empty element has no
        ///     line.
        /// </summary>
        /// <param name="a">The text (see <see cref="split"/>).</param>
        /// <param name="keepends">Whether a line keeps its boundary, broadcast per element and read with Python's truth
        ///     rule (<c>PyObject_IsTrue</c>: <c>0.5</c>, <c>'x'</c> and NaN are true, <c>''</c> and 0 false); null (NumPy's
        ///     <c>None</c>) is false.</param>
        /// <returns>NumPy's object array of lists (<see cref="NDListArray"/>).</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (null too: Python's <c>None</c>); a
        ///     <c>StringDType</c> NA that is not a str
        ///     (<c>descriptor 'splitlines' for 'str' objects doesn't apply to a 'NoneType' object</c>).</exception>
        /// <exception cref="ValueError">A ragged list argument.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> element, keepends value or line above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="a"/> and <paramref name="keepends"/> do not broadcast.</exception>
        [NDAnchored]
        public NDListArray splitlines(object a, object keepends = null)
        {
            using var text = VecArgOfText(a, nameof(a));
            using var keep = keepends is null ? null : NDCharObjectPath.VecArg.Convert(keepends, nameof(keepends));
            return NDCharObjectPath.Splitlines(text, keep);
        }

        /// <summary>
        ///     NumPy's <c>np.char.join(sep, seq)</c>: every element of <paramref name="seq"/> — a STRING — with its
        ///     characters joined by the separator element (<c>join('-', 'osd')</c> is <c>'o-s-d'</c>), as Python's
        ///     <c>sep.join(seq)</c> iterates a string.
        /// </summary>
        /// <param name="sep">The separators: <c>S</c> / <c>U</c> / <c>StringDType</c> text or a text array_like.</param>
        /// <param name="seq">The strings, broadcast against <paramref name="sep"/>.</param>
        /// <returns>
        ///     The joined text as NumPy's <c>_to_bytes_or_str_array</c> builds it: of <paramref name="seq"/>'s class (<c>U</c>
        ///     / <c>S</c> as wide as the longest result, at least 1; <c>StringDType()</c> with default parameters), a fresh
        ///     C-ordered array; for an empty broadcast an empty array of <paramref name="seq"/>'s dtype, whatever it is.
        /// </returns>
        /// <exception cref="TypeError"><paramref name="sep"/> is not text (<c>string operation on non-string array</c>,
        ///     also for null: Python's <c>None</c>); per element, in NumPy's order: a <c>StringDType</c> NA separator that is
        ///     not a str (<c>descriptor 'join' for 'str' objects doesn't apply to a 'NoneType' object</c>), a non-iterable
        ///     sequence (a number or None — a null <paramref name="seq"/> is Python's <c>None</c> —: <c>can only join an
        ///     iterable</c>), a sequence of the other class (<c>sequence item 0: expected str instance,
        ///     int found</c> — iterating bytes gives ints —, <c>sequence item 0: expected a bytes-like object, str
        ///     found</c>), and an object <paramref name="seq"/> (a C# list holding null) once every element joined
        ///     (<see cref="NDCharObjectPath.ObjectDTypeText"/>, NumPy 2.4.2's own failure).</exception>
        /// <exception cref="ValueError">A ragged list argument.</exception>
        /// <exception cref="SystemError">A <c>U</c> separator or sequence that is one code unit above U+10FFFF, or a
        ///     <c>U</c> sequence holding such a unit (iterating it creates the character).</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        /// <exception cref="NotSupportedException"><paramref name="seq"/> is an <see cref="NDListArray"/> whose broadcast
        ///     is empty: NumPy returns an empty object array, which only the <see cref="join(object, NDListArray)"/> overload
        ///     can return.</exception>
        /// <remarks>
        ///     An <c>S</c> sequence joins only when empty (<c>b'-'.join(b'ab')</c> iterates ints), as in NumPy. A list of
        ///     words is not a sequence here: <c>np.asarray(['ab', 'cd'])</c> is a string array, so each word's characters are
        ///     joined.
        /// </remarks>
        [NDAnchored]
        public NDArray join(object sep, object seq)
        {
            if (seq is NDListArray lists)
            {
                var r = join(sep, lists);
                r.Dispose();
                throw new NotSupportedException("np.char.join over an empty object array returns an empty object array; call the " +
                                                "join(object, NDListArray) overload for it.");
            }
            using var separators = VecArgOfText(sep, nameof(sep));
            using var sequences = NDCharObjectPath.VecArg.Convert(seq, nameof(seq));
            var result = NDCharObjectPath.Join(separators, sequences, out var empty);
            if (result is null)
            {
                empty.Dispose();
                throw new NotSupportedException("np.char.join over an empty object array returns an empty object array, which " +
                                                "NumSharp's NDArray result cannot hold.");
            }
            return result;
        }

        /// <summary>
        ///     <c>np.char.join(sep, seq)</c> with <paramref name="seq"/> an object array of lists (<see cref="split"/>'s
        ///     result): each element's list joined, as NumPy's <c>_vec_string</c> joins it — and then NumPy 2.4.2's own
        ///     error, because <c>_to_bytes_or_str_array</c> calls <c>ObjectDType(n)</c>, which takes no argument. So the
        ///     idiom <c>np.char.join(' ', np.char.split(a))</c> raises in NumPy 2.4.2, and here.
        /// </summary>
        /// <param name="sep">The separators (text).</param>
        /// <param name="seq">The lists.</param>
        /// <returns>For an empty broadcast, NumPy's empty object array: an empty <see cref="NDListArray"/> of the broadcast
        ///     shape. A non-empty broadcast never returns (NumPy raises).</returns>
        /// <exception cref="TypeError"><paramref name="sep"/> is not text (null too); a null <paramref name="seq"/> is
        ///     Python's <c>None</c> (<c>can only join an iterable</c> per element); per element, a list item of the wrong class
        ///     (<c>sequence item 0: expected str instance, bytes found</c>, <c>sequence item 0: expected a bytes-like object,
        ///     str found</c>) or a <c>StringDType</c> NA separator; otherwise, for a non-empty broadcast, NumPy's
        ///     <see cref="NDCharObjectPath.ObjectDTypeText"/> after every element joined.</exception>
        /// <exception cref="SystemError">A one-unit <c>U</c> separator above U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        [NDAnchored]
        public NDListArray join(object sep, NDListArray seq)
        {
            using var separators = VecArgOfText(sep, nameof(sep));
            // A null seq is Python's None (C# binds a bare null here, the more specific overload): the object 0-d array
            // holding None, whose element join raises "can only join an iterable" — or an empty object array.
            using var lists = NDCharObjectPath.VecArg.Convert(seq, nameof(seq));
            var result = NDCharObjectPath.Join(separators, lists, out var empty);
            result?.Dispose();
            return empty;
        }

        /// <summary>The <c>split</c> / <c>rsplit</c> body: the arguments converted in NumPy's order, then the walk.</summary>
        /// <param name="a">The text.</param>
        /// <param name="sep">The separator, or null.</param>
        /// <param name="maxsplit">The limit, or null.</param>
        /// <param name="reverse">True for <c>rsplit</c>.</param>
        /// <returns>The lists.</returns>
        /// <exception cref="Exception">See <see cref="split"/>.</exception>
        private static NDListArray SplitCore(object a, object sep, object maxsplit, bool reverse)
        {
            using var text = VecArgOfText(a, nameof(a));
            // _vec_string's multi-iterator converts the other arguments after the text's class check, in order.
            using var separators = sep is null ? null : NDCharObjectPath.VecArg.Convert(sep, nameof(sep));
            using var limits = maxsplit is null ? null : NDCharObjectPath.VecArg.Convert(maxsplit, nameof(maxsplit));
            return NDCharObjectPath.Split(text, separators, limits, reverse);
        }

        /// <summary>
        ///     <c>_vec_string</c>'s first argument: converted (<c>PyArray_Converter</c>), then its class picks the method —
        ///     <c>S</c>, <c>U</c> or <c>StringDType</c>, anything else NumPy's TypeError.
        /// </summary>
        /// <param name="value">The argument.</param>
        /// <param name="name">The parameter name.</param>
        /// <returns>The converted text.</returns>
        /// <exception cref="TypeError">Not text: <c>string operation on non-string array</c> (null is Python's
        ///     <c>None</c>, an object array).</exception>
        /// <exception cref="ValueError">A ragged list.</exception>
        private static NDCharObjectPath.VecArg VecArgOfText(object value, string name)
        {
            var arg = NDCharObjectPath.VecArg.Convert(value, name);
            if (!arg.IsText)
            {
                arg.Dispose();
                throw new TypeError(NDCharObjectPath.NonStringArrayText);
            }
            return arg;
        }
    }
}
