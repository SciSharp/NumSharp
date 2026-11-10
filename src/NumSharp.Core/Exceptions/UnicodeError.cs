using System;
using System.Globalization;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>UnicodeError</c> — Python's base class for text encoding and decoding failures, a
    ///     subclass of <see cref="ValueError"/>. NumSharp raises its two concrete subclasses,
    ///     <see cref="UnicodeEncodeError"/> and <see cref="UnicodeDecodeError"/>, when a value crosses between a
    ///     fixed-width Unicode (<c>U</c>) and a byte-string (<c>S</c>) dtype the way NumPy's ASCII codec refuses it.
    /// </summary>
    /// <remarks>
    ///     Catching <see cref="ValueError"/> (or <see cref="ArgumentException"/>) catches these too, exactly as
    ///     <c>except ValueError</c> catches Python's Unicode errors. The message is CPython's verbatim
    ///     <c>UnicodeError.__str__</c> text, so error parity with NumPy is a string comparison.
    /// </remarks>
    public class UnicodeError : ValueError
    {
        /// <summary>Creates the error with CPython's message text.</summary>
        /// <param name="message">The verbatim message (<c>'ascii' codec can't encode …</c>).</param>
        public UnicodeError(string message) : base(message) { }

        /// <summary>Creates the error with CPython's message text and the failure that caused it.</summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The underlying failure.</param>
        public UnicodeError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     NumPy-compatible <c>UnicodeEncodeError</c>: a Unicode string held a character the target encoding cannot
    ///     represent. NumSharp raises it where NumPy's codecs do — a <c>U</c> or <c>StringDType</c> value with a code point
    ///     at or above 128 stored into an <c>S</c> (byte-string) array, cast to <c>S</c>, or given to
    ///     <c>np.array(…, dtype='S')</c>, and every codec of <c>np.strings.encode</c>.
    /// </summary>
    /// <remarks>
    ///     <para>The message follows CPython's <c>UnicodeEncodeError.__str__</c> exactly: one offending character reads
    ///     <c>'ascii' codec can't encode character '\xe9' in position 1: ordinal not in range(128)</c> (the character as
    ///     <c>\xNN</c>, <c>\uNNNN</c> or <c>\UNNNNNNNN</c>), a run of consecutive offending characters reads
    ///     <c>'ascii' codec can't encode characters in position 1-2: ordinal not in range(128)</c>. Positions count code
    ///     points of the element, not UTF-16 units, as Python does.</para>
    ///     <para>Like Python's, it carries <see cref="encoding"/>, <see cref="@object"/> (the element's text),
    ///     <see cref="start"/>, <see cref="end"/> and <see cref="reason"/>, so a caller reads where encoding failed without
    ///     parsing the message.</para>
    /// </remarks>
    public class UnicodeEncodeError : UnicodeError
    {
        /// <summary>
        ///     Creates the error from a message alone: <see cref="encoding"/>, <see cref="@object"/> and <see cref="reason"/>
        ///     are null and <see cref="start"/> / <see cref="end"/> are 0.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <remarks>Python has no such constructor (its <c>UnicodeEncodeError</c> takes the five arguments). Every NumSharp
        ///     raise site fills the five values, so a caught error carries them; this form is for code that holds a text
        ///     only.</remarks>
        public UnicodeEncodeError(string message) : base(message) { }

        /// <summary>
        ///     Python's <c>UnicodeEncodeError(encoding, object, start, end, reason)</c>: the message is CPython's
        ///     <c>UnicodeEncodeError.__str__</c> of the five values.
        /// </summary>
        /// <param name="encoding">The codec name (<c>ascii</c>, <c>utf-8</c>, <c>charmap</c>).</param>
        /// <param name="object">The text being encoded.</param>
        /// <param name="start">The code-point position of the first unencodable character.</param>
        /// <param name="end">One past the last one (a run of consecutive unencodable characters is reported together).</param>
        /// <param name="reason">CPython's reason text (<c>ordinal not in range(128)</c>).</param>
        /// <exception cref="TypeError"><paramref name="encoding"/>, <paramref name="object"/> or <paramref name="reason"/> is
        ///     null (CPython: <c>argument 1 must be str, not None</c>, and arguments 2 and 5 alike).</exception>
        /// <remarks>
        ///     Footgun: <paramref name="start"/> and <paramref name="end"/> count CODE POINTS, as Python's str indices do; in
        ///     the UTF-16 <paramref name="object"/> they stop matching char indices once a character past <c>U+FFFF</c> (a
        ///     surrogate pair) precedes them. The one-character message (<c>end == start + 1</c>, start inside the text) quotes
        ///     the code point at <paramref name="start"/> as <c>'\xe9'</c> / <c>'中'</c> / <c>'\U0001f600'</c>; any other
        ///     range reads <c>characters in position 0-2</c>, as CPython prints it.
        /// </remarks>
        public UnicodeEncodeError(string encoding, string @object, long start, long end, string reason)
            : this(encoding, @object, start, end, reason, CodePointAt(@object, start)) { }

        /// <summary>
        ///     The raise sites' form: the first unencodable code point is passed in, because a UCS-4 element knows it exactly,
        ///     where re-reading it from the UTF-16 <paramref name="object"/> would merge two lone surrogate code points
        ///     (<c>'\ud83d', '\ude00'</c>) into one character.
        /// </summary>
        /// <param name="encoding">The codec name.</param>
        /// <param name="object">The text being encoded.</param>
        /// <param name="start">The code-point position of the first unencodable character.</param>
        /// <param name="end">One past the last one.</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <param name="badCodePoint">The code point at <paramref name="start"/>, or -1 when <paramref name="start"/> is past
        ///     the end of the text (the message then takes the range form).</param>
        /// <exception cref="TypeError">A text argument is null.</exception>
        internal UnicodeEncodeError(string encoding, string @object, long start, long end, string reason, long badCodePoint)
            : base(FormatMessage(encoding, @object, start, end, reason, badCodePoint))
        {
            this.encoding = encoding;
            this.@object = @object;
            this.start = start;
            this.end = end;
            this.reason = reason;
        }

        /// <summary>The codec name (Python's <c>encoding</c>), or null when the error was created from a message alone.</summary>
        public string encoding { get; }

        /// <summary>The text being encoded (Python's <c>object</c>), or null when the error was created from a message alone.</summary>
        /// <remarks>The whole element NumPy handed the codec, not just the unencodable part.</remarks>
        public string @object { get; }

        /// <summary>The code-point position of the first unencodable character (Python's <c>start</c>).</summary>
        /// <remarks>A code-point index, not a UTF-16 one: see the five-argument constructor.</remarks>
        public long start { get; }

        /// <summary>One past the code-point position of the last unencodable character (Python's <c>end</c>).</summary>
        public long end { get; }

        /// <summary>CPython's reason text (Python's <c>reason</c>), or null when the error was created from a message alone.</summary>
        public string reason { get; }

        /// <summary>
        ///     CPython's <c>UnicodeEncodeError.__str__</c> (<c>Objects/exceptions.c</c>), after its argument checks.
        /// </summary>
        /// <param name="encoding">The codec name.</param>
        /// <param name="object">The text.</param>
        /// <param name="start">The first position.</param>
        /// <param name="end">One past the last.</param>
        /// <param name="reason">The reason.</param>
        /// <param name="badCodePoint">The code point at <paramref name="start"/>, or -1 when there is none.</param>
        /// <returns>The message.</returns>
        /// <exception cref="TypeError">A text argument is null.</exception>
        private static string FormatMessage(string encoding, string @object, long start, long end, string reason, long badCodePoint)
        {
            // CPython's PyArg_ParseTuple order: argument 1, then 2, then 5 (the positions are ints and cannot be None here).
            if (encoding is null)
                throw new TypeError("argument 1 must be str, not None");
            if (@object is null)
                throw new TypeError("argument 2 must be str, not None");
            if (reason is null)
                throw new TypeError("argument 5 must be str, not None");
            if (badCodePoint >= 0 && end == start + 1)
            {
                string escaped = badCodePoint <= 0xFF ? string.Create(CultureInfo.InvariantCulture, $"\\x{badCodePoint:x2}")
                    : badCodePoint <= 0xFFFF ? string.Create(CultureInfo.InvariantCulture, $"\\u{badCodePoint:x4}")
                    : string.Create(CultureInfo.InvariantCulture, $"\\U{badCodePoint:x8}");
                return string.Create(CultureInfo.InvariantCulture, $"'{encoding}' codec can't encode character '{escaped}' in position {start}: {reason}");
            }
            return string.Create(CultureInfo.InvariantCulture, $"'{encoding}' codec can't encode characters in position {start}-{end - 1}: {reason}");
        }

        /// <summary>
        ///     The code point at code-point position <paramref name="position"/> of <paramref name="text"/> (a surrogate pair
        ///     is one code point, a lone surrogate is its own), or -1 when the text is null or shorter.
        /// </summary>
        /// <param name="text">The text (null allowed: the message check reports it).</param>
        /// <param name="position">The code-point position.</param>
        /// <returns>The code point, or -1.</returns>
        private static long CodePointAt(string text, long position)
        {
            if (text is null || position < 0)
                return -1;
            long cp = 0;
            for (int i = 0; i < text.Length; i++, cp++)
            {
                bool pair = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]);
                if (cp == position)
                    return pair ? char.ConvertToUtf32(text[i], text[i + 1]) : text[i];
                if (pair)
                    i++;
            }
            return -1;
        }
    }

    /// <summary>
    ///     NumPy-compatible <c>UnicodeDecodeError</c>: bytes that the source encoding cannot read. NumSharp raises it where
    ///     NumPy's codecs do — an <c>S</c> (byte-string) value holding a byte at or above <c>0x80</c> stored into a
    ///     <c>U</c> array, cast to <c>U</c>, or given to <c>np.array(…, dtype='U')</c>, every codec of
    ///     <c>np.strings.decode</c>, and text that is not UTF-8 where NumPy reads UTF-8.
    /// </summary>
    /// <remarks>
    ///     <para>The message follows CPython's <c>UnicodeDecodeError.__str__</c>: one bad byte reads
    ///     <c>'ascii' codec can't decode byte 0xe9 in position 1: ordinal not in range(128)</c> (the ASCII codec reports the
    ///     FIRST bad byte only), a longer range <c>'utf-8' codec can't decode bytes in position 0-1: …</c>.</para>
    ///     <para>Like Python's, it carries <see cref="encoding"/>, <see cref="@object"/> (the element's bytes),
    ///     <see cref="start"/>, <see cref="end"/> and <see cref="reason"/>, so a caller reads where decoding failed without
    ///     parsing the message.</para>
    /// </remarks>
    public class UnicodeDecodeError : UnicodeError
    {
        /// <summary>
        ///     Creates the error from a message alone: <see cref="encoding"/>, <see cref="@object"/> and <see cref="reason"/>
        ///     are null and <see cref="start"/> / <see cref="end"/> are 0.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <remarks>Python has no such constructor (its <c>UnicodeDecodeError</c> takes the five arguments). Every NumSharp
        ///     raise site fills the five values, so a caught error carries them; this form is for code that holds a text
        ///     only.</remarks>
        public UnicodeDecodeError(string message) : base(message) { }

        /// <summary>
        ///     Python's <c>UnicodeDecodeError(encoding, object, start, end, reason)</c>: the message is CPython's
        ///     <c>UnicodeDecodeError.__str__</c> of the five values.
        /// </summary>
        /// <param name="encoding">The codec name (<c>utf-8</c>, <c>ascii</c>, <c>charmap</c>).</param>
        /// <param name="object">The bytes being decoded (the whole input, not just the bad range).</param>
        /// <param name="start">The first byte of the bad range.</param>
        /// <param name="end">One past its last byte.</param>
        /// <param name="reason">CPython's reason text (<c>invalid start byte</c>).</param>
        /// <exception cref="TypeError"><paramref name="encoding"/>, <paramref name="object"/> or <paramref name="reason"/> is
        ///     null (CPython: <c>argument 1 must be str, not None</c>, <c>a bytes-like object is required, not
        ///     'NoneType'</c>, <c>argument 5 must be str, not None</c>).</exception>
        /// <remarks>
        ///     The one-byte message (<c>end == start + 1</c>, start inside the bytes) quotes the byte:
        ///     <c>'utf-8' codec can't decode byte 0xff in position 0: invalid start byte</c>; any other range reads
        ///     <c>bytes in position 0-1</c>, as CPython prints it. The array is kept as given, not copied.
        /// </remarks>
        public UnicodeDecodeError(string encoding, byte[] @object, long start, long end, string reason)
            : base(FormatMessage(encoding, @object, start, end, reason))
        {
            this.encoding = encoding;
            this.@object = @object;
            this.start = start;
            this.end = end;
            this.reason = reason;
        }

        /// <summary>The codec name (Python's <c>encoding</c>), or null when the error was created from a message alone.</summary>
        public string encoding { get; }

        /// <summary>The bytes being decoded (Python's <c>object</c>), or null when the error was created from a message alone.</summary>
        /// <remarks>The whole input NumPy handed the codec: for a fixed-width <c>S</c> element, its bytes without the trailing
        ///     NULs, as NumPy reads an element into a Python <c>bytes</c>.</remarks>
        public byte[] @object { get; }

        /// <summary>The first byte of the bad range (Python's <c>start</c>).</summary>
        public long start { get; }

        /// <summary>One past the last byte of the bad range (Python's <c>end</c>).</summary>
        public long end { get; }

        /// <summary>CPython's reason text (Python's <c>reason</c>), or null when the error was created from a message alone.</summary>
        public string reason { get; }

        /// <summary>
        ///     CPython's <c>UnicodeDecodeError.__str__</c> (<c>Objects/exceptions.c</c>), after its argument checks.
        /// </summary>
        /// <param name="encoding">The codec name.</param>
        /// <param name="object">The bytes.</param>
        /// <param name="start">The first byte of the range.</param>
        /// <param name="end">One past its last byte.</param>
        /// <param name="reason">The reason.</param>
        /// <returns>The message.</returns>
        /// <exception cref="TypeError">A text argument or the bytes are null.</exception>
        private static string FormatMessage(string encoding, byte[] @object, long start, long end, string reason)
        {
            // CPython's argument order: the codec name, then the bytes (a buffer, hence its own text), then the reason.
            if (encoding is null)
                throw new TypeError("argument 1 must be str, not None");
            if (@object is null)
                throw new TypeError("a bytes-like object is required, not 'NoneType'");
            if (reason is null)
                throw new TypeError("argument 5 must be str, not None");
            if (start >= 0 && start < @object.Length && end == start + 1)
                return string.Create(CultureInfo.InvariantCulture, $"'{encoding}' codec can't decode byte 0x{@object[start]:x2} in position {start}: {reason}");
            return string.Create(CultureInfo.InvariantCulture, $"'{encoding}' codec can't decode bytes in position {start}-{end - 1}: {reason}");
        }

        /// <summary>
        ///     CPython's <c>repr()</c> of the error: <c>UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')</c>
        ///     with the five arguments, or <c>UnicodeDecodeError('&lt;message&gt;')</c> when only the message is known.
        /// </summary>
        /// <returns>The text.</returns>
        internal string PythonRepr()
            => encoding == null
                ? $"UnicodeDecodeError({NumSharp.Backends.Printing.PyRepr.Str(Message)})"
                : string.Create(CultureInfo.InvariantCulture, $"UnicodeDecodeError({NumSharp.Backends.Printing.PyRepr.Str(encoding)}, " +
                  $"{NumSharp.Backends.Printing.PyRepr.Bytes(@object)}, {start}, {end}, " +
                  $"{NumSharp.Backends.Printing.PyRepr.Str(reason)})");
    }
}
