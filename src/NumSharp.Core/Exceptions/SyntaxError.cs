using System;
using System.Globalization;
using System.IO;

namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to Python's <c>SyntaxError</c>: text that is not valid Python source. Catch it
    ///     where a string argument is parsed as Python literals and the caller wants to tell malformed text from a
    ///     well-formed value that is merely not allowed (that one is a <see cref="ValueError"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Raised by the string form of <see cref="np.matrix(string, DType, bool)"/> / <see cref="np.asmatrix(string, DType)"/>,
    ///     where NumPy's <c>_convert_from_string</c> hands every whitespace-separated token to
    ///     <c>ast.literal_eval</c>. A token that does not parse (<c>"1 + 2"</c> splits into the lone token
    ///     <c>+</c>, <c>"1L"</c> is not a number) is CPython's <c>SyntaxError</c> there, with CPython's own text
    ///     (<c>invalid syntax (&lt;unknown&gt;, line 1)</c>, <c>invalid decimal literal (&lt;unknown&gt;, line 1)</c>).
    ///     </para>
    ///     <para>
    ///     It is NOT a <see cref="ValueError"/>, exactly as in Python, so a <c>catch (ValueError)</c> written for
    ///     "Rows not the same size." does not swallow a typo in the literal.
    ///     </para>
    ///     <para>
    ///     Like Python's, it carries the location: <see cref="msg"/>, <see cref="filename"/>, <see cref="lineno"/>,
    ///     <see cref="offset"/>, <see cref="text"/>, <see cref="end_lineno"/> and <see cref="end_offset"/>, filled by
    ///     <see cref="SyntaxError(string, string, int?, int?, string, int?, int?)"/> (Python's
    ///     <c>SyntaxError(msg, (filename, lineno, offset, text, end_lineno, end_offset))</c>), whose
    ///     <see cref="Exception.Message"/> is CPython's <c>str()</c>: <c>invalid syntax (c.py, line 3)</c>. Python's
    ///     subclasses <see cref="IndentationError"/> and <see cref="TabError"/> derive from it.
    ///     </para>
    /// </remarks>
    public class SyntaxError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise SyntaxError</c>: <c>"None"</c>, as
        ///     <c>str(SyntaxError())</c> is <c>'None'</c> (CPython's <c>SyntaxError.__str__</c> prints its <c>msg</c>
        ///     attribute, which is None without arguments), unlike the other Python errors' empty text.
        /// </summary>
        /// <remarks>Prefer the overload that carries the parser's text: a bare <c>SyntaxError</c> tells the user nothing
        ///     about the token. Every location attribute, <see cref="msg"/> included, is null.</remarks>
        public SyntaxError() : base("None") { }

        /// <summary>Creates the error with the parser's message, which is also <see cref="msg"/>; the location attributes
        ///     stay null.</summary>
        /// <param name="message">CPython's message, including its <c>(&lt;unknown&gt;, line 1)</c> location suffix
        ///     where CPython prints one.</param>
        /// <remarks>Python's <c>SyntaxError('m')</c>: <c>msg</c> is <c>'m'</c> and the text is <c>m</c>, nothing parsed out
        ///     of it, so a message that already holds a location suffix keeps it in <see cref="msg"/> too.</remarks>
        public SyntaxError(string message) : base(message)
        {
            msg = message;
        }

        /// <summary>
        ///     Creates the error with the parser's message and the exception that caused it: Python's <c>raise
        ///     SyntaxError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as
        ///     Python keeps it in <c>__cause__</c>; <see cref="msg"/> is the message and the location attributes stay null.
        /// </summary>
        /// <param name="message">CPython's message, verbatim.</param>
        /// <param name="innerException">The exception that led to this one (a tokenizer's own failure), or null for
        ///     none.</param>
        public SyntaxError(string message, Exception innerException) : base(message, innerException)
        {
            msg = message;
        }

        /// <summary>
        ///     Python's <c>SyntaxError(msg, (filename, lineno, offset, text[, end_lineno, end_offset]))</c>: the message and
        ///     where the error is. <see cref="Exception.Message"/> is CPython's <c>str()</c> of the error:
        ///     <c>msg (file, line N)</c>, <c>msg (file)</c> without a line number, <c>msg (line N)</c> without a file name,
        ///     <c>msg</c> alone without either; the file is its base name.
        /// </summary>
        /// <param name="msg">The parser's message without the location (<c>invalid syntax</c>), or null (printed
        ///     <c>None</c>).</param>
        /// <param name="filename">The source's name (<c>&lt;unknown&gt;</c> for <c>ast.literal_eval</c>, a path), or null for
        ///     none.</param>
        /// <param name="lineno">The 1-based line, or null for none.</param>
        /// <param name="offset">The 1-based column (CPython reports 0 for some tokenizer errors), or null for none.</param>
        /// <param name="text">The source line, or null for none.</param>
        /// <param name="end_lineno">The line the error ends on, or null for none.</param>
        /// <param name="end_offset">The column it ends at, or null for none; required when <paramref name="end_lineno"/>
        ///     is given.</param>
        /// <exception cref="TypeError"><paramref name="end_lineno"/> is given without <paramref name="end_offset"/>
        ///     (CPython: <c>end_offset must be provided when end_lineno is provided</c>).</exception>
        /// <remarks>
        ///     The base name follows the operating system NumSharp runs on, as CPython's does (its <c>my_basename</c> splits
        ///     at the platform's <c>SEP</c>): on Windows <c>C:\a\b\c.py</c> reads <c>c.py</c> and <c>a/b/c.py</c> stays whole;
        ///     elsewhere <c>a/b/c.py</c> reads <c>c.py</c> and <c>C:\a\b\c.py</c> stays whole. A name that ends in the
        ///     separator reads as empty (<c>m (, line 3)</c>), as in CPython.
        /// </remarks>
        public SyntaxError(string msg, string filename, int? lineno, int? offset, string text, int? end_lineno = null,
            int? end_offset = null)
            : base(FormatMessage(msg, filename, lineno, end_lineno, end_offset))
        {
            this.msg = msg;
            this.filename = filename;
            this.lineno = lineno;
            this.offset = offset;
            this.text = text;
            this.end_lineno = end_lineno;
            this.end_offset = end_offset;
        }

        /// <summary>The parser's message without the location (Python's <c>msg</c>), or null after the bare
        ///     constructor.</summary>
        public string msg { get; }

        /// <summary>The name of the source the error is in (Python's <c>filename</c>), or null when not given.</summary>
        /// <remarks>The full name as given; only <see cref="Exception.Message"/> shortens it to the base name.</remarks>
        public string filename { get; }

        /// <summary>The 1-based line of the error (Python's <c>lineno</c>), or null when not given.</summary>
        public int? lineno { get; }

        /// <summary>The column of the error (Python's <c>offset</c>), or null when not given.</summary>
        public int? offset { get; }

        /// <summary>The source line holding the error (Python's <c>text</c>), or null when not given.</summary>
        public string text { get; }

        /// <summary>The line the error ends on (Python's <c>end_lineno</c>), or null when not given.</summary>
        public int? end_lineno { get; }

        /// <summary>The column the error ends at (Python's <c>end_offset</c>), or null when not given.</summary>
        public int? end_offset { get; }

        /// <summary>
        ///     CPython's <c>SyntaxError.__str__</c> (<c>Objects/exceptions.c</c>), after <c>SyntaxError_init</c>'s check of
        ///     the end position.
        /// </summary>
        /// <param name="msg">The message, or null.</param>
        /// <param name="filename">The source's name, or null.</param>
        /// <param name="lineno">The line, or null.</param>
        /// <param name="end_lineno">The end line, or null.</param>
        /// <param name="end_offset">The end column, or null.</param>
        /// <returns>The text.</returns>
        /// <exception cref="TypeError">An end line without an end column.</exception>
        private static string FormatMessage(string msg, string filename, int? lineno, int? end_lineno, int? end_offset)
        {
            if (end_lineno.HasValue && !end_offset.HasValue)
                throw new TypeError("end_offset must be provided when end_lineno is provided");
            // %S of a missing msg prints None.
            string m = msg ?? "None";
            string file = filename is null ? null : BaseName(filename);
            if (file is null && !lineno.HasValue)
                return m;
            if (file != null && lineno.HasValue)
                return string.Create(CultureInfo.InvariantCulture, $"{m} ({file}, line {lineno.Value})");
            if (file != null)
                return $"{m} ({file})";
            return string.Create(CultureInfo.InvariantCulture, $"{m} (line {lineno.Value})");
        }

        /// <summary>
        ///     CPython's <c>my_basename</c>: the text after the LAST platform separator (<c>\</c> on Windows, <c>/</c>
        ///     elsewhere), or the whole name when it has none.
        /// </summary>
        /// <param name="name">The file name.</param>
        /// <returns>Its base name (empty when it ends in the separator).</returns>
        private static string BaseName(string name)
        {
            // Only SEP counts, not ALTSEP: CPython's loop compares against SEP alone, so Windows keeps "a/b/c.py" whole.
            int i = name.LastIndexOf(Path.DirectorySeparatorChar);
            return i < 0 ? name : name.Substring(i + 1);
        }
    }

    /// <summary>
    ///     Python's <c>IndentationError</c>: source whose indentation is wrong (<c>unexpected indent</c>, <c>expected an
    ///     indented block</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="SyntaxError"/>, as in Python, with the same location attributes and the same text rules
    ///     (a bare one reads <c>None</c>). Its own subclass is <see cref="TabError"/>.
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it (NumPy's literal parsing never sees indentation); the class exists so a port of code
    ///     that raises or catches it compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class IndentationError : SyntaxError
    {
        /// <summary>Creates the error with Python's text for a bare <c>raise IndentationError</c>: <c>"None"</c>, as for
        ///     <see cref="SyntaxError()"/>.</summary>
        /// <remarks>Every location attribute, <see cref="SyntaxError.msg"/> included, is null.</remarks>
        public IndentationError() : base() { }

        /// <summary>Creates the error with the parser's message, which is also <see cref="SyntaxError.msg"/>.</summary>
        /// <param name="message">The text, verbatim.</param>
        public IndentationError(string message) : base(message) { }

        /// <summary>Creates the error with the parser's message and the exception that caused it (Python's <c>raise ...
        ///     from innerException</c>).</summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public IndentationError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="SyntaxError(string, string, int?, int?, string, int?, int?)"/>
        public IndentationError(string msg, string filename, int? lineno, int? offset, string text, int? end_lineno = null,
            int? end_offset = null)
            : base(msg, filename, lineno, offset, text, end_lineno, end_offset) { }
    }

    /// <summary>
    ///     Python's <c>TabError</c>: indentation that mixes tabs and spaces inconsistently (<c>inconsistent use of tabs and
    ///     spaces in indentation</c>).
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="IndentationError"/> and so of <see cref="SyntaxError"/>, as in Python, with the same
    ///     location attributes and text rules. NumSharp never raises it; the class exists for ports.
    /// </remarks>
    public class TabError : IndentationError
    {
        /// <summary>Creates the error with Python's text for a bare <c>raise TabError</c>: <c>"None"</c>, as for
        ///     <see cref="SyntaxError()"/>.</summary>
        /// <remarks>Every location attribute, <see cref="SyntaxError.msg"/> included, is null.</remarks>
        public TabError() : base() { }

        /// <summary>Creates the error with the parser's message, which is also <see cref="SyntaxError.msg"/>.</summary>
        /// <param name="message">The text, verbatim.</param>
        public TabError(string message) : base(message) { }

        /// <summary>Creates the error with the parser's message and the exception that caused it (Python's <c>raise ...
        ///     from innerException</c>).</summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public TabError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="SyntaxError(string, string, int?, int?, string, int?, int?)"/>
        public TabError(string msg, string filename, int? lineno, int? offset, string text, int? end_lineno = null,
            int? end_offset = null)
            : base(msg, filename, lineno, offset, text, end_lineno, end_offset) { }
    }
}
