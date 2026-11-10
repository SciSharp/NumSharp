using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>SystemError</c> — Python's "the interpreter found an internal inconsistency" error. NumPy
    ///     raises it when an operation meets data that no Python object can represent, and NumSharp raises it in the same
    ///     places with CPython's verbatim text.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The one place today: reading an element of a fixed-width Unicode (<c>U</c>) array whose UCS-4 unit is above
    ///     <c>U+10FFFF</c> (only reachable by viewing raw <c>uint32</c> data as <c>U</c>). Python's <c>str</c> cannot hold
    ///     such a code point and NumPy's <c>UNICODE_getitem</c> fails inside <c>PyUnicode_New</c> with
    ///     <c>invalid maximum character passed to PyUnicode_New</c>; a .NET <see cref="string"/> cannot hold it either.
    ///     Copies and string↔string casts never decode, so they move such units unchanged, exactly as NumPy does.
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/> (not from <see cref="System.SystemException"/>, which .NET
    ///     reserves for runtime faults), like the other Python-named errors (<see cref="RuntimeError"/>,
    ///     <see cref="TypeError"/>).
    ///     </para>
    /// </remarks>
    public class SystemError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise SystemError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(SystemError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>SystemError</c>.</remarks>
        public SystemError() : base(string.Empty) { }

        /// <summary>Creates the error with CPython's message text.</summary>
        /// <param name="message">The verbatim message (<c>invalid maximum character passed to PyUnicode_New</c>).</param>
        public SystemError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise SystemError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public SystemError(string message, Exception innerException) : base(message, innerException) { }
    }
}
