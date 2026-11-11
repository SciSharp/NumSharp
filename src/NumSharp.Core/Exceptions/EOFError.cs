using System;
using System.IO;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>EOFError</c>: a read that hit the end of its input with nothing to read. NumSharp raises it where
    ///     NumPy does: <c>np.load</c> on an empty file, or past the last array of a file that holds several
    ///     (<c>No data left in file</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from .NET's <see cref="EndOfStreamException"/>, the exception with the same meaning (and through it
    ///     <see cref="IOException"/>), so a <c>catch (EndOfStreamException)</c> or <c>catch (IOException)</c> written for
    ///     .NET code catches it. Python's <c>EOFError</c> is NOT an <c>OSError</c>, and NumSharp's is not an
    ///     <see cref="OSError"/> either: <c>catch (OSError)</c> misses it, as <c>except OSError</c> does.
    ///     </para>
    ///     <para>
    ///     A loop that reads arrays from one stream until the end (<c>while True: np.load(f)</c>) ends on this error, in
    ///     both languages.
    ///     </para>
    /// </remarks>
    public class EOFError : EndOfStreamException, INumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise EOFError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(EOFError())</c> is <c>''</c> (not .NET's
        ///     <see cref="EndOfStreamException"/> default, "Unable to read beyond the end of the stream.").
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>EOFError</c>.</remarks>
        public EOFError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error (<c>No data left in
        ///     file</c>).</param>
        public EOFError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise EOFError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one (the .NET reader's
        ///     <see cref="EndOfStreamException"/>, for example), or null for none.</param>
        public EOFError(string message, Exception innerException) : base(message, innerException) { }
    }
}
