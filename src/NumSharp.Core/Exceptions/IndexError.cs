using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's IndexError.
    /// Raised when a sequence subscript is out of range, or when an index type is invalid
    /// (e.g. float/complex index on an ndarray).
    /// </summary>
    public class IndexError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise IndexError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(IndexError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>IndexError</c>; prefer a message that names the index and the bound.</remarks>
        public IndexError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public IndexError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise IndexError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public IndexError(string message, Exception innerException) : base(message, innerException) { }
    }
}
