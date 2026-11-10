using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's TypeError.
    /// Raised when an operation or function receives an argument of inappropriate type.
    /// </summary>
    public class TypeError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise TypeError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(TypeError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>TypeError</c>; prefer a message that names the refused type.</remarks>
        public TypeError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public TypeError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise TypeError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public TypeError(string message, Exception innerException) : base(message, innerException) { }
    }
}
