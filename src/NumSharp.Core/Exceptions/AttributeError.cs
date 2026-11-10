using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's AttributeError.
    /// Raised when an attribute reference or assignment fails.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's AttributeError for API compatibility. In NumSharp this is raised where NumPy
    ///     itself raises <c>AttributeError</c> — notably <c>np.savetxt(fname, X, fmt=[...])</c> when the
    ///     list/tuple of formats is not the same length as the number of columns, which reaches
    ///     <c>_npyio_impl.py</c>'s <c>raise AttributeError(f'fmt has wrong shape.  {str(fmt)}')</c>.
    /// </remarks>
    public class AttributeError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise AttributeError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(AttributeError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>AttributeError</c>; prefer a message that names the attribute.</remarks>
        public AttributeError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public AttributeError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     AttributeError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as
        ///     Python keeps it in <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public AttributeError(string message, Exception innerException) : base(message, innerException) { }
    }
}
