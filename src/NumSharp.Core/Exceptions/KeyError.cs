using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's KeyError.
    /// Raised when a mapping (dict) key is not found.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's KeyError for API compatibility. In NumSharp this is raised where NumPy
    ///     itself raises <c>KeyError</c> — notably <c>np.einsum_path(..., optimize="&lt;unknown&gt;")</c>,
    ///     which reaches <c>einsumfunc.py</c>'s <c>raise KeyError("Path name %s not found", path_type)</c>.
    ///     <para>
    ///     The message is used verbatim: Python's <c>KeyError.__str__</c> quotes a lone argument
    ///     (<c>str(KeyError('a'))</c> is <c>"'a'"</c>), so a port that wants Python's text passes it already quoted.
    ///     </para>
    /// </remarks>
    public class KeyError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise KeyError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(KeyError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>KeyError</c>; prefer a message that names the missing key.</remarks>
        public KeyError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public KeyError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise KeyError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public KeyError(string message, Exception innerException) : base(message, innerException) { }
    }
}
