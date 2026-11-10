using System;

namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to Python/NumPy's <c>NameError</c>.
    ///     Raised when a name referenced by value is not defined.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's <c>NameError</c> for API compatibility. In NumSharp this is raised where
    ///     NumPy itself raises <c>NameError</c> — notably <c>np.bmat("A,B; C,D", …)</c>, whose string
    ///     form resolves each whitespace/comma-separated token as a variable name and reaches
    ///     <c>defmatrix.py</c>'s <c>raise NameError(f"name {col!r} is not defined")</c> when a token is
    ///     absent from the supplied name dictionaries (this also fires for numeric literals, since a
    ///     token such as <c>"1"</c> is treated as a name).
    /// </remarks>
    public class NameError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise NameError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(NameError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>NameError</c>; prefer a message that names the undefined name.</remarks>
        public NameError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public NameError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise NameError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public NameError(string message, Exception innerException) : base(message, innerException) { }
    }
}
