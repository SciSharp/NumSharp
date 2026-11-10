using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's RuntimeError.
    /// Raised when an error is detected that does not fall into any of the other categories.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's RuntimeError for API compatibility. Common cases:
    ///     - np.cov: "cannot handle multidimensional fweights"
    ///     - np.cov: "incompatible numbers of samples and fweights"
    ///     <para>
    ///     Python's subclasses derive from it here too: <see cref="NotImplementedError"/> and <see cref="TooHardError"/>, so
    ///     a <c>catch (RuntimeError)</c> catches them as <c>except RuntimeError</c> does.
    ///     </para>
    /// </remarks>
    public class RuntimeError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise RuntimeError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(RuntimeError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>RuntimeError</c>.</remarks>
        public RuntimeError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public RuntimeError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise RuntimeError(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public RuntimeError(string message, Exception innerException) : base(message, innerException) { }
    }
}
