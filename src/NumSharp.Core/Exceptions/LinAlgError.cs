using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>numpy.linalg.LinAlgError</c>.
    ///     Raised by <see cref="np.linalg"/> when a matrix is unsuitable for the requested
    ///     factorisation — the wrong rank, not square, singular, or not positive definite.
    /// </summary>
    /// <remarks>
    ///     Mirrors <c>numpy.linalg.LinAlgError</c>, which derives from Python's <c>ValueError</c>
    ///     — so this derives from <see cref="ValueError"/> and a <c>catch (ValueError)</c> sees it,
    ///     exactly as <c>except ValueError</c> does upstream.
    ///     <para>
    ///     The messages NumPy 2.4.2 raises verbatim (probed):
    ///     <list type="bullet">
    ///     <item><c>{n}-dimensional array given. Array must be at least two-dimensional</c></item>
    ///     <item><c>{n}-dimensional array given. Array must be two-dimensional</c></item>
    ///     <item><c>Last 2 dimensions of the array must be square</c></item>
    ///     <item><c>Singular matrix</c></item>
    ///     <item><c>Matrix is not positive definite</c></item>
    ///     <item><c>Array must not contain infs or NaNs</c></item>
    ///     </list>
    ///     </para>
    /// </remarks>
    public class LinAlgError : ValueError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise LinAlgError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(np.linalg.LinAlgError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>LinAlgError</c>; prefer one of NumPy's texts listed on the class.</remarks>
        public LinAlgError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message.</summary>
        /// <param name="message">The text NumPy raises, verbatim (<c>Singular matrix</c>).</param>
        public LinAlgError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with NumPy's message and the exception that caused it: Python's <c>raise
        ///     LinAlgError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text NumPy raises, verbatim.</param>
        /// <param name="innerException">The exception that led to this one (a backend's own failure), or null for
        ///     none.</param>
        public LinAlgError(string message, Exception innerException) : base(message, innerException) { }
    }
}
