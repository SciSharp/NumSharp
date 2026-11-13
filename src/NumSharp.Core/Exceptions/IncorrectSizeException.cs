using System;

namespace NumSharp
{
    /// <summary>
    ///     Raised when an array that is not of size 1 is read as one scalar: <c>item()</c> with NumPy's text
    ///     (<c>can only convert an array of size 1 to a Python scalar</c>), and <c>np.asscalar</c> and the scalar
    ///     conversions in <c>Converts</c> with NumSharp's own (<c>Unable to convert NDArray to scalar because size is not
    ///     1.</c>).
    /// </summary>
    /// <remarks>
    ///     Footgun: NumPy raises a <c>ValueError</c> for these failures, but this type derives from
    ///     <see cref="NumSharpException"/> only, so a <c>catch (ValueError)</c> ported from Python's
    ///     <c>except ValueError</c> misses it; catch this type, <see cref="NumSharpException"/>, or
    ///     <see cref="INumSharpException"/>.
    /// </remarks>
    public class IncorrectSizeException : NumSharpException
    {
        /// <summary>Creates the exception with the message as given (NumPy's text where a NumPy function is mirrored).</summary>
        /// <param name="message">The message, kept verbatim.</param>
        public IncorrectSizeException(string message) : base(message) { }

        /// <summary>
        ///     Creates the exception with the message and the exception that caused it, .NET's standard constructor every
        ///     NumSharp exception has (Python's <c>raise ... from innerException</c>).
        /// </summary>
        /// <param name="message">The message, kept verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public IncorrectSizeException(string message, Exception innerException) : base(message, innerException) { }
    }
}
