using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.ma.MAError</c>: the base exception for masked-array errors. Distinct from the
    ///     ordinary NumSharp/NumPy exceptions so a caller can catch masked-array-specific failures without also
    ///     catching unrelated <see cref="Exception"/>s. NumSharp's <c>np.ma</c> surface does not yet RAISE this
    ///     itself (the current failures leak the underlying <c>np.*</c> exception), so it exists for API/type
    ///     parity and for callers that want to throw it from their own masked-array code.
    /// </summary>
    public class MAError : Exception
    {
        /// <summary>Creates the error with no message.</summary>
        public MAError() { }

        /// <summary>Creates the error with a human-readable <paramref name="message"/>.</summary>
        /// <param name="message">The error text.</param>
        public MAError(string message) : base(message) { }

        /// <summary>Creates the error wrapping an <paramref name="innerException"/>.</summary>
        /// <param name="message">The error text.</param>
        /// <param name="innerException">The underlying cause.</param>
        public MAError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     NumPy's <c>numpy.ma.MaskError</c>: a mask-specific <see cref="MAError"/> (raised by NumPy when an
    ///     operation cannot honor the mask). Extends <see cref="MAError"/> exactly as NumPy's <c>MaskError</c>
    ///     extends <c>MAError</c>, so <c>catch (MAError)</c> also catches it.
    /// </summary>
    public class MaskError : MAError
    {
        /// <summary>Creates the error with no message.</summary>
        public MaskError() { }

        /// <summary>Creates the error with a human-readable <paramref name="message"/>.</summary>
        /// <param name="message">The error text.</param>
        public MaskError(string message) : base(message) { }

        /// <summary>Creates the error wrapping an <paramref name="innerException"/>.</summary>
        /// <param name="message">The error text.</param>
        /// <param name="innerException">The underlying cause.</param>
        public MaskError(string message, Exception innerException) : base(message, innerException) { }
    }
}
