using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>BufferError</c>: a buffer could not be exported in the form a caller asked for. NumPy's
    ///     <c>ndarray(shape, buffer=memoryview(a[::2]))</c> raises it, because the buffer protocol request asks for a
    ///     contiguous buffer and a strided <c>memoryview</c> refuses (<c>memoryview: underlying buffer is not
    ///     contiguous</c>).
    /// </summary>
    /// <remarks>
    ///     A non-contiguous <see cref="NDArray"/> passed directly is a <see cref="ValueError"/> instead
    ///     (<c>ndarray is not contiguous</c>): NumPy's own array exporter raises that one. The two texts and types are
    ///     NumPy's, kept apart as NumPy keeps them.
    /// </remarks>
    public class BufferError : NumSharpException
    {
        /// <summary>Creates the exception with the default message <c>BufferError</c>.</summary>
        public BufferError() : base("BufferError") { }

        /// <summary>Creates the exception with NumPy's message.</summary>
        /// <param name="message">The message, verbatim from NumPy where NumPy has one.</param>
        public BufferError(string message) : base(message) { }

        /// <summary>Creates the exception with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The cause.</param>
        public BufferError(string message, Exception innerException) : base(message, innerException) { }
    }
}
