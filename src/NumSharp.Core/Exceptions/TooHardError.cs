using System;

namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to NumPy's <c>numpy.exceptions.TooHardError</c> — raised by
    ///     <see cref="np.shares_memory(NDArray,NDArray,long)"/> when the bounded memory-overlap search
    ///     exhausts its <c>max_work</c> candidate budget before it can decide the question exactly.
    /// </summary>
    /// <remarks>
    ///     Derives from <see cref="RuntimeError"/> so that callers written against NumPy's hierarchy —
    ///     where <c>TooHardError</c> subclasses <c>RuntimeError</c> — keep catching it with a
    ///     <c>catch (RuntimeError)</c>. It signals "undecided within the work cap", NOT "the arrays
    ///     overlap": assigning a finite <c>max_work</c> is what turns an otherwise-answerable query
    ///     into this failure. <see cref="np.may_share_memory(NDArray,NDArray,long)"/> never throws it —
    ///     it treats the undecided outcome as a conservative <c>true</c> instead.
    /// </remarks>
    public class TooHardError : RuntimeError
    {
        /// <summary>
        ///     Creates a <see cref="TooHardError"/> with Python's text for a bare <c>raise TooHardError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(np.exceptions.TooHardError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks>NumSharp's own raise site passes NumPy's text, "Exceeded max_work".</remarks>
        public TooHardError() : base(string.Empty) { }

        /// <summary>Creates a <see cref="TooHardError"/> with a specific message.</summary>
        /// <param name="message">The message that describes the error (NumPy uses "Exceeded max_work").</param>
        public TooHardError(string message) : base(message) { }

        /// <summary>
        ///     Creates a <see cref="TooHardError"/> with a message and the exception that caused it: Python's <c>raise
        ///     TooHardError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public TooHardError(string message, Exception innerException) : base(message, innerException) { }
    }
}
