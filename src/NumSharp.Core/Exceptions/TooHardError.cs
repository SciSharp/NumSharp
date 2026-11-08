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
        /// <summary>Creates a <see cref="TooHardError"/> with the default "TooHardError" message.</summary>
        public TooHardError() : base("TooHardError") { }

        /// <summary>Creates a <see cref="TooHardError"/> with a specific message.</summary>
        /// <param name="message">The message that describes the error (NumPy uses "Exceeded max_work").</param>
        public TooHardError(string message) : base(message) { }
    }
}
