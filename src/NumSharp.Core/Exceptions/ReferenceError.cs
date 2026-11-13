using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>ReferenceError</c>: a weak-reference proxy was used after its referent was collected
    ///     (<c>weakly-referenced object no longer exists</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/>: .NET has no exception with the same meaning (a
    ///     <see cref="WeakReference{T}"/> reports a collected target by returning false, not by throwing).
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it; the class exists so a port of code that raises or catches it compiles and behaves as
    ///     in Python.
    ///     </para>
    /// </remarks>
    public class ReferenceError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ReferenceError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ReferenceError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>ReferenceError</c>.</remarks>
        public ReferenceError() : base(string.Empty) { }

        /// <summary>Creates the error with the interpreter's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim (<c>weakly-referenced object no longer exists</c>).</param>
        public ReferenceError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise ReferenceError(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ReferenceError(string message, Exception innerException) : base(message, innerException) { }
    }
}
