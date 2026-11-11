using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>RecursionError</c>: the interpreter's recursion limit was exceeded (<c>maximum recursion depth
    ///     exceeded</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="RuntimeError"/>, as in Python (<c>RecursionError → RuntimeError → Exception</c>), so
    ///     <c>catch (RuntimeError)</c> catches it.
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it: .NET has no recursion limit to exceed, and a stack that does run out ends the process
    ///     (a <see cref="StackOverflowException"/> cannot be caught). Code that guards its own depth
    ///     (<see cref="System.Runtime.CompilerServices.RuntimeHelpers.EnsureSufficientExecutionStack"/> throws
    ///     <see cref="InsufficientExecutionStackException"/>) can translate to this class where Python's code raises it.
    ///     </para>
    /// </remarks>
    public class RecursionError : RuntimeError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise RecursionError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(RecursionError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>RecursionError</c>.</remarks>
        public RecursionError() : base(string.Empty) { }

        /// <summary>Creates the error with the interpreter's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim (<c>maximum recursion depth exceeded</c>).</param>
        public RecursionError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise RecursionError(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> (an
        ///     <see cref="InsufficientExecutionStackException"/>, for example).
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public RecursionError(string message, Exception innerException) : base(message, innerException) { }
    }
}
