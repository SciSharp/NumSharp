using System;

namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to Python's <c>AssertionError</c>, raised where NumPy's own code ends in an
    ///     <c>assert</c> or in <c>typing.assert_never</c> for an argument it does not handle.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy raises it from library code, not as part of a documented contract, so a caller that ports NumPy code
    ///     sees it only where NumPy's validation is an exhaustive <c>match</c>: <c>np.pad</c>'s dict <c>pad_width</c>
    ///     matches each width against <c>int</c> and a 2-tuple of ints and sends anything else to
    ///     <c>typing.assert_never</c> ("Expected code to be unreachable, but got: 1.5").
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/> like the other Python-named house types, NOT from
    ///     <see cref="ArgumentException"/>: Python's AssertionError is no ValueError / TypeError, so a handler for the
    ///     argument errors does not catch it.
    ///     </para>
    ///     <para>
    ///     A port of Python's bare <c>assert condition</c> throws <c>new AssertionError()</c>: its message is empty, exactly
    ///     as <c>str()</c> of the <c>AssertionError</c> a failed bare <c>assert</c> raises, so error texts compared with
    ///     Python's match.
    ///     </para>
    /// </remarks>
    public class AssertionError : NumSharpException
    {
        /// <summary>
        ///     Creates the exception with Python's text for a failed bare <c>assert</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(AssertionError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>AssertionError</c>. Pass a message wherever the failure has something to say: an empty one tells a
        ///     reader of the log nothing.</remarks>
        public AssertionError() : base(string.Empty) { }

        /// <summary>Creates the exception with NumPy's message.</summary>
        /// <param name="message">The text NumPy raises, verbatim.</param>
        public AssertionError(string message) : base(message) { }

        /// <summary>Creates the exception with NumPy's message and the exception that caused it.</summary>
        /// <param name="message">The text NumPy raises, verbatim.</param>
        /// <param name="innerException">The exception that led to this one.</param>
        public AssertionError(string message, Exception innerException) : base(message, innerException) { }
    }
}
