using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible ValueError exception.
    ///     Raised when an operation receives an argument with the right type but inappropriate value.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's ValueError / numpy's ValueError for API compatibility.
    ///     Common cases:
    ///     - negative dimensions: "negative dimensions are not allowed"
    ///     - invalid seed: "Seed must be between 0 and 2**32 - 1"
    ///     - randint bounds: "high is out of bounds for int32"
    /// </remarks>
    public class ValueError : ArgumentException, INumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ValueError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ValueError())</c> is <c>''</c> (not .NET's
        ///     <see cref="ArgumentException"/> default, "Value does not fall within the expected range.").
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>ValueError</c>; prefer a message that names the refused value. The subclasses' parameterless forms
        ///     (<see cref="LinAlgError"/>, <see cref="UnicodeError"/>) read the same.</remarks>
        public ValueError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public ValueError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the .NET parameter it is about. Footgun: .NET appends
        ///     <c>" (Parameter '…')"</c> to <see cref="Exception.Message"/>, so the text no longer equals NumPy's; use it only
        ///     where no caller compares the message with Python's.
        /// </summary>
        /// <param name="message">The text.</param>
        /// <param name="paramName">The name of the refused parameter, reported by <see cref="ArgumentException.ParamName"/>.</param>
        public ValueError(string message, string paramName) : base(message, paramName) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise ValueError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ValueError(string message, Exception innerException) : base(message, innerException) { }
    }
}
