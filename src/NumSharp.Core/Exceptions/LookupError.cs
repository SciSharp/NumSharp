using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>LookupError</c>: a name that a registry does not know. NumSharp raises it where Python's codec
    ///     registry does — an encoding name <c>np.savetxt</c> / <c>np.loadtxt</c> cannot resolve (<c>unknown encoding:
    ///     bytes</c>: <c>np.savetxt</c> has no special <c>'bytes'</c> encoding, unlike <c>np.loadtxt</c>).
    /// </summary>
    /// <remarks>
    ///     Python's <c>LookupError</c> is the base of <c>KeyError</c> and <c>IndexError</c>; NumSharp's <see cref="KeyError"/>
    ///     and <see cref="IndexError"/> keep their own bases (changing them would break existing catch sites), so this class
    ///     carries only the codec lookup. It derives from <see cref="ArgumentException"/>: the unknown name is an argument,
    ///     and a .NET caller that catches <see cref="ArgumentException"/> around an encoding choice keeps working.
    /// </remarks>
    public class LookupError : ArgumentException, INumSharpException
    {
        /// <summary>Creates the error with Python's message text.</summary>
        /// <param name="message">The verbatim message (<c>unknown encoding: bytes</c>).</param>
        public LookupError(string message) : base(message) { }

        /// <summary>Creates the error with Python's message text and the exception that caused it.</summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The .NET exception behind the failed lookup.</param>
        public LookupError(string message, Exception innerException) : base(message, innerException) { }
    }
}
