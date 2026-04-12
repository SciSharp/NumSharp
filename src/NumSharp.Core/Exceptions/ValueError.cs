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
        public ValueError() : base() { }

        public ValueError(string message) : base(message) { }

        public ValueError(string message, string paramName) : base(message, paramName) { }

        public ValueError(string message, Exception innerException) : base(message, innerException) { }
    }
}
