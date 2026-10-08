using System;

namespace NumSharp
{
    /// <summary>
    ///     The legacy axis exception, with a parenthesized text ("axis (5) is out of bounds for array of dimension (2)")
    ///     NumPy never prints. NumSharp no longer raises it: every axis check raises NumPy's <see cref="AxisError"/>, with
    ///     NumPy's text.
    /// </summary>
    /// <remarks>
    ///     It derives from <see cref="AxisError"/> (and through it from <see cref="ArgumentOutOfRangeException"/>), so code
    ///     that still raises it is caught by a handler for NumPy's type. A handler written for THIS type catches nothing
    ///     NumSharp raises any more: catch <see cref="AxisError"/> instead.
    /// </remarks>
    [Obsolete("NumSharp raises AxisError, NumPy's exception, for an axis out of range; catch AxisError (or ArgumentOutOfRangeException).")]
    public class AxisOutOfRangeException : AxisError, INumSharpException
    {
        /// <summary>The legacy text for an axis and a rank.</summary>
        /// <param name="ndim">The rank.</param>
        /// <param name="axis">The axis.</param>
        public AxisOutOfRangeException(int ndim, int axis) : base($"axis ({axis}) is out of bounds for array of dimension ({ndim})") { }

        /// <summary>The legacy text for a 64-bit axis and a rank.</summary>
        /// <param name="ndim">The rank.</param>
        /// <param name="axis">The axis.</param>
        public AxisOutOfRangeException(int ndim, long axis) : base($"axis ({axis}) is out of bounds for array of dimension ({ndim})") { }

        /// <summary>The legacy text without an axis or a rank.</summary>
        public AxisOutOfRangeException() : base("axis is out of bounds for array of dimension") { }

        /// <summary>A caller-given text.</summary>
        /// <param name="message">The text.</param>
        public AxisOutOfRangeException(string message) : base(message) { }
    }
}
