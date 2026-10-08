using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible AxisError exception.
    ///     Raised when an axis argument is out of bounds for the array's dimensions.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Mirrors <c>numpy.exceptions.AxisError</c>. Its text is NumPy's <c>AxisError.__str__</c>:
    ///     <c>"axis {axis} is out of bounds for array of dimension {ndim}"</c>, preceded by <c>"{msg_prefix}: "</c> when
    ///     the raising function names the argument (<c>np.moveaxis</c>'s <c>source</c> / <c>destination</c>,
    ///     <c>np.swapaxes</c>' <c>axis1</c> / <c>axis2</c>, <c>np.cross</c>' <c>axisa</c>, <c>np.average</c>'s
    ///     <c>axis</c>). NumPy raises it from <c>normalize_axis_index</c> (C: <c>check_and_adjust_axis_msg</c>), so every
    ///     NumSharp function that normalizes an axis raises it through
    ///     <see cref="Backends.DefaultEngine.check_and_adjust_axis(int, int, string)"/>.
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="ArgumentOutOfRangeException"/>, so a handler written for
    ///     <see cref="ArgumentOutOfRangeException"/> or <see cref="ArgumentException"/> still catches it. The .NET base
    ///     appends <c>" (Parameter '…')"</c> and an <c>"Actual value was …"</c> line to <see cref="Exception.Message"/>;
    ///     NumPy's text is the part before them.
    ///     </para>
    /// </remarks>
    public class AxisError : ArgumentOutOfRangeException, INumSharpException
    {
        /// <summary>The axis as the caller gave it (negative values are not wrapped); 0 for a message-only error.</summary>
        public int Axis { get; }

        /// <summary>The rank the axis was checked against; 0 for a message-only error.</summary>
        public int NDim { get; }

        /// <summary>
        ///     NumPy's <c>AxisError(axis, ndim)</c>: <c>"axis {axis} is out of bounds for array of dimension {ndim}"</c>.
        /// </summary>
        /// <param name="axis">The axis as the caller gave it.</param>
        /// <param name="ndim">The rank it was checked against.</param>
        public AxisError(int axis, int ndim)
            : this(axis, ndim, null)
        {
        }

        /// <summary>
        ///     NumPy's <c>AxisError(axis, ndim, msg_prefix)</c>: the out-of-bounds text preceded by
        ///     <c>"{msgPrefix}: "</c>, as NumPy raises it for a named argument.
        /// </summary>
        /// <param name="axis">The axis as the caller gave it.</param>
        /// <param name="ndim">The rank it was checked against.</param>
        /// <param name="msgPrefix">The argument name NumPy puts in front of the text (<c>"source"</c>, <c>"axis1"</c>),
        ///     or <c>null</c> for none. It is also the exception's <see cref="ArgumentException.ParamName"/>
        ///     (<c>"axis"</c> without a prefix).</param>
        /// <remarks>
        ///     An empty prefix is not the same as none: NumPy tests <c>msg_prefix is not None</c>, so <c>""</c> gives
        ///     <c>": axis 5 is …"</c>, which this constructor reproduces.
        /// </remarks>
        public AxisError(int axis, int ndim, string msgPrefix)
            : base(string.IsNullOrEmpty(msgPrefix) ? "axis" : msgPrefix, axis, Format(axis, ndim, msgPrefix))
        {
            Axis = axis;
            NDim = ndim;
        }

        /// <summary>
        ///     NumPy's message-only <c>AxisError(msg)</c> (for example np.rollaxis' <c>"'start' arg requires … but … was
        ///     passed in"</c>): the text is <paramref name="message"/> verbatim.
        /// </summary>
        /// <param name="message">The whole text.</param>
        public AxisError(string message) : base("axis", message) { }

        /// <summary>An AxisError with no text (the parameterless form .NET serialization and reflection expect).</summary>
        public AxisError() : base() { }

        /// <summary>NumPy's <c>AxisError.__str__</c> for an axis and a rank: the prefix (when given), then the bounds text.</summary>
        /// <param name="axis">The axis as the caller gave it.</param>
        /// <param name="ndim">The rank.</param>
        /// <param name="msgPrefix">The argument name, or <c>null</c>.</param>
        /// <returns>The message text.</returns>
        private static string Format(int axis, int ndim, string msgPrefix)
        {
            string text = $"axis {axis} is out of bounds for array of dimension {ndim}";
            return msgPrefix is null ? text : $"{msgPrefix}: {text}";
        }
    }
}
