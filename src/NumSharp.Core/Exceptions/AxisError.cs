using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible AxisError exception.
    ///     Raised when an axis argument is out of bounds for the array's dimensions.
    /// </summary>
    /// <remarks>
    ///     Mirrors numpy.exceptions.AxisError for API compatibility.
    ///     Error format: "axis {axis} is out of bounds for array of dimension {ndim}"
    /// </remarks>
    public class AxisError : ArgumentOutOfRangeException, INumSharpException
    {
        public int Axis { get; }
        public int NDim { get; }

        public AxisError(int axis, int ndim)
            : base("axis", axis, $"axis {axis} is out of bounds for array of dimension {ndim}")
        {
            Axis = axis;
            NDim = ndim;
        }

        public AxisError(string message) : base("axis", message) { }

        public AxisError() : base() { }
    }
}
