using System;

namespace NumSharp
{
    public class AxisOutOfRangeException : ArgumentOutOfRangeException, INumSharpException
    {
        public AxisOutOfRangeException(int ndim, int axis) : base(nameof(axis), $"axis ({axis}) is out of bounds for array of dimension ({ndim})") { }

        public AxisOutOfRangeException() : base("axis", "axis is out of bounds for array of dimension") { }

        public AxisOutOfRangeException(string message) : base("axis", message) { }
    }
}
