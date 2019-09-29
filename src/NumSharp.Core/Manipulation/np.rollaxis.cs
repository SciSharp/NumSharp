namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Roll the specified axis backwards, until it lies in a given position. <br></br>
        ///     This function continues to be supported for backward compatibility, but you should prefer moveaxis. The moveaxis function was added in NumPy 1.11.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">The axis to roll backwards. The positions of the other axes do not change relative to one another.</param>
        /// <param name="start">The axis is rolled until it lies before this position. The default, 0, results in a “complete” roll.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.rollaxis.html</remarks>
        public static NDArray rollaxis(in NDArray a, int axis, int start = 0)
            => a.TensorEngine.RollAxis(a, axis, start);
    }
}