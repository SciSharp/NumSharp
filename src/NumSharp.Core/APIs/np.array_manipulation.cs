using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray ndarray(Shape shape, Type dtype = null, Array buffer = null, char order = 'F')
            => BackendFactory.GetEngine().CreateNDArray(shape, dtype: dtype, buffer: buffer, order: order);

        /// <summary>
        /// Roll array elements along a given axis.
        /// 
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// </summary>
        public static int roll(NDArray nd, int shift, int axis = -1)
            => (int) nd.roll(shift, axis);

        public static NDArray transpose(NDArray x, int[] axes = null)
            => x.TensorEngine.Transpose(x, axes: axes);

        /// <summary>
        ///     Find the unique elements of an array.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html</remarks>
        public static NDArray unique(NDArray a)
            => a.unique();

    }
}
