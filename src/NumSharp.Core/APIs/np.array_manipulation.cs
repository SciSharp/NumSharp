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
            => (int)nd.roll(shift, axis);
    }
}
