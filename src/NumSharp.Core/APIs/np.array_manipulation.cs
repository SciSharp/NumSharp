using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray ndarray(Shape shape, Type dtype = null, Array buffer = null, char order = 'F')
            => BackendFactory.GetEngine().CreateNDArray(shape, dtype: dtype, buffer: buffer, order: order);
    }
}
