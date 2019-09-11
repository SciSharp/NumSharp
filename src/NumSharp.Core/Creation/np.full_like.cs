using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full_like.html</remarks>
        public static NDArray full_like(NDArray a, object fill_value, Type dtype = null)
        {
            var typeCode = (dtype ?? fill_value?.GetType() ?? a.dtype).GetTypeCode();
            var shape = new Shape((int[])a.shape.Clone());
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, Converts.ChangeType(fill_value, (TypeCode) typeCode)), shape));
        }
    }
}
