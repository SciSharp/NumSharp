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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        public static NDArray full_like(NDArray a, object fill_value, Type dtype = null, string device = null)
            => full_like(a, fill_value, dtype, 'K', device);

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html</remarks>
        public static NDArray full_like(NDArray a, object fill_value, Type dtype, char order, string device = null)
        {
            ValidateDevice(device);
            var typeCode = (dtype ?? fill_value?.GetType() ?? a.dtype).GetTypeCode();
            char physical = OrderResolver.Resolve(order, a.Shape);
            var shape = new Shape((long[])a.shape.Clone(), physical);

            // Allocates from shape.size directly (not via UnmanagedStorage.Allocate), so the
            // dimension guard has to be explicit here — see np.full.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, Converts.ChangeType(fill_value, typeCode)), shape));
        }
    }
}
