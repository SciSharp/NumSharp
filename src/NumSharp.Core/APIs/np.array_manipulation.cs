using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Create an array.
        /// </summary>
        /// <param name="shape">Shape of the array.</param>
        /// <param name="dtype">Data type. Default is float32.</param>
        /// <param name="buffer">Optional buffer to use for data. If null, allocates new memory filled with zeros.</param>
        /// <param name="order">Memory order. Note: Only C-order is supported, F-order parameter is accepted but ignored.</param>
        /// <returns>New NDArray with the specified shape and dtype.</returns>
        /// <remarks>
        /// This function creates an NDArray directly without going through TensorEngine.
        /// Memory allocation is not backend-specific - all backends use the same unmanaged memory.
        /// </remarks>
        public static NDArray ndarray(Shape shape, Type dtype = null, Array buffer = null, char order = 'F')
        {
            dtype ??= typeof(float); // Default to float32

            if (buffer == null)
            {
                // Allocate new array with zeros (order accepted but ignored - C-order only)
                return new NDArray(dtype, shape, order);
            }
            else
            {
                // Use provided buffer (order accepted but ignored - C-order only)
                return new NDArray(buffer, shape, order);
            }
        }
    }
}
