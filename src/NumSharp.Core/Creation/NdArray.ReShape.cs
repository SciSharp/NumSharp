using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape newShape)
        {
            return reshape(ref newShape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data, filling values in the specified order.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        /// <param name="order">
        ///     Read/write order for the reshape.
        ///     'C' (default) - row-major, 'F' - column-major,
        ///     'A' - preserve source layout when possible, 'K' - memory order.
        ///     When 'F', values are both read in F-order from the source and written in F-order
        ///     to the destination, producing an F-contiguous result with NumPy-aligned values.
        /// </param>
        /// <returns>Reshaped array. For order='F' this is always a newly-allocated F-contiguous copy.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape newShape, char order)
        {
            char physical = OrderResolver.Resolve(order, this.Shape);
            if (physical != 'F')
                return reshape(ref newShape);

            // F-order reshape: read source column-major, write destination column-major.
            // Equivalent to placing flatten('F') memory into an F-contiguous shape.
            var fFlat = this.flatten('F');
            var dims = (long[])newShape.Dimensions.Clone();
            var fShape = new Shape(dims, 'F');
            return new NDArray(new UnmanagedStorage(fFlat.Storage.InternalArray, fShape));
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(ref Shape newShape)
        {
            // NumPy: reshape returns a view when possible (contiguous), otherwise a copy
            // For non-contiguous arrays (transposed/sliced), we must copy to get correct values
            if (!Shape.IsContiguous)
            {
                // Clone data to contiguous, then reshape the clean copy
                var copy = new NDArray(CloneData(), Shape.Clean());
                return copy.reshape(ref newShape);
            }

            var ret = Storage.Alias();
            ret.Reshape(ref newShape, false);
            return new NDArray(ret) {TensorEngine = TensorEngine};
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray reshape(int[] shape)
        {
            return reshape(Shape.ComputeLongShape(shape));
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray reshape(params long[] shape)
        {
            // NumPy: reshape returns a view when possible (contiguous), otherwise a copy
            // For non-contiguous arrays (transposed/sliced), we must copy to get correct values
            if (!Shape.IsContiguous)
            {
                // Clone data to contiguous, then reshape the clean copy
                var copy = new NDArray(CloneData(), Shape.Clean());
                return copy.reshape(shape);
            }

            var ret = Storage.Alias();
            ret.Reshape(shape, false);
            return new NDArray(ret) {TensorEngine = TensorEngine};
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape_unsafe(Shape newshape)
        {
            return reshape_unsafe(ref newshape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape_unsafe(ref Shape newshape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newshape, true);
            return new NDArray(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray reshape_unsafe(int[] shape)
        {
            return reshape_unsafe(Shape.ComputeLongShape(shape));
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray reshape_unsafe(params long[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, true);
            return new NDArray(ret) { TensorEngine = TensorEngine };
        }
    }
}
