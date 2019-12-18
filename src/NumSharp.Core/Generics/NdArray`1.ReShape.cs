using System.Diagnostics.CodeAnalysis;

namespace NumSharp.Generic
{
    public partial class NDArray<TDType>
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape(Shape newShape)
        {
            return reshape(ref newShape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape(ref Shape newShape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newShape, false);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
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
        public new NDArray<TDType> reshape(params int[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, false);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape_unsafe(Shape newshape)
        {
            return reshape_unsafe(ref newshape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape_unsafe(ref Shape newShape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newShape, true);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
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
        public new NDArray<TDType> reshape_unsafe(params int[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, true);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }
    }
}
