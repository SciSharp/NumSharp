using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="retShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape retShape)
        {
            return reshape(ref retShape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="retShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(ref Shape retShape)
        {
            if (retShape.size == 0 && size != 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(retShape));

            InferMissingDimension(ref retShape);

            if (size != retShape.size)
                throw new IncorrectShapeException($"Given shape size ({retShape.size}) does not match the size of the given storage size ({size})");


            var storage = Shape.IsSliced ? new UnmanagedStorage(Storage.CloneData(), retShape) : Storage.Alias(retShape);
            return new NDArray(storage) {TensorEngine = TensorEngine};
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
        public NDArray reshape(params int[] shape)
        {
            var retShape = new Shape(shape);

            InferMissingDimension(ref retShape);

            if (size != retShape.size)
                throw new IncorrectShapeException($"Given shape size ({retShape.size}) does not match the size of the given storage size ({size})");

            var storage = Shape.IsSliced ? new UnmanagedStorage(Storage.CloneData(), retShape) : Storage.Alias(retShape);
            return new NDArray(storage) {TensorEngine = TensorEngine};
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape_unsafe(Shape newshape)
        {
            return reshape_unsafe(ref newshape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        internal NDArray reshape_unsafe(ref Shape newshape)
        {
            InferMissingDimension(ref newshape);
            return new NDArray(Shape.IsSliced ? UnmanagedStorage.CreateBroadcastedUnsafe(Storage.CloneData(), newshape) : Storage.Alias(newshape)) {TensorEngine = TensorEngine};
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
        internal NDArray reshape_unsafe(params int[] shape)
        {
            var retShape = new Shape(shape);
            InferMissingDimension(ref retShape);
            return new NDArray(Shape.IsSliced ? UnmanagedStorage.CreateBroadcastedUnsafe(Storage.CloneData(), retShape) : Storage.Alias(retShape)) {TensorEngine = TensorEngine};
        }

        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        private void InferMissingDimension(ref Shape shape)
        {
            var indexOfNegOne = -1;
            int product = 1;
            for (int i = 0; i < shape.NDim; i++)
            {
                if (shape[i] == -1)
                {
                    if (indexOfNegOne != -1)
                        throw new ArgumentException("Only allowed to pass one shape dimension as -1");
                    indexOfNegOne = i;
                }
                else
                {
                    product *= shape[i];
                }
            }

            if (indexOfNegOne == -1)
            {
                return;
            }
            else
            {
                int missingValue = this.size / product;
                if (missingValue * product != this.size)
                {
                    throw new ArgumentException("Bad shape: missing dimension would have to be non-integer");
                }

                shape.dimensions[indexOfNegOne] = missingValue;
                shape.ComputeHashcode();
            }
        }
    }
}
