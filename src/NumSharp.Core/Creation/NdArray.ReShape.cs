using System;
using System.Collections.Generic;
using System.ComponentModel;
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
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(ref Shape newshape)
        {
            if (newshape.size == 0 && size != 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(newshape));

            if (Shape.IsSliced)
                Console.WriteLine("reshaping a sliced NDArray performs performs a copy.");

            var storage = newshape.IsSliced ? new UnmanagedStorage(Storage.CloneData(), newshape) : Storage.Alias(newshape);
            return new NDArray(storage) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape newshape)
        {
            return reshape(ref newshape);
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
        public NDArray reshape(params int[] shapeIn)
        {
            int[] shape = InferMissingDimension(shapeIn);

            if (shape.Length == 0 && size != 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(shape));

            return reshape(new Shape(shape));
        }

        // TODO: Unit test reshape with missing dimension
        int[] InferMissingDimension(int[] shape)
        {
            var indexOfNegOne = -1;
            int product = 1;
            for (int i = 0; i < shape.Length; i++)
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
                return shape;
            }
            else
            {
                int missingValue = this.size / product;
                if (missingValue * product != this.size)
                {
                    throw new ArgumentException("Bad shape: missing dimension would have to be non-integer");
                }
                int[] result = (int[])shape.Clone();
                result[indexOfNegOne] = missingValue;
                return result;
            }
        }
    }
}
