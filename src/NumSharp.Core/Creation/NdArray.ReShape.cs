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
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(params int[] shape)
        {
            if (shape.Length == 0 && size != 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(shape));

            return reshape(new Shape(shape));
        }

        protected static int FindNegativeIndex(params int[] shape)
        {
            var count = shape.Length;
            var negOne = false;
            var indexOfNegOne = -1;
            for (int i = 0; i < count; i++)
            {
                if (shape[i] == -1)
                {
                    if (negOne)
                        throw new ArgumentException("Only allowed to pass one shape dimension as -1");

                    negOne = true;
                    indexOfNegOne = i;
                }
            }

            return indexOfNegOne;
        }

        protected static int[] CalculateNegativeShape(int negativeIndex, IList<int> currentShape, params int[] shapeParams)
        {
            var currentShapeCount = currentShape.Count;
            var shapeParamCount = shapeParams.Length;
            var newShape = new List<int>();
            var curShapeVolume = currentShape.Aggregate((x, y) => x * y);
            if (negativeIndex > -1)
            {
                int x = shapeParams[0];
                int y = 0;
                if (shapeParamCount >= 1)
                    y = shapeParams[1];
                if (shapeParamCount > 2)
                    throw new ArgumentException("We cannot currently handle reshapes of more than 2 dimensions");

                if (negativeIndex == 0 && shapeParamCount == 2)
                {
                    var mod = curShapeVolume % y == 0;
                    if (!mod)
                        throw new ArgumentException($"Wrong Reshape. {curShapeVolume} is not evenly divisible by {y}");
                    else
                    {
                        var a = curShapeVolume / y;
                        var b = y;
                        newShape.Add(a);
                        newShape.Add(b);
                    }
                }
                else if (negativeIndex == 1 && shapeParamCount == 2)
                {
                    var mod = curShapeVolume % x == 0;
                    if (!mod)
                        throw new ArgumentException($"Wrong Reshape. {curShapeVolume} is not evenly divisible by {x}");
                    else
                    {
                        var a = x;
                        var b = curShapeVolume / x;
                        newShape.Add(a);
                        newShape.Add(b);
                    }
                }
            }
            else
                return currentShape.ToArray();

            return newShape.ToArray();
        }
    }
}
