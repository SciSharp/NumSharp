using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Represents a shape of an N-D array.
    /// </summary>
    /// <remarks>Handles slicing, indexing based on coordinates or linear offset and broadcastted indexing.</remarks>
    public partial struct Shape
    {
        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        /// <param name="unsafe">When true, then guards are skipped.</param>
        [MethodImpl((MethodImplOptions)768)]
        public readonly Shape Reshape(Shape newShape, bool @unsafe = true)
        {
            if (IsBroadcasted)
            {
                return _reshapeBroadcast(newShape, @unsafe);
            }

            // Handle -1 in reshape - returns new shape with inferred dimension
            newShape = _inferMissingDimension(newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
            }

            // NumPy-aligned: Create new shape with preserved offset and bufferSize
            int bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(
                (int[])newShape.dimensions.Clone(),
                (int[])newShape.strides.Clone(),
                offset,
                bufSize
            );
        }

        /// <summary>
        ///     Changes the shape representing this storage (broadcast version).
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        private readonly Shape _reshapeBroadcast(Shape newShape, bool @unsafe = true)
        {
            // Handle -1 in reshape
            newShape = _inferMissingDimension(newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
            }

            // NumPy-aligned: preserve bufferSize from original shape for broadcast tracking
            int bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(
                (int[])newShape.dimensions.Clone(),
                (int[])newShape.strides.Clone(),
                0,
                bufSize
            );
        }

        /// <summary>
        ///     Infers missing dimension (-1) and returns a new shape with correct dimensions/strides.
        /// </summary>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        private readonly Shape _inferMissingDimension(Shape shape)
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
                return shape; // No -1 to infer

            if (this.IsBroadcasted)
            {
                throw new NotSupportedException("Reshaping a broadcasted array with a -1 (unknown) dimension is not supported.");
            }

            int missingValue = this.size / product;
            if (missingValue * product != this.size)
            {
                throw new ArgumentException("Bad shape: missing dimension would have to be non-integer");
            }

            // Create new dimensions array with inferred value
            var newDims = (int[])shape.dimensions.Clone();
            newDims[indexOfNegOne] = missingValue;

            // Compute new strides for the corrected dimensions
            var newStrides = ComputeContiguousStrides(newDims);

            return new Shape(newDims, newStrides, 0, 0);
        }

        /// <summary>
        ///     Expands a specific <paramref name="axis"/> with 1 dimension.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "LocalVariableHidesMember")]
        public readonly Shape ExpandDimension(int axis)
        {
            int[] newDims;
            int[] newStrides;

            if (IsScalar)
            {
                newDims = new int[] { 1 };
                newStrides = new int[] { 0 };
            }
            else
            {
                newDims = (int[])dimensions.Clone();
                newStrides = (int[])strides.Clone();

                // Allow negative axis specification
                if (axis < 0)
                {
                    axis = dimensions.Length + 1 + axis;
                    if (axis < 0)
                        throw new ArgumentException($"Effective axis {axis} is less than 0");
                }

                Arrays.Insert(ref newDims, axis, 1);
                Arrays.Insert(ref newStrides, axis, 0);
            }

            // Create new shape with preserved bufferSize
            int bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(newDims, newStrides, offset, bufSize);
        }
    }
}
