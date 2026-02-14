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
        [MethodImpl(OptimizeAndInline)]
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
                // Check if this is a scalar reshape (ndim=0 shape from default constructor or empty dims)
                bool isScalarShape = (newShape.dimensions == null || newShape.dimensions.Length == 0);

                if (isScalarShape)
                {
                    // Scalar shapes are valid only when reshaping from size 1
                    if (size != 1)
                        throw new IncorrectShapeException($"Cannot reshape array of size {size} into scalar shape");
                }
                else
                {
                    // For non-scalar shapes, check for empty collection and size match
                    if (newShape.size == 0 && size != 0)
                        throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                    if (size != newShape.size)
                        throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
                }
            }

            // NumPy-aligned: Create new shape with preserved offset and bufferSize
            int bufSize = bufferSize > 0 ? bufferSize : size;

            // Handle scalar shape (null/empty dimensions from default constructor)
            var newDims = newShape.dimensions ?? Array.Empty<int>();
            var newStrides = newShape.strides ?? Array.Empty<int>();

            return new Shape(
                newDims.Length > 0 ? (int[])newDims.Clone() : newDims,
                newStrides.Length > 0 ? (int[])newStrides.Clone() : newStrides,
                offset,
                bufSize
            );
        }

        /// <summary>
        ///     Changes the shape representing this storage (broadcast version).
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        private readonly Shape _reshapeBroadcast(Shape newShape, bool @unsafe = true)
        {
            // Handle -1 in reshape
            newShape = _inferMissingDimension(newShape);

            if (!@unsafe)
            {
                // Check if this is a scalar reshape (ndim=0 shape from default constructor or empty dims)
                bool isScalarShape = (newShape.dimensions == null || newShape.dimensions.Length == 0);

                if (isScalarShape)
                {
                    // Scalar shapes are valid only when reshaping from size 1
                    if (size != 1)
                        throw new IncorrectShapeException($"Cannot reshape array of size {size} into scalar shape");
                }
                else
                {
                    // For non-scalar shapes, check for empty collection and size match
                    if (newShape.size == 0 && size != 0)
                        throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                    if (size != newShape.size)
                        throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
                }
            }

            // NumPy-aligned: preserve bufferSize from original shape for broadcast tracking
            int bufSize = bufferSize > 0 ? bufferSize : size;

            // Handle scalar shape (null/empty dimensions from default constructor)
            var newDims = newShape.dimensions ?? Array.Empty<int>();
            var newStrides = newShape.strides ?? Array.Empty<int>();

            return new Shape(
                newDims.Length > 0 ? (int[])newDims.Clone() : newDims,
                newStrides.Length > 0 ? (int[])newStrides.Clone() : newStrides,
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
            // Handle uninitialized shape (from default constructor) or scalar shapes
            if (shape.dimensions == null || shape.dimensions.Length == 0)
                return shape;

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

                // Calculate proper stride for C-contiguous layout
                int newStride;
                if (axis >= dimensions.Length)
                {
                    // Appending at the end - use 1 (innermost stride)
                    newStride = 1;
                }
                else
                {
                    // Inserting before existing dimension
                    newStride = dimensions[axis] * strides[axis];
                }
                Arrays.Insert(ref newStrides, axis, newStride);
            }

            // Create new shape with preserved bufferSize
            int bufSize = bufferSize > 0 ? bufferSize : size;
            return new Shape(newDims, newStrides, offset, bufSize);
        }
    }
}
