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
        public Shape Reshape(Shape newShape, bool @unsafe = true)
        {
            if (IsBroadcasted)
            {
                _reshapeBroadcast(ref newShape, @unsafe);
                return newShape;
            }

            //handle -1 in reshape
            _inferMissingDimension(ref newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
            }

            if (IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() { ParentShape = this, Slices = null };

            return newShape;
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl((MethodImplOptions)768)]
        private void _reshapeBroadcast(ref Shape newShape, bool @unsafe = true)
        {
            //handle -1 in reshape
            _inferMissingDimension(ref newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({size})");
            }

            if (IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() { ParentShape = this, Slices = null };

            newShape.BroadcastInfo = IsBroadcasted ? BroadcastInfo.Clone() : new BroadcastInfo(this);
        }

        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        private void _inferMissingDimension(ref Shape shape)
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
                return;


            if (this.IsBroadcasted)
            {
                /* //TODO: the following case needs to be handled.
                 *  a = np.arange(4).reshape(1,2,2)
                 *  print(a.strides)
                 *
                 *  b = np.broadcast_to(a, (2,2,2))
                 *  print(b.strides)
                 *
                 *  c = np.reshape(b, (2, -1))
                 *  print(c.strides)
                 *
                 *  c = np.reshape(b, (-1, 2))
                 *  print(c.strides)
                 *  
                 *  (16, 8, 4)
                 *  (0, 8, 4)
                 *  (0, 4) //here it handles broadcast
                 *  (8, 4) //here can be seen that numpy performs a copy
                 */
                //var originalReshaped = np.broadcast_to(this.BroadcastInfo.OriginalShape.Reshape(new int[] { this.BroadcastInfo.OriginalShape.size }), (Shape) new int[] { this.size });
                throw new NotSupportedException("Reshaping a broadcasted array with a -1 (unknown) dimension is not supported.");
            }

            int missingValue = this.size / product;
            if (missingValue * product != this.size)
            {
                throw new ArgumentException("Bad shape: missing dimension would have to be non-integer");
            }

            shape.dimensions[indexOfNegOne] = missingValue;
            var strides = shape.strides;
            var dimensions = shape.dimensions;



            if (indexOfNegOne == strides.Length - 1)
                strides[strides.Length - 1] = 1;

            for (int idx = indexOfNegOne; idx >= 1; idx--)
                strides[idx - 1] = strides[idx] * dimensions[idx];

            shape.ComputeHashcode();
        }

        /// <summary>
        ///     Expands a specific <paramref name="axis"/> with 1 dimension.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "LocalVariableHidesMember")]
        public Shape ExpandDimension(int axis)
        {
            Shape ret;
            if (IsScalar)
            {
                ret = Vector(1);
                ret.strides[0] = 0;
            }
            else
            {
                ret = Clone(true, true, false);
            }

            var dimensions = ret.dimensions;
            var strides = ret.strides;
            // Allow negative axis specification
            if (axis < 0)
            {
                axis = dimensions.Length + 1 + axis;
                if (axis < 0)
                {
                    throw new ArgumentException($"Effective axis {axis} is less than 0");
                }
            }

            Arrays.Insert(ref dimensions, axis, 1);
            Arrays.Insert(ref strides, axis, 0);
            ret.dimensions = dimensions;
            ret.strides = strides;
            if (IsSliced)
            {
                ret.ViewInfo = new ViewInfo() { ParentShape = this, Slices = null };
            }

            if (IsBroadcasted)
            {
                ret.BroadcastInfo = new BroadcastInfo(!BroadcastInfo.OriginalShape.IsEmpty ? BroadcastInfo.OriginalShape : this);
                ret.BroadcastInfo.UnreducedBroadcastedShape = null; //reset so it will be recomupted.
            }

            ret.ComputeHashcode();
            return ret;
        }

    }
}
