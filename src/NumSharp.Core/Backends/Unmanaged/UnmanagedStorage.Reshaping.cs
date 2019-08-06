using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Shaping

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        public void Reshape(params int[] dimensions)
        {
            Reshape(dimensions, false);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="dimensions"/>'s size == 0</exception>
        public void Reshape(int[] dimensions, bool @unsafe)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            Shape newShape = new Shape(dimensions);

            //handle -1 in reshape
            InferMissingDimension(ref newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && _shape.size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (_shape.size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({_shape.size})");
            }

            if (_shape.IsBroadcasted)
                throw new NotSupportedException("Reshaping an already broadcasted shape is not supported.");

            if (_shape.IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() {ParentShape = _shape, Slices = null};

            SetShapeUnsafe(ref newShape);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl((MethodImplOptions)768)]
        public void Reshape(Shape newShape, bool @unsafe = false)
        {
            Reshape(ref newShape, @unsafe);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl((MethodImplOptions)768)]
        public void Reshape(ref Shape newShape, bool @unsafe = false)
        {
            //handle -1 in reshape
            InferMissingDimension(ref newShape);

            if (!@unsafe)
            {
                if (newShape.size == 0 && _shape.size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (_shape.size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({_shape.size})");
            }

            if (_shape.IsBroadcasted)
                throw new NotSupportedException("Reshaping an already broadcasted shape is not supported.");

            if (_shape.IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() {ParentShape = _shape, Slices = null};

            SetShapeUnsafe(ref newShape);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl((MethodImplOptions)768)]
        public void ReshapeBroadcastedUnsafe(ref Shape newShape, bool guards = false, Shape? original = null)
        {
            //handle -1 in reshape
            InferMissingDimension(ref newShape);

            if (guards)
            {
                if (newShape.size == 0 && _shape.size != 0)
                    throw new ArgumentException("Value cannot be an empty collection.", nameof(newShape));

                if (_shape.size != newShape.size)
                    throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the given storage size ({_shape.size})");
            }

            if (_shape.IsSliced)
                // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                newShape.ViewInfo = new ViewInfo() {ParentShape = _shape, Slices = null};

            if (original.HasValue)
                newShape.BroadcastInfo = new BroadcastInfo(original.Value);

            SetShapeUnsafe(ref newShape);
        }

        /// <summary>
        ///     Set the shape of this storage without checking if sizes match.
        /// </summary>
        /// <remarks>Used during broadcasting</remarks>
        protected internal void SetShapeUnsafe(Shape shape)
        {
            SetShapeUnsafe(ref shape);
        }

        /// <summary>
        ///     Set the shape of this storage without checking if sizes match.
        /// </summary>
        /// <remarks>Used during broadcasting</remarks>
        protected internal void SetShapeUnsafe(ref Shape shape)
        {
            _shape = shape;
            Count = _shape.size;
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
                return;

            int missingValue = _shape.size / product;
            if (missingValue * product != _shape.size)
            {
                throw new ArgumentException("Bad shape: missing dimension would have to be non-integer");
            }

            shape.dimensions[indexOfNegOne] = missingValue;
            shape.ComputeHashcode();
        }

        protected internal void ExpandDimension(int axis)
        {
            _shape = _shape.ExpandDimension(axis);
        }

        #endregion
    }
}
