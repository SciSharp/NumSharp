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
            if (dimensions == null || dimensions.Length == 0)
                throw new ArgumentException(nameof(dimensions));

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

            SetShapeUnsafe(_shape.Reshape(dimensions, @unsafe));
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
            this.SetShapeUnsafe(_shape.Reshape(newShape, @unsafe));
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

        protected internal void ExpandDimension(int axis)
        {
            _shape = _shape.ExpandDimension(axis);
        }

        #endregion
    }
}
