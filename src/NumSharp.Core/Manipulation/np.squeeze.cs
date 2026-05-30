using System;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        public static NDArray squeeze(NDArray a)
        {
            return a.reshape(a.shape.Where(x => x != 1).ToArray());
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        /// <exception cref="IncorrectShapeException">If axis is not None, and an axis being squeezed is not of length 1</exception>
        public static NDArray squeeze(NDArray a, int axis)
        {
            while (axis < 0)
                axis = a.ndim + axis; //handle negative axis

            if (axis >= a.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (a.shape[axis] != 1)
                throw new IncorrectShapeException($"Unable to squeeze axis {axis} because it is of length {a.shape[axis]} and not 1.");

            return a.reshape(squeeze_fast(a.Shape, axis));
        }

        /// <summary>
        ///     Remove single-dimensional entries from a shape.
        /// </summary>
        /// <param name="shape">Input shape.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        public static Shape squeeze(Shape shape)
        {
            //TODO! what will happen if its a slice?
            return new Shape(shape.dimensions.Where(d => d != 1).ToArray());
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        /// <exception cref="IncorrectShapeException">If axis is not None, and an axis being squeezed is not of length 1</exception>
        [MethodImpl(Inline)]
        internal static NDArray squeeze_fast(NDArray a, int axis)
        {
            return a.reshape(squeeze_fast(a.Shape, axis));
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        /// <exception cref="IncorrectShapeException">If axis is not None, and an axis being squeezed is not of length 1</exception>
        [MethodImpl(Inline)]
        internal static Shape squeeze_fast(Shape a, int axis)
        {
            var r = a.dimensions.RemoveAt(axis);
            // NumPy squeeze(axis) removes ONLY the named axis. Only collapse to 0-D when that was the
            // last remaining axis (r.Length == 0); a remaining length-1 dimension must be kept (e.g.
            // squeeze([1,1], axis=0) -> [1], not scalar) — over-collapsing it diverges from NumPy and
            // breaks the matmul 1-D-promotion squeeze.
            if (r.Length == 0)
                return Shape.Scalar;

            return new Shape(r);
        }
    }
}
