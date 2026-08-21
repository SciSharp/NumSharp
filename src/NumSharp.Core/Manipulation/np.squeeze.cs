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
        /// <remarks>
        ///     A pure VIEW like NumPy's <c>PyArray_Squeeze</c>: the length-one axes are dropped from the
        ///     dims AND strides while offset/buffer stay — never a reshape (which rebuilds C-strides and
        ///     so lost F-contiguity, and MATERIALIZED non-contiguous inputs where NumPy shares memory).
        ///     Probed: <c>np.asfortranarray(zeros((3,1,4))).squeeze()</c> is F-contiguous, a transposed
        ///     input stays a strided view, and a broadcast input keeps stride-0 (read-only).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        /// </remarks>
        public static NDArray squeeze(NDArray a)
        {
            var dims = a.Shape.dimensions;
            int keep = 0;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] != 1)
                    keep++;
            if (keep == dims.Length)
                return new NDArray(a.Storage.Alias()); // nothing to remove — still a fresh view object

            return SqueezeView(a, keep, onlyAxis: -1);
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

            // Stride-dropping VIEW (see squeeze(NDArray)) removing ONLY the named axis.
            return SqueezeView(a, a.ndim - 1, onlyAxis: axis);
        }

        /// <summary>
        ///     Builds the squeezed VIEW: dims and strides minus the removed length-one axes (all of them,
        ///     or only <paramref name="onlyAxis"/>), offset and buffer preserved — NumPy's
        ///     <c>PyArray_Squeeze</c>/<c>SelectedSqueeze</c>. Writeability/read-onlyness inherit through
        ///     <see cref="UnmanagedStorage.Alias(Shape)"/>.
        /// </summary>
        private static NDArray SqueezeView(NDArray a, int keep, int onlyAxis)
        {
            var dims = a.Shape.dimensions;
            var strides = a.Shape.strides;

            if (keep == 0)
            {
                var scalar = new Shape(Array.Empty<long>(), Array.Empty<long>(), a.Shape.offset, a.Shape.BufferSize);
                return new NDArray(a.Storage.Alias(scalar));
            }

            var newDims = new long[keep];
            var newStrides = new long[keep];
            int j = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                if (onlyAxis >= 0 ? i == onlyAxis : dims[i] == 1)
                    continue;
                newDims[j] = dims[i];
                newStrides[j] = strides[i];
                j++;
            }

            return new NDArray(a.Storage.Alias(new Shape(newDims, newStrides, a.Shape.offset, a.Shape.BufferSize)));
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
