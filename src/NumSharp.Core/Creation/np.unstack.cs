using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Split an array into a sequence of arrays along the given axis.
        ///     The <paramref name="axis"/> parameter specifies the dimension along
        ///     which the array will be split. For example, if <c>axis=0</c> (the
        ///     default) it will be the first dimension and if <c>axis=-1</c> it
        ///     will be the last dimension.
        ///     Added in NumPy 2.1.
        /// </summary>
        /// <param name="x">The array to be unstacked.</param>
        /// <param name="axis">Axis along which the array will be split. Default: 0.</param>
        /// <returns>
        ///     The unstacked arrays — <c>x.shape[axis]</c> VIEWS into
        ///     <paramref name="x"/> (shared memory, matching NumPy), each with
        ///     shape equal to <c>x.shape</c> with the <paramref name="axis"/>
        ///     entry removed.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unstack.html
        ///     <br></br>
        ///     <c>unstack</c> serves as the reverse operation of <see cref="stack"/>:
        ///     <c>stack(unstack(x, axis), axis) == x</c>. Semantically equivalent to
        ///     NumPy's <c>tuple(np.moveaxis(x, axis, 0))</c> — moving <c>axis</c>
        ///     to the front and then indexing it away leaves the remaining axes in
        ///     their original relative order, so each view is constructed directly
        ///     by dropping <c>axis</c> from dims/strides and advancing the offset
        ///     by <c>i * strides[axis]</c>; no iterator or data movement anywhere.
        /// </remarks>
        public static NDArray[] unstack(NDArray x, int axis = 0)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            // NumPy: ValueError("Input array must be at least 1-d.")
            int ndim = x.ndim;
            if (ndim == 0)
                throw new ValueError("Input array must be at least 1-d.");

            // AxisError parity (moveaxis would raise the same for a bad axis).
            axis = DefaultEngine.check_and_adjust_axis(ndim, axis);

            var shape = x.Shape;
            var result = new NDArray[checked((int)shape.dimensions[axis])];
            if (result.Length == 0)
                return result;

            // Every child shares the same dims/strides — x's with `axis` removed.
            // Shape treats the arrays as immutable, so ONE pair serves all views.
            var subDims = new long[ndim - 1];
            var subStrides = new long[ndim - 1];
            for (int j = 0, k = 0; j < ndim; j++)
            {
                if (j == axis) continue;
                subDims[k] = shape.dimensions[j];
                subStrides[k] = shape.strides[j];
                k++;
            }

            if (x.size == 0)
            {
                // Some OTHER axis is zero-length (a zero `axis` yields an empty
                // result array above). There is no data to alias — each entry is
                // an empty array of the remaining shape.
                for (int i = 0; i < result.Length; i++)
                    result[i] = new NDArray(x.dtype, new Shape(subDims));
                return result;
            }

            long axisStride = shape.strides[axis];
            long baseOffset = shape.offset;
            var storage = x.Storage;
            var engine = x.TensorEngine;
            for (int i = 0; i < result.Length; i++)
            {
                // Alias() inherits writeability, so unstacking a read-only array
                // (broadcast view / 'r' memmap) yields read-only views.
                var sub = new Shape(subDims, subStrides, baseOffset + i * axisStride, shape.bufferSize);
                result[i] = new NDArray(storage.Alias(sub), engine, skipEngineResolve: true);
            }

            return result;
        }
    }
}
