using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <exception cref="ArgumentException">If indices_or_sections is an integer and does not result in equal division.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // array_split's argument validation handles section<=0; we validate first
            // so the equal-division check below doesn't divide by zero (NumPy bug:
            // raw int /% 0 throws ZeroDivisionError, but our ArgumentException is
            // clearer and consistent with array_split's own check).
            if (indices_or_sections <= 0)
                throw new ArgumentException("number sections must be larger than 0.");

            int ndim = ary.ndim;
            int ax = NormalizeSplitAxis(axis, ndim);

            long N = ary.Shape.dimensions[ax];
            if (N % indices_or_sections != 0)
                throw new ArgumentException("array split does not result in an equal division");

            return ArraySplitByInt(ary, indices_or_sections, ax);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, long[] indices, int axis = 0)
        {
            return array_split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, int[] indices, int axis = 0)
        {
            return array_split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N sub-arrays along axis.
        /// If N does not divide the array equally, it returns l % n sub-arrays of size
        /// l//n + 1 and the rest of size l//n.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// The only difference between split and array_split is that array_split allows
        /// indices_or_sections to be an integer that does not equally divide the axis.
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices_or_sections <= 0)
                throw new ArgumentException("number sections must be larger than 0.");

            int ndim = ary.ndim;
            int ax = NormalizeSplitAxis(axis, ndim);

            return ArraySplitByInt(ary, indices_or_sections, ax);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// If an index exceeds the dimension of the array along axis, an empty sub-array
        /// is returned correspondingly.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, long[] indices, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            long Ntotal = ary.Shape.dimensions[ax];

            return SplitByIndicesDirect(ary, indices, ax, Ntotal);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// If an index exceeds the dimension of the array along axis, an empty sub-array
        /// is returned correspondingly.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int[] indices, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            long Ntotal = ary.Shape.dimensions[ax];

            return SplitByIndicesDirect(ary, indices, ax, Ntotal);
        }

        /// <summary>
        ///     Normalises a possibly-negative axis to the [0, ndim) range. Throws when
        ///     the array is 0-d (no axes to split on) or the axis is out of range.
        /// </summary>
        private static int NormalizeSplitAxis(int axis, int ndim)
        {
            if (ndim == 0)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");

            int adjusted = axis < 0 ? axis + ndim : axis;
            if (adjusted < 0 || adjusted >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");
            return adjusted;
        }

        /// <summary>
        ///     Int-sections split with no div_points scratch. Each section size is
        ///     <c>Ntotal/Nsections + (i &lt; extras ? 1 : 0)</c> per NumPy's
        ///     <c>array_split</c>; we accumulate the running cursor inline and
        ///     hand each (start, end) straight to <see cref="BuildSubArrayView"/>.
        /// </summary>
        private static NDArray[] ArraySplitByInt(NDArray ary, int Nsections, int axis)
        {
            long Ntotal = ary.Shape.dimensions[axis];
            long Neach = Ntotal / Nsections;
            long extras = Ntotal % Nsections;

            var sub_arys = new NDArray[Nsections];

            var srcShape = ary.Shape;
            long[] srcDims = srcShape.dimensions;
            long[] srcStrides = srcShape.strides;
            long axisStride = srcStrides[axis];
            long baseOffset = srcShape.offset;
            long bufSize = srcShape.bufferSize > 0 ? srcShape.bufferSize : srcShape.size;
            int ndim = srcDims.Length;
            var engine = ary.TensorEngine;
            var storage = ary.Storage;

            long cursor = 0;
            for (int i = 0; i < Nsections; i++)
            {
                long size = (i < extras) ? Neach + 1 : Neach;
                long st = cursor;
                long end = cursor + size;

                sub_arys[i] = BuildSubArrayView(
                    storage, engine, srcDims, srcStrides,
                    ndim, axis, axisStride, baseOffset, bufSize, st, end);

                cursor = end;
            }

            return sub_arys;
        }

        /// <summary>
        ///     Indices-mode split that walks the indices array directly without
        ///     allocating a div_points scratch buffer. Walks the boundary list
        ///     <c>0, indices[0], indices[1], ..., indices[^1], Ntotal</c> with two
        ///     "cursors" (prev, cur) instead of materialising it. Saves one
        ///     <c>long[Nsections+1]</c> alloc on the indices path.
        /// </summary>
        private static NDArray[] SplitByIndicesDirect(NDArray ary, long[] indices, int axis, long Ntotal)
        {
            int Nsections = indices.Length + 1;
            var sub_arys = new NDArray[Nsections];

            var srcShape = ary.Shape;
            long[] srcDims = srcShape.dimensions;
            long[] srcStrides = srcShape.strides;
            long axisStride = srcStrides[axis];
            long baseOffset = srcShape.offset;
            long bufSize = srcShape.bufferSize > 0 ? srcShape.bufferSize : srcShape.size;
            int ndim = srcDims.Length;
            var engine = ary.TensorEngine;
            var storage = ary.Storage;
            long[] sharedStrides = srcStrides;

            long prev = 0;
            for (int i = 0; i < Nsections; i++)
            {
                long raw = (i == indices.Length) ? Ntotal : indices[i];
                long cur = ClampSlicePoint(raw, Ntotal);
                long st = prev;
                long end = cur < st ? st : cur;

                sub_arys[i] = BuildSubArrayView(
                    storage, engine, srcDims, sharedStrides,
                    ndim, axis, axisStride, baseOffset, bufSize, st, end);

                prev = cur;
            }

            return sub_arys;
        }

        /// <inheritdoc cref="SplitByIndicesDirect(NDArray, long[], int, long)"/>
        private static NDArray[] SplitByIndicesDirect(NDArray ary, int[] indices, int axis, long Ntotal)
        {
            int Nsections = indices.Length + 1;
            var sub_arys = new NDArray[Nsections];

            var srcShape = ary.Shape;
            long[] srcDims = srcShape.dimensions;
            long[] srcStrides = srcShape.strides;
            long axisStride = srcStrides[axis];
            long baseOffset = srcShape.offset;
            long bufSize = srcShape.bufferSize > 0 ? srcShape.bufferSize : srcShape.size;
            int ndim = srcDims.Length;
            var engine = ary.TensorEngine;
            var storage = ary.Storage;
            long[] sharedStrides = srcStrides;

            long prev = 0;
            for (int i = 0; i < Nsections; i++)
            {
                long raw = (i == indices.Length) ? Ntotal : indices[i];
                long cur = ClampSlicePoint(raw, Ntotal);
                long st = prev;
                long end = cur < st ? st : cur;

                sub_arys[i] = BuildSubArrayView(
                    storage, engine, srcDims, sharedStrides,
                    ndim, axis, axisStride, baseOffset, bufSize, st, end);

                prev = cur;
            }

            return sub_arys;
        }

        /// <summary>
        ///     One sub-array view: clone dims (per-Shape ownership required since
        ///     <c>dims[axis]</c> varies), patch <c>dims[axis] = end - start</c>,
        ///     advance offset by <c>start * axisStride</c>, share strides, alias
        ///     storage. Inlined; the three split entry points all call it.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static NDArray BuildSubArrayView(
            UnmanagedStorage storage, TensorEngine engine,
            long[] srcDims, long[] sharedStrides,
            int ndim, int axis, long axisStride, long baseOffset, long bufSize,
            long st, long end)
        {
            var dims = new long[ndim];
            Array.Copy(srcDims, dims, ndim);
            dims[axis] = end - st;

            long newOffset = baseOffset + st * axisStride;
            var newShape = new Shape(dims, sharedStrides, newOffset, bufSize);

            return new NDArray(storage.Alias(newShape)) { TensorEngine = engine };
        }

        /// <summary>
        ///     NumPy slice clamping: <c>n &lt; 0 ? max(0, n + N) : min(n, N)</c>.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static long ClampSlicePoint(long n, long N)
        {
            if (n < 0)
            {
                long wrapped = n + N;
                return wrapped < 0 ? 0 : wrapped;
            }
            return n > N ? N : n;
        }
    }
}
