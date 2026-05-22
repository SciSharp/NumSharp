using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take elements from an array along an axis. Equivalent to fancy
        ///     indexing along the specified axis.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="indices">Integer array of indices to take.</param>
        /// <param name="axis">
        ///     Axis along which to take. <c>null</c> (default) flattens <paramref name="a"/>
        ///     and treats <paramref name="indices"/> as flat indices.
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default — throw on OOB), <c>"wrap"</c>
        ///     (modulo with sign correction), or <c>"clip"</c> (saturate).
        /// </param>
        /// <returns>
        ///     New array with shape:
        ///     <list type="bullet">
        ///       <item><c>axis=None</c>: same as <paramref name="indices"/>.</item>
        ///       <item><c>axis=k</c>: <c>a.shape[:k] + indices.shape + a.shape[k+1:]</c>.</item>
        ///     </list>
        ///     Dtype matches <paramref name="a"/>.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.take.html</remarks>
        public static NDArray take(NDArray a, NDArray indices, int? axis = null, string mode = "raise")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int modeInt = ParseMode(mode, nameof(mode));

            // 0-d source — treat as 1-element 1-D for the gather. Output shape =
            // indices.shape regardless of axis (NumPy parity).
            if (a.ndim == 0)
            {
                var flat0d = a.reshape(new Shape(1));
                return TakeFlat(flat0d, indices, modeInt);
            }

            // axis=None ⇒ flat take from the C-order flattening of a.
            if (axis == null)
                return TakeFlat(a, indices, modeInt);

            int ax = axis.Value;
            if (ax < 0) ax += a.ndim;
            if (ax < 0 || ax >= a.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis.Value} is out of bounds for array of dimension {a.ndim}");

            return TakeAxis(a, indices, ax, modeInt);
        }

        /// <summary>
        ///     Scalar convenience overload — take a single element by flat index.
        /// </summary>
        public static NDArray take(NDArray a, long index, int? axis = null, string mode = "raise")
        {
            // Wrap the scalar in a 0-d NDArray so the array overload's shape-preserving
            // logic emits a 0-d result (NumPy semantic for scalar input).
            var idxArr = NDArray.Scalar(index);
            return take(a, idxArr, axis, mode);
        }

        private static unsafe NDArray TakeFlat(NDArray a, NDArray indices, int mode)
        {
            // Materialise non-contig source to C-contig so the kernel can walk linearly.
            NDArray sourceOwned = null;
            NDArray source = a;
            if (!a.Shape.IsContiguous)
            {
                sourceOwned = np.ascontiguousarray(a);
                source = sourceOwned;
            }

            // Cast indices to contig int64.
            var idx64 = CastIndicesToInt64(indices, out bool ownIdx);

            // Output shape = indices.shape, dtype = a.dtype, C-contig allocated.
            var outShape = new Shape((long[])indices.Shape.dimensions.Clone());
            var result = new NDArray(a.typecode, outShape, false);

            long indicesCount = indices.size;
            long maxItem = a.size;
            long elemBytes = a.dtypesize;

            if (indicesCount == 0)
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
                return result;
            }

            if (maxItem == 0)
                throw new ArgumentException("cannot do a non-empty take from an empty array.", nameof(a));

            try
            {
                ExecuteTakeKernel(source, idx64, result,
                    outerSize: 1, indicesCount: indicesCount,
                    maxItem: maxItem, innerSize: elemBytes, mode: mode);
            }
            finally
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
            }

            return result;
        }

        private static unsafe NDArray TakeAxis(NDArray a, NDArray indices, int axis, int mode)
        {
            NDArray sourceOwned = null;
            NDArray source = a;
            if (!a.Shape.IsContiguous)
            {
                sourceOwned = np.ascontiguousarray(a);
                source = sourceOwned;
            }

            var idx64 = CastIndicesToInt64(indices, out bool ownIdx);

            // Compute outer × inner factorisation around `axis`.
            long outerSize = 1;
            for (int d = 0; d < axis; d++) outerSize *= a.Shape.dimensions[d];
            long maxItem = a.Shape.dimensions[axis];
            long innerCount = 1;
            for (int d = axis + 1; d < a.ndim; d++) innerCount *= a.Shape.dimensions[d];
            long elemBytes = a.dtypesize;
            long innerBytes = innerCount * elemBytes;

            // Output shape = a.shape[:axis] + indices.shape + a.shape[axis+1:].
            int outNdim = a.ndim + indices.ndim - 1;
            var outDims = new long[outNdim];
            int outIdx = 0;
            for (int d = 0; d < axis; d++) outDims[outIdx++] = a.Shape.dimensions[d];
            for (int d = 0; d < indices.ndim; d++) outDims[outIdx++] = indices.Shape.dimensions[d];
            for (int d = axis + 1; d < a.ndim; d++) outDims[outIdx++] = a.Shape.dimensions[d];

            var result = new NDArray(a.typecode, new Shape(outDims), false);

            long indicesCount = indices.size;
            if (indicesCount == 0 || outerSize == 0)
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
                return result;
            }

            if (maxItem == 0)
                throw new ArgumentException("cannot do a non-empty take from an empty axis.", nameof(a));

            try
            {
                ExecuteTakeKernel(source, idx64, result,
                    outerSize: outerSize, indicesCount: indicesCount,
                    maxItem: maxItem, innerSize: innerBytes, mode: mode);
            }
            finally
            {
                if (ownIdx) idx64.Dispose();
                sourceOwned?.Dispose();
            }

            return result;
        }

        private static unsafe void ExecuteTakeKernel(
            NDArray source, NDArray idx64, NDArray result,
            long outerSize, long indicesCount, long maxItem, long innerSize, int mode)
        {
            var kernel = ILKernelGenerator.GetTakeKernel();
            if (kernel == null)
                throw new NotSupportedException("np.take: IL kernel unavailable");

            byte* srcPtr = (byte*)source.Storage.Address + source.Shape.offset * source.dtypesize;
            long* idxPtr = (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* dstPtr = (byte*)result.Storage.Address;

            long status = kernel(srcPtr, idxPtr, indicesCount, outerSize,
                                  maxItem, innerSize, mode, dstPtr);
            long expected = outerSize * indicesCount;
            if (status < expected)
            {
                long failPair = status;
                long badJ = failPair % indicesCount;
                long badVal = idxPtr[badJ];
                throw new ArgumentOutOfRangeException(
                    nameof(idx64),
                    $"index {badVal} is out of bounds for axis with size {maxItem}");
            }
        }

        /// <summary>
        ///     Casts <paramref name="indices"/> to contig int64 if needed. Sets
        ///     <paramref name="owned"/> to true when the returned array is a fresh
        ///     allocation that the caller must Dispose. The result shape matches
        ///     the input shape but always has contig strides.
        /// </summary>
        private static NDArray CastIndicesToInt64(NDArray indices, out bool owned)
        {
            if (indices.GetTypeCode == NPTypeCode.Int64 && indices.Shape.IsContiguous)
            {
                owned = false;
                return indices;
            }

            if (indices.GetTypeCode == NPTypeCode.Int64)
            {
                var c = np.ascontiguousarray(indices);
                owned = !ReferenceEquals(c, indices);
                return c;
            }

            owned = true;
            return indices.astype(NPTypeCode.Int64);
        }
    }
}
