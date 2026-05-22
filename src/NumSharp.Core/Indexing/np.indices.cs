using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array representing the indices of a grid. Dense form —
        ///     for the sparse form, see <see cref="indices_sparse"/>.
        /// </summary>
        /// <param name="dimensions">Shape of the grid.</param>
        /// <param name="dtype">Element type of the result. Default is <see cref="long"/>.</param>
        /// <returns>
        ///     Single dense array of shape <c>(len(dimensions), *dimensions)</c>. The
        ///     d-th sub-array contains the d-th coordinate of each output position.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     NumPy's <c>sparse=True</c> mode is exposed as the separate method
        ///     <see cref="indices_sparse"/> — C# can't change return type based on a
        ///     parameter the way Python's dynamic typing does, so splitting the API is
        ///     clearer than throwing from a shared signature.
        ///     </para>
        /// </remarks>
        public static NDArray indices(int[] dimensions, NPTypeCode dtype = NPTypeCode.Int64)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));

            // Special case: empty dimensions tuple → numpy returns shape (0,).
            if (dimensions.Length == 0)
                return new NDArray(dtype, new Shape(0), false);

            return BuildDenseIndices(dimensions, dtype);
        }

        /// <summary>
        ///     Sparse counterpart to <see cref="indices"/> — returns a tuple of
        ///     broadcast-shaped arrays where each axis-d array has shape
        ///     <c>(1, …, 1, dimensions[d], 1, …, 1)</c>. Equivalent to NumPy's
        ///     <c>np.indices(dimensions, sparse=True)</c>.
        /// </summary>
        public static NDArray[] indices_sparse(int[] dimensions, NPTypeCode dtype = NPTypeCode.Int64)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                return Array.Empty<NDArray>();

            return BuildSparseIndicesAsArray(dimensions, dtype);
        }

        private static unsafe NDArray BuildDenseIndices(int[] dimensions, NPTypeCode dtype)
        {
            int ndim = dimensions.Length;

            // Compute prod = total elements per slab and validate non-negative dims.
            long prod = 1;
            for (int d = 0; d < ndim; d++)
            {
                int dim = dimensions[d];
                if (dim < 0)
                    throw new ArgumentException(
                        $"negative dimensions are not allowed (got dim[{d}] = {dim}).",
                        nameof(dimensions));
                long next = unchecked(prod * dim);
                if (dim != 0 && next / dim != prod)
                    throw new ArgumentException(
                        "invalid dimensions: array size larger than the maximum possible size.",
                        nameof(dimensions));
                prod = next;
            }

            // Result shape: (ndim, *dimensions).
            var resultShape = new long[ndim + 1];
            resultShape[0] = ndim;
            for (int d = 0; d < ndim; d++) resultShape[d + 1] = dimensions[d];

            // Allocate as Int64 first (the kernel's native dtype). If the caller asked
            // for a different dtype, astype afterwards. The dense-fill IL kernel only
            // emits long-typed stores.
            var int64Result = new NDArray(NPTypeCode.Int64, new Shape(resultShape), false);

            // Fast-exit for shapes with size 0 — the buffer is already zero-allocated
            // and no slab fill is needed.
            if (prod == 0)
                return dtype == NPTypeCode.Int64 ? int64Result : int64Result.astype(dtype);

            // Compute dimStrides: dimStrides[ndim-1] = 1; dimStrides[d] = dimStrides[d+1] * dims[d+1].
            Span<long> dimStrides = stackalloc long[ndim];
            Span<long> dimsSpan = stackalloc long[ndim];
            dimStrides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                dimStrides[d] = dimStrides[d + 1] * dimensions[d + 1];
            for (int d = 0; d < ndim; d++) dimsSpan[d] = dimensions[d];

            var kernel = ILKernelGenerator.GetIndicesKernel();
            if (kernel == null)
                throw new NotSupportedException("np.indices: IL kernel unavailable");

            long* resPtr = (long*)int64Result.Storage.Address;
            fixed (long* dsPtr = dimStrides)
            fixed (long* dPtr = dimsSpan)
            {
                kernel(resPtr, dsPtr, dPtr, ndim, prod);
            }

            return dtype == NPTypeCode.Int64 ? int64Result : int64Result.astype(dtype);
        }

        private static NDArray[] BuildSparseIndicesAsArray(int[] dimensions, NPTypeCode dtype)
        {
            int ndim = dimensions.Length;
            var result = new NDArray[ndim];

            for (int d = 0; d < ndim; d++)
            {
                int dim = dimensions[d];
                if (dim < 0)
                    throw new ArgumentException(
                        $"negative dimensions are not allowed (got dim[{d}] = {dim}).",
                        nameof(dimensions));

                // axis-d array has shape (1, ..., 1, dim, 1, ..., 1) with `dim` along axis d.
                var axisShape = new long[ndim];
                for (int k = 0; k < ndim; k++) axisShape[k] = (k == d) ? dim : 1;

                // Build via arange then reshape.
                var arr = np.arange(0, dim).reshape(axisShape);
                if (arr.GetTypeCode != dtype)
                    arr = arr.astype(dtype);
                result[d] = arr;
            }

            return result;
        }
    }
}
