using System;
using NumSharp.Generic;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static NDArray<long>[] NonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => nonzeros<T>(nd.MakeGeneric<T>());

        private static long CountNonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => count_nonzero<T>(nd.MakeGeneric<T>());

        private static void CountNonZeroAxisDispatch<T>(NDArray nd, NDArray result, int axis) where T : unmanaged
            => count_nonzero_axis<T>(nd.MakeGeneric<T>(), result, axis);

        public override NDArray<long>[] NonZero(NDArray nd)
        {
            return NpFunc.Invoke(nd.typecode, NonZeroDispatch<int>, nd);
        }

        /// <summary>
        /// Generic implementation of nonzero.
        ///
        /// Routing:
        ///   * Empty → tuple of empty index arrays (one per dimension).
        ///   * Contiguous + <c>T=bool</c> → IL-emitted SIMD path:
        ///     1. <c>IsAllZeroBoolKernel</c> prescan (closes the NumPy all-false short-circuit gap).
        ///     2. <c>NonZeroFlatBoolKernel</c> bit-scan that writes flat indices into a stackalloc/heap buffer.
        ///     3. Single flat→coord conversion pass at the end.
        ///   * Contiguous other dtypes → existing <see cref="ILKernelGenerator.NonZeroSimdHelper{T}"/>.
        ///   * Non-contiguous (sliced/transposed/broadcast) → strided coordinate-based helper.
        ///
        /// Why the bool path matters: the most common nonzero caller is <c>np.where(condition)</c>,
        /// which always coerces to bool. The IL kernels are emitted once on first use and cached
        /// for the process lifetime via <see cref="ILKernelGenerator.GetIsAllZeroBoolKernel"/>
        /// and <see cref="ILKernelGenerator.GetNonZeroFlatBoolKernel"/>.
        /// </summary>
        private static unsafe NDArray<long>[] nonzeros<T>(NDArray<T> x) where T : unmanaged
        {
            // Ensure at least 1D (NumPy behavior)
            x = np.atleast_1d(x).MakeGeneric<T>();
            var shape = x.Shape;
            var size = x.size;
            var ndim = x.ndim;

            // Handle empty arrays: return tuple of empty arrays (one per dimension)
            // NumPy: np.nonzero(np.array([])) -> (array([], dtype=int64),)
            if (size == 0)
                return MakeEmptyNonZeroResult(ndim);

            if (shape.IsContiguous)
            {
                T* basePtr = (T*)x.Address + shape.offset;

                // Bool-specific IL-emitted fast path. Coerces bool* to byte* for SIMD treatment
                // since .NET's Vector*<bool> isn't supported but Vector*<byte> is.
                if (typeof(T) == typeof(bool))
                {
                    byte* maskPtr = (byte*)basePtr;

                    // Stage 1: IL-emitted any-true prescan — short-circuits when the entire mask
                    // is zero. NumPy's nonzero does the same to keep all-false 10M elements at
                    // sub-millisecond cost. Returns null when SIMD is unavailable (VectorBits==0),
                    // in which case the bit-scan kernel below will simply walk byte-by-byte.
                    var prescan = ILKernelGenerator.GetIsAllZeroBoolKernel();
                    if (prescan != null && prescan(maskPtr, size))
                        return MakeEmptyNonZeroResult(ndim);

                    // Stages 2+3: count → exact-size alloc → bit-scan write.
                    //
                    // The two-pass approach (count first, then collect) matches NumPy. It costs
                    // one extra SIMD scan of the mask, but lets us:
                    //   * Skip the buffer entirely if count == 0 (rare after a successful prescan,
                    //     but possible if SIMD is unavailable).
                    //   * Allocate result NDArrays of exactly `count` longs — no max-size waste
                    //     (10M bool * 8 = 80MB) for dense masks.
                    var countKernel = ILKernelGenerator.GetNonZeroCountBoolKernel();
                    var scanKernel = ILKernelGenerator.GetNonZeroFlatBoolKernel();
                    if (countKernel != null && scanKernel != null)
                    {
                        long count = countKernel(maskPtr, size);

                        if (count == 0)
                            return MakeEmptyNonZeroResult(ndim);

                        // 1D fast path: write flat indices directly into result[0] — no
                        // intermediate buffer, no flat→coord conversion.
                        if (ndim == 1)
                        {
                            var result = new NDArray<long>[1] { new NDArray<long>(count) };
                            long written;
                            long* dst = (long*)result[0].Address;
                            written = scanKernel(maskPtr, dst, size);
                            // Sanity: written should equal count. If not (corrupt mask between
                            // calls in a multithreaded scenario), the result NDArray will have
                            // garbage past `written`. We trust count for sizing.
                            return result;
                        }

                        // Multi-dim: write to a temp buffer of exact size `count`, then convert
                        // to per-dim coordinates.
                        var buffer = new long[count];
                        fixed (long* bufPtr = buffer)
                            scanKernel(maskPtr, bufPtr, size);

                        return BuildCoordinatesFromFlat(buffer, count, shape.dimensions);
                    }

                    // Fallback: no SIMD available — generic helper handles bool via its scalar
                    // branch (EqualityComparer<bool>.Default).
                    var flatBoolFb = new System.Collections.Generic.List<long>(EstimateNonZeroCapacity(size));
                    ILKernelGenerator.NonZeroSimdHelper(basePtr, size, flatBoolFb);
                    return ILKernelGenerator.ConvertFlatIndicesToCoordinates(flatBoolFb, shape.dimensions);
                }

                // Non-bool numeric dtypes: existing generic SIMD helper (V256/V128/scalar internally).
                // Avoids the per-non-zero long[ndim] allocation that the strided helper performs.
                var flat = new System.Collections.Generic.List<long>(EstimateNonZeroCapacity(size));
                ILKernelGenerator.NonZeroSimdHelper(basePtr, size, flat);
                return ILKernelGenerator.ConvertFlatIndicesToCoordinates(flat, shape.dimensions);
            }

            // Non-contiguous: strided coordinate-based helper.
            return ILKernelGenerator.FindNonZeroStridedHelper((T*)x.Address, shape.dimensions, shape.strides, shape.offset);
        }

        /// <summary>
        /// Build per-dimension coordinate <see cref="NDArray{Int64}"/> outputs from a flat-index
        /// buffer of length <paramref name="count"/>. Single pass with precomputed dim strides
        /// (no allocation per element).
        /// </summary>
        private static unsafe NDArray<long>[] BuildCoordinatesFromFlat(long[] flatBuffer, long count, long[] shape)
        {
            int ndim = shape.Length;

            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NDArray<long>(count);

            // count is bounded by size, which fits in long. The List path used (int)count safely
            // because List<T> capacity is int — here we know count <= size <= long.MaxValue.
            if (count == 0)
                return result;

            var addresses = new long*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (long*)result[d].Address;

            // Pre-compute dim-product strides (row-major / C-order).
            var dimStrides = new long[ndim];
            dimStrides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                dimStrides[d] = dimStrides[d + 1] * shape[d + 1];

            for (long i = 0; i < count; i++)
            {
                long flat = flatBuffer[i];
                for (int d = 0; d < ndim; d++)
                {
                    addresses[d][i] = flat / dimStrides[d];
                    flat %= dimStrides[d];
                }
            }

            return result;
        }

        // List<T> capacity is int-limited by the BCL. For very large arrays, start with a
        // reasonable capacity and let it grow rather than pre-sizing to int.MaxValue.
        private static int EstimateNonZeroCapacity(long size)
            => size <= int.MaxValue ? Math.Max(16, (int)(size / 4)) : 1 << 20;

        private static NDArray<long>[] MakeEmptyNonZeroResult(int ndim)
        {
            var result = new NDArray<long>[ndim];
            for (int i = 0; i < ndim; i++)
                result[i] = new NDArray<long>(0);
            return result;
        }

        /// <summary>
        /// Count the number of non-zero elements in the array.
        /// </summary>
        /// <remarks>
        /// NumPy-aligned: np.count_nonzero([0, 1, 0, 2]) = 2
        /// </remarks>
        public override long CountNonZero(NDArray nd)
        {
            if (nd.size == 0)
                return 0;

            return NpFunc.Invoke(nd.typecode, CountNonZeroDispatch<int>, nd);
        }

        /// <summary>
        /// Count non-zero elements along a specific axis.
        /// </summary>
        public override NDArray CountNonZero(NDArray nd, int axis, bool keepdims = false)
        {
            var shape = nd.Shape;

            // Normalize axis
            while (axis < 0)
                axis = nd.ndim + axis;
            if (axis >= nd.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            // Compute output shape
            var outputDims = new long[nd.ndim - 1];
            for (int d = 0, od = 0; d < nd.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Int64, outputShape, false);

            if (nd.size == 0)
            {
                // Already zeros from allocation
                if (keepdims)
                {
                    var ks = new long[nd.ndim];
                    for (int d = 0, sd = 0; d < nd.ndim; d++)
                        ks[d] = (d == axis) ? 1 : outputDims[sd++];
                    result.Storage.Reshape(new Shape(ks));
                }
                return result;
            }

            NpFunc.Invoke(nd.typecode, CountNonZeroAxisDispatch<int>, nd, result, axis);

            if (keepdims)
            {
                var ks = new long[nd.ndim];
                for (int d = 0, sd = 0; d < nd.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }

        /// <summary>
        /// Generic implementation of count_nonzero (element-wise).
        /// </summary>
        private static unsafe long count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;

            if (shape.IsContiguous)
            {
                // Fast path for contiguous arrays
                T* ptr = (T*)x.Address;
                T zero = default;
                long count = 0;
                for (long i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
                return count;
            }

            // Strided path: use NpyIter for layout-aware traversal.
            using var iter = NpyIterRef.New(x, NpyIterGlobalFlags.EXTERNAL_LOOP);
            return iter.ExecuteReducing<CountNonZeroKernel<T>, long>(default, 0L);
        }

        /// <summary>
        /// Count non-zero elements along an axis.
        /// </summary>
        private static unsafe void count_nonzero_axis<T>(NDArray<T> x, NDArray result, int axis) where T : unmanaged
        {
            var shape = x.Shape;
            long axisSize = shape.dimensions[axis];
            var outputSize = result.size;
            T zero = default;

            // Compute output dimension strides for coordinate calculation
            int outputNdim = x.ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * shape.dimensions[nextInputDim];
                }
            }

            long axisStride = shape.strides[axis];

            // Use direct pointer access to result array (result is contiguous Int64)
            long* resultPtr = (long*)result.Address;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to input coordinates
                long remaining = outIdx;
                long inputBaseOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                // Count non-zeros along axis
                long count = 0;
                T* basePtr = (T*)x.Address + shape.offset + inputBaseOffset;
                for (long i = 0; i < axisSize; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(basePtr[i * axisStride], zero))
                        count++;
                }

                // Write directly to result buffer using linear index
                resultPtr[outIdx] = count;
            }
        }

    }
}
