using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static long CountNonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => count_nonzero<T>(nd.MakeGeneric<T>());

        private static void CountNonZeroAxisDispatch<T>(NDArray nd, NDArray result, int axis) where T : unmanaged
            => count_nonzero_axis<T>(nd.MakeGeneric<T>(), result, axis);

        /// <summary>
        ///     <c>np.nonzero</c> — returns a tuple of <c>ndim</c> int64 arrays of length
        ///     <c>N</c>, containing the per-dim coordinates of non-zero elements in C-order.
        ///
        ///     <para>
        ///     <b>Implementation:</b> three IL-emitted kernels keyed off the element type
        ///     (<see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereFlatKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetNonZeroPerDimKernel"/>). The runtime call
        ///     site has zero <c>typeof(T)</c> branches: it looks the kernels up in the
        ///     per-dtype <see cref="System.Collections.Concurrent.ConcurrentDictionary{Type,Object}"/>
        ///     cache and invokes them. Every loop (SIMD popcount, SIMD bit-scan, coord
        ///     expand + carry chain) lives inside the emitted IL.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Two-pass pre-size-then-fill:</b> the SIMD popcount sizes the result
        ///     exactly, the SIMD bit-scan writes flat indices either straight into the
        ///     ndim==1 result buffer or into a temp <c>long[]</c> which the IL per-dim
        ///     expand kernel walks once to emit the per-axis columns. Mirrors the design
        ///     of <see cref="DefaultEngine.Argwhere"/>.
        ///     </para>
        ///
        ///     <para>
        ///     Routing:
        ///     <list type="bullet">
        ///       <item>0-d → promote via <c>atleast_1d</c>, recurse
        ///             (truthy→<c>([0],)</c>, falsy→<c>([],)</c>).</item>
        ///       <item>size == 0 → tuple of <c>ndim</c> empty int64 arrays.</item>
        ///       <item>Contiguous → IL count + IL flat-scan + (ndim==1 ? direct write : IL per-dim expand).</item>
        ///       <item>Non-contiguous → materialize via <c>ascontiguousarray</c> then
        ///             same path as contig.</item>
        ///     </list>
        ///     </para>
        /// </summary>
        public override unsafe NDArray<long>[] NonZero(NDArray nd)
        {
            // Boundary scope: the yielded tuple is the column VIEWS; the shared multi-index base
            // matrix (and the 0-d promotion view) are reclaimed at exit — ARC keeps the buffer
            // alive through the yielded views until the caller releases them.
            using var scope = NDScope.Open();

            // 0-d: NumPy 2.4 raises ValueError, but its own error message suggests
            // `np.atleast_1d(scalar).nonzero()`, which is exactly the result our
            // historical implementation has always produced. Preserve that semantic
            // (otherwise this becomes a breaking-change PR for downstream callers).
            if (nd.ndim == 0)
                return scope.Returns(NonZero(np.atleast_1d(nd)));

            int ndim = nd.ndim;
            long size = nd.size;

            // Empty input → tuple of `ndim` empty int64 arrays.
            // Covers shape (0,), (0, 3), (2, 0, 4), ….
            if (size == 0)
                return scope.Returns(MakeEmptyNonZeroResult(ndim));

            // Materialize non-contig to C-contig. For contig inputs we read from `nd`
            // directly (no copy); the local `materialized` is only set on the non-contig
            // branch and disposed in `finally` — see the ARC note at the bottom of the
            // method for why the explicit Dispose matters here.
            NDArray materialized = null;
            NDArray source = nd;
            if (!nd.Shape.IsContiguous)
            {
                materialized = np.ascontiguousarray(nd);
                source = materialized;
            }
            var sourceShape = source.Shape;

            byte* basePtr = (byte*)source.Storage.Address + sourceShape.offset * nd.dtypesize;

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(nd.dtype);
            var flatKernel = DirectILKernelGenerator.GetArgwhereFlatKernel(nd.dtype);
            if (countKernel == null || flatKernel == null)
                throw new NotSupportedException($"np.nonzero: no IL kernel available for {nd.dtype.Name}");

            try
            {
                long count = countKernel(basePtr, size);
                if (count == 0)
                    return scope.Returns(MakeEmptyNonZeroResult(ndim));

                // NumPy's PyArray_Nonzero fills ONE (count, ndim) C-order multi-index buffer and
                // the tuple entries are its COLUMNS — strided views (shape (count,), byte stride
                // ndim*8, owndata=False) all sharing that base (probed 2.4.2: tuple[j].base is one
                // (count, ndim) array; ndim==1 collapses to a unit-stride contiguous view of a
                // (count, 1) base). The base matrix is byte-identical to np.argwhere's result, so
                // the ndim>1 fill drives the SAME row-expand IL kernel (one sequential write
                // stream — better locality than ndim separate column streams).
                var multiIndex = new NDArray(NPTypeCode.Int64, new Shape(count, ndim), false);
                long* resPtr = (long*)multiIndex.Storage.Address;

                // ndim == 1: the (count, 1) matrix's single column IS the flat index list —
                // scan straight into the base buffer.
                if (ndim == 1)
                {
                    flatKernel(basePtr, size, resPtr);
                    return scope.Returns(MakeNonZeroColumnViews(multiIndex, count, ndim));
                }

                // ndim > 1: scan into a temp flat buffer then row-expand into the shared matrix.
                var flatBuffer = new long[count];
                fixed (long* flatPtr = flatBuffer)
                fixed (long* dimsPtr = sourceShape.dimensions)
                {
                    flatKernel(basePtr, size, flatPtr);

                    // Pre-compute dim strides for the expand kernel:
                    //   dimStrides[ndim-1] = 1
                    //   dimStrides[d]      = dimStrides[d+1] * dims[d+1]
                    // Cheap O(ndim) prologue. Mirrors the argwhere expand-kernel prologue.
                    Span<long> dimStrides = stackalloc long[ndim];
                    dimStrides[ndim - 1] = 1;
                    for (int d = ndim - 2; d >= 0; d--)
                        dimStrides[d] = dimStrides[d + 1] * sourceShape.dimensions[d + 1];

                    fixed (long* dimStridesPtr = dimStrides)
                    {
                        var expandKernel = DirectILKernelGenerator.GetArgwhereExpandKernel();
                        if (expandKernel == null)
                            throw new NotSupportedException("np.nonzero: row-expand IL kernel unavailable");

                        expandKernel(flatPtr, count, dimsPtr, dimStridesPtr, ndim, resPtr);
                    }
                }

                return scope.Returns(MakeNonZeroColumnViews(multiIndex, count, ndim));
            }
            finally
            {
                // ARC: source.Storage.Address is an unmanaged pointer that does NOT
                // keep `materialized` GC-alive. Under repeated nonzero() calls on a
                // non-contig input, the JIT can decide `materialized` is dead after
                // basePtr is computed — then the result-NDArray allocations below
                // trigger GC, the freshly materialized array's UnmanagedStorage gets
                // its refcount finalized to zero, and the buffer behind basePtr is
                // freed mid-IL-scan (the bench harness reproduces this as a Release-
                // mode AccessViolationException in IL_ArgwhereFlat_*).
                //
                // Explicit Dispose() is the established ARC-release pattern in this
                // codebase (see commits 392529f2, 294d4329) — it forces the JIT to
                // keep `materialized` rooted until the call site here. For the contig
                // fast path materialized is null and Dispose is skipped.
                materialized?.Dispose();
            }
        }

        private static NDArray<long>[] MakeEmptyNonZeroResult(int ndim)
        {
            // NumPy: even the empty result is columns of a shared (0, ndim) multi-index buffer —
            // each tuple entry keeps byte stride ndim*8 and owndata=False (probed 2.4.2).
            return MakeNonZeroColumnViews(
                new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false), 0, ndim);
        }

        /// <summary>
        ///     Wrap the shared (count, ndim) multi-index matrix as NumPy's nonzero tuple: entry
        ///     <c>d</c> is the d-th COLUMN — a strided view (shape (count,), element stride ndim,
        ///     element offset d) whose storage aliases the matrix, so <c>owndata=False</c> and all
        ///     entries report one shared base, exactly like <c>PyArray_Nonzero</c>'s
        ///     <c>ret[..., j]</c> views. ndim==1 collapses to a unit-stride view (C=F-contiguous),
        ///     matching NumPy's (count, 1) base there; size-0 views keep the stride/offset pattern
        ///     (never dereferenced).
        /// </summary>
        private static NDArray<long>[] MakeNonZeroColumnViews(NDArray multiIndex, long count, int ndim)
        {
            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
            {
                var view = new Shape(new long[] { count }, new long[] { ndim }, d, count * (long)ndim);
                result[d] = new NDArray<long>(multiIndex.Storage.Alias(view)) { TensorEngine = multiIndex.TensorEngine };
            }

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
        ///
        /// Contig fast path reuses <see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>
        /// which is the SAME SIMD popcount kernel used by <c>np.nonzero</c>'s pre-size
        /// pass: load a Vector&lt;T&gt;, compare-ne-zero, ExtractMostSignificantBits,
        /// PopCount the inverted mask. The earlier scalar
        /// <see cref="EqualityComparer{T}.Default.Equals"/> per-element loop was a
        /// 109× regression vs NumPy (2.1 ms vs 19 µs on 1 M bool).
        /// </summary>
        private static unsafe long count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;

            if (shape.IsContiguous)
            {
                var ilKernel = DirectILKernelGenerator.GetArgwhereCountKernel(typeof(T));
                if (ilKernel != null)
                {
                    // Sliced views: Address ignores shape.offset; advance manually so
                    // the kernel sees the live element window.
                    byte* basePtr = (byte*)x.Address + shape.offset * sizeof(T);
                    return ilKernel(basePtr, size);
                }

                // Fallback (only if IL kernel generation is disabled).
                T* ptr = (T*)x.Address + shape.offset;
                T zero = default;
                long count = 0;
                for (long i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
                return count;
            }

            // Strided path: use NDIter for layout-aware traversal.
            using var iter = NDIterRef.New(x, NDIterGlobalFlags.EXTERNAL_LOOP);
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
