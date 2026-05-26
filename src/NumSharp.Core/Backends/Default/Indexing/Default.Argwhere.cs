using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.argwhere</c> — returns the (N, ndim) int64 array of coordinates of
        ///     non-zero elements, traversed in C-order. Equivalent to
        ///     <c>np.transpose(np.nonzero(a))</c> with NumPy's 0-d special case
        ///     (truthy → (1,0), falsy → (0,0)).
        ///
        ///     <para>
        ///     <b>Implementation:</b> three IL-emitted kernels keyed off the element type
        ///     (<see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereFlatKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereExpandKernel"/>). The runtime call
        ///     site has zero <c>typeof(T)</c> branches: it looks the kernels up in the
        ///     per-dtype <see cref="System.Collections.Concurrent.ConcurrentDictionary{Type,Object}"/>
        ///     cache and invokes them. Every loop (SIMD body, scalar tail, coord-expand
        ///     carry chain) lives inside the emitted IL.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Two-pass pre-size-then-fill:</b> a SIMD popcount sizes the result
        ///     exactly, the SIMD bit-scan writes directly into the typed result buffer
        ///     for ndim==1 (no temp), and into a temp <c>long[]</c> for ndim&gt;1 which
        ///     the IL expand kernel walks once. The two-pass design avoids the
        ///     "allocate max-size temp" pathology that a one-pass upper-bound design
        ///     would pay on the memcpy back to a properly-sized result (~equivalent to
        ///     the count pass on dense outputs, far worse on dense large arrays).
        ///     </para>
        ///
        ///     <para>
        ///     Routing:
        ///     <list type="bullet">
        ///       <item>0-d → shape <c>(1,0)</c> truthy / <c>(0,0)</c> falsy via
        ///             <c>atleast_1d</c> + nonzero count.</item>
        ///       <item>size == 0 → shape <c>(0, ndim)</c>.</item>
        ///       <item>Contiguous → IL count + IL scan + (ndim==1 ? direct write : IL expand).</item>
        ///       <item>Non-contiguous → materialize via <c>ascontiguousarray</c> then
        ///             same path as contig.</item>
        ///     </list>
        ///     </para>
        /// </summary>
        public override unsafe NDArray Argwhere(NDArray nd)
        {
            // 0-d: NumPy promotes via atleast_1d then strips the dim with [:, :0].
            // Net result is (1, 0) for truthy, (0, 0) for falsy. Route the truthiness
            // check through NonZero so we don't add yet another dtype dispatch here.
            if (nd.ndim == 0)
            {
                var promoted = np.atleast_1d(nd);
                var nz = NonZero(promoted);
                long n0 = nz[0].size;
                return new NDArray(NPTypeCode.Int64, new Shape(n0, 0), false);
            }

            return ArgwhereContiguousOrMaterialize(nd);
        }

        /// <summary>
        ///     Single dispatch path for all contig and non-contig inputs. Non-contig is
        ///     materialized to a fresh C-contig buffer first so the same IL kernels apply
        ///     to every layout. Flat indices into the contig buffer map back to user-facing
        ///     coords through the shape dims (C-order traversal preserved).
        /// </summary>
        private static unsafe NDArray ArgwhereContiguousOrMaterialize(NDArray nd)
        {
            int ndim = nd.ndim;
            long size = nd.size;

            // Empty input → shape (0, ndim). Includes shape (0,3), (2,0,4), …
            if (size == 0)
                return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

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
                throw new NotSupportedException($"np.argwhere: no IL kernel available for {nd.dtype.Name}");

            try
            {
                long count = countKernel(basePtr, size);
                if (count == 0)
                    return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                var result = new NDArray(NPTypeCode.Int64, new Shape(count, ndim), false);
                long* resPtr = (long*)result.Storage.Address;

                // ndim == 1: flat index IS the coord — scan straight into the result buffer.
                if (ndim == 1)
                {
                    flatKernel(basePtr, size, resPtr);
                    return result;
                }

                // ndim > 1: scan into a temp flat buffer then expand via IL kernel.
                var flatBuffer = new long[count];
                fixed (long* flatPtr = flatBuffer)
                fixed (long* dimsPtr = sourceShape.dimensions)
                {
                    flatKernel(basePtr, size, flatPtr);

                    // Pre-compute dim strides for the expand kernel: dimStrides[ndim-1] = 1;
                    // dimStrides[d] = dimStrides[d+1] * dims[d+1]. Cheap O(ndim) prologue.
                    Span<long> dimStrides = stackalloc long[ndim];
                    dimStrides[ndim - 1] = 1;
                    for (int d = ndim - 2; d >= 0; d--)
                        dimStrides[d] = dimStrides[d + 1] * sourceShape.dimensions[d + 1];

                    fixed (long* dimStridesPtr = dimStrides)
                    {
                        var expandKernel = DirectILKernelGenerator.GetArgwhereExpandKernel();
                        if (expandKernel == null)
                            throw new NotSupportedException("np.argwhere: expand IL kernel unavailable");

                        expandKernel(flatPtr, count, dimsPtr, dimStridesPtr, ndim, resPtr);
                    }
                }

                return result;
            }
            finally
            {
                // ARC: source.Storage.Address is an unmanaged pointer that does NOT
                // keep `materialized` GC-alive. Under repeated argwhere() calls on a
                // non-contig input, the JIT can decide `materialized` is dead after
                // basePtr is computed — then the result-NDArray allocations below
                // trigger GC, the freshly materialized array's UnmanagedStorage gets
                // its refcount finalized to zero, and the buffer behind basePtr is
                // freed mid-IL-scan (the np.nonzero bench harness reproduces this
                // exact pattern as a Release-mode AccessViolationException; argwhere
                // shares the same fix preventively).
                //
                // Explicit Dispose() is the established ARC-release pattern in this
                // codebase (see commits 392529f2, 294d4329) — it forces the JIT to
                // keep `materialized` rooted until the call site here. For the contig
                // fast path materialized is null and Dispose is skipped.
                materialized?.Dispose();
            }
        }
    }
}
