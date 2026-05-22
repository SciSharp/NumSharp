using System;
using System.Collections.Generic;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Find the indices of array elements that are non-zero, grouped by element.
        ///     Returns a 2-D <c>int64</c> array of shape <c>(N, ndim)</c> where each row holds
        ///     the coordinates of one non-zero element (C-order traversal).
        ///
        ///     Mirrors NumPy 2.4.2's <c>np.argwhere</c>: equivalent to
        ///     <c>np.transpose(np.nonzero(a))</c> but avoids the per-dimension column
        ///     materialization + transpose by writing flat indices directly into the
        ///     <c>(N, ndim)</c> result buffer.
        ///
        ///     Routing (mirrors <see cref="NonZero"/>):
        ///         * 0-d            → shape <c>(1, 0)</c> for truthy, <c>(0, 0)</c> for falsy.
        ///         * size == 0      → shape <c>(0, ndim)</c>.
        ///         * Contig bool    → IL prescan + count + bit-scan flat indices, then expand.
        ///         * Contig numeric → <see cref="ILKernelGenerator.NonZeroSimdHelper{T}"/> flat scan, then expand.
        ///         * Non-contig     → <see cref="ILKernelGenerator.FindNonZeroStridedHelper{T}"/> per-dim columns,
        ///                              then transpose into the (N, ndim) buffer.
        /// </summary>
        public override unsafe NDArray Argwhere(NDArray nd)
        {
            // 0-d: NumPy promotes via atleast_1d then strips the dim back off with [:, :0].
            // Net effect: shape (1, 0) for truthy, (0, 0) for falsy. Reuse NonZero on the
            // promoted view to dispatch the truthiness check through the existing IL path
            // without re-routing dtype here.
            if (nd.ndim == 0)
            {
                var promoted = np.atleast_1d(nd);
                var nz = NonZero(promoted);
                long n0 = nz[0].size;
                return new NDArray(NPTypeCode.Int64, new Shape(n0, 0), false);
            }

            return NpFunc.Invoke(nd.typecode, ArgwhereDispatch<int>, nd);
        }

        private static NDArray ArgwhereDispatch<T>(NDArray nd) where T : unmanaged
            => argwhere_impl<T>(nd.MakeGeneric<T>());

        private static unsafe NDArray argwhere_impl<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;
            int ndim = x.ndim;

            // size == 0 (e.g. shape (0,3)) returns (0, ndim) per NumPy.
            if (size == 0)
                return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

            if (shape.IsContiguous)
            {
                T* basePtr = (T*)x.Address + shape.offset;

                // Bool-specific IL fast path — same kernels as np.nonzero.
                if (typeof(T) == typeof(bool))
                {
                    byte* maskPtr = (byte*)basePtr;

                    // Prescan: short-circuit on all-false mask (NumPy parity).
                    var prescan = ILKernelGenerator.GetIsAllZeroBoolKernel();
                    if (prescan != null && prescan(maskPtr, size))
                        return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                    var countKernel = ILKernelGenerator.GetNonZeroCountBoolKernel();
                    var scanKernel = ILKernelGenerator.GetNonZeroFlatBoolKernel();
                    if (countKernel != null && scanKernel != null)
                    {
                        long count = countKernel(maskPtr, size);
                        if (count == 0)
                            return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                        var result = new NDArray(NPTypeCode.Int64, new Shape(count, ndim), false);
                        long* resPtr = (long*)result.Storage.Address;

                        // ndim == 1: flat index IS the coord. Scan directly into the (count, 1)
                        // result buffer — no intermediate, no expand pass.
                        if (ndim == 1)
                        {
                            scanKernel(maskPtr, resPtr, size);
                            return result;
                        }

                        // ndim > 1: scan into a temp buffer, then expand flat → (i, d) into the
                        // result in a single pass.
                        var buffer = new long[count];
                        fixed (long* bufPtr = buffer)
                            scanKernel(maskPtr, bufPtr, size);

                        ExpandFlatToArgwhere(buffer, count, ndim, shape.dimensions, resPtr);
                        return result;
                    }

                    // SIMD-unavailable fallback (VectorBits == 0): generic helper.
                    var flatBoolFb = new List<long>(EstimateNonZeroCapacity(size));
                    ILKernelGenerator.NonZeroSimdHelper(basePtr, size, flatBoolFb);
                    return BuildArgwhereFromList(flatBoolFb, ndim, shape.dimensions);
                }

                // Non-bool contiguous numeric dtypes: two-pass count + scan, mirroring the
                // bool path. Skips the List<long> growth + pointer-bookkeeping that
                // NonZeroSimdHelper pays per element, dropping the dense case to a tight
                // SIMD scan + pointer store.
                long countT = ILKernelGenerator.NonZeroCountHelper(basePtr, size);
                if (countT == 0)
                    return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                var resultT = new NDArray(NPTypeCode.Int64, new Shape(countT, ndim), false);
                long* resPtrT = (long*)resultT.Storage.Address;

                if (ndim == 1)
                {
                    // 1-D: flat index IS the coord — write the scan output straight into
                    // the (count, 1) result buffer.
                    ILKernelGenerator.NonZeroFlatHelper(basePtr, size, resPtrT);
                    return resultT;
                }

                // ndim > 1: scan into a temp, then expand flat → row-major in one pass.
                var bufferT = new long[countT];
                fixed (long* bufPtr = bufferT)
                    ILKernelGenerator.NonZeroFlatHelper(basePtr, size, bufPtr);

                ExpandFlatToArgwhere(bufferT, countT, ndim, shape.dimensions, resPtrT);
                return resultT;
            }

            // Non-contiguous (sliced / transposed / broadcasted / neg-stride): materialize
            // to a fresh C-contig buffer and route through the SIMD path. Flat indices into
            // the contig buffer map back to user-facing coordinates via the shape dims (not
            // the original strides), so this preserves the C-order traversal semantics
            // NumPy guarantees while picking up the SIMD count + scan kernels.
            //
            // Trades: 1 extra O(N) materialize pass for a 4–6× speedup on the strided/
            // transposed/broadcast cases vs the scalar coord-iter helper. Wins for any
            // non-trivial density; for extreme sparsity the strided helper would short-
            // circuit faster, but the ascontiguousarray cost is bounded by memory bandwidth
            // and stays well under NumPy's per-call overhead on small-to-medium arrays.
            var contig = np.ascontiguousarray(x).MakeGeneric<T>();
            var contigShape = contig.Shape;
            T* contigPtr = (T*)contig.Address + contigShape.offset;

            if (typeof(T) == typeof(bool))
            {
                byte* maskPtr2 = (byte*)contigPtr;

                var prescan2 = ILKernelGenerator.GetIsAllZeroBoolKernel();
                if (prescan2 != null && prescan2(maskPtr2, size))
                    return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                var countKernel2 = ILKernelGenerator.GetNonZeroCountBoolKernel();
                var scanKernel2 = ILKernelGenerator.GetNonZeroFlatBoolKernel();
                if (countKernel2 != null && scanKernel2 != null)
                {
                    long count2 = countKernel2(maskPtr2, size);
                    if (count2 == 0)
                        return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

                    var result2 = new NDArray(NPTypeCode.Int64, new Shape(count2, ndim), false);
                    long* resPtr2 = (long*)result2.Storage.Address;

                    if (ndim == 1)
                    {
                        scanKernel2(maskPtr2, resPtr2, size);
                        return result2;
                    }

                    var buffer2 = new long[count2];
                    fixed (long* bufPtr2 = buffer2)
                        scanKernel2(maskPtr2, bufPtr2, size);

                    ExpandFlatToArgwhere(buffer2, count2, ndim, shape.dimensions, resPtr2);
                    return result2;
                }

                // SIMD unavailable — fall back to the strided helper (correctness only).
                var cols = ILKernelGenerator.FindNonZeroStridedHelper((T*)x.Address, shape.dimensions, shape.strides, shape.offset);
                return BuildArgwhereFromColumns(cols, ndim);
            }

            long countTS = ILKernelGenerator.NonZeroCountHelper(contigPtr, size);
            if (countTS == 0)
                return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

            var resultTS = new NDArray(NPTypeCode.Int64, new Shape(countTS, ndim), false);
            long* resPtrTS = (long*)resultTS.Storage.Address;

            if (ndim == 1)
            {
                ILKernelGenerator.NonZeroFlatHelper(contigPtr, size, resPtrTS);
                return resultTS;
            }

            var bufferTS = new long[countTS];
            fixed (long* bufPtrTS = bufferTS)
                ILKernelGenerator.NonZeroFlatHelper(contigPtr, size, bufPtrTS);

            ExpandFlatToArgwhere(bufferTS, countTS, ndim, shape.dimensions, resPtrTS);
            return resultTS;
        }

        /// <summary>
        ///     Convert a flat-index buffer to the (<paramref name="count"/>, <paramref name="ndim"/>)
        ///     argwhere result, writing row-major into <paramref name="resPtr"/>.
        ///
        ///     Incremental coord advance: <paramref name="flatBuffer"/> is monotonic ascending
        ///     (C-order scan output), so we add the per-element delta to the innermost coord
        ///     and carry into outer dims only on overflow. For dense outputs (delta == 1) this
        ///     is a branch-predictable increment + carry-check — no divmod. For sparse outputs
        ///     where delta is large, the inner-loop divmod still applies but only at carry
        ///     boundaries, not per element. Net: dense argwhere drops from ~10 ms/Kelt (divmod)
        ///     to ~1 ms/Kelt (carry chain).
        /// </summary>
        private static unsafe void ExpandFlatToArgwhere(long[] flatBuffer, long count, int ndim, long[] dims, long* resPtr)
        {
            Span<long> coords = stackalloc long[ndim];
            // Seed coords from the first flat index via a single divmod pass — cheap one-time
            // cost regardless of density.
            {
                long flat = flatBuffer[0];
                long stride = 1;
                Span<long> dimStrides = stackalloc long[ndim];
                dimStrides[ndim - 1] = 1;
                for (int d = ndim - 2; d >= 0; d--)
                    dimStrides[d] = dimStrides[d + 1] * dims[d + 1];
                for (int d = 0; d < ndim; d++)
                {
                    long s = dimStrides[d];
                    coords[d] = flat / s;
                    flat %= s;
                }

                long* row0 = resPtr;
                for (int d = 0; d < ndim; d++)
                    row0[d] = coords[d];
                stride = 1; // suppress "unused" diagnostic when ndim==1
                _ = stride;
            }

            long lastFlat = flatBuffer[0];
            for (long i = 1; i < count; i++)
            {
                long flat = flatBuffer[i];
                long delta = flat - lastFlat;
                lastFlat = flat;

                // Advance the innermost coord by delta, carrying into outer dims as needed.
                // For dense (delta=1), this is a single increment + one carry check per row.
                long innerSize = dims[ndim - 1];
                long newInner = coords[ndim - 1] + delta;
                if (newInner < innerSize)
                {
                    coords[ndim - 1] = newInner;
                }
                else
                {
                    long carry = newInner / innerSize;
                    coords[ndim - 1] = newInner % innerSize;
                    for (int d = ndim - 2; d >= 0 && carry > 0; d--)
                    {
                        long axisSize = dims[d];
                        long sum = coords[d] + carry;
                        if (sum < axisSize)
                        {
                            coords[d] = sum;
                            carry = 0;
                        }
                        else
                        {
                            coords[d] = sum % axisSize;
                            carry = sum / axisSize;
                        }
                    }
                }

                long* row = resPtr + i * ndim;
                for (int d = 0; d < ndim; d++)
                    row[d] = coords[d];
            }
        }

        /// <summary>
        ///     Allocate the (count, ndim) result and expand flat indices held in a List&lt;long&gt;
        ///     (the SIMD-unavailable fallback path). For ndim == 1 the flat indices already are
        ///     the coords, so a single tight copy suffices.
        /// </summary>
        private static unsafe NDArray BuildArgwhereFromList(List<long> flatBuffer, int ndim, long[] dims)
        {
            long count = flatBuffer.Count;
            if (count == 0)
                return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

            var result = new NDArray(NPTypeCode.Int64, new Shape(count, ndim), false);
            long* resPtr = (long*)result.Storage.Address;

            if (ndim == 1)
            {
                for (long i = 0; i < count; i++)
                    resPtr[i] = flatBuffer[(int)i];
                return result;
            }

            Span<long> dimStrides = stackalloc long[ndim];
            dimStrides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                dimStrides[d] = dimStrides[d + 1] * dims[d + 1];

            for (long i = 0; i < count; i++)
            {
                long flat = flatBuffer[(int)i];
                long* row = resPtr + i * ndim;
                for (int d = 0; d < ndim; d++)
                {
                    long s = dimStrides[d];
                    row[d] = flat / s;
                    flat %= s;
                }
            }
            return result;
        }

        /// <summary>
        ///     Transpose per-dimension coordinate columns (the output of <see cref="ILKernelGenerator.FindNonZeroStridedHelper{T}"/>)
        ///     into the row-major (N, ndim) result. Column pointers are hoisted into a local
        ///     so the inner loop reads from cached locals, not array slots.
        /// </summary>
        private static unsafe NDArray BuildArgwhereFromColumns(NDArray<long>[] cols, int ndim)
        {
            long count = cols[0].size;
            if (count == 0)
                return new NDArray(NPTypeCode.Int64, new Shape(0, ndim), false);

            var result = new NDArray(NPTypeCode.Int64, new Shape(count, ndim), false);
            long* resPtr = (long*)result.Storage.Address;

            // Hoist column pointers once. Each cols[d] is a fresh contiguous (count,) NDArray<long>,
            // so the cached address stays valid for the inner loop.
            var colPtrs = new long*[ndim];
            for (int d = 0; d < ndim; d++)
                colPtrs[d] = (long*)cols[d].Address;

            for (long i = 0; i < count; i++)
            {
                long* row = resPtr + i * ndim;
                for (int d = 0; d < ndim; d++)
                    row[d] = colPtrs[d][i];
            }
            return result;
        }
    }
}
