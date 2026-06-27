using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static unsafe void CopyMaskedDispatch<T>(nint arr, nint mask, nint result, long size) where T : unmanaged
            => DirectILKernelGenerator.CopyMaskedElementsHelper((T*)arr, (bool*)mask, (T*)result, size);

        // =====================================================================
        //  Boolean mask GET — NumPy boolean indexing (mapping.c
        //  array_boolean_subscript). The mask matches a leading PREFIX of arr's
        //  shape (mask.ndim <= arr.ndim). For each True position (in C-order) the
        //  trailing sub-tensor arr.shape[mask.ndim:] is gathered. Result shape:
        //  (count_true,) + arr.shape[mask.ndim:].
        //
        //  All three NumPy cases collapse onto ONE NpyIter-driven kernel:
        //    • Full element mask (mask.ndim == arr.ndim)  → block size B = 1.
        //    • Axis-0 / partial   (mask.ndim <  arr.ndim) → B = prod(trailing);
        //      the mask is broadcast across the trailing dims (stride 0) so the
        //      gather reduces to a full element mask. The iterator coalesces the
        //      contiguous trailing run into one inner loop, so a C-contiguous arr
        //      gets ONE bulk block-copy per selected position (see the kernel's
        //      stride-0 fast path) instead of per-element work.
        // =====================================================================

        /// <summary>
        /// Apply a boolean mask to select elements / sub-tensors from an array.
        /// </summary>
        /// <param name="arr">Source array.</param>
        /// <param name="mask">Boolean mask matching a leading prefix of <paramref name="arr"/>'s shape.</param>
        /// <returns>Array of shape (count_true,) + arr.shape[mask.ndim:].</returns>
        public override NDArray BooleanMask(NDArray arr, NDArray mask)
        {
            if (mask.typecode != NPTypeCode.Boolean)
                throw new ArgumentException("Mask must be boolean array", nameof(mask));

            int leadNdim = mask.ndim;

            // Trailing block size B = prod(arr.shape[leadNdim:]) (1 for a full element mask).
            long blockSize = 1;
            for (int d = leadNdim; d < arr.ndim; d++)
                blockSize *= arr.shape[d];

            long trueCount = CountMaskTrue(mask.MakeGeneric<bool>());

            if (trueCount == 0)
                return new NDArray(arr.dtype, BooleanMaskResultShape(0, arr, leadNdim));

            // Flat gather buffer of trueCount*B elements, reshaped to (count,)+trailing below.
            var result = new NDArray(arr.dtype, new Shape(trueCount * blockSize));

            bool simd = DirectILKernelGenerator.Enabled && DirectILKernelGenerator.VectorBits > 0;

            if (blockSize == 1 && simd && arr.Shape.IsContiguous && mask.Shape.IsContiguous)
            {
                // SIMD fast path: contiguous full element mask.
                BooleanMaskSimdFill(arr, mask.MakeGeneric<bool>(), result);
            }
            else
            {
                // General NpyIter gather (layout-aware: honors strides/offset/broadcast).
                NDArray gatherMask = blockSize == 1 ? mask : BroadcastMaskAcrossBlock(mask, arr);
                BooleanMaskGather(arr, gatherMask, result);
            }

            return leadNdim == arr.ndim
                ? result
                : result.reshape(BooleanMaskResultShape(trueCount, arr, leadNdim));
        }

        // =====================================================================
        //  Boolean mask SET — NumPy boolean assignment (mapping.c
        //  array_assign_boolean_subscript). Streams the value into arr through an
        //  NpyIter scatter: walk (arr[READWRITE], mask[READONLY]) in C-order and,
        //  for each True, consume the next block from the value buffer. No
        //  per-row NDArray allocation, no nonzero index materialization.
        // =====================================================================

        /// <summary>
        /// Assign <paramref name="value"/> into <paramref name="arr"/> where the
        /// boolean <paramref name="mask"/> (matching a leading prefix of arr's
        /// shape) is True. <paramref name="value"/> broadcasts to the selection
        /// shape (count_true,) + arr.shape[mask.ndim:] (NumPy semantics).
        /// </summary>
        public override void BooleanMaskSet(NDArray arr, NDArray mask, NDArray value)
        {
            if (mask.typecode != NPTypeCode.Boolean)
                throw new ArgumentException("Mask must be boolean array", nameof(mask));

            int leadNdim = mask.ndim;

            long trueCount = CountMaskTrue(mask.MakeGeneric<bool>());
            if (trueCount == 0)
            {
                // The selection is EMPTY ((0,)+trailing). NumPy still requires the value to
                // broadcast to that indexing-result shape — a non-scalar that cannot broadcast
                // raises ValueError (e.g. arr[allFalseMask] = [93,1,39] into a (0,4) selection);
                // it is NOT silently a no-op. A scalar (size 1) always broadcasts.
                if (value.size != 1)
                {
                    var emptySel = BooleanMaskResultShape(0, arr, leadNdim);
                    string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
                    try { np.broadcast_to(value, emptySel); }
                    catch (IncorrectShapeException)
                    {
                        throw new ValueError($"shape mismatch: value array of shape {Tup(value.Shape.dimensions)} " +
                                             $"could not be broadcast to indexing result of shape {Tup(emptySel.dimensions)}");
                    }
                }
                return;
            }

            // A scalar (size-1) value splats to every selected slot from a clean
            // 1-element buffer; otherwise broadcast to the selection shape and
            // materialize a contiguous arr-dtype buffer consumed in C-order
            // (== NumPy selection order). Both forms are offset-0 contiguous.
            bool splat = value.size == 1;
            NDArray vflat = splat
                ? MaterializeMaskValue(value, value.Shape, arr)
                : MaterializeMaskValue(value, BooleanMaskResultShape(trueCount, arr, leadNdim), arr);

            NDArray scatterMask = leadNdim == arr.ndim ? mask : BroadcastMaskAcrossBlock(mask, arr);
            BooleanMaskScatter(arr, scatterMask, vflat, splat);
        }

        // ---------------------------------------------------------------------
        //  Shared helpers
        // ---------------------------------------------------------------------

        /// <summary>Count True values in a mask, layout-aware (SIMD when contiguous, else NpyIter).</summary>
        private static unsafe long CountMaskTrue(NDArray<bool> mask)
        {
            if (DirectILKernelGenerator.Enabled && DirectILKernelGenerator.VectorBits > 0 && mask.Shape.IsContiguous)
            {
                // Offset-hardened base (offset is folded into Address for contiguous
                // arrays today, so this is a no-op now and a guard against future change).
                bool* basePtr = (bool*)mask.Address + mask.Shape.offset;
                return DirectILKernelGenerator.CountTrueSimdHelper(basePtr, mask.size);
            }

            using var maskIter = NpyIterRef.New(mask, NpyIterGlobalFlags.EXTERNAL_LOOP);
            return maskIter.ExecuteReducing<CountNonZeroKernel<bool>, long>(default, 0L);
        }

        /// <summary>Result/selection shape (count,) + arr.shape[leadNdim:].</summary>
        private static Shape BooleanMaskResultShape(long count, NDArray arr, int leadNdim)
        {
            int trailing = arr.ndim - leadNdim;
            var dims = new long[1 + trailing];
            dims[0] = count;
            for (int i = 0; i < trailing; i++)
                dims[i + 1] = arr.shape[leadNdim + i];
            return new Shape(dims);
        }

        /// <summary>
        /// Reshape the mask to mask.shape + [1]*(arr.ndim-mask.ndim) and broadcast
        /// it to arr.shape — a read-only stride-0 trailing view that turns a
        /// partial/axis-0 mask into a full element mask over arr.
        /// </summary>
        private static NDArray BroadcastMaskAcrossBlock(NDArray mask, NDArray arr)
        {
            int trailing = arr.ndim - mask.ndim;
            var reshaped = new long[mask.ndim + trailing];
            for (int i = 0; i < mask.ndim; i++)
                reshaped[i] = mask.shape[i];
            for (int i = 0; i < trailing; i++)
                reshaped[mask.ndim + i] = 1;
            return np.broadcast_to(mask.reshape(reshaped), arr.Shape);
        }

        /// <summary>
        /// Broadcast <paramref name="value"/> to <paramref name="target"/> (raises the
        /// NumPy broadcast error on mismatch) and materialize a contiguous, C-order,
        /// arr-dtype buffer of target.size elements for the scatter cursor to consume.
        /// </summary>
        private static NDArray MaterializeMaskValue(NDArray value, Shape target, NDArray arr)
        {
            var flat = np.broadcast_to(value, target).flatten(); // contiguous C-order copy
            if (flat.typecode != arr.typecode)
                flat = flat.astype(arr.typecode);
            return flat;
        }

        /// <summary>SIMD-optimized fill for a contiguous full element mask.</summary>
        private unsafe void BooleanMaskSimdFill(NDArray arr, NDArray<bool> mask, NDArray result)
        {
            long size = arr.size;
            // Offset-hardened base pointers (offset == 0 for contiguous arrays today).
            nint arrBase = (nint)((byte*)arr.Address + arr.Shape.offset * arr.dtypesize);
            nint maskBase = (nint)((bool*)mask.Address + mask.Shape.offset);
            NpFunc.Invoke(arr.typecode, CopyMaskedDispatch<int>, arrBase, maskBase, (nint)result.Address, size);
        }

        /// <summary>
        /// NpyIter gather: copy arr elements where the (possibly trailing-broadcast)
        /// mask is True into the flat <paramref name="result"/>. NPY_CORDER forces
        /// logical C-order traversal (NumPy boolean-indexing semantics).
        /// </summary>
        private unsafe void BooleanMaskGather(NDArray arr, NDArray mask, NDArray result)
        {
            using var iter = NpyIterRef.MultiNew(
                2, new[] { arr, mask },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.NO_BROADCAST, NpyIterPerOpFlags.READONLY });

            var accum = new BooleanMaskGatherAccumulator
            {
                DestPtr = (IntPtr)result.Address,
                ElemSize = arr.dtypesize,
                DestIdx = 0,
            };
            iter.ExecuteReducing<BooleanMaskGatherKernel, BooleanMaskGatherAccumulator>(default, accum);
        }

        /// <summary>
        /// NpyIter scatter: write the value buffer into arr where the (possibly
        /// trailing-broadcast) mask is True, in C-order. arr is READWRITE so any
        /// buffered masked-off slots survive copy-out.
        /// </summary>
        private unsafe void BooleanMaskScatter(NDArray arr, NDArray mask, NDArray value, bool splat)
        {
            using var iter = NpyIterRef.MultiNew(
                2, new[] { arr, mask },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.NO_BROADCAST, NpyIterPerOpFlags.READONLY });

            var accum = new BooleanMaskScatterAccumulator
            {
                ValPtr = (IntPtr)value.Address,
                ValIdx = 0,
                ElemSize = arr.dtypesize,
                Splat = splat ? 1 : 0,
            };
            iter.ExecuteReducing<BooleanMaskScatterKernel, BooleanMaskScatterAccumulator>(default, accum);
        }

        // ---------------------------------------------------------------------
        //  Gather kernel
        // ---------------------------------------------------------------------

        /// <summary>
        /// Accumulator threading the destination byte pointer and write cursor
        /// through the multi-op gather loop.
        /// </summary>
        private struct BooleanMaskGatherAccumulator
        {
            public IntPtr DestPtr;
            public long DestIdx;
            public int ElemSize;
        }

        /// <summary>
        /// Inner loop: for each position, if mask is true, copy the arr element
        /// into result[destIdx] and increment destIdx.
        /// A stride-0 mask (trailing-broadcast partial/axis-0 mask) means the whole
        /// chunk is uniformly selected/skipped — handled by a single bulk block-copy.
        /// The element-wise path is specialized on element size to avoid
        /// Buffer.MemoryCopy per-element overhead for the small fixed sizes that
        /// cover all 15 NumSharp dtypes (1, 2, 4, 8, 16 bytes).
        /// </summary>
        private readonly struct BooleanMaskGatherKernel : INpyReducingInnerLoop<BooleanMaskGatherAccumulator>
        {
            public unsafe bool Execute(void** dataptrs, long* strides, long count, ref BooleanMaskGatherAccumulator accum)
            {
                byte* srcPtr = (byte*)dataptrs[0];
                byte* maskPtr = (byte*)dataptrs[1];
                long srcStride = strides[0];
                long maskStride = strides[1];
                byte* destBase = (byte*)accum.DestPtr;
                long destIdx = accum.DestIdx;
                int elemSize = accum.ElemSize;

                // Broadcast mask (stride 0): the whole run is one selected/skipped block.
                if (maskStride == 0)
                {
                    if (*(bool*)maskPtr)
                    {
                        byte* dest = destBase + destIdx * elemSize;
                        if (srcStride == elemSize)
                        {
                            long bytes = count * elemSize;
                            System.Buffer.MemoryCopy(srcPtr, dest, bytes, bytes);
                        }
                        else
                        {
                            for (long i = 0; i < count; i++)
                                System.Buffer.MemoryCopy(srcPtr + i * srcStride, dest + i * elemSize, elemSize, elemSize);
                        }
                        destIdx += count;
                    }
                    accum.DestIdx = destIdx;
                    return true;
                }

                switch (elemSize)
                {
                    case 1:
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                                *(destBase + destIdx++) = *(srcPtr + i * srcStride);
                        }
                        break;
                    case 2:
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                                *((short*)destBase + destIdx++) = *(short*)(srcPtr + i * srcStride);
                        }
                        break;
                    case 4:
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                                *((int*)destBase + destIdx++) = *(int*)(srcPtr + i * srcStride);
                        }
                        break;
                    case 8:
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                                *((long*)destBase + destIdx++) = *(long*)(srcPtr + i * srcStride);
                        }
                        break;
                    case 16:
                        // 16 bytes covers Complex (2 × double) and Decimal — copy as two longs.
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                            {
                                long* d = (long*)destBase + destIdx * 2;
                                long* s = (long*)(srcPtr + i * srcStride);
                                d[0] = s[0];
                                d[1] = s[1];
                                destIdx++;
                            }
                        }
                        break;
                    default:
                        // Any unexpected element size falls back to the byte copy.
                        for (long i = 0; i < count; i++)
                        {
                            if (*(bool*)(maskPtr + i * maskStride))
                            {
                                System.Buffer.MemoryCopy(
                                    srcPtr + i * srcStride,
                                    destBase + destIdx * elemSize,
                                    elemSize, elemSize);
                                destIdx++;
                            }
                        }
                        break;
                }

                accum.DestIdx = destIdx;
                return true;
            }
        }

        // ---------------------------------------------------------------------
        //  Scatter kernel
        // ---------------------------------------------------------------------

        /// <summary>
        /// Accumulator threading the value buffer pointer and read cursor through
        /// the multi-op scatter loop. <see cref="Splat"/> != 0 writes Val[0] to
        /// every selected slot (scalar value) without advancing the cursor.
        /// </summary>
        private struct BooleanMaskScatterAccumulator
        {
            public IntPtr ValPtr;
            public long ValIdx;
            public int ElemSize;
            public int Splat;
        }

        /// <summary>
        /// Inner loop: for each position, if mask is true, write the next value
        /// block (or the splat scalar) into arr[i]. Mirrors the gather: a stride-0
        /// mask copies a whole block at once; the element-wise path is specialized
        /// on element size.
        /// </summary>
        private readonly struct BooleanMaskScatterKernel : INpyReducingInnerLoop<BooleanMaskScatterAccumulator>
        {
            public unsafe bool Execute(void** dataptrs, long* strides, long count, ref BooleanMaskScatterAccumulator accum)
            {
                byte* dstPtr = (byte*)dataptrs[0];
                byte* maskPtr = (byte*)dataptrs[1];
                long dstStride = strides[0];
                long maskStride = strides[1];
                byte* valBase = (byte*)accum.ValPtr;
                long valIdx = accum.ValIdx;
                int elemSize = accum.ElemSize;
                bool splat = accum.Splat != 0;

                // Broadcast mask (stride 0): the whole run is one selected/skipped block.
                if (maskStride == 0)
                {
                    if (*(bool*)maskPtr)
                    {
                        if (splat)
                        {
                            for (long i = 0; i < count; i++)
                                System.Buffer.MemoryCopy(valBase, dstPtr + i * dstStride, elemSize, elemSize);
                        }
                        else
                        {
                            byte* v = valBase + valIdx * elemSize;
                            if (dstStride == elemSize)
                            {
                                long bytes = count * elemSize;
                                System.Buffer.MemoryCopy(v, dstPtr, bytes, bytes);
                            }
                            else
                            {
                                for (long i = 0; i < count; i++)
                                    System.Buffer.MemoryCopy(v + i * elemSize, dstPtr + i * dstStride, elemSize, elemSize);
                            }
                            valIdx += count;
                        }
                    }
                    accum.ValIdx = valIdx;
                    return true;
                }

                switch (elemSize)
                {
                    case 1:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                                *(dstPtr + i * dstStride) = *(valBase + (splat ? 0 : valIdx++));
                        break;
                    case 2:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                                *(short*)(dstPtr + i * dstStride) = *((short*)valBase + (splat ? 0 : valIdx++));
                        break;
                    case 4:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                                *(int*)(dstPtr + i * dstStride) = *((int*)valBase + (splat ? 0 : valIdx++));
                        break;
                    case 8:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                                *(long*)(dstPtr + i * dstStride) = *((long*)valBase + (splat ? 0 : valIdx++));
                        break;
                    case 16:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                            {
                                long* d = (long*)(dstPtr + i * dstStride);
                                long* s = (long*)valBase + (splat ? 0 : valIdx++) * 2;
                                d[0] = s[0];
                                d[1] = s[1];
                            }
                        break;
                    default:
                        for (long i = 0; i < count; i++)
                            if (*(bool*)(maskPtr + i * maskStride))
                            {
                                System.Buffer.MemoryCopy(
                                    valBase + (splat ? 0 : valIdx++) * elemSize,
                                    dstPtr + i * dstStride,
                                    elemSize, elemSize);
                            }
                        break;
                }

                accum.ValIdx = valIdx;
                return true;
            }
        }
    }
}
