using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Buffer management for NpyIter.
    /// Handles allocation, copy-in, and copy-out of iteration buffers.
    /// </summary>
    public static unsafe class NpyIterBufferManager
    {
        /// <summary>
        /// Default buffer size (number of elements).
        /// </summary>
        public const long DefaultBufferSize = 8192;

        /// <summary>
        /// Required alignment for SIMD operations.
        /// </summary>
        public const int Alignment = 64;  // Cache line size, good for AVX-512

        /// <summary>
        /// Allocate aligned buffer for an operand.
        /// </summary>
        public static void* AllocateAligned(long elements, NPTypeCode dtype)
        {
            long bytes = elements * InfoOf.GetSize(dtype);
            return NativeMemory.AlignedAlloc((nuint)bytes, Alignment);
        }

        /// <summary>
        /// Free aligned buffer.
        /// </summary>
        public static void FreeAligned(void* buffer)
        {
            if (buffer != null)
                NativeMemory.AlignedFree(buffer);
        }

        /// <summary>
        /// Determine optimal buffer size based on array sizes and cache.
        /// </summary>
        public static long DetermineBufferSize(ref NpyIterState state, long requestedSize)
        {
            if (requestedSize > 0)
                return requestedSize;

            // Use L2 cache size heuristic
            const long L2CacheSize = 256 * 1024;  // 256 KB

            long totalElementSize = 0;
            for (int op = 0; op < state.NOp; op++)
            {
                totalElementSize += state.GetElementSize(op);
            }

            if (totalElementSize == 0)
                return DefaultBufferSize;

            // Target: buffers fit in L2 cache
            long maxElements = L2CacheSize / totalElementSize;

            // Round down to SIMD vector multiple
            int vectorSize = 32;  // AVX2
            maxElements = (maxElements / vectorSize) * vectorSize;

            return Math.Max(vectorSize, Math.Min(maxElements, DefaultBufferSize));
        }

        /// <summary>
        /// Allocate buffers for all operands that need buffering.
        /// </summary>
        public static bool AllocateBuffers(ref NpyIterState state, long bufferSize)
        {
            if (bufferSize <= 0)
                bufferSize = DetermineBufferSize(ref state, 0);

            state.BufferSize = bufferSize;

            for (int op = 0; op < state.NOp; op++)
            {
                var opFlags = state.GetOpFlags(op);
                var dtype = state.GetOpDType(op);

                // Skip if operand doesn't need buffering
                if ((opFlags & NpyIterOpFlags.BUFNEVER) != 0)
                    continue;

                // Check if operand needs buffering (non-contiguous or needs cast)
                if ((opFlags & (NpyIterOpFlags.CAST | NpyIterOpFlags.CONTIG)) != 0 ||
                    !IsOperandContiguous(ref state, op))
                {
                    var buffer = AllocateAligned(bufferSize, dtype);
                    if (buffer == null)
                    {
                        // Cleanup already allocated buffers
                        FreeBuffers(ref state);
                        return false;
                    }

                    state.SetBuffer(op, buffer);
                    state.BufStrides[op] = state.GetElementSize(op);
                }
            }

            return true;
        }

        /// <summary>
        /// Free all allocated buffers.
        /// </summary>
        public static void FreeBuffers(ref NpyIterState state)
        {
            for (int op = 0; op < state.NOp; op++)
            {
                var buffer = state.GetBuffer(op);
                if (buffer != null)
                {
                    FreeAligned(buffer);
                    state.SetBuffer(op, null);
                }
            }
        }

        /// <summary>
        /// Check if an operand is contiguous in the current iteration space.
        /// </summary>
        private static bool IsOperandContiguous(ref NpyIterState state, int op)
        {
            if (state.NDim == 0)
                return true;

            long expected = 1;

            // Access dynamically allocated arrays directly (not fixed arrays)
            var shape = state.Shape;
            var strides = state.Strides;
            int stridesNDim = state.StridesNDim;

            for (int axis = state.NDim - 1; axis >= 0; axis--)
            {
                long dim = shape[axis];
                if (dim == 0)
                    return true;

                long stride = strides[op * stridesNDim + axis];

                if (dim != 1)
                {
                    if (stride != expected)
                        return false;
                    expected *= dim;
                }
            }

            return true;
        }

        /// <summary>
        /// Copy data from operand to buffer (strided to contiguous).
        /// If operand needs casting, performs type conversion during copy.
        /// Runtime dtype dispatch version - handles any NumSharp dtype.
        /// </summary>
        public static void CopyToBuffer(ref NpyIterState state, int op, long count)
        {
            // Check if casting is needed
            if (state.NeedsCast(op))
            {
                CopyToBufferWithCast(ref state, op, count);
                return;
            }

            // No casting - use same-type copy
            var dtype = state.GetOpDType(op);

            switch (dtype)
            {
                case NPTypeCode.Boolean: CopyToBuffer<bool>(ref state, op, count); break;
                case NPTypeCode.Byte: CopyToBuffer<byte>(ref state, op, count); break;
                case NPTypeCode.Int16: CopyToBuffer<short>(ref state, op, count); break;
                case NPTypeCode.UInt16: CopyToBuffer<ushort>(ref state, op, count); break;
                case NPTypeCode.Int32: CopyToBuffer<int>(ref state, op, count); break;
                case NPTypeCode.UInt32: CopyToBuffer<uint>(ref state, op, count); break;
                case NPTypeCode.Int64: CopyToBuffer<long>(ref state, op, count); break;
                case NPTypeCode.UInt64: CopyToBuffer<ulong>(ref state, op, count); break;
                case NPTypeCode.Single: CopyToBuffer<float>(ref state, op, count); break;
                case NPTypeCode.Double: CopyToBuffer<double>(ref state, op, count); break;
                case NPTypeCode.Decimal: CopyToBuffer<decimal>(ref state, op, count); break;
                case NPTypeCode.Char: CopyToBuffer<char>(ref state, op, count); break;
                default: throw new NotSupportedException($"Buffer copy not supported for dtype {dtype}");
            }
        }

        /// <summary>
        /// Copy data from buffer to operand (contiguous to strided).
        /// If operand needs casting, performs type conversion during copy.
        /// Runtime dtype dispatch version - handles any NumSharp dtype.
        /// </summary>
        public static void CopyFromBuffer(ref NpyIterState state, int op, long count)
        {
            // Check if casting is needed
            if (state.NeedsCast(op))
            {
                CopyFromBufferWithCast(ref state, op, count);
                return;
            }

            // No casting - use same-type copy
            var dtype = state.GetOpDType(op);

            switch (dtype)
            {
                case NPTypeCode.Boolean: CopyFromBuffer<bool>(ref state, op, count); break;
                case NPTypeCode.Byte: CopyFromBuffer<byte>(ref state, op, count); break;
                case NPTypeCode.Int16: CopyFromBuffer<short>(ref state, op, count); break;
                case NPTypeCode.UInt16: CopyFromBuffer<ushort>(ref state, op, count); break;
                case NPTypeCode.Int32: CopyFromBuffer<int>(ref state, op, count); break;
                case NPTypeCode.UInt32: CopyFromBuffer<uint>(ref state, op, count); break;
                case NPTypeCode.Int64: CopyFromBuffer<long>(ref state, op, count); break;
                case NPTypeCode.UInt64: CopyFromBuffer<ulong>(ref state, op, count); break;
                case NPTypeCode.Single: CopyFromBuffer<float>(ref state, op, count); break;
                case NPTypeCode.Double: CopyFromBuffer<double>(ref state, op, count); break;
                case NPTypeCode.Decimal: CopyFromBuffer<decimal>(ref state, op, count); break;
                case NPTypeCode.Char: CopyFromBuffer<char>(ref state, op, count); break;
                default: throw new NotSupportedException($"Buffer copy not supported for dtype {dtype}");
            }
        }

        /// <summary>
        /// Copy data from operand to buffer with type conversion.
        /// </summary>
        public static void CopyToBufferWithCast(ref NpyIterState state, int op, long count)
        {
            var buffer = state.GetBuffer(op);
            if (buffer == null || count <= 0)
                return;

            var srcType = state.GetOpSrcDType(op);
            var dstType = state.GetOpDType(op);
            var src = state.GetDataPtr(op);

            if (src == null)
                return;

            if (state.NDim == 0)
            {
                // Scalar - just convert one value
                NpyIterCasting.ConvertValue(src, buffer, srcType, dstType);
                return;
            }

            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                // Simple 1D copy with cast
                long stride = stridePtr[0];
                NpyIterCasting.CopyWithCast(src, stride, srcType, buffer, 1, dstType, count);
            }
            else
            {
                // Multi-dimensional strided copy with cast
                NpyIterCasting.CopyStridedToContiguousWithCast(
                    src, stridePtr, srcType,
                    buffer, dstType,
                    state.GetShapePointer(), state.NDim, count);
            }
        }

        /// <summary>
        /// Copy data from buffer to operand with type conversion.
        /// </summary>
        public static void CopyFromBufferWithCast(ref NpyIterState state, int op, long count)
        {
            var buffer = state.GetBuffer(op);
            if (buffer == null)
                return;

            var opFlags = state.GetOpFlags(op);
            if ((opFlags & NpyIterOpFlags.WRITE) == 0)
                return;  // Read-only operand

            var srcType = state.GetOpDType(op);  // Buffer dtype
            var dstType = state.GetOpSrcDType(op);  // Array dtype
            var dst = state.GetDataPtr(op);
            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                // Simple 1D copy with cast
                long stride = stridePtr[0];
                NpyIterCasting.CopyWithCast(buffer, 1, srcType, dst, stride, dstType, count);
            }
            else
            {
                // Multi-dimensional strided copy with cast
                NpyIterCasting.CopyContiguousToStridedWithCast(
                    buffer, srcType,
                    dst, stridePtr, dstType,
                    state.GetShapePointer(), state.NDim, count);
            }
        }

        /// <summary>
        /// Copy data from operand to buffer (strided to contiguous).
        /// </summary>
        public static void CopyToBuffer<T>(
            ref NpyIterState state,
            int op,
            long count)
            where T : unmanaged
        {
            var buffer = (T*)state.GetBuffer(op);
            if (buffer == null)
                return;

            var src = (T*)state.GetDataPtr(op);
            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                // Simple 1D copy
                long stride = stridePtr[0];
                if (stride == 1)
                {
                    // Contiguous
                    Unsafe.CopyBlock(buffer, src, (uint)(count * sizeof(T)));
                }
                else
                {
                    // Strided
                    for (long i = 0; i < count; i++)
                    {
                        buffer[i] = src[i * stride];
                    }
                }
            }
            else
            {
                // Multi-dimensional strided copy
                CopyStridedToContiguous<T>(src, buffer, state.GetShapePointer(), stridePtr, state.NDim, count);
            }
        }

        /// <summary>
        /// Copy data from buffer to operand (contiguous to strided).
        /// </summary>
        public static void CopyFromBuffer<T>(
            ref NpyIterState state,
            int op,
            long count)
            where T : unmanaged
        {
            var buffer = (T*)state.GetBuffer(op);
            if (buffer == null)
                return;

            var opFlags = state.GetOpFlags(op);
            if ((opFlags & NpyIterOpFlags.WRITE) == 0)
                return;  // Read-only operand

            var dst = (T*)state.GetDataPtr(op);
            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                long stride = stridePtr[0];
                if (stride == 1)
                {
                    Unsafe.CopyBlock(dst, buffer, (uint)(count * sizeof(T)));
                }
                else
                {
                    for (long i = 0; i < count; i++)
                    {
                        dst[i * stride] = buffer[i];
                    }
                }
            }
            else
            {
                CopyContiguousToStrided<T>(buffer, dst, state.GetShapePointer(), stridePtr, state.NDim, count);
            }
        }

        /// <summary>
        /// Copy strided data to contiguous buffer.
        /// </summary>
        private static void CopyStridedToContiguous<T>(
            T* src,
            T* dst,
            long* shape,
            long* strides,
            int ndim,
            long count)
            where T : unmanaged
        {
            // Use coordinate-based iteration
            var coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            for (long i = 0; i < count; i++)
            {
                // Calculate source offset
                long srcOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    srcOffset += coords[d] * strides[d];
                }

                dst[i] = src[srcOffset];

                // Advance coordinates (ripple carry)
                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        /// <summary>
        /// Copy contiguous buffer to strided destination.
        /// </summary>
        private static void CopyContiguousToStrided<T>(
            T* src,
            T* dst,
            long* shape,
            long* strides,
            int ndim,
            long count)
            where T : unmanaged
        {
            var coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            for (long i = 0; i < count; i++)
            {
                long dstOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    dstOffset += coords[d] * strides[d];
                }

                dst[dstOffset] = src[i];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        // =========================================================================
        // GROWINNER Optimization
        // =========================================================================
        // When GROWINNER flag is set, the iterator tries to make the inner loop
        // as large as possible by coalescing contiguous dimensions into the buffer.
        // This maximizes SIMD efficiency at the cost of larger buffers.
        // =========================================================================

        /// <summary>
        /// Calculate optimal inner loop size with GROWINNER optimization.
        /// NumPy: npyiter_grow_buffers() in nditer_api.c
        ///
        /// When GROWINNER is enabled, we try to grow the inner loop to include
        /// as many elements as possible while still fitting in the buffer.
        /// </summary>
        public static long CalculateGrowInnerSize(ref NpyIterState state, long bufferSize)
        {
            // If GROWINNER is not set, return the innermost dimension size
            if ((state.ItFlags & (uint)NpyIterFlags.GROWINNER) == 0)
            {
                return state.NDim > 0 ? state.Shape[state.NDim - 1] : 1;
            }

            // Try to fit as many elements as possible in the buffer
            long innerSize = 1;
            for (int d = state.NDim - 1; d >= 0; d--)
            {
                long dimSize = state.Shape[d];
                long newSize = innerSize * dimSize;

                if (newSize > bufferSize)
                    break;

                // Check if all operands are contiguous up to this dimension
                bool allContiguous = true;
                long expectedStride = 1;

                for (int op = 0; op < state.NOp; op++)
                {
                    // Only check operands that are being buffered
                    if (state.GetBuffer(op) == null)
                        continue;

                    for (int axis = state.NDim - 1; axis >= d; axis--)
                    {
                        long stride = state.GetStride(axis, op);
                        if (state.Shape[axis] > 1 && stride != expectedStride)
                        {
                            allContiguous = false;
                            break;
                        }
                        expectedStride *= state.Shape[axis];
                    }

                    if (!allContiguous)
                        break;
                }

                if (!allContiguous)
                    break;

                innerSize = newSize;
            }

            return Math.Min(innerSize, bufferSize);
        }

        // =========================================================================
        // Buffer Reuse Tracking
        // =========================================================================
        // The BUF_REUSABLE flag indicates that a buffer's contents are still valid
        // and can be reused without re-copying from the source array. This is useful
        // for reduction operations where the same input is used multiple times.
        // =========================================================================

        /// <summary>
        /// Mark operand buffer as reusable (contents are still valid).
        /// Call this after CopyToBuffer when the source data hasn't changed.
        /// </summary>
        public static void MarkBufferReusable(ref NpyIterState state, int op)
        {
            var flags = state.GetOpFlags(op);
            state.SetOpFlags(op, flags | NpyIterOpFlags.BUF_REUSABLE);
        }

        /// <summary>
        /// Check if operand buffer can be reused (contents still valid).
        /// </summary>
        public static bool IsBufferReusable(ref NpyIterState state, int op)
        {
            return (state.GetOpFlags(op) & NpyIterOpFlags.BUF_REUSABLE) != 0;
        }

        /// <summary>
        /// Clear buffer reusable flag (contents are no longer valid).
        /// Call this when the source data or iteration position changes.
        /// </summary>
        public static void InvalidateBuffer(ref NpyIterState state, int op)
        {
            var flags = state.GetOpFlags(op);
            state.SetOpFlags(op, flags & ~NpyIterOpFlags.BUF_REUSABLE);
        }

        /// <summary>
        /// Invalidate all buffers (e.g., after Reset or GotoIterIndex).
        /// </summary>
        public static void InvalidateAllBuffers(ref NpyIterState state)
        {
            for (int op = 0; op < state.NOp; op++)
            {
                InvalidateBuffer(ref state, op);
            }
        }

        /// <summary>
        /// Copy data to buffer only if not reusable.
        /// Returns true if copy was performed, false if buffer was reused.
        /// </summary>
        public static bool CopyToBufferIfNeeded(ref NpyIterState state, int op, long count)
        {
            if (IsBufferReusable(ref state, op))
                return false;

            CopyToBuffer(ref state, op, count);
            MarkBufferReusable(ref state, op);
            return true;
        }

        /// <summary>
        /// Prepare buffers for an iteration block.
        /// Handles GROWINNER and buffer reuse.
        /// </summary>
        public static long PrepareBuffers(ref NpyIterState state)
        {
            long innerSize = CalculateGrowInnerSize(ref state, state.BufferSize);

            // Copy input operands to buffers (with reuse check)
            for (int op = 0; op < state.NOp; op++)
            {
                if (state.GetBuffer(op) == null)
                    continue;

                var flags = state.GetOpFlags(op);
                if ((flags & NpyIterOpFlags.READ) != 0)
                {
                    CopyToBufferIfNeeded(ref state, op, innerSize);
                }
            }

            return innerSize;
        }

        /// <summary>
        /// Finalize buffers after an iteration block.
        /// Writes back output operands.
        /// </summary>
        public static void FinalizeBuffers(ref NpyIterState state, long count)
        {
            // Copy output operands from buffers
            for (int op = 0; op < state.NOp; op++)
            {
                if (state.GetBuffer(op) == null)
                    continue;

                var flags = state.GetOpFlags(op);
                if ((flags & NpyIterOpFlags.WRITE) != 0)
                {
                    CopyFromBuffer(ref state, op, count);
                    // Output buffers are no longer reusable after write-back
                    InvalidateBuffer(ref state, op);
                }
            }
        }
    }
}
