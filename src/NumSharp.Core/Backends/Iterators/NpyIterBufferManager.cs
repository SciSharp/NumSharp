using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;
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
        ///
        /// Buffering criterion (NumPy nditer parity): an operand is buffered when
        /// it needs a CAST, when the user requested CONTIG, or when its memory is
        /// not a single linear walk across the whole (coalesced) iteration space
        /// (<see cref="IsOperandIterLinear"/>). Linear operands — contiguous OR
        /// constant-stride 1-D views OR fully-broadcast scalars — stay unbuffered
        /// (BUFNEVER-style) and the kernel reads/writes the array directly through
        /// its true stride, which the Tier-3B kernels handle (incl. AVX2 gather).
        ///
        /// For unbuffered operands, <see cref="NpyIterState.BufStrides"/> is set to
        /// the operand's TRUE inner-axis byte stride so that
        /// <c>GetInnerLoopByteStrides()</c> exposes one consistent stride array to
        /// kernels under BUFFER. (Buffered operands get the tight element size.)
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
                {
                    SetUnbufferedBufStride(ref state, op);
                    continue;
                }

                // Buffered REDUCE keeps the historical criterion (its double-loop
                // machinery owns BufStrides/DataPtr swapping for every operand,
                // incl. the stride-0 accumulator slot). The windowed non-reduce
                // path uses the linearity criterion so linear strided operands
                // stay unbuffered and keep their true strides.
                bool reduceIter = (state.ItFlags & (uint)NpyIterFlags.REDUCE) != 0;
                bool needsBuffer =
                    state.NeedsCast(op) ||
                    (opFlags & (NpyIterOpFlags.CAST | NpyIterOpFlags.CONTIG | NpyIterOpFlags.REDUCE)) != 0 ||
                    (reduceIter ? !IsOperandContiguous(ref state, op)
                                : !IsOperandIterLinear(ref state, op));

                if (needsBuffer)
                {
                    // A buffered WRITEMASKED write means the flush will be a
                    // MASKED copy-back reading the ARRAYMASK operand byte-wise.
                    // NumPy validates the mask dtype when it builds that masked
                    // transfer function (nditer_constr.c:3470-3494 →
                    // dtype_transfer.c, TypeError) — same here, same text. Both
                    // the mask's array dtype and its buffer dtype must be a
                    // 1-byte nonzero-test type or the byte-wise reads are junk.
                    if ((opFlags & NpyIterOpFlags.WRITEMASKED) != 0 &&
                        (opFlags & NpyIterOpFlags.WRITE) != 0 &&
                        state.MaskOp >= 0)
                    {
                        var maskSrcType = state.GetOpSrcDType(state.MaskOp);
                        var maskBufType = state.GetOpDType(state.MaskOp);
                        if ((maskSrcType != NPTypeCode.Boolean && maskSrcType != NPTypeCode.Byte) ||
                            (maskBufType != NPTypeCode.Boolean && maskBufType != NPTypeCode.Byte))
                        {
                            throw new NotSupportedException(
                                "Only bool and uint8 masks are supported.");
                        }
                    }

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
                else
                {
                    SetUnbufferedBufStride(ref state, op);
                }
            }

            return true;
        }

        /// <summary>
        /// Record the true inner-axis byte stride for an operand that is NOT
        /// buffered, so kernels see correct strides via BufStrides under BUFFER.
        /// </summary>
        private static void SetUnbufferedBufStride(ref NpyIterState state, int op)
        {
            if (state.NDim == 0)
            {
                state.BufStrides[op] = 0;
                return;
            }

            long innerElemStride = state.GetStride(state.NDim - 1, op);
            state.BufStrides[op] = innerElemStride * state.GetSrcElementSize(op);
        }

        /// <summary>
        /// True when the operand's memory positions form one arithmetic
        /// progression over the entire (coalesced) iteration space — i.e. a
        /// single 1-D walk at the inner stride covers every element in order.
        /// Contiguous operands, constant-stride 1-D views, and fully-broadcast
        /// (all-stride-0) operands qualify; genuinely multi-dimensional strided
        /// operands do not.
        /// </summary>
        public static bool IsOperandIterLinear(ref NpyIterState state, int op)
        {
            if (state.NDim <= 1)
                return true;

            long inner = state.GetStride(state.NDim - 1, op);
            long expected = inner * state.Shape[state.NDim - 1];

            for (int d = state.NDim - 2; d >= 0; d--)
            {
                long dim = state.Shape[d];
                if (dim == 0)
                    return true;  // empty iteration — trivially linear

                if (dim != 1)
                {
                    if (state.GetStride(d, op) != expected)
                        return false;
                }

                expected *= dim;
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
                case NPTypeCode.SByte: CopyToBuffer<sbyte>(ref state, op, count); break;
                case NPTypeCode.Int16: CopyToBuffer<short>(ref state, op, count); break;
                case NPTypeCode.UInt16: CopyToBuffer<ushort>(ref state, op, count); break;
                case NPTypeCode.Int32: CopyToBuffer<int>(ref state, op, count); break;
                case NPTypeCode.UInt32: CopyToBuffer<uint>(ref state, op, count); break;
                case NPTypeCode.Int64: CopyToBuffer<long>(ref state, op, count); break;
                case NPTypeCode.UInt64: CopyToBuffer<ulong>(ref state, op, count); break;
                case NPTypeCode.Half: CopyToBuffer<Half>(ref state, op, count); break;
                case NPTypeCode.Single: CopyToBuffer<float>(ref state, op, count); break;
                case NPTypeCode.Double: CopyToBuffer<double>(ref state, op, count); break;
                case NPTypeCode.Decimal: CopyToBuffer<decimal>(ref state, op, count); break;
                case NPTypeCode.Complex: CopyToBuffer<System.Numerics.Complex>(ref state, op, count); break;
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
                case NPTypeCode.SByte: CopyFromBuffer<sbyte>(ref state, op, count); break;
                case NPTypeCode.Int16: CopyFromBuffer<short>(ref state, op, count); break;
                case NPTypeCode.UInt16: CopyFromBuffer<ushort>(ref state, op, count); break;
                case NPTypeCode.Int32: CopyFromBuffer<int>(ref state, op, count); break;
                case NPTypeCode.UInt32: CopyFromBuffer<uint>(ref state, op, count); break;
                case NPTypeCode.Int64: CopyFromBuffer<long>(ref state, op, count); break;
                case NPTypeCode.UInt64: CopyFromBuffer<ulong>(ref state, op, count); break;
                case NPTypeCode.Half: CopyFromBuffer<Half>(ref state, op, count); break;
                case NPTypeCode.Single: CopyFromBuffer<float>(ref state, op, count); break;
                case NPTypeCode.Double: CopyFromBuffer<double>(ref state, op, count); break;
                case NPTypeCode.Decimal: CopyFromBuffer<decimal>(ref state, op, count); break;
                case NPTypeCode.Complex: CopyFromBuffer<System.Numerics.Complex>(ref state, op, count); break;
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
        // Windowed buffered iteration (NumPy npyiter_copy_to_buffers /
        // npyiter_copy_from_buffers / npyiter_buffered_iternext equivalents)
        // =========================================================================
        // A "window" is one buffer fill: BufTransferSize elements starting at the
        // iterator's current position (IterIndex/Coords — which stay parked at the
        // fill start while the kernel consumes the window; BufferedNext flushes,
        // jumps, and refills).
        //
        // Per-operand semantics per fill (matching NumPy):
        //   • buffered + READ      → strided/casting copy into the buffer
        //   • buffered + WRITEONLY → no copy-in; kernel produces the contents
        //   • buffered + WRITE     → flushed back to the array on iternext/Dispose
        //   • unbuffered (linear)  → DataPtr stays in the array; BufStrides holds
        //                            its true inner byte stride
        //
        // Fills are capped at BufferSize and, for NDim > 1, row-aligned (end on an
        // inner-dimension boundary) — observed NumPy behavior: a (100,100) strided
        // source with buffersize=4096 fills in chunks of 4000/4000/2000, with and
        // without GROWINNER.
        // =========================================================================

        /// <summary>
        /// Number of elements the next buffer fill should cover, from the
        /// iterator's current position. Capped at <see cref="NpyIterState.BufferSize"/>;
        /// for multi-dim iterations the window ends on an inner-row boundary
        /// unless a single row exceeds the buffer (then a partial row is used).
        /// </summary>
        public static long ComputeTransferSize(ref NpyIterState state)
        {
            long remaining = state.IterEnd - state.IterIndex;
            if (remaining <= 0)
                return 0;

            long cap = Math.Min(remaining, state.BufferSize);
            if (state.NDim <= 1 || cap == remaining)
                return cap;

            long inner = state.Shape[state.NDim - 1];
            if (inner <= 0)
                return cap;

            long posInRow = state.Coords[state.NDim - 1];
            long firstRun = inner - posInRow;
            if (cap <= firstRun)
                return cap;  // can't finish the current row — partial-row fill

            long wholeRows = (cap - firstRun) / inner;
            return firstRun + wholeRows * inner;
        }

        /// <summary>
        /// Fill all buffered READ operands from the iterator's current position
        /// for <paramref name="count"/> elements, record the array write-back
        /// positions, swap buffered operands' DataPtrs to their buffers, and set
        /// the window bookkeeping (BufTransferSize / BufIterEnd / BufFlushed).
        /// </summary>
        public static void FillBufferWindow(ref NpyIterState state, long count)
        {

            // Save the array positions of the fill start: flush targets, and the
            // restore points for unbuffered operands are simply unchanged.
            for (int op = 0; op < state.NOp; op++)
                state.SetArrayWritebackPtr(op, state.GetDataPtr(op));

            for (int op = 0; op < state.NOp; op++)
            {
                var buffer = state.GetBuffer(op);
                if (buffer == null)
                    continue;

                if ((state.GetOpFlags(op) & NpyIterOpFlags.READ) != 0 && count > 0)
                    CopyWindowToBuffer(ref state, op, count);

                state.SetDataPtr(op, buffer);
            }

            state.BufTransferSize = count;
            state.BufIterEnd = state.IterIndex + count;
            state.BufFlushed = 0;
        }

        /// <summary>
        /// Write the current window's buffered WRITE operands back to their
        /// arrays. Idempotent (BufFlushed) so iternext, the single-inner-loop
        /// fast paths, and Dispose can all call it.
        /// </summary>
        public static void FlushBufferWindow(ref NpyIterState state)
        {
            if (state.BufFlushed != 0)
                return;
            state.BufFlushed = 1;

            long count = state.BufTransferSize;
            if (count <= 0)
                return;

            for (int op = 0; op < state.NOp; op++)
            {
                var buffer = state.GetBuffer(op);
                if (buffer == null)
                    continue;

                var opFlags = state.GetOpFlags(op);
                if ((opFlags & NpyIterOpFlags.WRITE) != 0)
                {
                    // WRITEMASKED: only elements whose ARRAYMASK byte is true
                    // reach the array; everything else in the buffer is
                    // discarded. This is the ONLY place masking is enforced —
                    // an unbuffered WRITEMASKED operand writes the array
                    // directly and the mask is the kernel's contract, exactly
                    // like NumPy (verified 2.4.2: 'buffered' + contiguous
                    // same-dtype operands write unmasked slots too).
                    // NumPy npyiter_copy_from_buffers (nditer_api.c:2001-2026).
                    if ((opFlags & NpyIterOpFlags.WRITEMASKED) != 0 && state.MaskOp >= 0)
                        CopyWindowFromBufferMasked(ref state, op, count);
                    else
                        CopyWindowFromBuffer(ref state, op, count);
                }
            }
        }

        /// <summary>
        /// Strided/casting gather of <paramref name="count"/> elements from the
        /// operand's array (starting at the fill-start position recorded in
        /// ArrayWritebackPtrs, walking the iteration space from the current
        /// Coords) into its tight contiguous buffer. Decomposes the window into
        /// inner-axis runs and uses the SIMD IL cast kernels per run.
        /// </summary>
        private static void CopyWindowToBuffer(ref NpyIterState state, int op, long count)
        {
            var srcType = state.GetOpSrcDType(op);
            var dstType = state.GetOpDType(op);
            int srcSize = state.GetSrcElementSize(op);
            int dstSize = state.GetElementSize(op);

            byte* src = (byte*)state.GetArrayWritebackPtr(op);
            byte* dst = (byte*)state.GetBuffer(op);

            if (state.NDim <= 1)
            {
                long stride = state.NDim == 0 ? 0 : state.GetStride(0, op);
                CopyRunToBuffer(src, stride, srcType, srcSize, dst, dstType, count);
                return;
            }

            int ndim = state.NDim;
            int innerAxis = ndim - 1;
            long innerLen = state.Shape[innerAxis];
            long innerStride = state.GetStride(innerAxis, op);

            // Local walk state — must NOT mutate the iterator's Coords.
            long* coords;
            long* coordsHeap = null;
            if (ndim <= NpyIterState.StackAllocThreshold)
            {
                long* coordsStack = stackalloc long[ndim];
                coords = coordsStack;
            }
            else
            {
                coordsHeap = (long*)NativeMemory.Alloc((nuint)(ndim * sizeof(long)));
                coords = coordsHeap;
            }
            try
            {
                for (int d = 0; d < ndim; d++)
                    coords[d] = state.Coords[d];

                long remaining = count;
                while (remaining > 0)
                {
                    long run = Math.Min(remaining, innerLen - coords[innerAxis]);
                    CopyRunToBuffer(src, innerStride, srcType, srcSize, dst, dstType, run);

                    dst += run * dstSize;
                    remaining -= run;
                    if (remaining <= 0)
                        break;

                    // Row finished — ripple-carry to the next position, adjusting
                    // the source pointer incrementally (same math as Advance()).
                    src += run * innerStride * srcSize;
                    coords[innerAxis] += run;

                    for (int d = innerAxis; d >= 0; d--)
                    {
                        if (coords[d] < state.Shape[d])
                            break;

                        coords[d] = 0;
                        src -= state.GetStride(d, op) * state.Shape[d] * srcSize;
                        if (d == 0)
                            break;
                        coords[d - 1]++;
                        src += state.GetStride(d - 1, op) * srcSize;
                    }
                }
            }
            finally
            {
                if (coordsHeap != null)
                    NativeMemory.Free(coordsHeap);
            }
        }

        /// <summary>
        /// Scatter of <paramref name="count"/> elements from the operand's tight
        /// contiguous buffer back into its array — the mirror of
        /// <see cref="CopyWindowToBuffer"/> (same window: ArrayWritebackPtrs +
        /// current Coords, which still describe the fill start).
        /// </summary>
        private static void CopyWindowFromBuffer(ref NpyIterState state, int op, long count)
        {
            var bufType = state.GetOpDType(op);
            var arrType = state.GetOpSrcDType(op);
            int bufSize = state.GetElementSize(op);
            int arrSize = state.GetSrcElementSize(op);

            byte* buf = (byte*)state.GetBuffer(op);
            byte* dst = (byte*)state.GetArrayWritebackPtr(op);

            if (state.NDim <= 1)
            {
                long stride = state.NDim == 0 ? 0 : state.GetStride(0, op);
                CopyRunFromBuffer(buf, bufType, bufSize, dst, stride, arrType, count);
                return;
            }

            int ndim = state.NDim;
            int innerAxis = ndim - 1;
            long innerLen = state.Shape[innerAxis];
            long innerStride = state.GetStride(innerAxis, op);

            long* coords;
            long* coordsHeap = null;
            if (ndim <= NpyIterState.StackAllocThreshold)
            {
                long* coordsStack = stackalloc long[ndim];
                coords = coordsStack;
            }
            else
            {
                coordsHeap = (long*)NativeMemory.Alloc((nuint)(ndim * sizeof(long)));
                coords = coordsHeap;
            }
            try
            {
                for (int d = 0; d < ndim; d++)
                    coords[d] = state.Coords[d];

                long remaining = count;
                while (remaining > 0)
                {
                    long run = Math.Min(remaining, innerLen - coords[innerAxis]);
                    CopyRunFromBuffer(buf, bufType, bufSize, dst, innerStride, arrType, run);

                    buf += run * bufSize;
                    remaining -= run;
                    if (remaining <= 0)
                        break;

                    dst += run * innerStride * arrSize;
                    coords[innerAxis] += run;

                    for (int d = innerAxis; d >= 0; d--)
                    {
                        if (coords[d] < state.Shape[d])
                            break;

                        coords[d] = 0;
                        dst -= state.GetStride(d, op) * state.Shape[d] * arrSize;
                        if (d == 0)
                            break;
                        coords[d - 1]++;
                        dst += state.GetStride(d - 1, op) * arrSize;
                    }
                }
            }
            finally
            {
                if (coordsHeap != null)
                    NativeMemory.Free(coordsHeap);
            }
        }

        /// <summary>
        /// Masked mirror of <see cref="CopyWindowFromBuffer"/> for WRITEMASKED
        /// operands: scatters the window back to the array but ONLY where the
        /// ARRAYMASK operand's byte is nonzero. The mask is read from its
        /// buffer when it was buffered, else from its array at the window's
        /// fill-start position (ArrayWritebackPtrs — saved for every operand
        /// by <see cref="FillBufferWindow"/>) — NumPy chooses the same way by
        /// BUFNEVER (nditer_api.c:2002-2014). The mask byte stride is
        /// <see cref="NpyIterState.BufStrides"/>: the tight element size for a
        /// buffered mask, the TRUE single linear byte stride for an unbuffered
        /// one (unbuffered ⇒ IsOperandIterLinear, so window element k sits at
        /// exactly k strides from the window start — incl. stride 0 for a
        /// fully-broadcast mask).
        /// </summary>
        private static void CopyWindowFromBufferMasked(ref NpyIterState state, int op, long count)
        {
            int maskOp = state.MaskOp;
            byte* maskBuffer = (byte*)state.GetBuffer(maskOp);
            byte* mask = maskBuffer != null ? maskBuffer : (byte*)state.GetArrayWritebackPtr(maskOp);
            long maskStride = maskBuffer != null
                ? state.GetElementSize(maskOp)
                : state.BufStrides[maskOp];

            var bufType = state.GetOpDType(op);
            var arrType = state.GetOpSrcDType(op);
            int bufSize = state.GetElementSize(op);
            int arrSize = state.GetSrcElementSize(op);

            byte* buf = (byte*)state.GetBuffer(op);
            byte* dst = (byte*)state.GetArrayWritebackPtr(op);

            if (state.NDim <= 1)
            {
                long stride = state.NDim == 0 ? 0 : state.GetStride(0, op);
                CopyRunFromBufferMasked(buf, bufType, bufSize, dst, stride, arrType, arrSize,
                    count, mask, maskStride);
                return;
            }

            int ndim = state.NDim;
            int innerAxis = ndim - 1;
            long innerLen = state.Shape[innerAxis];
            long innerStride = state.GetStride(innerAxis, op);

            long* coords;
            long* coordsHeap = null;
            if (ndim <= NpyIterState.StackAllocThreshold)
            {
                long* coordsStack = stackalloc long[ndim];
                coords = coordsStack;
            }
            else
            {
                coordsHeap = (long*)NativeMemory.Alloc((nuint)(ndim * sizeof(long)));
                coords = coordsHeap;
            }
            try
            {
                for (int d = 0; d < ndim; d++)
                    coords[d] = state.Coords[d];

                long remaining = count;
                while (remaining > 0)
                {
                    long run = Math.Min(remaining, innerLen - coords[innerAxis]);
                    CopyRunFromBufferMasked(buf, bufType, bufSize, dst, innerStride, arrType, arrSize,
                        run, mask, maskStride);

                    buf += run * bufSize;
                    mask += run * maskStride;
                    remaining -= run;
                    if (remaining <= 0)
                        break;

                    dst += run * innerStride * arrSize;
                    coords[innerAxis] += run;

                    for (int d = innerAxis; d >= 0; d--)
                    {
                        if (coords[d] < state.Shape[d])
                            break;

                        coords[d] = 0;
                        dst -= state.GetStride(d, op) * state.Shape[d] * arrSize;
                        if (d == 0)
                            break;
                        coords[d - 1]++;
                        dst += state.GetStride(d - 1, op) * arrSize;
                    }
                }
            }
            finally
            {
                if (coordsHeap != null)
                    NativeMemory.Free(coordsHeap);
            }
        }

        /// <summary>
        /// One masked inner-axis run: tight buffer → array, writing only where
        /// the mask byte is nonzero. Decomposes the run into contiguous TRUE
        /// stretches and hands each to the unmasked <see cref="CopyRunFromBuffer"/>
        /// (memcpy / SIMD cast kernels stay effective for dense masks) — the
        /// same structure as NumPy's _strided_masked_wrapper
        /// (dtype_transfer.c: skip false runs via mask scan, transfer true runs).
        /// The mask reads are 1-byte nonzero tests, guaranteed by the
        /// bool/uint8 mask validation in <see cref="AllocateBuffers"/>.
        /// </summary>
        private static void CopyRunFromBufferMasked(
            byte* buf, NPTypeCode bufType, int bufSize,
            byte* dst, long dstStride, NPTypeCode arrType, int arrSize,
            long count, byte* mask, long maskStride)
        {
            if (count <= 0)
                return;

            // Broadcast mask (stride 0): one value gates the whole run.
            if (maskStride == 0)
            {
                if (mask[0] != 0)
                    CopyRunFromBuffer(buf, bufType, bufSize, dst, dstStride, arrType, count);
                return;
            }

            long i = 0;
            while (i < count)
            {
                while (i < count && mask[i * maskStride] == 0)
                    i++;
                long start = i;
                while (i < count && mask[i * maskStride] != 0)
                    i++;
                if (i > start)
                    CopyRunFromBuffer(
                        buf + start * bufSize, bufType, bufSize,
                        dst + start * dstStride * arrSize, dstStride, arrType,
                        i - start);
            }
        }

        /// <summary>
        /// One inner-axis run: array (element stride <paramref name="srcStride"/>)
        /// → tight buffer. SIMD IL cast kernels when available; same-type memcpy /
        /// typed loop otherwise; scalar ConvertValue as the last resort
        /// (Decimal/Half/Complex pairs the IL generator does not cover).
        /// </summary>
        private static void CopyRunToBuffer(
            byte* src, long srcStride, NPTypeCode srcType, int srcSize,
            byte* dst, NPTypeCode dstType, long count)
        {
            if (count <= 0)
                return;


            if (srcType == dstType)
            {
                if (srcStride == 1)
                {
                    Buffer.MemoryCopy(src, dst, count * srcSize, count * srcSize);
                    return;
                }

                CopySameTypeStridedRun(src, srcStride, dst, 1, srcType, count);
                return;
            }

            if (srcStride == 1)
            {
                var contig = DirectILKernelGenerator.TryGetCastKernel(srcType, dstType);
                if (contig != null)
                {
                    contig(src, dst, count);
                    return;
                }
            }
            else
            {
                var strided = DirectILKernelGenerator.TryGetStridedCastKernel(srcType, dstType);
                if (strided != null)
                {
                    long srcStrideLocal = srcStride;
                    long dstStrideLocal = 1;
                    long shapeLocal = count;
                    strided(src, dst, &srcStrideLocal, &dstStrideLocal, &shapeLocal, 1);
                    return;
                }
            }

            NpyIterCasting.CopyWithCast(src, srcStride, srcType, dst, 1, dstType, count);
        }

        /// <summary>
        /// One inner-axis run: tight buffer → array (element stride
        /// <paramref name="dstStride"/>). Mirror of <see cref="CopyRunToBuffer"/>.
        /// </summary>
        private static void CopyRunFromBuffer(
            byte* buf, NPTypeCode bufType, int bufSize,
            byte* dst, long dstStride, NPTypeCode arrType, long count)
        {
            if (count <= 0)
                return;

            if (bufType == arrType)
            {
                if (dstStride == 1)
                {
                    Buffer.MemoryCopy(buf, dst, count * bufSize, count * bufSize);
                    return;
                }

                CopySameTypeStridedRun(buf, 1, dst, dstStride, bufType, count);
                return;
            }

            if (dstStride == 1)
            {
                var contig = DirectILKernelGenerator.TryGetCastKernel(bufType, arrType);
                if (contig != null)
                {
                    contig(buf, dst, count);
                    return;
                }
            }
            else
            {
                var strided = DirectILKernelGenerator.TryGetStridedCastKernel(bufType, arrType);
                if (strided != null)
                {
                    long srcStrideLocal = 1;
                    long dstStrideLocal = dstStride;
                    long shapeLocal = count;
                    strided(buf, dst, &srcStrideLocal, &dstStrideLocal, &shapeLocal, 1);
                    return;
                }
            }

            NpyIterCasting.CopyWithCast(buf, 1, bufType, dst, dstStride, arrType, count);
        }

        /// <summary>
        /// Same-dtype strided 1-D run copy (src/dst element strides). Mirrors the
        /// file's existing per-dtype dispatch style.
        /// </summary>
        private static void CopySameTypeStridedRun(
            byte* src, long srcStride, byte* dst, long dstStride, NPTypeCode dtype, long count)
        {
            switch (dtype)
            {
                case NPTypeCode.Boolean: StridedRun<bool>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Byte: StridedRun<byte>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.SByte: StridedRun<sbyte>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Int16: StridedRun<short>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.UInt16: StridedRun<ushort>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Int32: StridedRun<int>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.UInt32: StridedRun<uint>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Int64: StridedRun<long>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.UInt64: StridedRun<ulong>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Char: StridedRun<char>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Half: StridedRun<Half>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Single: StridedRun<float>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Double: StridedRun<double>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Decimal: StridedRun<decimal>(src, srcStride, dst, dstStride, count); break;
                case NPTypeCode.Complex: StridedRun<System.Numerics.Complex>(src, srcStride, dst, dstStride, count); break;
                default: throw new NotSupportedException($"Buffer run copy not supported for dtype {dtype}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void StridedRun<T>(byte* src, long srcStride, byte* dst, long dstStride, long count)
            where T : unmanaged
        {
            var s = (T*)src;
            var d = (T*)dst;
            for (long i = 0; i < count; i++)
                d[i * dstStride] = s[i * srcStride];
        }

        // NOTE: the pre-Wave-4 GROWINNER block sizing (CalculateGrowInnerSize) and
        // its PrepareBuffers/FinalizeBuffers drivers were removed. They had no
        // callers after the windowed machinery (ComputeTransferSize +
        // FillBufferWindow/FlushBufferWindow) replaced them, and
        // CalculateGrowInnerSize carried a latent bug: its expectedStride
        // accumulator was declared outside the per-operand loop, so operand 2+
        // was checked against operand 1's accumulated stride product.

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

    }
}
