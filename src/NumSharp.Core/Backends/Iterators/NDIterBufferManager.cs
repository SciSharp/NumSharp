using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Buffer management for NDIter.
    /// Handles allocation, copy-in, and copy-out of iteration buffers.
    /// </summary>
    public static unsafe partial class NDIterBufferManager
    {
        /// <summary>
        /// Built-in default buffer size (number of elements) — NumPy's <c>NPY_BUFSIZE</c>.
        /// Used whenever neither an explicit nditer <c>buffersize</c> nor a per-thread
        /// override (<see cref="NumSharp.np.setbufsize(long)"/>) applies.
        /// </summary>
        public const long DefaultBufferSize = 8192;

        /// <summary>
        /// Smallest valid ufunc buffer size (NumPy: must be ≥ 5 AND a multiple of 16, so the
        /// effective minimum is 16). See <c>extobj.c</c>'s <c>extobj_make_extobj</c>.
        /// </summary>
        internal const long MinBufferSize = 16;

        /// <summary>
        /// Largest valid ufunc buffer size (NumPy's <c>10e6</c>, inclusive).
        /// </summary>
        internal const long MaxBufferSize = 10_000_000;

        /// <summary>
        /// Per-thread override for the ufunc/NDIter default buffer size, set through
        /// <see cref="NumSharp.np.setbufsize(long)"/>. Zero means "unset" — the thread then
        /// falls back to <see cref="DefaultBufferSize"/>. Thread-local to mirror NumPy 2.x's
        /// context-local extobj (a <c>setbufsize</c> on one thread never leaks into another).
        /// </summary>
        [ThreadStatic] private static long _userBufferSize;

        /// <summary>
        /// The current default NDIter/ufunc buffer size for the calling thread — exactly the
        /// value <see cref="NumSharp.np.getbufsize"/> returns (its whole body), and what the engine
        /// reads where NumPy's C code reads the extobj's buffer size. Equals the last
        /// <see cref="NumSharp.np.setbufsize(long)"/> value on this thread, or
        /// <see cref="DefaultBufferSize"/> (8192) when none was set.
        /// </summary>
        /// <remarks>
        /// Changing it changes the buffer and chunk size of buffered iteration, and with it every
        /// answer that follows NumPy's inner-loop calls, as it does in NumPy: the summation order of
        /// a buffered flat sum, which call of the min / max reductions decides a tie or a NaN's bits,
        /// the fills a cast reduction or an out= round trip runs in, np.power's zero-exponent-step test.
        /// </remarks>
        public static long CurrentBufferSize
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { long v = _userBufferSize; return v > 0 ? v : DefaultBufferSize; }
            internal set => _userBufferSize = value; // validated by np.setbufsize before it reaches here
        }

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
        ///     Allocates an aligned buffer of <paramref name="elements"/> elements of <paramref name="elementSize"/> bytes —
        ///     the form every buffer of the iterator uses, because a fixed-width string's element size is its descriptor's
        ///     (<c>&lt;U5</c> is 20 bytes) and its storage lane names none.
        /// </summary>
        /// <param name="elements">The element count.</param>
        /// <param name="elementSize">The bytes per element.</param>
        /// <returns>The buffer (free it with <see cref="FreeAligned"/>).</returns>
        /// <exception cref="OutOfMemoryException">The allocation failed.</exception>
        public static void* AllocateAligned(long elements, int elementSize)
        {
            long bytes = elements * elementSize;
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
        public static long DetermineBufferSize(ref NDIterState state, long requestedSize)
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
        ///     NumPy's <c>npyiter_allocate_buffers</c>: one buffer of <see cref="NDIterState.BufferSize"/> elements for every
        ///     operand the buffering setup did not mark <see cref="NDIterOpFlags.BUFNEVER"/>; an operand that already has its
        ///     buffer keeps it. A <c>StringDType</c> buffer starts zeroed (every cell the empty string, owning nothing — NumPy's
        ///     <c>NPY_NEEDS_INIT</c> memset) with an allocator of its own.
        /// </summary>
        /// <param name="state">The iterator state, after <see cref="FindBufferingSetup"/> (which decides who is buffered and
        ///     how many elements a fill covers).</param>
        /// <param name="bufferSize">The element count for a state the setup never sized (a caller outside the iterator's own
        ///     construction): used only when <see cref="NDIterState.BufferSize"/> is not positive, and then recorded there
        ///     (0 or less picks <see cref="DetermineBufferSize"/>'s heuristic).</param>
        /// <returns>True when every buffer exists; false when an allocation failed (every buffer allocated so far is freed
        ///     again, as NumPy's failure path does).</returns>
        /// <remarks>
        ///     The strides an inner loop sees are the setup's (<see cref="NDIterState.BufStrides"/>): a buffered operand's buffer
        ///     element size, an unbuffered operand's own array stride. This method only provides the memory. A buffer is
        ///     allocated for at least one element, so an empty iteration (a zero-element fill) still owns valid pointers.
        /// </remarks>
        public static bool AllocateBuffers(ref NDIterState state, long bufferSize)
        {
            if (state.BufferSize <= 0)
                state.BufferSize = bufferSize > 0 ? bufferSize : DetermineBufferSize(ref state, 0);

            long elements = state.BufferSize;
            var vstrings = NDIterVStringContext.Of(ref state);
            for (int op = 0; op < state.NOp; op++)
            {
                if ((state.GetOpFlags(op) & NDIterOpFlags.BUFNEVER) != 0 || state.GetBuffer(op) != null)
                    continue;

                // Sized by the stored element size (a fixed-width string's item size included).
                var buffer = AllocateAligned(Math.Max(elements, 1), state.GetElementSize(op));
                if (buffer == null)
                {
                    FreeBuffers(ref state);
                    return false;
                }

                state.SetBuffer(op, buffer);

                // A StringDType iteration side: zeroed cells (the empty string, owning nothing) and an allocator of the
                // buffer's own (NDIterVStringContext.AllocateBuffer).
                if (state.IsVStringOp(op))
                    vstrings?.AllocateBuffer(op, buffer, elements);
            }

            return true;
        }

        /// <summary>
        /// True when the operand's memory positions form one arithmetic
        /// progression over the entire (coalesced) iteration space — i.e. a
        /// single 1-D walk at the inner stride covers every element in order.
        /// Contiguous operands, constant-stride 1-D views, and fully-broadcast
        /// (all-stride-0) operands qualify; genuinely multi-dimensional strided
        /// operands do not.
        /// </summary>
        public static bool IsOperandIterLinear(ref NDIterState state, int op)
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
        public static void FreeBuffers(ref NDIterState state)
        {
            var vstrings = NDIterVStringContext.Of(ref state);
            for (int op = 0; op < state.NOp; op++)
            {
                var buffer = state.GetBuffer(op);
                if (buffer != null)
                {
                    // A StringDType buffer's strings (heap strings, the arena) go with it: NumPy's clear loop + free_allocator.
                    vstrings?.FreeBuffer(op, buffer, state.BufferSize);
                    FreeAligned(buffer);
                    state.SetBuffer(op, null);
                }
            }
        }

        /// <summary>
        /// Copy data from operand to buffer (strided to contiguous).
        /// If operand needs casting, performs type conversion during copy.
        /// Runtime dtype dispatch version - handles any NumSharp dtype.
        /// </summary>
        public static void CopyToBuffer(ref NDIterState state, int op, long count)
        {
            // A fixed-width string on either side moves through the string transfer: its width and class live in the
            // state's element sizes and string kinds, not in the (shared) String type code.
            if (state.IsStringOp(op))
            {
                CopyStringOperandFromStart(ref state, op, count, toBuffer: true);
                return;
            }

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
        public static void CopyFromBuffer(ref NDIterState state, int op, long count)
        {
            if (state.IsStringOp(op))
            {
                CopyStringOperandFromStart(ref state, op, count, toBuffer: false);
                return;
            }

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
        public static void CopyToBufferWithCast(ref NDIterState state, int op, long count)
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
                NDIterCasting.ConvertValue(src, buffer, srcType, dstType);
                return;
            }

            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                // Simple 1D copy with cast
                long stride = stridePtr[0];
                NDIterCasting.CopyWithCast(src, stride, srcType, buffer, 1, dstType, count);
            }
            else
            {
                // Multi-dimensional strided copy with cast
                NDIterCasting.CopyStridedToContiguousWithCast(
                    src, stridePtr, srcType,
                    buffer, dstType,
                    state.GetShapePointer(), state.NDim, count);
            }
        }

        /// <summary>
        /// Copy data from buffer to operand with type conversion.
        /// </summary>
        public static void CopyFromBufferWithCast(ref NDIterState state, int op, long count)
        {
            var buffer = state.GetBuffer(op);
            if (buffer == null)
                return;

            var opFlags = state.GetOpFlags(op);
            if ((opFlags & NDIterOpFlags.WRITE) == 0)
                return;  // Read-only operand

            var srcType = state.GetOpDType(op);  // Buffer dtype
            var dstType = state.GetOpSrcDType(op);  // Array dtype
            var dst = state.GetDataPtr(op);
            var stridePtr = state.GetStridesPointer(op);

            if (state.NDim == 1)
            {
                // Simple 1D copy with cast
                long stride = stridePtr[0];
                NDIterCasting.CopyWithCast(buffer, 1, srcType, dst, stride, dstType, count);
            }
            else
            {
                // Multi-dimensional strided copy with cast
                NDIterCasting.CopyContiguousToStridedWithCast(
                    buffer, srcType,
                    dst, stridePtr, dstType,
                    state.GetShapePointer(), state.NDim, count);
            }
        }

        /// <summary>
        /// Copy data from operand to buffer (strided to contiguous).
        /// </summary>
        public static void CopyToBuffer<T>(
            ref NDIterState state,
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
            ref NDIterState state,
            int op,
            long count)
            where T : unmanaged
        {
            var buffer = (T*)state.GetBuffer(op);
            if (buffer == null)
                return;

            var opFlags = state.GetOpFlags(op);
            if ((opFlags & NDIterOpFlags.WRITE) == 0)
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

        /// <summary>
        /// One masked inner-axis run: tight buffer → array, writing only where
        /// the mask byte is nonzero. Decomposes the run into contiguous TRUE
        /// stretches and hands each to the unmasked <see cref="CopyRunFromBuffer"/>
        /// (memcpy / SIMD cast kernels stay effective for dense masks) — the
        /// same structure as NumPy's _strided_masked_wrapper
        /// (dtype_transfer.c: skip false runs via mask scan, transfer true runs).
        /// The mask reads are 1-byte nonzero tests, guaranteed by the
        /// bool/uint8 mask validation of the construction
        /// (<see cref="ValidateWriteMaskedTransfers"/>).
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

            NDIterCasting.CopyWithCast(src, srcStride, srcType, dst, 1, dstType, count);
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

            NDIterCasting.CopyWithCast(buf, 1, bufType, dst, dstStride, arrType, count);
        }

        /// <summary>
        ///     The cast of operand <paramref name="op"/>'s transfer in one direction - NumPy's per-operand transfer function
        ///     (<c>transferinfo[iop].read</c> / <c>.write</c>, set up through <c>PyArray_GetDTypeTransferFunction</c>): the strided
        ///     loop of the class pair's casting impl between the operand's array descriptor and its iteration (buffer)
        ///     descriptor. A <c>StringDType</c> side's cast is the one the iterator's <see cref="NDIterVStringContext"/> keeps
        ///     (built once per operand and direction, over the allocators of both sides); a fixed-width side's is resolved for
        ///     the caller's walk, from the descriptors the state rebuilds (<see cref="NDIterState.GetOpSrcDescr"/> /
        ///     <see cref="NDIterState.GetOpDescr"/>).
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">A string operand (a fixed-width or <c>StringDType</c> side, at least).</param>
        /// <param name="toBuffer">True for the fill (array to buffer), false for the write-back (buffer to array).</param>
        /// <returns>The cast. A fixed-width one is resolved per call, so call this once per walk (one fill, one write-back),
        ///     not per run: the resolution allocates (a parse loop owns its scratch).</returns>
        /// <exception cref="InvalidOperationException">A <c>StringDType</c> operand without its iterator context, or a
        ///     <c>StringDType</c> buffer without its allocator (routing bugs).</exception>
        /// <exception cref="TypeError">No casting implementation joins the two classes (the descriptor plan admitted the
        ///     operand, so this does not occur).</exception>
        internal static NDCastInfo StringTransfer(ref NDIterState state, int op, bool toBuffer)
        {
            if (state.IsVStringOp(op))
            {
                var ctx = NDIterVStringContext.Of(ref state)
                          ?? throw new InvalidOperationException("A StringDType iterator operand has no StringDType context.");
                return ctx.Transfer(op, toBuffer);
            }

            // The state keeps a fixed-width side's class and width; the descriptors rebuilt from them are native, as the
            // buffers and every NumSharp storage are, so the class pair's loop takes them as they are.
            DType arr = state.GetOpSrcDescr(op), buf = state.GetOpDescr(op);
            var (from, to) = toBuffer ? (arr, buf) : (buf, arr);
            return DTypeCasting.GetDTypeTransferFunction(true, from.itemsize, to.itemsize, from, to, false, out _);
        }

        /// <summary>
        ///     One run of a string operand (a fixed-width or <c>StringDType</c> side, at least) between its array and its
        ///     buffer: the operand's cast (<see cref="StringTransfer"/>) over the run - NumPy's <c>string_to_string</c> zero-pad
        ///     copy and its <c>S ↔ U</c> casts, a number's <c>str()</c> and a text's parse (the legacy casts of astype), and
        ///     the <c>StringDType</c> casts (every string duplicated into the destination's allocator, never bytes).
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The operand.</param>
        /// <param name="cast">The operand's cast in this direction (<see cref="StringTransfer"/>).</param>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcElemStride">The source stride in ELEMENTS of the source side.</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstElemStride">The destination stride in ELEMENTS of the destination side.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="toBuffer">True for array → buffer (the fill), false for buffer → array (the flush).</param>
        /// <exception cref="ValueError">Text → number meets a text that is not a number of the numeric side's kind (NumPy's
        ///     <c>int()</c> / <c>float()</c> / <c>complex()</c> texts, quoting <c>np.str_('…')</c>), or a <c>StringDType</c> NA
        ///     element meets a numeric side.</exception>
        /// <exception cref="OverflowException">Text → integer meets a value outside the numeric side.</exception>
        /// <exception cref="SystemError">A <c>U</c> text converted to a number is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U → S</c> element (or a <c>StringDType</c> text into <c>S</c>) holds a
        ///     code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S → U</c> element holds a byte at or above <c>0x80</c>, or an
        ///     <c>S</c> text converted to complex is not UTF-8.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> side cannot allocate a string (NumPy's
        ///     <c>MemoryError</c> texts).</exception>
        internal static void CopyStringRun(ref NDIterState state, int op, NDCastInfo cast, byte* src, long srcElemStride,
                                           byte* dst, long dstElemStride, long count, bool toBuffer)
        {
            if (count <= 0)
                return;
            // The loop takes byte strides: each side's element size converts its stride (the source is the array for a fill,
            // the buffer for a flush). A StringDType loop locks both sides' allocators for the call itself.
            int fromSize = toBuffer ? state.GetSrcElementSize(op) : state.GetElementSize(op);
            int toSize = toBuffer ? state.GetElementSize(op) : state.GetSrcElementSize(op);
            cast.Run(src, srcElemStride * fromSize, dst, dstElemStride * toSize, count);
        }

        /// <summary>
        ///     The masked flush of one run of a string operand — <see cref="CopyRunFromBufferMasked"/>'s structure (each
        ///     stretch of true mask bytes moved as one run) over <see cref="CopyStringRun"/>.
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The WRITEMASKED string operand.</param>
        /// <param name="cast">The operand's write-back cast (<see cref="StringTransfer"/>).</param>
        /// <param name="buf">The run's first buffer element.</param>
        /// <param name="dst">The run's first array element.</param>
        /// <param name="dstElemStride">The array stride in elements.</param>
        /// <param name="count">The run length.</param>
        /// <param name="mask">The run's first mask byte.</param>
        /// <param name="maskStride">The mask byte stride (0 for a broadcast mask).</param>
        /// <exception cref="Exception">The cast's conversion errors (see <see cref="CopyStringRun"/>).</exception>
        private static void CopyStringRunFromBufferMasked(ref NDIterState state, int op, NDCastInfo cast, byte* buf, byte* dst,
                                                          long dstElemStride, long count, byte* mask, long maskStride)
        {
            if (count <= 0)
                return;
            int bufSize = state.GetElementSize(op);
            int arrSize = state.GetSrcElementSize(op);

            if (maskStride == 0)
            {
                if (mask[0] != 0)
                    CopyStringRun(ref state, op, cast, buf, 1, dst, dstElemStride, count, toBuffer: false);
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
                    CopyStringRun(ref state, op, cast, buf + start * bufSize, 1,
                                  dst + start * dstElemStride * arrSize, dstElemStride, i - start, toBuffer: false);
            }
        }

        /// <summary>
        ///     The from-the-start transfer of a string operand behind <see cref="CopyToBuffer(ref NDIterState, int, long)"/>
        ///     / <see cref="CopyFromBuffer(ref NDIterState, int, long)"/>: <paramref name="count"/> elements from the operand's
        ///     current data pointer, walking the iteration space from coordinate zero — the layout
        ///     <see cref="CopyToBuffer{T}"/> / <see cref="CopyFromBuffer{T}"/> assume — in inner-axis runs of
        ///     <see cref="CopyStringRun"/>, all through one cast (<see cref="StringTransfer"/>). (The iterator's own buffered
        ///     iteration walks from the buffer start's coordinates instead: <c>TransferBuffer</c> in
        ///     NDIterBufferManager.Buffering.cs.)
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The string operand.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="toBuffer">True to fill the buffer, false to flush it (a read-only operand is not flushed).</param>
        /// <exception cref="Exception">The cast's conversion errors (see <see cref="CopyStringRun"/>).</exception>
        private static void CopyStringOperandFromStart(ref NDIterState state, int op, long count, bool toBuffer)
        {
            byte* buffer = (byte*)state.GetBuffer(op);
            if (buffer == null || count <= 0)
                return;
            if (!toBuffer && (state.GetOpFlags(op) & NDIterOpFlags.WRITE) == 0)
                return;

            byte* arr = (byte*)state.GetDataPtr(op);
            int arrSize = state.GetSrcElementSize(op);
            int bufSize = state.GetElementSize(op);
            int ndim = state.NDim;
            // The operand's cast in this direction, resolved once for the whole walk (every run below runs it).
            var cast = StringTransfer(ref state, op, toBuffer);
            if (ndim == 0)
            {
                if (toBuffer)
                    CopyStringRun(ref state, op, cast, arr, 0, buffer, 1, 1, toBuffer: true);
                else
                    CopyStringRun(ref state, op, cast, buffer, 1, arr, 0, 1, toBuffer: false);
                return;
            }

            int innerAxis = ndim - 1;
            long innerLen = state.Shape[innerAxis];
            long innerStride = state.GetStride(innerAxis, op);

            // The coordinates live on the stack, as in CopyStridedToContiguous<T> (the numeric twin of this walk).
            long* coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            long remaining = count;
            while (remaining > 0)
            {
                long run = Math.Min(remaining, innerLen - coords[innerAxis]);
                if (toBuffer)
                    CopyStringRun(ref state, op, cast, arr, innerStride, buffer, 1, run, toBuffer: true);
                else
                    CopyStringRun(ref state, op, cast, buffer, 1, arr, innerStride, run, toBuffer: false);

                buffer += run * bufSize;
                remaining -= run;
                if (remaining <= 0)
                    break;

                // Row finished: ripple-carry to the next position, moving the array pointer incrementally.
                arr += run * innerStride * arrSize;
                coords[innerAxis] += run;
                for (int d = innerAxis; d >= 0; d--)
                {
                    if (coords[d] < state.Shape[d])
                        break;
                    coords[d] = 0;
                    arr -= state.GetStride(d, op) * state.Shape[d] * arrSize;
                    if (d == 0)
                        break;
                    coords[d - 1]++;
                    arr += state.GetStride(d - 1, op) * arrSize;
                }
            }
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
        // callers after the windowed machinery replaced them, and
        // CalculateGrowInnerSize carried a latent bug: its expectedStride
        // accumulator was declared outside the per-operand loop, so operand 2+
        // was checked against operand 1's accumulated stride product. The windowed
        // machinery (ComputeTransferSize / FillBufferWindow / FlushBufferWindow, fills
        // of whole inner rows up to the buffer size) and the legacy buffered-reduce
        // double loop were in turn replaced by the port of NumPy's buffering itself
        // (NDIterBufferManager.Buffering.cs): NumPy's core / outer split, the reduce
        // double loop, the reuse rule, GROWINNER.

        // =========================================================================
        // Buffer Reuse Tracking
        // =========================================================================
        // The BUF_REUSABLE flag indicates that a buffer's contents are still valid
        // and can be reused without re-copying from the source array. The iterator's
        // own fills set and read it (CopyToBuffers: NumPy's reuse rule of the reduce
        // double loop); these helpers expose the flag to callers that drive buffers
        // themselves.
        // =========================================================================

        /// <summary>
        /// Mark operand buffer as reusable (contents are still valid).
        /// Call this after CopyToBuffer when the source data hasn't changed.
        /// </summary>
        public static void MarkBufferReusable(ref NDIterState state, int op)
        {
            var flags = state.GetOpFlags(op);
            state.SetOpFlags(op, flags | NDIterOpFlags.BUF_REUSABLE);
        }

        /// <summary>
        /// Check if operand buffer can be reused (contents still valid).
        /// </summary>
        public static bool IsBufferReusable(ref NDIterState state, int op)
        {
            return (state.GetOpFlags(op) & NDIterOpFlags.BUF_REUSABLE) != 0;
        }

        /// <summary>
        /// Clear buffer reusable flag (contents are no longer valid).
        /// Call this when the source data or iteration position changes.
        /// </summary>
        public static void InvalidateBuffer(ref NDIterState state, int op)
        {
            var flags = state.GetOpFlags(op);
            state.SetOpFlags(op, flags & ~NDIterOpFlags.BUF_REUSABLE);
        }

        /// <summary>
        /// Invalidate all buffers (e.g., after Reset or GotoIterIndex).
        /// </summary>
        public static void InvalidateAllBuffers(ref NDIterState state)
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
        public static bool CopyToBufferIfNeeded(ref NDIterState state, int op, long count)
        {
            if (IsBufferReusable(ref state, op))
                return false;

            CopyToBuffer(ref state, op, count);
            MarkBufferReusable(ref state, op);
            return true;
        }

    }
}
