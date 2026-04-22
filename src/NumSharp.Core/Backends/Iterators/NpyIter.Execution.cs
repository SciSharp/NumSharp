using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NpyIter.Execution.cs — Kernel Integration Layer (DESIGN)
// =============================================================================
//
// RATIONALE
// ---------
// NumPy's nditer is written in C++ with templates: each ufunc plugs a typed
// inner-loop function into the iterator and calls it in the canonical loop:
//
//     do { inner(dataptrs, strides, count, auxdata); } while (iternext(iter));
//
// NumSharp has two halves that need to meet:
//   - NpyIter produces data pointers, strides, buffers, reduction scheduling
//   - ILKernelGenerator produces type-specific SIMD kernels by emitting IL
//
// This partial class is the bridge. It exposes NumPy-style APIs where a caller
// supplies (or lets NumSharp synthesize via IL) the inner-loop kernel, and the
// iterator drives it.
//
// LAYERS (bottom to top)
// ----------------------
// 1. ForEach(NpyInnerLoopFunc, auxdata)
//      Canonical NumPy iteration. Caller-supplied native kernel runs per inner
//      loop. EXLOOP aware. This is the raw power user entry point.
//
// 2. ExecuteGeneric<TKernel>(TKernel kernel)
//      Struct-generic dispatch with zero-alloc. TKernel is a struct implementing
//      INpyInnerLoop; JIT inlines the call site. Same capability as ForEach but
//      branch-free through the iteration driver.
//
// 3. ExecuteBinary/Unary/Comparison/Reduction/Scan(Op op)
//      High-level "please run this ufunc". Picks path via
//      NpyIter.DetectExecutionPath and materializes the matching IL kernel.
//      Handles reduction first-visit init, buffered cast write-back, etc.
//
// BUG NOTES DISCOVERED DURING DESIGN
// ----------------------------------
// (a) `Iternext()` calls `state.Advance()` unconditionally. That ignores the
//     EXLOOP flag, so callers iterating with EXTERNAL_LOOP see NDim-1 extra
//     iterations and read past buffer end. The bridge below uses
//     `GetIterNext()` (which picks the correct advancer) and never touches the
//     broken wrapper.
//
// (b) Buffered-with-cast: after `CopyToBuffer`, the buffer is tight-packed at
//     the buffer dtype (e.g. float64), but `Strides[op]` still holds the
//     source-array stride (e.g. 1 element = 4 bytes for int32). `state.Advance`
//     multiplies by `ElementSizes[op]` which is now the buffer element size
//     (8 bytes), producing the wrong pointer delta. The bridge below routes
//     buffered paths through BufStrides, which NpyIterBufferManager already
//     sets to the buffer element size.
//
// Both bugs are fixable in NpyIter.cs. The bridge is careful not to trip them
// so it works on the existing iterator, and exposing it will make the fixes
// enforceable by tests.
//
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    // -------------------------------------------------------------------------
    // Inner-loop delegate shapes
    // -------------------------------------------------------------------------

    /// <summary>
    /// Inner-loop callback matching NumPy's <c>PyUFuncGenericFunction</c>.
    /// Invoked once per outer iteration; processes <paramref name="count"/>
    /// elements starting at <paramref name="dataptrs"/>[op] with per-operand
    /// byte stride <paramref name="strides"/>[op].
    /// </summary>
    /// <param name="dataptrs">One byte-pointer per operand (NOp entries).</param>
    /// <param name="strides">Byte stride per operand for the inner loop (NOp).</param>
    /// <param name="count">Number of elements to process this inner loop.</param>
    /// <param name="auxdata">Opaque user cookie (may be null).</param>
    public unsafe delegate void NpyInnerLoopFunc(
        void** dataptrs, long* strides, long count, void* auxdata);

    /// <summary>
    /// Struct-generic inner loop — zero-alloc alternative to
    /// <see cref="NpyInnerLoopFunc"/>. Implementations should be
    /// <c>readonly struct</c>; JIT specializes <see cref="NpyIterRef.ExecuteGeneric"/>
    /// per type and inlines the call.
    /// </summary>
    public unsafe interface INpyInnerLoop
    {
        void Execute(void** dataptrs, long* strides, long count);
    }

    /// <summary>
    /// Reduction variant — the accumulator is threaded through the outer loop
    /// so each inner-loop invocation can accumulate into the same scalar.
    /// Return false to abort iteration (early exit for Any/All).
    /// </summary>
    public unsafe interface INpyReducingInnerLoop<TAccum> where TAccum : unmanaged
    {
        bool Execute(void** dataptrs, long* strides, long count, ref TAccum accumulator);
    }

    // -------------------------------------------------------------------------
    // Execution partial of NpyIterRef
    // -------------------------------------------------------------------------

    public unsafe ref partial struct NpyIterRef
    {
        // =====================================================================
        // Layer 1: Canonical NumPy-style ForEach
        // =====================================================================

        /// <summary>
        /// Drive the iterator with a user-supplied inner-loop kernel. Matches
        /// the pattern used by NumPy ufuncs in C:
        ///
        ///     do { inner(dataptrs, strides, count, aux); } while (iternext);
        ///
        /// The iterator decides the semantics:
        ///   • Fully coalesced + contiguous → 1 call covering IterSize elements.
        ///   • EXTERNAL_LOOP → 1 call per outer index, count = inner dim size.
        ///   • Buffered → 1 call per buffer fill, count = BufIterEnd.
        ///   • Otherwise → 1 call per element, count = 1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void ForEach(NpyInnerLoopFunc kernel, void* auxdata = null)
        {
            if (kernel is null) throw new ArgumentNullException(nameof(kernel));

            void** dataptrs = GetDataPtrArray();
            long* byteStrides = GetInnerLoopByteStrides();
            long innerSize = ResolveInnerLoopCount();

            if (IsSingleInnerLoop())
            {
                kernel(dataptrs, byteStrides, innerSize, auxdata);
                return;
            }

            var iternext = GetIterNext();

            // Buffered fills can change size at the tail, so re-read per call.
            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                long* bufSize = GetInnerLoopSizePtr();
                do
                {
                    kernel(dataptrs, byteStrides, *bufSize, auxdata);
                } while (iternext(ref *_state));
                return;
            }

            // EXLOOP and non-EXLOOP both have a stable innerSize across iterations.
            do
            {
                kernel(dataptrs, byteStrides, innerSize, auxdata);
            } while (iternext(ref *_state));
        }

        /// <summary>
        /// Returns the number of elements the kernel processes per inner-loop
        /// invocation, in a way that is correct regardless of which iterator
        /// flags are set:
        ///
        /// <list type="bullet">
        ///   <item>BUFFER: size of the current buffer fill (callers that can
        ///     observe per-iteration changes should re-read it from
        ///     <see cref="GetInnerLoopSizePtr"/>).</item>
        ///   <item>EXTERNAL_LOOP (EXLOOP): innermost coalesced shape dimension —
        ///     the iterator advances in strides of that size.</item>
        ///   <item>Otherwise: 1 — the iterator's <c>iternext</c> increments
        ///     <see cref="NpyIterState.IterIndex"/> by one per call, so the
        ///     kernel processes one element per invocation.</item>
        /// </list>
        ///
        /// Fixes the pre-existing inconsistency where
        /// <see cref="GetInnerLoopSizePtr"/> on a non-BUFFER, non-EXLOOP
        /// iterator reported <c>Shape[NDim - 1]</c> (the innermost dimension)
        /// while <c>Iternext</c> only advanced by one element — causing the
        /// kernel to over-read past the end of the array.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private long ResolveInnerLoopCount()
        {
            uint f = _state->ItFlags;
            if ((f & (uint)NpyIterFlags.BUFFER) != 0) return _state->BufIterEnd;
            if ((f & (uint)NpyIterFlags.EXLOOP) != 0) return _state->Shape[_state->NDim - 1];
            return 1;
        }

        /// <summary>
        /// Struct-generic overload — the JIT devirtualizes and inlines the
        /// kernel call through the TKernel type parameter. Preferred when the
        /// kernel is known at call site.
        ///
        /// Performance note: the single-iteration fast path (coalesced + EXLOOP
        /// or ONEITERATION) avoids the do/while + delegate call so the JIT can
        /// autovectorize the kernel body.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ExecuteGeneric<TKernel>(TKernel kernel) where TKernel : struct, INpyInnerLoop
        {
            if (IsSingleInnerLoop())
                ExecuteGenericSingle(kernel);
            else
                ExecuteGenericMulti(kernel);
        }

        /// <summary>
        /// Fast path: the whole iteration is one inner-loop kernel call. This
        /// method is tiny and has no delegate calls or loops, so the JIT can
        /// inline it into the caller and autovectorize the kernel's own loop.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void ExecuteGenericSingle<TKernel>(TKernel kernel) where TKernel : struct, INpyInnerLoop
        {
            kernel.Execute(GetDataPtrArray(), GetInnerLoopByteStrides(), ResolveInnerLoopCount());
        }

        /// <summary>Multi-loop path with do/while driver.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void ExecuteGenericMulti<TKernel>(TKernel kernel) where TKernel : struct, INpyInnerLoop
        {
            void** dataptrs = GetDataPtrArray();
            long* byteStrides = GetInnerLoopByteStrides();
            var iternext = GetIterNext();

            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                long* bufSize = GetInnerLoopSizePtr();
                do
                {
                    kernel.Execute(dataptrs, byteStrides, *bufSize);
                } while (iternext(ref *_state));
                return;
            }

            long innerSize = ResolveInnerLoopCount();
            do
            {
                kernel.Execute(dataptrs, byteStrides, innerSize);
            } while (iternext(ref *_state));
        }

        /// <summary>
        /// True when the iterator is guaranteed to complete in exactly one
        /// inner-loop kernel invocation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool IsSingleInnerLoop()
        {
            uint f = _state->ItFlags;
            // ONEITERATION: iter size <= 1.
            if ((f & (uint)NpyIterFlags.ONEITERATION) != 0) return true;
            // Fully coalesced to one axis + EXLOOP: whole iteration is one inner loop.
            if ((f & (uint)NpyIterFlags.EXLOOP) != 0 && _state->NDim <= 1) return true;
            // Buffered and whole iteration fits in one buffer fill.
            if ((f & (uint)NpyIterFlags.BUFFER) != 0 && _state->BufIterEnd >= _state->IterSize) return true;
            return false;
        }

        /// <summary>
        /// Reducing variant. The accumulator is passed by reference; return
        /// <c>false</c> from the kernel to abort (used by All/Any early exit).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public TAccum ExecuteReducing<TKernel, TAccum>(TKernel kernel, TAccum init)
            where TKernel : struct, INpyReducingInnerLoop<TAccum>
            where TAccum : unmanaged
        {
            void** dataptrs = GetDataPtrArray();
            long* byteStrides = GetInnerLoopByteStrides();
            TAccum accum = init;

            if (IsSingleInnerLoop())
            {
                kernel.Execute(dataptrs, byteStrides, ResolveInnerLoopCount(), ref accum);
                return accum;
            }

            var iternext = GetIterNext();

            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                long* bufSize = GetInnerLoopSizePtr();
                do
                {
                    if (!kernel.Execute(dataptrs, byteStrides, *bufSize, ref accum))
                        break;
                } while (iternext(ref *_state));
                return accum;
            }

            long innerSize = ResolveInnerLoopCount();
            do
            {
                if (!kernel.Execute(dataptrs, byteStrides, innerSize, ref accum))
                    break;
            } while (iternext(ref *_state));
            return accum;
        }

        // =====================================================================
        // Layer 2: Typed helpers — generate and run an ILKernelGenerator kernel
        // =====================================================================

        /// <summary>
        /// Run a binary ufunc over three operands [in0, in1, out].
        /// Picks SimdFull / SimdScalarRight / SimdScalarLeft / SimdChunk /
        /// General based on the iterator's stride picture after coalescing.
        /// </summary>
        public void ExecuteBinary(BinaryOp op)
        {
            if (_state->NOp != 3)
                throw new InvalidOperationException(
                    $"ExecuteBinary requires 3 operands (in0, in1, out); got {_state->NOp}.");

            // Buffered path needs the whole-array kernel signature because the
            // iterator writes into aligned buffers whose strides == elementSize.
            if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            {
                RunBufferedBinary(op);
                return;
            }

            var key = new MixedTypeKernelKey(
                _state->GetOpDType(0),
                _state->GetOpDType(1),
                _state->GetOpDType(2),
                op,
                DetectExecutionPath());

            var kernel = ILKernelGenerator.GetMixedTypeKernel(key);

            // Gather byte-stride arrays per operand, sized NDim.
            int ndim = _state->NDim;
            long* lhsStrides = stackalloc long[Math.Max(1, ndim)];
            long* rhsStrides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, lhsStrides, ndim);
            FillElementStrides(1, rhsStrides, ndim);

            kernel(
                _state->GetDataPtr(0),
                _state->GetDataPtr(1),
                _state->GetDataPtr(2),
                lhsStrides,
                rhsStrides,
                _state->Shape,
                ndim,
                _state->IterSize);
        }

        /// <summary>
        /// Run a unary op over [in, out].
        /// </summary>
        public void ExecuteUnary(UnaryOp op)
        {
            if (_state->NOp != 2)
                throw new InvalidOperationException(
                    $"ExecuteUnary requires 2 operands (in, out); got {_state->NOp}.");

            int ndim = _state->NDim;
            bool isContig = (_state->ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0;

            var key = new UnaryKernelKey(
                _state->GetOpDType(0),
                _state->GetOpDType(1),
                op,
                isContig);

            var kernel = ILKernelGenerator.GetUnaryKernel(key);

            long* strides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, strides, ndim);

            kernel(
                _state->GetDataPtr(0),
                _state->GetDataPtr(1),
                strides,
                _state->Shape,
                ndim,
                _state->IterSize);
        }

        /// <summary>
        /// Reduce a single operand to a scalar of type <typeparamref name="TResult"/>.
        /// If the iterator has BUFFER + REDUCE set, the double-loop reduction
        /// schedule is used via <see cref="BufferedReduce{TResult}"/>. Otherwise
        /// we let the IL kernel iterate the array directly.
        /// </summary>
        public TResult ExecuteReduction<TResult>(ReductionOp op) where TResult : unmanaged
        {
            if (_state->NOp != 1)
                throw new InvalidOperationException(
                    $"ExecuteReduction requires 1 operand; got {_state->NOp}.");

            uint f = _state->ItFlags;
            bool isContig = (f & (uint)NpyIterFlags.CONTIGUOUS) != 0;

            var srcType = _state->GetOpSrcDType(0);
            var accumType = DetermineAccumulatorType(srcType, op, typeof(TResult));

            var key = new ElementReductionKernelKey(srcType, accumType, op, isContig);
            var kernel = ILKernelGenerator.GetTypedElementReductionKernel<TResult>(key);

            int ndim = _state->NDim;
            long* strides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, strides, ndim);

            return kernel(_state->GetDataPtr(0), strides, _state->Shape, ndim, _state->IterSize);
        }

        /// <summary>
        /// Reduction variant that honors REDUCE + BUFFER: uses
        /// <see cref="NpyIterState.BufferedReduceAdvance"/> and
        /// <see cref="IsFirstVisit"/> to initialize the accumulator once per
        /// output slot. This is the NumPy-parity path for axis reductions that
        /// span multiple output elements.
        /// </summary>
        public void BufferedReduce<TKernel, TAccum>(TKernel kernel)
            where TKernel : struct, INpyReducingInnerLoop<TAccum>
            where TAccum : unmanaged
        {
            if ((_state->ItFlags & ((uint)NpyIterFlags.BUFFER | (uint)NpyIterFlags.REDUCE))
                != ((uint)NpyIterFlags.BUFFER | (uint)NpyIterFlags.REDUCE))
            {
                throw new InvalidOperationException(
                    "BufferedReduce requires BUFFER + REDUCE flags on the iterator.");
            }

            void** dataptrs = GetDataPtrArray();
            long* strides = GetInnerLoopByteStrides();
            long* innerSize = GetInnerLoopSizePtr();

            // The reduce-accumulator operand's pointer stays pinned while input
            // advances, so *dataptrs[reduce_op] is the accumulator slot.
            // Caller sees current output slot via IsFirstVisit(reduce_op).
            TAccum accum = default;
            do
            {
                // Kernel decides whether to re-init (IsFirstVisit) or continue.
                if (!kernel.Execute(dataptrs, strides, *innerSize, ref accum))
                    break;
            } while (Iternext());  // Iternext picks BufferedReduceIternext internally.
        }

        /// <summary>
        /// Element-wise comparison → bool output. Same 3-operand shape as
        /// ExecuteBinary but the output is always Boolean.
        /// </summary>
        public void ExecuteComparison(ComparisonOp op)
        {
            if (_state->NOp != 3)
                throw new InvalidOperationException(
                    $"ExecuteComparison requires 3 operands; got {_state->NOp}.");

            var key = new ComparisonKernelKey(
                _state->GetOpDType(0),
                _state->GetOpDType(1),
                op,
                DetectExecutionPath());

            var kernel = ILKernelGenerator.GetComparisonKernel(key);

            int ndim = _state->NDim;
            long* lhsStrides = stackalloc long[Math.Max(1, ndim)];
            long* rhsStrides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, lhsStrides, ndim);
            FillElementStrides(1, rhsStrides, ndim);

            kernel(
                _state->GetDataPtr(0),
                _state->GetDataPtr(1),
                (bool*)_state->GetDataPtr(2),
                lhsStrides,
                rhsStrides,
                _state->Shape,
                ndim,
                _state->IterSize);
        }

        /// <summary>
        /// Cumulative scan (CumSum, CumProd) over [in, out].
        /// </summary>
        public void ExecuteScan(ReductionOp op)
        {
            if (_state->NOp != 2)
                throw new InvalidOperationException(
                    $"ExecuteScan requires 2 operands (in, out); got {_state->NOp}.");

            int ndim = _state->NDim;
            bool isContig = (_state->ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0;

            var key = new CumulativeKernelKey(
                _state->GetOpDType(0),
                _state->GetOpDType(1),
                op,
                isContig);

            var kernel = ILKernelGenerator.GetCumulativeKernel(key);

            long* strides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, strides, ndim);

            kernel(
                _state->GetDataPtr(0),
                _state->GetDataPtr(1),
                strides,
                _state->Shape,
                ndim,
                _state->IterSize);
        }

        /// <summary>
        /// Same-type copy with broadcast. When both operands are contiguous
        /// the kernel collapses to <c>cpblk</c>.
        /// </summary>
        public void ExecuteCopy()
        {
            if (_state->NOp != 2)
                throw new InvalidOperationException(
                    $"ExecuteCopy requires 2 operands; got {_state->NOp}.");

            var dtype = _state->GetOpDType(1);  // target dtype
            bool bothContig = (_state->ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0;
            var path = bothContig ? CopyExecutionPath.Contiguous : CopyExecutionPath.General;
            var kernel = ILKernelGenerator.GetCopyKernel(new CopyKernelKey(dtype, path));

            int ndim = _state->NDim;
            long* srcStrides = stackalloc long[Math.Max(1, ndim)];
            long* dstStrides = stackalloc long[Math.Max(1, ndim)];
            FillElementStrides(0, srcStrides, ndim);
            FillElementStrides(1, dstStrides, ndim);

            kernel(
                _state->GetDataPtr(0),
                _state->GetDataPtr(1),
                srcStrides,
                dstStrides,
                _state->Shape,
                ndim,
                _state->IterSize);
        }

        // =====================================================================
        // Path detection & helpers
        // =====================================================================

        /// <summary>
        /// Pick the right <see cref="ExecutionPath"/> for MixedType/Comparison
        /// kernel selection by scanning the post-coalesce stride picture.
        /// </summary>
        public ExecutionPath DetectExecutionPath()
        {
            if ((_state->ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0)
                return ExecutionPath.SimdFull;

            int ndim = _state->NDim;
            if (ndim == 0)
                return ExecutionPath.SimdFull;

            // "Scalar" = every stride is 0 across all dims (0-d or fully broadcast).
            bool op0Scalar = OperandIsScalar(0);
            bool op1Scalar = _state->NOp >= 2 && OperandIsScalar(1);

            if (op1Scalar && OperandIsContiguous(0)) return ExecutionPath.SimdScalarRight;
            if (op0Scalar && _state->NOp >= 2 && OperandIsContiguous(1)) return ExecutionPath.SimdScalarLeft;

            // Inner-dim contiguous for all operands = chunkable
            bool chunkable = true;
            for (int op = 0; op < _state->NOp; op++)
            {
                long inner = _state->GetStride(ndim - 1, op);
                if (inner != 0 && inner != 1) { chunkable = false; break; }
            }
            if (chunkable) return ExecutionPath.SimdChunk;

            return ExecutionPath.General;
        }

        private bool OperandIsScalar(int op)
        {
            for (int d = 0; d < _state->NDim; d++)
                if (_state->GetStride(d, op) != 0) return false;
            return true;
        }

        private bool OperandIsContiguous(int op)
        {
            long expected = 1;
            for (int d = _state->NDim - 1; d >= 0; d--)
            {
                long dim = _state->Shape[d];
                if (dim == 0) return true;
                if (dim != 1)
                {
                    if (_state->GetStride(d, op) != expected) return false;
                    expected *= dim;
                }
            }
            return true;
        }

        /// <summary>
        /// Copy operand <paramref name="op"/>'s post-coalesce element strides
        /// into <paramref name="dst"/>. The destination buffer must hold at
        /// least <paramref name="ndim"/> longs.
        ///
        /// ILKernelGenerator kernels expect ELEMENT strides (they multiply by
        /// elementSize internally). Do NOT convert to bytes here.
        /// </summary>
        private void FillElementStrides(int op, long* dst, int ndim)
        {
            for (int d = 0; d < ndim; d++)
                dst[d] = _state->GetStride(d, op);
        }

        /// <summary>
        /// Unified view of the inner-loop strides as bytes, regardless of
        /// whether the iterator is buffered. For buffered operands we reuse
        /// <see cref="NpyIterState.BufStrides"/> (already bytes); for
        /// non-buffered we convert element strides.
        /// </summary>
        private long* GetInnerLoopByteStrides()
        {
            bool buffered = (_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0;
            if (buffered)
                return _state->BufStrides;  // already bytes

            // Element strides for innermost axis × element size.
            // Stash in a heap buffer that lives as long as the state.
            // (Cheap: one per operand, reused across ForEach calls.)
            int nop = _state->NOp;
            long* cache = _state->InnerStrides;  // repurposed — filled below in bytes
            int inner = _state->NDim - 1;
            if (_state->NDim == 0)
            {
                for (int op = 0; op < nop; op++) cache[op] = 0;
            }
            else
            {
                for (int op = 0; op < nop; op++)
                    cache[op] = _state->GetStride(inner, op) * _state->ElementSizes[op];
            }
            return cache;
        }

        /// <summary>
        /// Determine the accumulator dtype given the source dtype and op.
        /// Mirrors NEP50 widening (int32→int64 for Sum/Prod/CumSum, etc.).
        /// </summary>
        private static NPTypeCode DetermineAccumulatorType(NPTypeCode src, ReductionOp op, Type result)
        {
            // Sum/Prod/CumSum widen integer inputs to int64/uint64.
            if (op == ReductionOp.Sum || op == ReductionOp.Prod ||
                op == ReductionOp.CumSum || op == ReductionOp.CumProd)
            {
                return src switch
                {
                    NPTypeCode.Boolean => NPTypeCode.Int64,
                    NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.Int32 => NPTypeCode.Int64,
                    NPTypeCode.UInt16 or NPTypeCode.UInt32 => NPTypeCode.UInt64,
                    _ => src,
                };
            }
            // Mean/Var/Std always compute in double.
            if (op == ReductionOp.Mean || op == ReductionOp.Var || op == ReductionOp.Std)
                return NPTypeCode.Double;
            return src;
        }

        // =====================================================================
        // Buffered binary path — avoids the Strides/ElementSizes mismatch bug
        // =====================================================================

        /// <summary>
        /// When BUFFERED is set, run the inner loop against the buffer instead
        /// of the source array, using BufStrides (already element-size-matched
        /// to the buffer dtype). After the kernel fills the output buffer,
        /// write-back happens via NpyIterBufferManager.CopyFromBuffer on the
        /// WRITE operand.
        /// </summary>
        private void RunBufferedBinary(BinaryOp op)
        {
            var key = new MixedTypeKernelKey(
                _state->GetOpDType(0),
                _state->GetOpDType(1),
                _state->GetOpDType(2),
                op,
                ExecutionPath.SimdFull);  // buffers are always contiguous
            var kernel = ILKernelGenerator.GetMixedTypeKernel(key);

            // Single-axis byte strides for each operand = element size (buffer is tight).
            long s0 = _state->BufStrides[0];
            long s1 = _state->BufStrides[1];
            long s2 = _state->BufStrides[2];
            long* lhsStr = &s0;
            long* rhsStr = &s1;

            long shape0 = _state->BufIterEnd;
            long* shape = &shape0;

            // Drive the outer loop across buffer fills.
            do
            {
                kernel(
                    _state->GetBuffer(0),
                    _state->GetBuffer(1),
                    _state->GetBuffer(2),
                    lhsStr, rhsStr, shape, 1, _state->BufIterEnd);

                // Flush the output buffer back into its array slot.
                NpyIterBufferManager.CopyFromBuffer(ref *_state, 2, _state->BufIterEnd);
            } while (Iternext());  // Iternext re-fills input buffers on each pass.
        }

        // =====================================================================
        // Test-visible accessors (internal) — let the bridge tests poke state.
        // =====================================================================

        public NpyIterState* RawState => _state;
    }
}
