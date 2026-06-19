using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Reduction.Axis.Widening.cs — Widening SIMD Axis Reduction
// =============================================================================
//
// Covers axis reductions whose accumulator is WIDER than the input dtype
// (NEP50: sum/prod of narrow ints accumulate in int64/uint64, mean of any
// int accumulates in double). One generic core serves every pair:
//
//   input row (TIn, contiguous) ──VPMOVSX/VPMOVZX/VCVT──► Vector256<TAcc>
//                                                            │ op
//   output slab (TAcc) ◄───────── combine + store ───────────┘
//
// Loop structure (leading axis, axis < ndim-1 or contig-inner-slab axis 0):
//
//   for each 8192-element column block of the output slab:   (64 KB — L2-hot)
//       init block to identity
//       for each axis row:                                    (input STREAMS)
//           block[j] = op(block[j], widen(row[j]))            (4x unrolled SIMD)
//
// The block (the output itself) is the accumulator and stays cache-resident;
// the input is read exactly once, sequentially, at full DRAM bandwidth. This
// replaces the previous column-tiled scheme (register accumulators walking
// all rows per 32-column tile) whose row-stride jumps defeated the hardware
// prefetcher beyond page granularity and left it latency-bound — measured
// 24 ms vs 5.3 ms NumPy on sum(int32 3162x3162, axis=0); the same-T kernels
// (AxisReductionLeadingTyped) already stream this way and beat NumPy.
//
// Element ORDER is preserved by construction: every widen loads exactly four
// consecutive TIn elements and produces one element-ordered Vector256<TAcc>
// (x86 PMOVSX/PMOVZX/CVTPS2PD semantics). The previous uint32 path built
// accumulators via Avx2.UnpackLow/UnpackHigh, which interleave PER 128-BIT
// LANE — accumulator slots held columns (0,1,4,5)/(2,3,6,7) but were stored
// to (0..3)/(4..7), silently swapping columns 2,3 with 4,5 in every group of
// eight. That bug is structurally impossible here.
//
// Coverage (Sum/Prod/Min/Max; Mean when the accumulator is Double):
//   sbyte/short/int            -> long     (sign-extend)
//   byte/ushort/uint           -> ulong    (zero-extend; also serves Int64
//                                           accumulators — bit-identical)
//   bool -> byte, char -> ushort            (alias routes)
//   sbyte/byte/short/ushort/int/uint/float -> double
//   (u)long -> double is NOT covered (no AVX2 64-bit-int <-> double convert);
//   it falls back to the typed scalar path, as before.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Accumulator elements per leading-axis column block. 8192 × 8 bytes
        /// = 64 KB — resident in L2 across all axis rows, while input row
        /// segments stream through. Mirrors NumPy's 8192-element buffersize.
        /// </summary>
        private const long WideningColumnBlock = 8192;

        // ---------------------------------------------------------------------
        // Public dispatcher: try widening SIMD for (input, accum) promotion.
        // Returns null when the pair/op isn't covered — caller falls back to
        // the general scalar kernels.
        // ---------------------------------------------------------------------
        internal static unsafe AxisReductionKernel TryGetAxisReductionWideningKernel(
            AxisReductionKernelKey key)
        {
            var op = key.Op;

            // Mean accumulates as Sum and post-divides; it is only routed here
            // when the accumulator is double (mean of any int dtype). Integer
            // Mean accumulators (mean(int16, dtype=int32) etc.) stay scalar.
            if (op == ReductionOp.Mean)
            {
                if (key.AccumulatorType != NPTypeCode.Double) return null;
            }
            else if (op != ReductionOp.Sum && op != ReductionOp.Prod &&
                     op != ReductionOp.Min && op != ReductionOp.Max)
            {
                return null;
            }

            if (!Avx2.IsSupported) return null;

            // Pair table — single source of truth. Alias routes reinterpret
            // bit-identical pointers (char==ushort, bool==byte storage, and
            // Int64 accumulators for zero-extended unsigned inputs).
            return (key.InputType, key.AccumulatorType) switch
            {
                // ---- integer -> 64-bit integer accumulator ----
                (NPTypeCode.Int16, NPTypeCode.Int64) => MakeWideningKernel<short, long, WidenI16ToI64>(op),
                (NPTypeCode.UInt16, NPTypeCode.UInt64) => MakeWideningKernel<ushort, ulong, WidenU16ToU64>(op),
                (NPTypeCode.UInt16, NPTypeCode.Int64) => MakeWideningKernel<ushort, ulong, WidenU16ToU64>(op),
                (NPTypeCode.Char, NPTypeCode.UInt64) => MakeWideningKernel<ushort, ulong, WidenU16ToU64>(op),
                (NPTypeCode.Char, NPTypeCode.Int64) => MakeWideningKernel<ushort, ulong, WidenU16ToU64>(op),
                (NPTypeCode.SByte, NPTypeCode.Int64) => MakeWideningKernel<sbyte, long, WidenI8ToI64>(op),
                (NPTypeCode.Byte, NPTypeCode.UInt64) => MakeWideningKernel<byte, ulong, WidenU8ToU64>(op),
                (NPTypeCode.Byte, NPTypeCode.Int64) => MakeWideningKernel<byte, ulong, WidenU8ToU64>(op),
                (NPTypeCode.Boolean, NPTypeCode.Int64) => MakeWideningKernel<byte, ulong, WidenU8ToU64>(op),
                (NPTypeCode.Boolean, NPTypeCode.UInt64) => MakeWideningKernel<byte, ulong, WidenU8ToU64>(op),
                (NPTypeCode.Int32, NPTypeCode.Int64) => MakeWideningKernel<int, long, WidenI32ToI64>(op),
                (NPTypeCode.UInt32, NPTypeCode.UInt64) => MakeWideningKernel<uint, ulong, WidenU32ToU64>(op),
                (NPTypeCode.UInt32, NPTypeCode.Int64) => MakeWideningKernel<uint, ulong, WidenU32ToU64>(op),

                // ---- anything -> double accumulator (sum/prod/min/max/mean) ----
                (NPTypeCode.Single, NPTypeCode.Double) => MakeWideningKernel<float, double, WidenF32ToF64>(op),
                (NPTypeCode.Int32, NPTypeCode.Double) => MakeWideningKernel<int, double, WidenI32ToF64>(op),
                (NPTypeCode.UInt32, NPTypeCode.Double) => MakeWideningKernel<uint, double, WidenU32ToF64>(op),
                (NPTypeCode.Int16, NPTypeCode.Double) => MakeWideningKernel<short, double, WidenI16ToF64>(op),
                (NPTypeCode.UInt16, NPTypeCode.Double) => MakeWideningKernel<ushort, double, WidenU16ToF64>(op),
                (NPTypeCode.Char, NPTypeCode.Double) => MakeWideningKernel<ushort, double, WidenU16ToF64>(op),
                (NPTypeCode.SByte, NPTypeCode.Double) => MakeWideningKernel<sbyte, double, WidenI8ToF64>(op),
                (NPTypeCode.Byte, NPTypeCode.Double) => MakeWideningKernel<byte, double, WidenU8ToF64>(op),
                (NPTypeCode.Boolean, NPTypeCode.Double) => MakeWideningKernel<byte, double, WidenU8ToF64>(op),

                _ => null
            };
        }

        private static unsafe AxisReductionKernel MakeWideningKernel<TIn, TAcc, TW>(ReductionOp op)
            where TIn : unmanaged
            where TAcc : unmanaged
            where TW : struct, IWidenLoad<TAcc>
        {
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                RunWideningPair<TIn, TAcc, TW>(
                    (TIn*)input, (TAcc*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize, op);
            };
        }

        // ---------------------------------------------------------------------
        // Per-pair driver: layout detection + op dispatch happen ONCE, then the
        // hot loop runs fully specialized (no branches on op/type inside).
        // ---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RunWideningPair<TIn, TAcc, TW>(
            TIn* input, TAcc* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where TIn : unmanaged
            where TAcc : unmanaged
            where TW : struct, IWidenLoad<TAcc>
        {
            // Mean accumulates as Sum, divides at the end.
            var loopOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;

            // Fast path 1: leading axis (axis < ndim-1) with C-contig array,
            // or axis 0 with a contiguous inner slab (covers a[::2,:],
            // a[100:900,100:900], reversed outer, etc).
            bool isLeadingCContig = axis < ndim - 1 && IsCContig(inputStrides, inputShape, ndim);
            bool isLeadingInnerSlab = axis == 0 && ndim >= 2 &&
                                      IsInnerSlabCContig(inputStrides, inputShape, 0, ndim);

            if (isLeadingCContig || isLeadingInnerSlab)
            {
                long innerSize = 1;
                for (int d = axis + 1; d < ndim; d++) innerSize *= inputShape[d];
                long outerSize = 1;
                for (int d = 0; d < axis; d++) outerSize *= inputShape[d];
                long axisStride = inputStrides[axis];

                // Sum (the hot op) dispatches to fully concrete per-pair loops
                // — see the "Concrete Sum kernels" region for why generics are
                // banned from this hot path. Falls through to the generic
                // kernels for the →double pairs (Mean) and rare ops.
                if (loopOp == ReductionOp.Sum &&
                    TrySumLeadingConcrete<TIn, TAcc>(input, output, outerSize, axisSize, innerSize, axisStride))
                {
                    // handled
                }
                else switch (loopOp)
                {
                    case ReductionOp.Sum:
                        WideningLeading<TIn, TAcc, TW, WAddOp<TAcc>>(input, output, outerSize, axisSize, innerSize, axisStride);
                        break;
                    case ReductionOp.Prod:
                        WideningLeading<TIn, TAcc, TW, WMulOp<TAcc>>(input, output, outerSize, axisSize, innerSize, axisStride);
                        break;
                    case ReductionOp.Min:
                        WideningLeading<TIn, TAcc, TW, WMinOp<TAcc>>(input, output, outerSize, axisSize, innerSize, axisStride);
                        break;
                    case ReductionOp.Max:
                        WideningLeading<TIn, TAcc, TW, WMaxOp<TAcc>>(input, output, outerSize, axisSize, innerSize, axisStride);
                        break;
                }

                if (op == ReductionOp.Mean)
                    DivideOutputByCount(output, outputSize, axisSize);
                return;
            }

            // Fast path 2: innermost axis (axis == ndim-1) with C-contig array.
            // Each output reduces a contiguous run of axisSize inputs.
            if (axis == ndim - 1 && IsCContig(inputStrides, inputShape, ndim))
            {
                if (loopOp == ReductionOp.Sum &&
                    TrySumInnermostConcrete<TIn, TAcc>(input, output, outputSize, axisSize))
                {
                    // handled
                }
                else switch (loopOp)
                {
                    case ReductionOp.Sum:
                        WideningInnermost<TIn, TAcc, TW, WAddOp<TAcc>>(input, output, outputSize, axisSize);
                        break;
                    case ReductionOp.Prod:
                        WideningInnermost<TIn, TAcc, TW, WMulOp<TAcc>>(input, output, outputSize, axisSize);
                        break;
                    case ReductionOp.Min:
                        WideningInnermost<TIn, TAcc, TW, WMinOp<TAcc>>(input, output, outputSize, axisSize);
                        break;
                    case ReductionOp.Max:
                        WideningInnermost<TIn, TAcc, TW, WMaxOp<TAcc>>(input, output, outputSize, axisSize);
                        break;
                }

                if (op == ReductionOp.Mean)
                    DivideOutputByCount(output, outputSize, axisSize);
                return;
            }

            // Fast path 3: 2-D non-C-contiguous inputs with a stride-1 axis
            // (F-order, transposed, sliced-along-the-strided-axis). The leading/
            // innermost SIMD kernels above already handle an arbitrary axisStride
            // with a contiguous inner slab (WideningLeading) or consecutive
            // contiguous blocks (WideningInnermost) — they were simply never
            // REACHED because the gates above key off C-contiguity / axis position.
            // Reading the orientation from the strides instead lets F-order axis0/
            // axis1 and a.T stop falling to the scalar path: the int32->int64
            // widening-sum layout cliff (F-order/transposed measured 14-20x slower
            // than NumPy, while C-contig wins ~10x — same kernel, just not routed).
            if (ndim == 2)
            {
                int other = 1 - axis;
                long axisStride = inputStrides[axis];

                // SLAB: the non-reduced axis is contiguous (stride 1) → it is the
                // inner output slab and the reduced axis is streamed with axisStride.
                // Covers F-order reduce-last-axis (inner = the leading contiguous
                // axis). One 2-D slab ⇒ outerSize = 1. axisStride may be 0
                // (broadcast reduce) — WideningLeading folds it correctly.
                if (inputStrides[other] == 1L)
                {
                    long innerSize = inputShape[other];
                    if (loopOp == ReductionOp.Sum &&
                        TrySumLeadingConcrete<TIn, TAcc>(input, output, 1, axisSize, innerSize, axisStride))
                    {
                        // handled by the concrete sum kernel
                    }
                    else switch (loopOp)
                    {
                        case ReductionOp.Sum:  WideningLeading<TIn, TAcc, TW, WAddOp<TAcc>>(input, output, 1, axisSize, innerSize, axisStride); break;
                        case ReductionOp.Prod: WideningLeading<TIn, TAcc, TW, WMulOp<TAcc>>(input, output, 1, axisSize, innerSize, axisStride); break;
                        case ReductionOp.Min:  WideningLeading<TIn, TAcc, TW, WMinOp<TAcc>>(input, output, 1, axisSize, innerSize, axisStride); break;
                        case ReductionOp.Max:  WideningLeading<TIn, TAcc, TW, WMaxOp<TAcc>>(input, output, 1, axisSize, innerSize, axisStride); break;
                    }
                    if (op == ReductionOp.Mean) DivideOutputByCount(output, outputSize, axisSize);
                    return;
                }

                // PINNED: the reduced axis is contiguous (stride 1) AND tiles into
                // CONSECUTIVE blocks (stride[other] == axisSize), so each output
                // cell reduces a contiguous run of axisSize elements. Covers F-order
                // reduce-axis0 and a.T reduce-axis0. The block-consecutive guard
                // rejects sliced F-order views (block gaps) — they stay scalar.
                if (axisStride == 1L && inputStrides[other] == axisSize)
                {
                    if (loopOp == ReductionOp.Sum &&
                        TrySumInnermostConcrete<TIn, TAcc>(input, output, outputSize, axisSize))
                    {
                        // handled by the concrete sum kernel
                    }
                    else switch (loopOp)
                    {
                        case ReductionOp.Sum:  WideningInnermost<TIn, TAcc, TW, WAddOp<TAcc>>(input, output, outputSize, axisSize); break;
                        case ReductionOp.Prod: WideningInnermost<TIn, TAcc, TW, WMulOp<TAcc>>(input, output, outputSize, axisSize); break;
                        case ReductionOp.Min:  WideningInnermost<TIn, TAcc, TW, WMinOp<TAcc>>(input, output, outputSize, axisSize); break;
                        case ReductionOp.Max:  WideningInnermost<TIn, TAcc, TW, WMaxOp<TAcc>>(input, output, outputSize, axisSize); break;
                    }
                    if (op == ReductionOp.Mean) DivideOutputByCount(output, outputSize, axisSize);
                    return;
                }
            }

            // General layout (strided axis, transposed, sliced inner dim):
            // typed scalar with promotion — handles Mean's divide internally.
            AxisReductionScalarHelper<TIn, TAcc>(
                input, output, inputStrides, inputShape, outputStrides,
                axis, axisSize, ndim, outputSize, op);
        }

        // =====================================================================
        // Concrete Sum kernels — the hot tier.
        //
        // GENERICS ARE BANNED FROM THIS HOT PATH, with measurements (int16
        // axis-0 sum, 10M, this machine):
        //   fully concrete loops:                        ~6.5 ms  (this tier)
        //   + int32-chunked accumulation (used below):   ~3.6 ms  (beats NumPy 4.55)
        //   one interface hop (static abstract OR        ~10-11 ms
        //   instance, byte* or Vector256 params):
        //   two interface hops (op + load):              ~14.5 ms
        //   generic struct / static generic helper       ~23-45 ms
        //   with typeof(T) chains:
        // The JIT neither folds typeof chains nor fully inlines interface
        // dispatch at this call density (one widen per 4 elements — 4x the
        // density of the same-T kernels, where the identical pattern is
        // tolerable). See also the Add256<T> post-mortem in
        // DirectILKernelGenerator.Reduction.Axis.Simd.cs.
        //
        // Loop structure (leading): the output slab is the accumulator and
        // stays L2-resident per 8192-column block; rows stream sequentially.
        // 8/16-bit inputs accumulate into an int32/uint32 SCRATCH block
        // (halves the accumulator traffic and widens 8 lanes per instruction
        // instead of 4), drained into the int64 output every CHUNK rows —
        // chunk sizes are chosen so a scratch lane cannot overflow:
        //   int16:  16384 rows * 32768  = 2^29  < 2^31
        //   uint16: 32768 rows * 65535  < 2^32
        //   int8:   8M    rows * 128    < 2^31      (one drain in practice)
        //   uint8:  8M    rows * 255    < 2^32
        // 32-bit inputs widen directly to 64-bit (VPMOVSX/ZXDQ + VPADDQ).
        // =====================================================================

        /// <summary>Rows accumulated in int32/uint32 scratch before draining to 64-bit.</summary>
        private const long ChunkI16 = 16384;
        private const long ChunkU16 = 32768;
        private const long ChunkI8 = 8_388_608;
        private const long ChunkU8 = 8_388_608;

        /// <summary>
        /// Dispatch Sum to a concrete per-pair leading-axis kernel. The typeof
        /// chain runs ONCE per call (not per element). Returns false for pairs
        /// without a concrete kernel (the →double pairs) — caller falls back
        /// to the generic tier.
        /// </summary>
        private static unsafe bool TrySumLeadingConcrete<TIn, TAcc>(
            TIn* input, TAcc* output, long outerSize, long axisSize, long innerSize, long axisStride)
            where TIn : unmanaged where TAcc : unmanaged
        {
            if (typeof(TIn) == typeof(short) && typeof(TAcc) == typeof(long))
            { SumLeadingI16I64((short*)input, (long*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            if (typeof(TIn) == typeof(ushort) && typeof(TAcc) == typeof(ulong))
            { SumLeadingU16U64((ushort*)input, (ulong*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            if (typeof(TIn) == typeof(sbyte) && typeof(TAcc) == typeof(long))
            { SumLeadingI8I64((sbyte*)input, (long*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            if (typeof(TIn) == typeof(byte) && typeof(TAcc) == typeof(ulong))
            { SumLeadingU8U64((byte*)input, (ulong*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            if (typeof(TIn) == typeof(int) && typeof(TAcc) == typeof(long))
            { SumLeadingI32I64((int*)input, (long*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            if (typeof(TIn) == typeof(uint) && typeof(TAcc) == typeof(ulong))
            { SumLeadingU32U64((uint*)input, (ulong*)output, outerSize, axisSize, innerSize, axisStride); return true; }
            return false;
        }

        /// <summary>Innermost-axis counterpart of <see cref="TrySumLeadingConcrete{TIn,TAcc}"/>.</summary>
        private static unsafe bool TrySumInnermostConcrete<TIn, TAcc>(
            TIn* input, TAcc* output, long outputSize, long axisSize)
            where TIn : unmanaged where TAcc : unmanaged
        {
            if (typeof(TIn) == typeof(short) && typeof(TAcc) == typeof(long))
            { SumInnermostI16I64((short*)input, (long*)output, outputSize, axisSize); return true; }
            if (typeof(TIn) == typeof(ushort) && typeof(TAcc) == typeof(ulong))
            { SumInnermostU16U64((ushort*)input, (ulong*)output, outputSize, axisSize); return true; }
            if (typeof(TIn) == typeof(sbyte) && typeof(TAcc) == typeof(long))
            { SumInnermostI8I64((sbyte*)input, (long*)output, outputSize, axisSize); return true; }
            if (typeof(TIn) == typeof(byte) && typeof(TAcc) == typeof(ulong))
            { SumInnermostU8U64((byte*)input, (ulong*)output, outputSize, axisSize); return true; }
            if (typeof(TIn) == typeof(int) && typeof(TAcc) == typeof(long))
            { SumInnermostI32I64((int*)input, (long*)output, outputSize, axisSize); return true; }
            if (typeof(TIn) == typeof(uint) && typeof(TAcc) == typeof(ulong))
            { SumInnermostU32U64((uint*)input, (ulong*)output, outputSize, axisSize); return true; }
            return false;
        }

        // ---------------------------------------------------------------------
        // Leading axis, 16-bit inputs — int32/uint32 chunked scratch.
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingI16I64(
            short* input, long* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;
            int* scratch = stackalloc int[(int)WideningColumnBlock];

            for (long o = 0; o < outerSize; o++)
            {
                short* inBase = input + o * inputSlab;
                long* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    long* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<long>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long aChunk = 0; aChunk < axisSize; aChunk += ChunkI16)
                    {
                        long aEnd = Math.Min(aChunk + ChunkI16, axisSize);

                        j = 0;
                        for (; j + 8 <= bw; j += 8) Vector256.Store(Vector256<int>.Zero, scratch + j);
                        for (; j < bw; j++) scratch[j] = 0;

                        for (long a = aChunk; a < aEnd; a++)
                        {
                            short* row = inBase + a * axisStride + c0;
                            j = 0;
                            long ue = bw - 16;
                            for (; j <= ue; j += 16)
                            {
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.Load(row + j))), scratch + j);
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j + 8), Avx2.ConvertToVector256Int32(Vector128.Load(row + j + 8))), scratch + j + 8);
                            }
                            for (; j + 8 <= bw; j += 8)
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.Load(row + j))), scratch + j);
                            for (; j < bw; j++) scratch[j] += row[j];
                        }

                        j = 0;
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(scratch + j))), outB + j);
                        for (; j < bw; j++) outB[j] += scratch[j];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingU16U64(
            ushort* input, ulong* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;
            uint* scratch = stackalloc uint[(int)WideningColumnBlock];

            for (long o = 0; o < outerSize; o++)
            {
                ushort* inBase = input + o * inputSlab;
                ulong* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    ulong* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<ulong>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long aChunk = 0; aChunk < axisSize; aChunk += ChunkU16)
                    {
                        long aEnd = Math.Min(aChunk + ChunkU16, axisSize);

                        j = 0;
                        for (; j + 8 <= bw; j += 8) Vector256.Store(Vector256<uint>.Zero, scratch + j);
                        for (; j < bw; j++) scratch[j] = 0;

                        for (long a = aChunk; a < aEnd; a++)
                        {
                            ushort* row = inBase + a * axisStride + c0;
                            j = 0;
                            long ue = bw - 16;
                            for (; j <= ue; j += 16)
                            {
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.Load(row + j)).AsUInt32()), scratch + j);
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j + 8), Avx2.ConvertToVector256Int32(Vector128.Load(row + j + 8)).AsUInt32()), scratch + j + 8);
                            }
                            for (; j + 8 <= bw; j += 8)
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.Load(row + j)).AsUInt32()), scratch + j);
                            for (; j < bw; j++) scratch[j] += row[j];
                        }

                        j = 0;
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(scratch + j)).AsUInt64()), outB + j);
                        for (; j < bw; j++) outB[j] += scratch[j];
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Leading axis, 8-bit inputs — int32/uint32 chunked scratch (8 lanes
        // per VPMOVSX/ZXBD; one drain in practice given the huge chunk).
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingI8I64(
            sbyte* input, long* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;
            int* scratch = stackalloc int[(int)WideningColumnBlock];

            for (long o = 0; o < outerSize; o++)
            {
                sbyte* inBase = input + o * inputSlab;
                long* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    long* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<long>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long aChunk = 0; aChunk < axisSize; aChunk += ChunkI8)
                    {
                        long aEnd = Math.Min(aChunk + ChunkI8, axisSize);

                        j = 0;
                        for (; j + 8 <= bw; j += 8) Vector256.Store(Vector256<int>.Zero, scratch + j);
                        for (; j < bw; j++) scratch[j] = 0;

                        for (long a = aChunk; a < aEnd; a++)
                        {
                            sbyte* row = inBase + a * axisStride + c0;
                            j = 0;
                            long ue = bw - 16;
                            for (; j <= ue; j += 16)
                            {
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j)).AsSByte())), scratch + j);
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j + 8), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j + 8)).AsSByte())), scratch + j + 8);
                            }
                            for (; j + 8 <= bw; j += 8)
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j)).AsSByte())), scratch + j);
                            for (; j < bw; j++) scratch[j] += row[j];
                        }

                        j = 0;
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(scratch + j))), outB + j);
                        for (; j < bw; j++) outB[j] += scratch[j];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingU8U64(
            byte* input, ulong* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;
            uint* scratch = stackalloc uint[(int)WideningColumnBlock];

            for (long o = 0; o < outerSize; o++)
            {
                byte* inBase = input + o * inputSlab;
                ulong* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    ulong* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<ulong>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long aChunk = 0; aChunk < axisSize; aChunk += ChunkU8)
                    {
                        long aEnd = Math.Min(aChunk + ChunkU8, axisSize);

                        j = 0;
                        for (; j + 8 <= bw; j += 8) Vector256.Store(Vector256<uint>.Zero, scratch + j);
                        for (; j < bw; j++) scratch[j] = 0;

                        for (long a = aChunk; a < aEnd; a++)
                        {
                            byte* row = inBase + a * axisStride + c0;
                            j = 0;
                            long ue = bw - 16;
                            for (; j <= ue; j += 16)
                            {
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j)).AsByte()).AsUInt32()), scratch + j);
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j + 8), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j + 8)).AsByte()).AsUInt32()), scratch + j + 8);
                            }
                            for (; j + 8 <= bw; j += 8)
                                Vector256.Store(Avx2.Add(Vector256.Load(scratch + j), Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + j)).AsByte()).AsUInt32()), scratch + j);
                            for (; j < bw; j++) scratch[j] += row[j];
                        }

                        j = 0;
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(scratch + j)).AsUInt64()), outB + j);
                        for (; j < bw; j++) outB[j] += scratch[j];
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Leading axis, 32-bit inputs — direct widen to 64-bit, no scratch.
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingI32I64(
            int* input, long* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;

            for (long o = 0; o < outerSize; o++)
            {
                int* inBase = input + o * inputSlab;
                long* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    long* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<long>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long a = 0; a < axisSize; a++)
                    {
                        int* row = inBase + a * axisStride + c0;
                        j = 0;
                        long ue = bw - 16;
                        for (; j <= ue; j += 16)
                        {
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(row + j))), outB + j);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 4), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 4))), outB + j + 4);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 8), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 8))), outB + j + 8);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 12), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 12))), outB + j + 12);
                        }
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(row + j))), outB + j);
                        for (; j < bw; j++) outB[j] += row[j];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumLeadingU32U64(
            uint* input, ulong* output, long outerSize, long axisSize, long innerSize, long axisStride)
        {
            long inputSlab = axisSize * axisStride;

            for (long o = 0; o < outerSize; o++)
            {
                uint* inBase = input + o * inputSlab;
                ulong* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    ulong* outB = outSlab + c0;
                    long j = 0;
                    for (; j + 4 <= bw; j += 4) Vector256.Store(Vector256<ulong>.Zero, outB + j);
                    for (; j < bw; j++) outB[j] = 0;

                    for (long a = 0; a < axisSize; a++)
                    {
                        uint* row = inBase + a * axisStride + c0;
                        j = 0;
                        long ue = bw - 16;
                        for (; j <= ue; j += 16)
                        {
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(row + j)).AsUInt64()), outB + j);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 4), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 4)).AsUInt64()), outB + j + 4);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 8), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 8)).AsUInt64()), outB + j + 8);
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j + 12), Avx2.ConvertToVector256Int64(Vector128.Load(row + j + 12)).AsUInt64()), outB + j + 12);
                        }
                        for (; j + 4 <= bw; j += 4)
                            Vector256.Store(Avx2.Add(Vector256.Load(outB + j), Avx2.ConvertToVector256Int64(Vector128.Load(row + j)).AsUInt64()), outB + j);
                        for (; j < bw; j++) outB[j] += row[j];
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Innermost axis — per-output flat reduce. Narrow inputs accumulate in
        // int32/uint32 lanes per chunk, drained into a 64-bit vector
        // accumulator (chunk sizes are 16-multiples, so only the final tail
        // is scalar).
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostI16I64(
            short* input, long* output, long outputSize, long axisSize)
        {
            for (long o = 0; o < outputSize; o++)
            {
                short* row = input + o * axisSize;
                var acc64 = Vector256<long>.Zero;
                long i = 0;
                while (i + 16 <= axisSize)
                {
                    long cend = Math.Min(i + ChunkI16, axisSize);
                    var s0 = Vector256<int>.Zero;
                    var s1 = Vector256<int>.Zero;
                    long ve = cend - 16;
                    for (; i <= ve; i += 16)
                    {
                        s0 = Avx2.Add(s0, Avx2.ConvertToVector256Int32(Vector128.Load(row + i)));
                        s1 = Avx2.Add(s1, Avx2.ConvertToVector256Int32(Vector128.Load(row + i + 8)));
                    }
                    var s = Avx2.Add(s0, s1);
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetLower()));
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetUpper()));
                }
                long sum = acc64.GetElement(0) + acc64.GetElement(1) + acc64.GetElement(2) + acc64.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostU16U64(
            ushort* input, ulong* output, long outputSize, long axisSize)
        {
            for (long o = 0; o < outputSize; o++)
            {
                ushort* row = input + o * axisSize;
                var acc64 = Vector256<ulong>.Zero;
                long i = 0;
                while (i + 16 <= axisSize)
                {
                    long cend = Math.Min(i + ChunkU16, axisSize);
                    var s0 = Vector256<uint>.Zero;
                    var s1 = Vector256<uint>.Zero;
                    long ve = cend - 16;
                    for (; i <= ve; i += 16)
                    {
                        s0 = Avx2.Add(s0, Avx2.ConvertToVector256Int32(Vector128.Load(row + i)).AsUInt32());
                        s1 = Avx2.Add(s1, Avx2.ConvertToVector256Int32(Vector128.Load(row + i + 8)).AsUInt32());
                    }
                    var s = Avx2.Add(s0, s1);
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetLower()).AsUInt64());
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetUpper()).AsUInt64());
                }
                ulong sum = acc64.GetElement(0) + acc64.GetElement(1) + acc64.GetElement(2) + acc64.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostI8I64(
            sbyte* input, long* output, long outputSize, long axisSize)
        {
            for (long o = 0; o < outputSize; o++)
            {
                sbyte* row = input + o * axisSize;
                var acc64 = Vector256<long>.Zero;
                long i = 0;
                while (i + 16 <= axisSize)
                {
                    long cend = Math.Min(i + ChunkI8, axisSize);
                    var s0 = Vector256<int>.Zero;
                    var s1 = Vector256<int>.Zero;
                    long ve = cend - 16;
                    for (; i <= ve; i += 16)
                    {
                        s0 = Avx2.Add(s0, Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + i)).AsSByte()));
                        s1 = Avx2.Add(s1, Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + i + 8)).AsSByte()));
                    }
                    var s = Avx2.Add(s0, s1);
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetLower()));
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetUpper()));
                }
                long sum = acc64.GetElement(0) + acc64.GetElement(1) + acc64.GetElement(2) + acc64.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostU8U64(
            byte* input, ulong* output, long outputSize, long axisSize)
        {
            for (long o = 0; o < outputSize; o++)
            {
                byte* row = input + o * axisSize;
                var acc64 = Vector256<ulong>.Zero;
                long i = 0;
                while (i + 16 <= axisSize)
                {
                    long cend = Math.Min(i + ChunkU8, axisSize);
                    var s0 = Vector256<uint>.Zero;
                    var s1 = Vector256<uint>.Zero;
                    long ve = cend - 16;
                    for (; i <= ve; i += 16)
                    {
                        s0 = Avx2.Add(s0, Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + i)).AsByte()).AsUInt32());
                        s1 = Avx2.Add(s1, Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(ulong*)(row + i + 8)).AsByte()).AsUInt32());
                    }
                    var s = Avx2.Add(s0, s1);
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetLower()).AsUInt64());
                    acc64 = Avx2.Add(acc64, Avx2.ConvertToVector256Int64(s.GetUpper()).AsUInt64());
                }
                ulong sum = acc64.GetElement(0) + acc64.GetElement(1) + acc64.GetElement(2) + acc64.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostI32I64(
            int* input, long* output, long outputSize, long axisSize)
        {
            long unrollEnd = axisSize - 16;
            long vecEnd = axisSize - 4;

            for (long o = 0; o < outputSize; o++)
            {
                int* row = input + o * axisSize;
                var a0 = Vector256<long>.Zero; var a1 = Vector256<long>.Zero;
                var a2 = Vector256<long>.Zero; var a3 = Vector256<long>.Zero;
                long i = 0;
                for (; i <= unrollEnd; i += 16)
                {
                    a0 = Avx2.Add(a0, Avx2.ConvertToVector256Int64(Vector128.Load(row + i)));
                    a1 = Avx2.Add(a1, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 4)));
                    a2 = Avx2.Add(a2, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 8)));
                    a3 = Avx2.Add(a3, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 12)));
                }
                var acc = Avx2.Add(Avx2.Add(a0, a1), Avx2.Add(a2, a3));
                for (; i <= vecEnd; i += 4)
                    acc = Avx2.Add(acc, Avx2.ConvertToVector256Int64(Vector128.Load(row + i)));
                long sum = acc.GetElement(0) + acc.GetElement(1) + acc.GetElement(2) + acc.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SumInnermostU32U64(
            uint* input, ulong* output, long outputSize, long axisSize)
        {
            long unrollEnd = axisSize - 16;
            long vecEnd = axisSize - 4;

            for (long o = 0; o < outputSize; o++)
            {
                uint* row = input + o * axisSize;
                var a0 = Vector256<ulong>.Zero; var a1 = Vector256<ulong>.Zero;
                var a2 = Vector256<ulong>.Zero; var a3 = Vector256<ulong>.Zero;
                long i = 0;
                for (; i <= unrollEnd; i += 16)
                {
                    a0 = Avx2.Add(a0, Avx2.ConvertToVector256Int64(Vector128.Load(row + i)).AsUInt64());
                    a1 = Avx2.Add(a1, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 4)).AsUInt64());
                    a2 = Avx2.Add(a2, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 8)).AsUInt64());
                    a3 = Avx2.Add(a3, Avx2.ConvertToVector256Int64(Vector128.Load(row + i + 12)).AsUInt64());
                }
                var acc = Avx2.Add(Avx2.Add(a0, a1), Avx2.Add(a2, a3));
                for (; i <= vecEnd; i += 4)
                    acc = Avx2.Add(acc, Avx2.ConvertToVector256Int64(Vector128.Load(row + i)).AsUInt64());
                ulong sum = acc.GetElement(0) + acc.GetElement(1) + acc.GetElement(2) + acc.GetElement(3);
                for (; i < axisSize; i++) sum += row[i];
                output[o] = sum;
            }
        }

        // =====================================================================
        // Generic tier — Prod/Min/Max and the →double pairs (Mean). One
        // interface hop costs ~40-55% in widen-density loops (see the concrete
        // tier's preamble), which is acceptable for these rarer ops: they are
        // still ~70-90x faster than the scalar promoted fallback.
        //
        // Leading-axis kernel — blocked row-streaming: the output slab IS the
        // accumulator; a 64 KB column block stays L2-resident across all axis
        // rows while input rows stream through sequentially. Input is read
        // exactly once.
        //
        // TW/TOp members are STATIC ABSTRACT and take concrete byte* — a
        // generic-pointee pointer (TIn*) in the interface signature defeats
        // constrained-call devirtualization entirely (boxed virtual call per
        // Load4); byte* keeps the signature concrete.
        // =====================================================================
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WideningLeading<TIn, TAcc, TW, TOp>(
            TIn* input, TAcc* output,
            long outerSize, long axisSize, long innerSize, long axisStride)
            where TIn : unmanaged
            where TAcc : unmanaged
            where TW : struct, IWidenLoad<TAcc>
            where TOp : struct, IWAccOp<TAcc>
        {
            TAcc identity = TOp.Identity();
            var identV = Vector256.Create(identity);

            int vl = Vector256<TAcc>.Count;        // 4 for all 8-byte accumulators
            long unrollStep = vl * 4;

            // Slab origin per outer index. For full C-contig, stride[axis] ==
            // innerSize * (product of dims after axis+1 ... ) such that
            // axisSize*axisStride equals the outer stride; for the inner-slab
            // case outerSize == 1 and the value is unused.
            long inputSlab = axisSize * axisStride;

            for (long o = 0; o < outerSize; o++)
            {
                TIn* inBase = input + o * inputSlab;
                TAcc* outSlab = output + o * innerSize;

                for (long c0 = 0; c0 < innerSize; c0 += WideningColumnBlock)
                {
                    long bw = Math.Min(WideningColumnBlock, innerSize - c0);
                    TAcc* outB = outSlab + c0;

                    // Init the block to the op identity.
                    long j = 0;
                    for (; j + vl <= bw; j += vl)
                        Vector256.Store(identV, outB + j);
                    for (; j < bw; j++)
                        outB[j] = identity;

                    // Stream every axis row through the block.
                    for (long a = 0; a < axisSize; a++)
                    {
                        TIn* row = inBase + a * axisStride + c0;
                        j = 0;
                        long unrollEnd = bw - unrollStep;
                        for (; j <= unrollEnd; j += unrollStep)
                        {
                            Vector256.Store(TOp.Combine(Vector256.Load(outB + j), TW.Load4((byte*)(row + j))), outB + j);
                            Vector256.Store(TOp.Combine(Vector256.Load(outB + j + vl), TW.Load4((byte*)(row + j + vl))), outB + j + vl);
                            Vector256.Store(TOp.Combine(Vector256.Load(outB + j + vl * 2), TW.Load4((byte*)(row + j + vl * 2))), outB + j + vl * 2);
                            Vector256.Store(TOp.Combine(Vector256.Load(outB + j + vl * 3), TW.Load4((byte*)(row + j + vl * 3))), outB + j + vl * 3);
                        }

                        for (; j + vl <= bw; j += vl)
                            Vector256.Store(TOp.Combine(Vector256.Load(outB + j), TW.Load4((byte*)(row + j))), outB + j);

                        for (; j < bw; j++)
                            outB[j] = TOp.CombineScalar(outB[j], TW.LoadScalar((byte*)(row + j)));
                    }
                }
            }
        }

        // =====================================================================
        // Innermost-axis widening kernel — per-output flat reduce.
        //
        // Eight independent vector accumulators break the dependency chain;
        // tree-merge, horizontal reduce, scalar tail.
        // =====================================================================
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WideningInnermost<TIn, TAcc, TW, TOp>(
            TIn* input, TAcc* output, long outputSize, long axisSize)
            where TIn : unmanaged
            where TAcc : unmanaged
            where TW : struct, IWidenLoad<TAcc>
            where TOp : struct, IWAccOp<TAcc>
        {
            var identV = Vector256.Create(TOp.Identity());

            int vl = Vector256<TAcc>.Count;        // 4
            long unrollStep = vl * 8;              // 32 input elements/iter
            long unrollEnd = axisSize - unrollStep;
            long vecEnd = axisSize - vl;

            for (long o = 0; o < outputSize; o++)
            {
                TIn* row = input + o * axisSize;
                long i = 0;

                var a0 = identV; var a1 = identV; var a2 = identV; var a3 = identV;
                var a4 = identV; var a5 = identV; var a6 = identV; var a7 = identV;

                for (; i <= unrollEnd; i += unrollStep)
                {
                    a0 = TOp.Combine(a0, TW.Load4((byte*)(row + i)));
                    a1 = TOp.Combine(a1, TW.Load4((byte*)(row + i + vl)));
                    a2 = TOp.Combine(a2, TW.Load4((byte*)(row + i + vl * 2)));
                    a3 = TOp.Combine(a3, TW.Load4((byte*)(row + i + vl * 3)));
                    a4 = TOp.Combine(a4, TW.Load4((byte*)(row + i + vl * 4)));
                    a5 = TOp.Combine(a5, TW.Load4((byte*)(row + i + vl * 5)));
                    a6 = TOp.Combine(a6, TW.Load4((byte*)(row + i + vl * 6)));
                    a7 = TOp.Combine(a7, TW.Load4((byte*)(row + i + vl * 7)));
                }

                // Tree merge: 8 -> 4 -> 2 -> 1
                var m0 = TOp.Combine(TOp.Combine(a0, a1), TOp.Combine(a2, a3));
                var m1 = TOp.Combine(TOp.Combine(a4, a5), TOp.Combine(a6, a7));
                var acc = TOp.Combine(m0, m1);

                for (; i <= vecEnd; i += vl)
                    acc = TOp.Combine(acc, TW.Load4((byte*)(row + i)));

                TAcc s = HorizontalReduceWidened<TAcc, TOp>(acc);
                for (; i < axisSize; i++)
                    s = TOp.CombineScalar(s, TW.LoadScalar((byte*)(row + i)));

                output[o] = s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static TAcc HorizontalReduceWidened<TAcc, TOp>(Vector256<TAcc> v)
            where TAcc : unmanaged
            where TOp : struct, IWAccOp<TAcc>
        {
            TAcc r = v.GetElement(0);
            for (int j = 1; j < Vector256<TAcc>.Count; j++)
                r = TOp.CombineScalar(r, v.GetElement(j));
            return r;
        }

        /// <summary>
        /// Mean post-pass: divide the accumulated sums by the axis length.
        /// Only double accumulators reach here (dispatcher gate).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DivideOutputByCount<TAcc>(TAcc* output, long count, long divisor)
            where TAcc : unmanaged
        {
            if (typeof(TAcc) != typeof(double))
                throw new InvalidOperationException("Widening Mean requires a double accumulator.");

            double* d = (double*)output;
            var dv = Vector256.Create((double)divisor);
            long i = 0;
            for (; i + 4 <= count; i += 4)
                Vector256.Store(Vector256.Divide(Vector256.Load(d + i), dv), d + i);
            for (; i < count; i++)
                d[i] /= divisor;
        }

        // =====================================================================
        // Accumulator op tags. TAcc is one of {long, ulong, double}; the
        // typeof chains are JIT-folded per generic specialization, so each
        // Combine compiles to a single SIMD instruction. Members are STATIC
        // ABSTRACT — constrained static dispatch devirtualizes and inlines
        // unconditionally.
        // =====================================================================
        internal interface IWAccOp<TAcc> where TAcc : unmanaged
        {
            static abstract Vector256<TAcc> Combine(Vector256<TAcc> a, Vector256<TAcc> b);
            static abstract TAcc CombineScalar(TAcc a, TAcc b);
            static abstract TAcc Identity();
        }

        internal readonly struct WAddOp<TAcc> : IWAccOp<TAcc> where TAcc : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<TAcc> Combine(Vector256<TAcc> a, Vector256<TAcc> b)
            {
                if (typeof(TAcc) == typeof(long) && Avx2.IsSupported)
                    return Avx2.Add(a.As<TAcc, long>(), b.As<TAcc, long>()).As<long, TAcc>();
                if (typeof(TAcc) == typeof(ulong) && Avx2.IsSupported)
                    return Avx2.Add(a.As<TAcc, ulong>(), b.As<TAcc, ulong>()).As<ulong, TAcc>();
                if (typeof(TAcc) == typeof(double) && Avx.IsSupported)
                    return Avx.Add(a.As<TAcc, double>(), b.As<TAcc, double>()).As<double, TAcc>();
                return Vector256.Add(a, b);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc CombineScalar(TAcc a, TAcc b)
            {
                if (typeof(TAcc) == typeof(long)) return (TAcc)(object)((long)(object)a + (long)(object)b);
                if (typeof(TAcc) == typeof(ulong)) return (TAcc)(object)((ulong)(object)a + (ulong)(object)b);
                if (typeof(TAcc) == typeof(double)) return (TAcc)(object)((double)(object)a + (double)(object)b);
                throw new NotSupportedException();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc Identity() => default;     // 0 / 0UL / 0.0
        }

        internal readonly struct WMulOp<TAcc> : IWAccOp<TAcc> where TAcc : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<TAcc> Combine(Vector256<TAcc> a, Vector256<TAcc> b)
            {
                // No AVX2 64-bit integer multiply; Vector256.Multiply emits the
                // vpmuludq decomposition (exact wraparound semantics).
                if (typeof(TAcc) == typeof(double) && Avx.IsSupported)
                    return Avx.Multiply(a.As<TAcc, double>(), b.As<TAcc, double>()).As<double, TAcc>();
                return Vector256.Multiply(a, b);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc CombineScalar(TAcc a, TAcc b)
            {
                if (typeof(TAcc) == typeof(long)) return (TAcc)(object)((long)(object)a * (long)(object)b);
                if (typeof(TAcc) == typeof(ulong)) return (TAcc)(object)((ulong)(object)a * (ulong)(object)b);
                if (typeof(TAcc) == typeof(double)) return (TAcc)(object)((double)(object)a * (double)(object)b);
                throw new NotSupportedException();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc Identity() => OneOf<TAcc>();
        }

        // Min/Max use the cross-platform Vector256.Min/Max: for double they
        // propagate NaN correctly (x86 vminpd/vmaxpd do not), for (u)long
        // they emit the compare+blend sequence (no native instruction).
        internal readonly struct WMinOp<TAcc> : IWAccOp<TAcc> where TAcc : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<TAcc> Combine(Vector256<TAcc> a, Vector256<TAcc> b) => Vector256.Min(a, b);

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc CombineScalar(TAcc a, TAcc b)
            {
                if (typeof(TAcc) == typeof(long)) return (TAcc)(object)Math.Min((long)(object)a, (long)(object)b);
                if (typeof(TAcc) == typeof(ulong)) return (TAcc)(object)Math.Min((ulong)(object)a, (ulong)(object)b);
                if (typeof(TAcc) == typeof(double)) return (TAcc)(object)Math.Min((double)(object)a, (double)(object)b);
                throw new NotSupportedException();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc Identity() => MaxValueOf<TAcc>();
        }

        internal readonly struct WMaxOp<TAcc> : IWAccOp<TAcc> where TAcc : unmanaged
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<TAcc> Combine(Vector256<TAcc> a, Vector256<TAcc> b) => Vector256.Max(a, b);

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc CombineScalar(TAcc a, TAcc b)
            {
                if (typeof(TAcc) == typeof(long)) return (TAcc)(object)Math.Max((long)(object)a, (long)(object)b);
                if (typeof(TAcc) == typeof(ulong)) return (TAcc)(object)Math.Max((ulong)(object)a, (ulong)(object)b);
                if (typeof(TAcc) == typeof(double)) return (TAcc)(object)Math.Max((double)(object)a, (double)(object)b);
                throw new NotSupportedException();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static TAcc Identity() => MinValueOf<TAcc>();
        }

        // =====================================================================
        // Widening loads. Each Load4 reads EXACTLY four consecutive input
        // elements (loads are sized to 4*sizeof(TIn) bytes — no overread) and
        // returns them widened, in element order, as one Vector256<TAcc>.
        //
        // PMOVSX/PMOVZX consume the LOW 4 elements of the source Vector128;
        // CreateScalarUnsafe materializes just those bytes (upper bytes are
        // undefined and unread).
        //
        // The pointer parameter is a concrete byte* on purpose: a TIn* with a
        // generic pointee in the interface signature blocks constrained-call
        // devirtualization (the call boxes and virtual-dispatches — measured
        // ~10 cycles/element). Each struct documents its input type and the
        // dispatcher's pair table is the single place that wires TIn to TW.
        // =====================================================================
        internal unsafe interface IWidenLoad<TAcc> where TAcc : unmanaged
        {
            static abstract Vector256<TAcc> Load4(byte* p);
            static abstract TAcc LoadScalar(byte* p);
        }

        // ---- integer -> 64-bit integer ----

        /// <summary>short -> long (VPMOVSXWQ).</summary>
        internal readonly unsafe struct WidenI16ToI64 : IWidenLoad<long>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<long> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(*(ulong*)p).AsInt16());

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static long LoadScalar(byte* p) => *(short*)p;
        }

        /// <summary>ushort -> ulong (VPMOVZXWQ).</summary>
        internal readonly unsafe struct WidenU16ToU64 : IWidenLoad<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<ulong> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(*(ulong*)p).AsUInt16()).AsUInt64();

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong LoadScalar(byte* p) => *(ushort*)p;
        }

        /// <summary>sbyte -> long (VPMOVSXBQ).</summary>
        internal readonly unsafe struct WidenI8ToI64 : IWidenLoad<long>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<long> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(*(uint*)p).AsSByte());

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static long LoadScalar(byte* p) => *(sbyte*)p;
        }

        /// <summary>byte -> ulong (VPMOVZXBQ).</summary>
        internal readonly unsafe struct WidenU8ToU64 : IWidenLoad<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<ulong> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.CreateScalarUnsafe(*(uint*)p).AsByte()).AsUInt64();

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong LoadScalar(byte* p) => *p;
        }

        /// <summary>int -> long (VPMOVSXDQ).</summary>
        internal readonly unsafe struct WidenI32ToI64 : IWidenLoad<long>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<long> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.Load((int*)p));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static long LoadScalar(byte* p) => *(int*)p;
        }

        /// <summary>uint -> ulong (VPMOVZXDQ).</summary>
        internal readonly unsafe struct WidenU32ToU64 : IWidenLoad<ulong>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<ulong> Load4(byte* p)
                => Avx2.ConvertToVector256Int64(Vector128.Load((uint*)p)).AsUInt64();

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong LoadScalar(byte* p) => *(uint*)p;
        }

        // ---- anything -> double ----

        /// <summary>float -> double (VCVTPS2PD).</summary>
        internal readonly unsafe struct WidenF32ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(Vector128.Load((float*)p));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(float*)p;
        }

        /// <summary>int -> double (VCVTDQ2PD).</summary>
        internal readonly unsafe struct WidenI32ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(Vector128.Load((int*)p));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(int*)p;
        }

        /// <summary>uint -> double (exact bias trick — no unsigned convert below AVX-512).</summary>
        internal readonly unsafe struct WidenU32ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
            {
                // (double)(int)(u ^ 0x80000000) + 2^31 == u for all uint32.
                var biased = Sse2.Xor(Vector128.Load((int*)p), Vector128.Create(int.MinValue));
                return Avx.Add(Avx.ConvertToVector256Double(biased), Vector256.Create(2147483648.0));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(uint*)p;
        }

        /// <summary>short -> double (VPMOVSXWD + VCVTDQ2PD).</summary>
        internal readonly unsafe struct WidenI16ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(
                    Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(*(ulong*)p).AsInt16()));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(short*)p;
        }

        /// <summary>ushort -> double (VPMOVZXWD + VCVTDQ2PD).</summary>
        internal readonly unsafe struct WidenU16ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(
                    Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(*(ulong*)p).AsUInt16()));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(ushort*)p;
        }

        /// <summary>sbyte -> double (VPMOVSXBD + VCVTDQ2PD).</summary>
        internal readonly unsafe struct WidenI8ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(
                    Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(*(uint*)p).AsSByte()));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *(sbyte*)p;
        }

        /// <summary>byte -> double (VPMOVZXBD + VCVTDQ2PD).</summary>
        internal readonly unsafe struct WidenU8ToF64 : IWidenLoad<double>
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static Vector256<double> Load4(byte* p)
                => Avx.ConvertToVector256Double(
                    Sse41.ConvertToVector128Int32(Vector128.CreateScalarUnsafe(*(uint*)p).AsByte()));

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double LoadScalar(byte* p) => *p;
        }
    }
}
