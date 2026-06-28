using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Reduction.Axis.cs - Axis Reduction Core
// =============================================================================
//
// RESPONSIBILITY:
//   - Axis reduction cache and API (TryGetAxisReductionKernel)
//   - Main dispatcher (CreateAxisReductionKernel)
//   - General axis reduction kernels (scalar loop with type conversion)
//
// RELATED FILES:
//   - DirectILKernelGenerator.Reduction.Axis.Arg.cs - ArgMax/ArgMin axis
//   - DirectILKernelGenerator.Reduction.Axis.Simd.cs - Typed SIMD kernels
//   - DirectILKernelGenerator.Reduction.Axis.VarStd.cs - Var/Std axis
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Axis Reduction SIMD Helpers

        /// <summary>
        /// Cache for axis reduction kernels (delegates that call SIMD helpers).
        /// </summary>
        internal static readonly System.Collections.Concurrent.ConcurrentDictionary<AxisReductionKernelKey, AxisReductionKernel> _axisReductionCache = new();

        /// <summary>
        /// Try to get an axis reduction kernel.
        /// Supports all reduction operations and all types including type promotion.
        /// Uses SIMD for capable types, scalar loop for others.
        /// </summary>
        public static AxisReductionKernel? TryGetAxisReductionKernel(AxisReductionKernelKey key)
        {
            if (!Enabled)
                return null;

            // Support Sum, Prod, Min, Max, Mean, Var, Std, ArgMax, ArgMin operations
            if (key.Op != ReductionOp.Sum && key.Op != ReductionOp.Prod &&
                key.Op != ReductionOp.Min && key.Op != ReductionOp.Max &&
                key.Op != ReductionOp.Mean && key.Op != ReductionOp.Var &&
                key.Op != ReductionOp.Std && key.Op != ReductionOp.ArgMax &&
                key.Op != ReductionOp.ArgMin)
            {
                return null;
            }

            // ArgMax/ArgMin always output Int64 (the index), regardless of input type
            // They use a different kernel path that tracks both value and index
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                return _axisReductionCache.GetOrAdd(key, CreateAxisArgReductionKernel);
            }

            // Var/Std use two-pass algorithm (mean, then squared differences)
            // They always output double for accuracy
            if (key.Op == ReductionOp.Var || key.Op == ReductionOp.Std)
            {
                return _axisReductionCache.GetOrAdd(key, CreateAxisVarStdReductionKernel);
            }

            // All types supported - SIMD for capable types, scalar for others
            return _axisReductionCache.GetOrAdd(key, CreateAxisReductionKernel);
        }

        /// <summary>
        /// Create an axis reduction kernel that dispatches to the appropriate helper.
        /// Handles all types including type promotion.
        /// </summary>
        private static AxisReductionKernel CreateAxisReductionKernel(AxisReductionKernelKey key)
        {
            // ─────────────────────────────────────────────────────────────────
            // PER-DTYPE MIN/MAX SPECIALIZATION — bool / char reinterpret to SIMD.
            //
            // bool and char have no fractional/NaN domain, and their total order is
            // bit-identical to the matching unsigned integer (bool ≡ uint8 over {0,1},
            // char ≡ uint16 over [0,65535]). So amin/amax over them is bit-identical to
            // the byte / uint16 SIMD reducer — vpminub / vpminuw — instead of the scalar
            // double-bridge in CreateAxisReductionKernelGeneral (ConvertToDouble →
            // Math.Min → ConvertFromDouble per element). The reinterpret routes the
            // operands, untouched, through the existing typed SIMD helper: measured
            // ~76× (bool) and ~219× (char) on a 1000×1000 axis reduce. The key's cache
            // identity stays (Boolean/Char, Op) — CreateAxisReductionKernelTyped only
            // reads key.Op and the type argument, never key.InputType.
            //
            // Half cannot be reinterpreted (IEEE-754 ordering ≠ uint16 bit ordering for
            // negatives/NaN); it gets its own boxing-free scalar loop in
            // CreateAxisReductionKernelGeneral below.
            // ─────────────────────────────────────────────────────────────────
            if ((key.Op == ReductionOp.Min || key.Op == ReductionOp.Max) && key.InputType == key.AccumulatorType)
            {
                if (key.InputType == NPTypeCode.Boolean) return CreateAxisReductionKernelTyped<byte>(key);
                if (key.InputType == NPTypeCode.Char) return CreateAxisReductionKernelTyped<ushort>(key);
            }

            // For type promotion cases or non-SIMD types, use the general dispatcher.
            // First try the widening SIMD fast path (int32->int64, float->double, etc).
            if (key.InputType != key.AccumulatorType || !CanUseSimd(key.InputType))
            {
                var wideningKernel = TryGetAxisReductionWideningKernel(key);
                if (wideningKernel != null) return wideningKernel;

                return CreateAxisReductionKernelGeneral(key);
            }

            // Same-type SIMD path - dispatch based on input type
            return key.InputType switch
            {
                NPTypeCode.Byte => CreateAxisReductionKernelTyped<byte>(key),
                NPTypeCode.SByte => CreateAxisReductionKernelTyped<sbyte>(key),
                NPTypeCode.Int16 => CreateAxisReductionKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisReductionKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisReductionKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisReductionKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisReductionKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisReductionKernelTyped<ulong>(key),
                NPTypeCode.Single => CreateAxisReductionKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisReductionKernelTyped<double>(key),
                _ => CreateAxisReductionKernelGeneral(key) // Fallback for Boolean, Char, Decimal, Half, Complex
            };
        }

        /// <summary>
        /// Create an axis ArgMax/ArgMin reduction kernel.
        /// These operations track both the value (for comparison) and index (for output).
        /// Output is always Int64 regardless of input type.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelGeneral(AxisReductionKernelKey key)
        {
            // Dispatch based on input and accumulator type combination
            return (key.InputType, key.AccumulatorType) switch
            {
                // Same-type scalar paths (for non-SIMD types like Decimal, Half, Complex)
                (NPTypeCode.Decimal, NPTypeCode.Decimal) => CreateAxisReductionKernelScalar<decimal, decimal>(key),
                (NPTypeCode.Boolean, NPTypeCode.Boolean) => CreateAxisReductionKernelScalar<bool, bool>(key),
                (NPTypeCode.Char, NPTypeCode.Char) => CreateAxisReductionKernelScalar<char, char>(key),
                // Half min/max get a boxing-free direct-Half loop (no double bridge);
                // sum/mean/prod keep the double-intermediate scalar path for accumulation precision.
                (NPTypeCode.Half, NPTypeCode.Half) => key.Op == ReductionOp.Min || key.Op == ReductionOp.Max
                    ? CreateAxisReductionKernelHalfMinMax(key)
                    : CreateAxisReductionKernelScalar<Half, Half>(key),
                (NPTypeCode.Complex, NPTypeCode.Complex) => CreateAxisReductionKernelScalar<System.Numerics.Complex, System.Numerics.Complex>(key),

                // Common type promotion paths (input -> wider accumulator)
                // byte -> int32/int64/double
                (NPTypeCode.Byte, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<byte, int>(key),
                (NPTypeCode.Byte, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<byte, long>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<byte, uint>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<byte, ulong>(key),
                (NPTypeCode.Byte, NPTypeCode.Double) => CreateAxisReductionKernelScalar<byte, double>(key),

                // sbyte -> int32/int64/double
                (NPTypeCode.SByte, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<sbyte, int>(key),
                (NPTypeCode.SByte, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<sbyte, long>(key),
                (NPTypeCode.SByte, NPTypeCode.Double) => CreateAxisReductionKernelScalar<sbyte, double>(key),

                // bool -> int64 (NEP50 sum/prod accumulator)
                (NPTypeCode.Boolean, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<bool, long>(key),

                // int16 -> int32/int64/double
                (NPTypeCode.Int16, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<short, int>(key),
                (NPTypeCode.Int16, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<short, long>(key),
                (NPTypeCode.Int16, NPTypeCode.Double) => CreateAxisReductionKernelScalar<short, double>(key),

                // uint16 -> int32/uint32/int64/uint64/double
                (NPTypeCode.UInt16, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<ushort, int>(key),
                (NPTypeCode.UInt16, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<ushort, uint>(key),
                (NPTypeCode.UInt16, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<ushort, long>(key),
                (NPTypeCode.UInt16, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<ushort, ulong>(key),
                (NPTypeCode.UInt16, NPTypeCode.Double) => CreateAxisReductionKernelScalar<ushort, double>(key),

                // int32 -> int64/double
                (NPTypeCode.Int32, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<int, long>(key),
                (NPTypeCode.Int32, NPTypeCode.Double) => CreateAxisReductionKernelScalar<int, double>(key),

                // uint32 -> int64/uint64/double
                (NPTypeCode.UInt32, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<uint, long>(key),
                (NPTypeCode.UInt32, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<uint, ulong>(key),
                (NPTypeCode.UInt32, NPTypeCode.Double) => CreateAxisReductionKernelScalar<uint, double>(key),

                // int64 -> double
                (NPTypeCode.Int64, NPTypeCode.Double) => CreateAxisReductionKernelScalar<long, double>(key),

                // uint64 -> double
                (NPTypeCode.UInt64, NPTypeCode.Double) => CreateAxisReductionKernelScalar<ulong, double>(key),

                // float -> double
                (NPTypeCode.Single, NPTypeCode.Double) => CreateAxisReductionKernelScalar<float, double>(key),

                // char -> int32/int64/uint64
                (NPTypeCode.Char, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<char, int>(key),
                (NPTypeCode.Char, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<char, long>(key),
                (NPTypeCode.Char, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<char, uint>(key),
                (NPTypeCode.Char, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<char, ulong>(key),

                // decimal -> double (for mean)
                (NPTypeCode.Decimal, NPTypeCode.Double) => CreateAxisReductionKernelScalar<decimal, double>(key),

                // Default fallback - use double accumulator
                _ => CreateAxisReductionKernelWithConversion(key)
            };
        }

        /// <summary>
        /// Create a fallback kernel using runtime type conversion.
        /// Used for rare type combinations not explicitly handled.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelWithConversion(AxisReductionKernelKey key)
        {
            // For rare combinations, use a runtime conversion approach via double
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisReductionWithConversionHelper(
                    input, output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.InputType, key.AccumulatorType, key.Op);
            };
        }

        /// <summary>
        /// Helper for axis reduction with runtime type conversion.
        /// </summary>
        private static unsafe void AxisReductionWithConversionHelper(
            void* input, void* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            NPTypeCode inputType, NPTypeCode accumType, ReductionOp op)
        {
            long axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();
            int outputElemSize = accumType.SizeOf();

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            byte* inputBytes = (byte*)input;
            byte* outputBytes = (byte*)output;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Reduce along axis using double as intermediate
                double accum = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => 0.0,
                    ReductionOp.Prod => 1.0,
                    ReductionOp.Min => double.PositiveInfinity,
                    ReductionOp.Max => double.NegativeInfinity,
                    _ => 0.0
                };

                for (long i = 0; i < axisSize; i++)
                {
                    long inputOffset = inputBaseOffset + i * axisStride;
                    double val = ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);

                    accum = op switch
                    {
                        ReductionOp.Sum or ReductionOp.Mean => accum + val,
                        ReductionOp.Prod => accum * val,
                        ReductionOp.Min => Math.Min(accum, val),
                        ReductionOp.Max => Math.Max(accum, val),
                        _ => accum
                    };
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    accum /= axisSize;

                // Write result
                WriteFromDouble(outputBytes + outputOffset * outputElemSize, accum, accumType);
            }
        }

        /// <summary>
        /// Read a value as double from typed memory.
        /// </summary>
        // Per-element helpers below run once per element inside the axis-reduction loops
        // (the generic typeof(T) branches constant-fold to a single cast per instantiation),
        // so inline them where possible and give them full tier-1 codegen otherwise.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe double ReadAsDouble(byte* ptr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.SByte => *(sbyte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.UInt64 => *(ulong*)ptr,
                NPTypeCode.Single => *(float*)ptr,
                NPTypeCode.Double => *(double*)ptr,
                NPTypeCode.Half => (double)*(Half*)ptr,
                NPTypeCode.Decimal => (double)*(decimal*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                NPTypeCode.Boolean => *(bool*)ptr ? 1.0 : 0.0,
                NPTypeCode.Complex => (*(System.Numerics.Complex*)ptr).Real, // Use real part for reductions
                _ => 0.0
            };
        }

        /// <summary>
        /// Write a double value to typed memory.
        /// </summary>
        private static unsafe void WriteFromDouble(byte* ptr, double value, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Byte: *(byte*)ptr = (byte)value; break;
                case NPTypeCode.SByte: *(sbyte*)ptr = (sbyte)value; break;
                case NPTypeCode.Int16: *(short*)ptr = (short)value; break;
                case NPTypeCode.UInt16: *(ushort*)ptr = (ushort)value; break;
                case NPTypeCode.Int32: *(int*)ptr = (int)value; break;
                case NPTypeCode.UInt32: *(uint*)ptr = (uint)value; break;
                case NPTypeCode.Int64: *(long*)ptr = (long)value; break;
                case NPTypeCode.UInt64: *(ulong*)ptr = (ulong)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = value; break;
                case NPTypeCode.Half: *(Half*)ptr = (Half)value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = (decimal)value; break;
                case NPTypeCode.Char: *(char*)ptr = (char)(int)value; break;
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
                case NPTypeCode.Complex: *(System.Numerics.Complex*)ptr = new System.Numerics.Complex(value, 0); break;
            }
        }

        /// <summary>
        /// Create a typed scalar axis reduction kernel with type promotion.
        /// Uses scalar loop - no SIMD, but handles type conversion at compile time.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelScalar<TInput, TAccum>(AxisReductionKernelKey key)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisReductionScalarHelper<TInput, TAccum>(
                    (TInput*)input, (TAccum*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// Scalar axis reduction helper with type promotion.
        /// </summary>
        internal static unsafe void AxisReductionScalarHelper<TInput, TAccum>(
            TInput* input, TAccum* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            long axisStride = inputStrides[axis];

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute offsets
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Reduce along axis with type conversion
                TAccum accum = GetIdentityValueTyped<TAccum>(op);
                TInput* axisStart = input + inputBaseOffset;

                for (long i = 0; i < axisSize; i++)
                {
                    TInput val = axisStart[i * axisStride];
                    accum = CombineScalarsPromoted<TInput, TAccum>(accum, val, op);
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    accum = DivideByCount<TAccum>(accum, axisSize);

                output[outputOffset] = accum;
            }
        }

        /// <summary>
        /// Boxing-free Half min/max axis reducer.
        ///
        /// The generic scalar path (CombineScalarsPromoted's Half branch) bridges every
        /// element through <see cref="double"/>: (double)(Half)accum → Math.Min →
        /// (Half)result. That round-trip dominated f16 amin/amax along an axis. Half has
        /// no Vector&lt;Half&gt; arithmetic in the BCL, but it exposes a hardware-backed
        /// total order via its comparison operators, so a direct Half loop with no double
        /// round-trip is ~7× faster than the bridge (9.99 ms → ~1.45 ms on a 1000×1000
        /// reduce) and beats NumPy's f16 loop (1.7–2.1 ms, which widens to float per
        /// element).
        ///
        /// NaN propagates to match Math.Min/Max and NumPy's reduce: the accumulator is
        /// seeded from element 0, and a NaN input is taken on sight (x &lt; acc is false
        /// when acc is NaN, so once NaN enters it sticks). Only Min/Max route here —
        /// Sum/Mean/Prod keep the double-intermediate path for accumulation precision.
        /// Offset / strided / negative-stride / transposed views are handled by the same
        /// coordinate walk as <see cref="AxisReductionScalarHelper{TInput,TAccum}"/>.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelHalfMinMax(AxisReductionKernelKey key)
        {
            bool isMin = key.Op == ReductionOp.Min;
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisReductionHalfMinMaxHelper(
                    (Half*)input, (Half*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize, isMin);
            };
        }

        /// <summary>
        /// Static body for <see cref="CreateAxisReductionKernelHalfMinMax"/>, mirroring the
        /// thin-lambda → static-helper shape of <see cref="AxisReductionSimdHelper{T}"/> /
        /// <see cref="AxisReductionScalarHelper{TInput,TAccum}"/> and marked AggressiveOptimization.
        /// ~1.45 ms (1000×1000) is the realistic floor for this delegate path: the data pointers
        /// arrive as method parameters rather than fixed-block locals, so the JIT emits a slightly
        /// more conservative inner loop than an equivalent standalone scan (~0.85 ms) — still well
        /// ahead of the double-bridge it replaces, and ahead of NumPy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionHalfMinMaxHelper(
            Half* input, Half* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize, bool isMin)
        {
            long axisStride = inputStrides[axis];

            int outputNdim = ndim - 1;
            Span<long> outputDimStrides = stackalloc long[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Linear output index → coordinates → input/output offsets.
                long remaining = outIdx;
                long inputBaseOffset = 0;
                long outputOffset = 0;
                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Seed from element 0 (axisSize >= 1: empty min/max throws upstream).
                Half* axisStart = input + inputBaseOffset;
                Half acc = *axisStart;
                if (axisStride == 1)
                {
                    // INNER-CONTIG fast path. With a runtime axisStride the JIT cannot prove
                    // unit stride, so axisStart[i*axisStride] (or p += axisStride) emits a
                    // variable-stride load and a per-element 64-bit multiply — that alone
                    // doubled this loop (1.5 ms → 0.85 ms on a 1000×1000 innermost reduce).
                    // Branching on axisStride==1 lets the JIT treat axisStart[i] as the
                    // sequential stream it is. Covers axis==ndim-1 on C-contig input.
                    if (isMin)
                        for (long i = 1; i < axisSize; i++) { Half x = axisStart[i]; if (x < acc || Half.IsNaN(x)) acc = x; }
                    else
                        for (long i = 1; i < axisSize; i++) { Half x = axisStart[i]; if (x > acc || Half.IsNaN(x)) acc = x; }
                }
                else
                {
                    // STRIDED path — advance the read pointer by axisStride each step (no
                    // i*stride multiply). Also covers negative-stride views (axisStride < 0).
                    Half* p = axisStart;
                    if (isMin)
                        for (long i = 1; i < axisSize; i++) { p += axisStride; Half x = *p; if (x < acc || Half.IsNaN(x)) acc = x; }
                    else
                        for (long i = 1; i < axisSize; i++) { p += axisStride; Half x = *p; if (x > acc || Half.IsNaN(x)) acc = x; }
                }
                output[outputOffset] = acc;
            }
        }

        /// <summary>
        /// Combine accumulator with input value, promoting input to accumulator type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static TAccum CombineScalarsPromoted<TInput, TAccum>(TAccum accum, TInput val, ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            // Special handling for Complex - cannot use double intermediate
            if (typeof(TAccum) == typeof(System.Numerics.Complex))
            {
                var cAccum = (System.Numerics.Complex)(object)accum;
                var cVal = typeof(TInput) == typeof(System.Numerics.Complex)
                    ? (System.Numerics.Complex)(object)val
                    : new System.Numerics.Complex(ConvertToDouble(val), 0);

                var cResult = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => cAccum + cVal,
                    ReductionOp.Prod => cAccum * cVal,
                    // NumPy parity: lex ordering on (Real, Imaginary); NaN-first-wins
                    // propagation (NaN-containing = Re or Im is NaN). Identity picked in
                    // GetIdentityValueTyped as (+inf,+inf)/(-inf,-inf) so first finite
                    // value beats it under lex comparison.
                    ReductionOp.Min => ComplexLexPick(cAccum, cVal, pickGreater: false),
                    ReductionOp.Max => ComplexLexPick(cAccum, cVal, pickGreater: true),
                    _ => cAccum
                };
                return (TAccum)(object)cResult;
            }

            // Special handling for Half - use double intermediate for precision
            if (typeof(TAccum) == typeof(Half))
            {
                double hAccum = (double)(Half)(object)accum;
                double hVal = typeof(TInput) == typeof(Half)
                    ? (double)(Half)(object)val
                    : ConvertToDouble(val);

                double hResult = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => hAccum + hVal,
                    ReductionOp.Prod => hAccum * hVal,
                    ReductionOp.Min => Math.Min(hAccum, hVal),
                    ReductionOp.Max => Math.Max(hAccum, hVal),
                    _ => hAccum
                };
                return (TAccum)(object)(Half)hResult;
            }

            // Integer accumulators: EXACT integer arithmetic. NumPy wraps integer sum/prod modulo
            // 2^width (e.g. prod of 70 twos = 0, prod of 63 twos = -2^63); a double intermediate
            // would lose precision / saturate for |x| > 2^53 — (long)(double)2^70 = Int64.MaxValue.
            // On x86 these (input,accum) pairs usually take the AVX2 widening kernel; this scalar
            // path is the non-AVX2 (e.g. ARM64) / uncovered-shape fallback. Min/Max also stay exact
            // here (large int64 magnitudes aren't representable in double). Add/mul wraparound is
            // bit-identical for signed and unsigned, so the read sign/zero-extends per input type.
            if (typeof(TAccum) == typeof(long))
            {
                long a = (long)(object)accum, v = unchecked((long)ConvertToInt64Bits(val));
                long r = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => unchecked(a + v),
                    ReductionOp.Prod => unchecked(a * v),
                    ReductionOp.Min => v < a ? v : a,
                    ReductionOp.Max => v > a ? v : a,
                    _ => a
                };
                return (TAccum)(object)r;
            }
            if (typeof(TAccum) == typeof(ulong))
            {
                ulong a = (ulong)(object)accum, v = ConvertToInt64Bits(val);
                ulong r = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => unchecked(a + v),
                    ReductionOp.Prod => unchecked(a * v),
                    ReductionOp.Min => v < a ? v : a,
                    ReductionOp.Max => v > a ? v : a,
                    _ => a
                };
                return (TAccum)(object)r;
            }
            if (typeof(TAccum) == typeof(int))
            {
                int a = (int)(object)accum, v = unchecked((int)ConvertToInt64Bits(val));
                int r = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => unchecked(a + v),
                    ReductionOp.Prod => unchecked(a * v),
                    ReductionOp.Min => v < a ? v : a,
                    ReductionOp.Max => v > a ? v : a,
                    _ => a
                };
                return (TAccum)(object)r;
            }
            if (typeof(TAccum) == typeof(uint))
            {
                uint a = (uint)(object)accum, v = unchecked((uint)ConvertToInt64Bits(val));
                uint r = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => unchecked(a + v),
                    ReductionOp.Prod => unchecked(a * v),
                    ReductionOp.Min => v < a ? v : a,
                    ReductionOp.Max => v > a ? v : a,
                    _ => a
                };
                return (TAccum)(object)r;
            }

            // Convert input to double for arithmetic, then to accumulator type
            double dAccum = ConvertToDouble(accum);
            double dVal = ConvertToDouble(val);

            double result = op switch
            {
                ReductionOp.Sum or ReductionOp.Mean => dAccum + dVal,
                ReductionOp.Prod => dAccum * dVal,
                ReductionOp.Min => Math.Min(dAccum, dVal),
                ReductionOp.Max => Math.Max(dAccum, dVal),
                _ => dAccum
            };

            return ConvertFromDouble<TAccum>(result);
        }

        /// <summary>
        /// Read an integer/bool/char value and return its 64-bit two's-complement representation:
        /// signed types sign-extend, unsigned types zero-extend. Used by the exact integer
        /// accumulator branches of <see cref="CombineScalarsPromoted{TInput,TAccum}"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static ulong ConvertToInt64Bits<T>(T value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)value;
            if (typeof(T) == typeof(sbyte)) return unchecked((ulong)(long)(sbyte)(object)value);
            if (typeof(T) == typeof(short)) return unchecked((ulong)(long)(short)(object)value);
            if (typeof(T) == typeof(ushort)) return (ushort)(object)value;
            if (typeof(T) == typeof(int)) return unchecked((ulong)(long)(int)(object)value);
            if (typeof(T) == typeof(uint)) return (uint)(object)value;
            if (typeof(T) == typeof(long)) return unchecked((ulong)(long)(object)value);
            if (typeof(T) == typeof(ulong)) return (ulong)(object)value;
            if (typeof(T) == typeof(char)) return (char)(object)value;
            if (typeof(T) == typeof(bool)) return (bool)(object)value ? 1UL : 0UL;
            return 0UL;
        }

        /// <summary>
        /// NumPy-parity pick for Complex Min/Max: NaN-containing operand (first wins) or
        /// lex-compared (Real, Imaginary). Shared with CombineScalarsPromoted's Complex path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static System.Numerics.Complex ComplexLexPick(System.Numerics.Complex a, System.Numerics.Complex b, bool pickGreater)
        {
            bool aNaN = double.IsNaN(a.Real) || double.IsNaN(a.Imaginary);
            if (aNaN) return a;
            bool bNaN = double.IsNaN(b.Real) || double.IsNaN(b.Imaginary);
            if (bNaN) return b;

            bool aGreater = a.Real > b.Real || (a.Real == b.Real && a.Imaginary > b.Imaginary);
            if (pickGreater)
                return aGreater ? a : b;
            return aGreater ? b : a;
        }

        /// <summary>
        /// Divide accumulator by count (for Mean).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static TAccum DivideByCount<TAccum>(TAccum accum, long count) where TAccum : unmanaged
        {
            // Special handling for Complex
            if (typeof(TAccum) == typeof(System.Numerics.Complex))
            {
                var cAccum = (System.Numerics.Complex)(object)accum;
                return (TAccum)(object)(cAccum / count);
            }

            // Special handling for Half
            if (typeof(TAccum) == typeof(Half))
            {
                double hAccum = (double)(Half)(object)accum;
                return (TAccum)(object)(Half)(hAccum / count);
            }

            double result = ConvertToDouble(accum) / count;
            return ConvertFromDouble<TAccum>(result);
        }

        /// <summary>
        /// Convert any numeric type to double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double ConvertToDouble<T>(T value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)value;
            if (typeof(T) == typeof(sbyte)) return (sbyte)(object)value;
            if (typeof(T) == typeof(short)) return (short)(object)value;
            if (typeof(T) == typeof(ushort)) return (ushort)(object)value;
            if (typeof(T) == typeof(int)) return (int)(object)value;
            if (typeof(T) == typeof(uint)) return (uint)(object)value;
            if (typeof(T) == typeof(long)) return (long)(object)value;
            if (typeof(T) == typeof(ulong)) return (ulong)(object)value;
            if (typeof(T) == typeof(float)) return (float)(object)value;
            if (typeof(T) == typeof(double)) return (double)(object)value;
            if (typeof(T) == typeof(Half)) return (double)(Half)(object)value;
            if (typeof(T) == typeof(decimal)) return (double)(decimal)(object)value;
            if (typeof(T) == typeof(char)) return (char)(object)value;
            if (typeof(T) == typeof(bool)) return (bool)(object)value ? 1.0 : 0.0;
            if (typeof(T) == typeof(System.Numerics.Complex)) return ((System.Numerics.Complex)(object)value).Real;
            return 0.0;
        }

        /// <summary>
        /// Convert double to target type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static T ConvertFromDouble<T>(double value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (T)(object)(byte)value;
            if (typeof(T) == typeof(sbyte)) return (T)(object)(sbyte)value;
            if (typeof(T) == typeof(short)) return (T)(object)(short)value;
            if (typeof(T) == typeof(ushort)) return (T)(object)(ushort)value;
            if (typeof(T) == typeof(int)) return (T)(object)(int)value;
            if (typeof(T) == typeof(uint)) return (T)(object)(uint)value;
            if (typeof(T) == typeof(long)) return (T)(object)(long)value;
            if (typeof(T) == typeof(ulong)) return (T)(object)(ulong)value;
            if (typeof(T) == typeof(float)) return (T)(object)(float)value;
            if (typeof(T) == typeof(double)) return (T)(object)value;
            if (typeof(T) == typeof(Half)) return (T)(object)(Half)value;
            if (typeof(T) == typeof(decimal)) return (T)(object)(decimal)value;
            if (typeof(T) == typeof(char)) return (T)(object)(char)(int)value;
            if (typeof(T) == typeof(bool)) return (T)(object)(value != 0);
            if (typeof(T) == typeof(System.Numerics.Complex)) return (T)(object)new System.Numerics.Complex(value, 0);
            return default;
        }

        /// <summary>
        /// Get typed identity value for reduction operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static T GetIdentityValueTyped<T>(ReductionOp op) where T : unmanaged
        {
            // Special handling for Complex
            if (typeof(T) == typeof(System.Numerics.Complex))
            {
                // Min/Max: use (±inf, ±inf) so any finite lex-compared element displaces
                // the identity on first combine (matches how double.PositiveInfinity works
                // for scalar Min above).
                var identity = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => System.Numerics.Complex.Zero,
                    ReductionOp.Prod => System.Numerics.Complex.One,
                    ReductionOp.Min => new System.Numerics.Complex(double.PositiveInfinity, double.PositiveInfinity),
                    ReductionOp.Max => new System.Numerics.Complex(double.NegativeInfinity, double.NegativeInfinity),
                    _ => System.Numerics.Complex.Zero
                };
                return (T)(object)identity;
            }

            // Special handling for Half
            if (typeof(T) == typeof(Half))
            {
                var identity = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => Half.Zero,
                    ReductionOp.Prod => (Half)1.0,
                    ReductionOp.Min => Half.PositiveInfinity,
                    ReductionOp.Max => Half.NegativeInfinity,
                    _ => Half.Zero
                };
                return (T)(object)identity;
            }

            // Special handling for Boolean — the double-bridge identity is WRONG for Max:
            // double.NegativeInfinity funneled through ConvertFromDouble<bool> (value != 0)
            // yields TRUE, so an all-False reduction group wrongly reduces to True (Math.Max
            // never drops below the seeded 1.0). Min coincidentally works because its
            // PositiveInfinity→True seed is the correct Min identity. Seed explicitly to match
            // NumPy: Max→false, Min→true. (Sum/Mean→false, Prod→true for completeness; bool
            // Sum/Prod promote to int64 per NEP50 and never reach this bool branch.)
            if (typeof(T) == typeof(bool))
            {
                bool identity = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => false,
                    ReductionOp.Prod => true,
                    ReductionOp.Min => true,
                    ReductionOp.Max => false,
                    _ => false
                };
                return (T)(object)identity;
            }

            double dIdentity = op switch
            {
                ReductionOp.Sum or ReductionOp.Mean => 0.0,
                ReductionOp.Prod => 1.0,
                ReductionOp.Min => double.PositiveInfinity,
                ReductionOp.Max => double.NegativeInfinity,
                _ => 0.0
            };
            return ConvertFromDouble<T>(dIdentity);
        }

        #endregion
    }
}
