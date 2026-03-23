using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.cs - Axis Reduction Core
// =============================================================================
//
// RESPONSIBILITY:
//   - Axis reduction cache and API (TryGetAxisReductionKernel)
//   - Main dispatcher (CreateAxisReductionKernel)
//   - General axis reduction kernels (scalar loop with type conversion)
//
// RELATED FILES:
//   - ILKernelGenerator.Reduction.Axis.Arg.cs - ArgMax/ArgMin axis
//   - ILKernelGenerator.Reduction.Axis.Simd.cs - Typed SIMD kernels
//   - ILKernelGenerator.Reduction.Axis.VarStd.cs - Var/Std axis
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Axis Reduction SIMD Helpers

        /// <summary>
        /// Cache for axis reduction kernels (delegates that call SIMD helpers).
        /// </summary>
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<AxisReductionKernelKey, AxisReductionKernel> _axisReductionCache = new();

        /// <summary>
        /// Number of axis reduction kernels in cache.
        /// </summary>
        public static int AxisReductionCachedCount => _axisReductionCache.Count;

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
            // For type promotion cases or non-SIMD types, use the general dispatcher
            if (key.InputType != key.AccumulatorType || !CanUseSimd(key.InputType))
            {
                return CreateAxisReductionKernelGeneral(key);
            }

            // Same-type SIMD path - dispatch based on input type
            return key.InputType switch
            {
                NPTypeCode.Byte => CreateAxisReductionKernelTyped<byte>(key),
                NPTypeCode.Int16 => CreateAxisReductionKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisReductionKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisReductionKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisReductionKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisReductionKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisReductionKernelTyped<ulong>(key),
                NPTypeCode.Single => CreateAxisReductionKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisReductionKernelTyped<double>(key),
                _ => CreateAxisReductionKernelGeneral(key) // Fallback for Boolean, Char, Decimal
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
                // Same-type scalar paths (for non-SIMD types like Decimal)
                (NPTypeCode.Decimal, NPTypeCode.Decimal) => CreateAxisReductionKernelScalar<decimal, decimal>(key),
                (NPTypeCode.Boolean, NPTypeCode.Boolean) => CreateAxisReductionKernelScalar<bool, bool>(key),
                (NPTypeCode.Char, NPTypeCode.Char) => CreateAxisReductionKernelScalar<char, char>(key),

                // Common type promotion paths (input -> wider accumulator)
                // byte -> int32/int64/double
                (NPTypeCode.Byte, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<byte, int>(key),
                (NPTypeCode.Byte, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<byte, long>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<byte, uint>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<byte, ulong>(key),
                (NPTypeCode.Byte, NPTypeCode.Double) => CreateAxisReductionKernelScalar<byte, double>(key),

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

                // char -> int32/int64
                (NPTypeCode.Char, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<char, int>(key),
                (NPTypeCode.Char, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<char, long>(key),
                (NPTypeCode.Char, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<char, uint>(key),

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
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
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
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            NPTypeCode inputType, NPTypeCode accumType, ReductionOp op)
        {
            int axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();
            int outputElemSize = accumType.SizeOf();

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
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

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
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

                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
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
        private static unsafe double ReadAsDouble(byte* ptr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.UInt64 => *(ulong*)ptr,
                NPTypeCode.Single => *(float*)ptr,
                NPTypeCode.Double => *(double*)ptr,
                NPTypeCode.Decimal => (double)*(decimal*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                NPTypeCode.Boolean => *(bool*)ptr ? 1.0 : 0.0,
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
                case NPTypeCode.Int16: *(short*)ptr = (short)value; break;
                case NPTypeCode.UInt16: *(ushort*)ptr = (ushort)value; break;
                case NPTypeCode.Int32: *(int*)ptr = (int)value; break;
                case NPTypeCode.UInt32: *(uint*)ptr = (uint)value; break;
                case NPTypeCode.Int64: *(long*)ptr = (long)value; break;
                case NPTypeCode.UInt64: *(ulong*)ptr = (ulong)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = (decimal)value; break;
                case NPTypeCode.Char: *(char*)ptr = (char)(int)value; break;
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
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
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
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
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            int axisStride = inputStrides[axis];

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
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

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute offsets
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Reduce along axis with type conversion
                TAccum accum = GetIdentityValueTyped<TAccum>(op);
                TInput* axisStart = input + inputBaseOffset;

                for (int i = 0; i < axisSize; i++)
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
        /// Combine accumulator with input value, promoting input to accumulator type.
        /// </summary>
        private static TAccum CombineScalarsPromoted<TInput, TAccum>(TAccum accum, TInput val, ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
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
        /// Divide accumulator by count (for Mean).
        /// </summary>
        private static TAccum DivideByCount<TAccum>(TAccum accum, int count) where TAccum : unmanaged
        {
            double result = ConvertToDouble(accum) / count;
            return ConvertFromDouble<TAccum>(result);
        }

        /// <summary>
        /// Convert any numeric type to double.
        /// </summary>
        private static double ConvertToDouble<T>(T value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)value;
            if (typeof(T) == typeof(short)) return (short)(object)value;
            if (typeof(T) == typeof(ushort)) return (ushort)(object)value;
            if (typeof(T) == typeof(int)) return (int)(object)value;
            if (typeof(T) == typeof(uint)) return (uint)(object)value;
            if (typeof(T) == typeof(long)) return (long)(object)value;
            if (typeof(T) == typeof(ulong)) return (ulong)(object)value;
            if (typeof(T) == typeof(float)) return (float)(object)value;
            if (typeof(T) == typeof(double)) return (double)(object)value;
            if (typeof(T) == typeof(decimal)) return (double)(decimal)(object)value;
            if (typeof(T) == typeof(char)) return (char)(object)value;
            if (typeof(T) == typeof(bool)) return (bool)(object)value ? 1.0 : 0.0;
            return 0.0;
        }

        /// <summary>
        /// Convert double to target type.
        /// </summary>
        private static T ConvertFromDouble<T>(double value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (T)(object)(byte)value;
            if (typeof(T) == typeof(short)) return (T)(object)(short)value;
            if (typeof(T) == typeof(ushort)) return (T)(object)(ushort)value;
            if (typeof(T) == typeof(int)) return (T)(object)(int)value;
            if (typeof(T) == typeof(uint)) return (T)(object)(uint)value;
            if (typeof(T) == typeof(long)) return (T)(object)(long)value;
            if (typeof(T) == typeof(ulong)) return (T)(object)(ulong)value;
            if (typeof(T) == typeof(float)) return (T)(object)(float)value;
            if (typeof(T) == typeof(double)) return (T)(object)value;
            if (typeof(T) == typeof(decimal)) return (T)(object)(decimal)value;
            if (typeof(T) == typeof(char)) return (T)(object)(char)(int)value;
            if (typeof(T) == typeof(bool)) return (T)(object)(value != 0);
            return default;
        }

        /// <summary>
        /// Get typed identity value for reduction operation.
        /// </summary>
        private static T GetIdentityValueTyped<T>(ReductionOp op) where T : unmanaged
        {
            double identity = op switch
            {
                ReductionOp.Sum or ReductionOp.Mean => 0.0,
                ReductionOp.Prod => 1.0,
                ReductionOp.Min => double.PositiveInfinity,
                ReductionOp.Max => double.NegativeInfinity,
                _ => 0.0
            };
            return ConvertFromDouble<T>(identity);
        }

        #endregion
    }
}
