using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.cs - Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - Axis reduction kernel generation (reduce along specific axis)
//   - Variance/StdDev axis reductions (two-pass algorithm)
//   - IKernelProvider interface implementation
//   - SIMD helpers for contiguous and strided axis operations
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
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
        /// Clear axis reduction cache.
        /// </summary>
        public static void ClearAxisReduction()
        {
            _axisReductionCache.Clear();
        }

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
        private static AxisReductionKernel CreateAxisArgReductionKernel(AxisReductionKernelKey key)
        {
            // Dispatch based on input type - output is always long (Int64)
            return key.InputType switch
            {
                NPTypeCode.Boolean => CreateAxisArgReductionKernelTyped<bool>(key),
                NPTypeCode.Byte => CreateAxisArgReductionKernelTyped<byte>(key),
                NPTypeCode.Int16 => CreateAxisArgReductionKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisArgReductionKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisArgReductionKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisArgReductionKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisArgReductionKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisArgReductionKernelTyped<ulong>(key),
                NPTypeCode.Char => CreateAxisArgReductionKernelTyped<char>(key),
                NPTypeCode.Single => CreateAxisArgReductionKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisArgReductionKernelTyped<double>(key),
                NPTypeCode.Decimal => CreateAxisArgReductionKernelTyped<decimal>(key),
                _ => throw new NotSupportedException($"ArgMax/ArgMin not supported for type {key.InputType}")
            };
        }

        /// <summary>
        /// Create a typed axis ArgMax/ArgMin kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisArgReductionKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisArgReductionHelper<T>(
                    (T*)input, (long*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// Helper for axis ArgMax/ArgMin reduction.
        /// Tracks both value (for comparison) and index (for output).
        /// </summary>
        internal static unsafe void AxisArgReductionHelper<T>(
            T* input, long* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where T : unmanaged
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

                // Find argmax or argmin along axis
                T* axisStart = input + inputBaseOffset;
                long resultIndex = ArgReduceAxis(axisStart, axisSize, axisStride, op);

                output[outputOffset] = resultIndex;
            }
        }

        /// <summary>
        /// Find the index of the max or min value along an axis.
        /// </summary>
        private static unsafe long ArgReduceAxis<T>(T* data, int size, int stride, ReductionOp op)
            where T : unmanaged
        {
            if (size == 0)
                return 0;

            if (size == 1)
                return 0;

            // Handle floating-point types with NaN awareness
            if (typeof(T) == typeof(float))
            {
                return ArgReduceAxisFloatNaN((float*)data, size, stride, op);
            }
            if (typeof(T) == typeof(double))
            {
                return ArgReduceAxisDoubleNaN((double*)data, size, stride, op);
            }
            // Handle boolean specially
            if (typeof(T) == typeof(bool))
            {
                return ArgReduceAxisBool((bool*)data, size, stride, op);
            }

            // Generic numeric types
            return ArgReduceAxisNumeric(data, size, stride, op);
        }

        /// <summary>
        /// ArgMax/ArgMin for float with NaN awareness.
        /// NumPy behavior: first NaN always wins.
        /// </summary>
        private static unsafe long ArgReduceAxisFloatNaN(float* data, int size, int stride, ReductionOp op)
        {
            float extreme = data[0];
            long extremeIdx = 0;

            for (int i = 1; i < size; i++)
            {
                float val = data[i * stride];

                // NumPy: first NaN always wins
                if (float.IsNaN(val) && !float.IsNaN(extreme))
                {
                    extreme = val;
                    extremeIdx = i;
                }
                else if (!float.IsNaN(extreme))
                {
                    if (op == ReductionOp.ArgMax)
                    {
                        if (val > extreme)
                        {
                            extreme = val;
                            extremeIdx = i;
                        }
                    }
                    else // ArgMin
                    {
                        if (val < extreme)
                        {
                            extreme = val;
                            extremeIdx = i;
                        }
                    }
                }
            }

            return extremeIdx;
        }

        /// <summary>
        /// ArgMax/ArgMin for double with NaN awareness.
        /// NumPy behavior: first NaN always wins.
        /// </summary>
        private static unsafe long ArgReduceAxisDoubleNaN(double* data, int size, int stride, ReductionOp op)
        {
            double extreme = data[0];
            long extremeIdx = 0;

            for (int i = 1; i < size; i++)
            {
                double val = data[i * stride];

                // NumPy: first NaN always wins
                if (double.IsNaN(val) && !double.IsNaN(extreme))
                {
                    extreme = val;
                    extremeIdx = i;
                }
                else if (!double.IsNaN(extreme))
                {
                    if (op == ReductionOp.ArgMax)
                    {
                        if (val > extreme)
                        {
                            extreme = val;
                            extremeIdx = i;
                        }
                    }
                    else // ArgMin
                    {
                        if (val < extreme)
                        {
                            extreme = val;
                            extremeIdx = i;
                        }
                    }
                }
            }

            return extremeIdx;
        }

        /// <summary>
        /// ArgMax/ArgMin for boolean.
        /// For ArgMax: True > False, find first True
        /// For ArgMin: False < True, find first False
        /// </summary>
        private static unsafe long ArgReduceAxisBool(bool* data, int size, int stride, ReductionOp op)
        {
            bool extreme = data[0];
            long extremeIdx = 0;

            for (int i = 1; i < size; i++)
            {
                bool val = data[i * stride];

                if (op == ReductionOp.ArgMax)
                {
                    // True > False: if val is True and extreme is False, update
                    if (val && !extreme)
                    {
                        extreme = val;
                        extremeIdx = i;
                    }
                }
                else // ArgMin
                {
                    // False < True: if val is False and extreme is True, update
                    if (!val && extreme)
                    {
                        extreme = val;
                        extremeIdx = i;
                    }
                }
            }

            return extremeIdx;
        }

        /// <summary>
        /// ArgMax/ArgMin for generic numeric types (non-NaN, non-boolean).
        /// </summary>
        private static unsafe long ArgReduceAxisNumeric<T>(T* data, int size, int stride, ReductionOp op)
            where T : unmanaged
        {
            // Use IComparer to compare values
            T extreme = data[0];
            long extremeIdx = 0;

            for (int i = 1; i < size; i++)
            {
                T val = data[i * stride];

                if (op == ReductionOp.ArgMax)
                {
                    if (CompareGreater(val, extreme))
                    {
                        extreme = val;
                        extremeIdx = i;
                    }
                }
                else // ArgMin
                {
                    if (CompareLess(val, extreme))
                    {
                        extreme = val;
                        extremeIdx = i;
                    }
                }
            }

            return extremeIdx;
        }

        /// <summary>
        /// Compare if a > b for numeric types.
        /// </summary>
        private static bool CompareGreater<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)a > (byte)(object)b;
            if (typeof(T) == typeof(short)) return (short)(object)a > (short)(object)b;
            if (typeof(T) == typeof(ushort)) return (ushort)(object)a > (ushort)(object)b;
            if (typeof(T) == typeof(int)) return (int)(object)a > (int)(object)b;
            if (typeof(T) == typeof(uint)) return (uint)(object)a > (uint)(object)b;
            if (typeof(T) == typeof(long)) return (long)(object)a > (long)(object)b;
            if (typeof(T) == typeof(ulong)) return (ulong)(object)a > (ulong)(object)b;
            if (typeof(T) == typeof(char)) return (char)(object)a > (char)(object)b;
            if (typeof(T) == typeof(decimal)) return (decimal)(object)a > (decimal)(object)b;
            // Float/double handled separately with NaN awareness
            throw new NotSupportedException($"CompareGreater not supported for type {typeof(T)}");
        }

        /// <summary>
        /// Compare if a < b for numeric types.
        /// </summary>
        private static bool CompareLess<T>(T a, T b) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)a < (byte)(object)b;
            if (typeof(T) == typeof(short)) return (short)(object)a < (short)(object)b;
            if (typeof(T) == typeof(ushort)) return (ushort)(object)a < (ushort)(object)b;
            if (typeof(T) == typeof(int)) return (int)(object)a < (int)(object)b;
            if (typeof(T) == typeof(uint)) return (uint)(object)a < (uint)(object)b;
            if (typeof(T) == typeof(long)) return (long)(object)a < (long)(object)b;
            if (typeof(T) == typeof(ulong)) return (ulong)(object)a < (ulong)(object)b;
            if (typeof(T) == typeof(char)) return (char)(object)a < (char)(object)b;
            if (typeof(T) == typeof(decimal)) return (decimal)(object)a < (decimal)(object)b;
            // Float/double handled separately with NaN awareness
            throw new NotSupportedException($"CompareLess not supported for type {typeof(T)}");
        }

        /// <summary>
        /// Create a general axis reduction kernel for type promotion or non-SIMD types.
        /// Uses scalar loop with type conversion.
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

        /// <summary>
        /// Create a typed axis reduction kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisReductionSimdHelper<T>(
                    (T*)input, (T*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// SIMD helper for axis reduction operations.
        /// Reduces along a specific axis, writing results to output array.
        /// </summary>
        /// <typeparam name="T">Element type</typeparam>
        /// <param name="input">Input data pointer</param>
        /// <param name="output">Output data pointer</param>
        /// <param name="inputStrides">Input strides (element units)</param>
        /// <param name="inputShape">Input shape</param>
        /// <param name="outputStrides">Output strides (element units)</param>
        /// <param name="axis">Axis to reduce along</param>
        /// <param name="axisSize">Size of the axis being reduced</param>
        /// <param name="ndim">Number of input dimensions</param>
        /// <param name="outputSize">Total number of output elements</param>
        /// <param name="op">Reduction operation</param>
        internal static unsafe void AxisReductionSimdHelper<T>(
            T* input, T* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where T : unmanaged
        {
            int axisStride = inputStrides[axis];

            // Check if the reduction axis is contiguous (stride == 1)
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            // Output has ndim-1 dimensions (axis removed)
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    // Map output dimension d to input dimension (d if d < axis, d+1 if d >= axis)
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            // Iterate over all output elements
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    // Map output dimension d to input dimension
                    int inputDim = d >= axis ? d + 1 : d;

                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];

                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Now reduce along the axis
                T* axisStart = input + inputBaseOffset;

                // For Mean, use Sum operation then divide
                ReductionOp actualOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;

                T result;
                if (axisContiguous)
                {
                    // Fast path: axis is contiguous, use SIMD
                    result = ReduceContiguousAxis(axisStart, axisSize, actualOp);
                }
                else
                {
                    // Strided path: axis is not contiguous
                    result = ReduceStridedAxis(axisStart, axisSize, axisStride, actualOp);
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    result = DivideByCountTyped(result, axisSize);

                output[outputOffset] = result;
            }
        }

        /// <summary>
        /// Divide a typed value by count (for Mean operation in SIMD path).
        /// </summary>
        private static T DivideByCountTyped<T>(T value, int count) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float result = (float)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double result = (double)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                // Integer division
                int result = (int)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long result = (long)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                byte result = (byte)((byte)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                short result = (short)((short)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort result = (ushort)((ushort)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint result = (uint)(object)value / (uint)count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong result = (ulong)(object)value / (ulong)count;
                return (T)(object)result;
            }
            // Fallback via double
            double dval = ConvertToDouble(value);
            return ConvertFromDouble<T>(dval / count);
        }

        /// <summary>
        /// Reduce a contiguous axis using SIMD.
        /// </summary>
        private static unsafe T ReduceContiguousAxis<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            if (size == 0)
            {
                return GetIdentityValue<T>(op);
            }

            if (size == 1)
            {
                return data[0];
            }

            // Use SIMD for Sum, Prod, Min, Max
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                return ReduceContiguousAxisSimd256(data, size, op);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && size >= Vector128<T>.Count)
            {
                return ReduceContiguousAxisSimd128(data, size, op);
            }
            else
            {
                return ReduceContiguousAxisScalar(data, size, op);
            }
        }

        /// <summary>
        /// Reduce contiguous axis using Vector256 SIMD with 4x unrolling.
        /// Uses 4 independent accumulators to break dependency chains.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd256<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            int vectorEnd = size - vectorCount;

            // Initialize 4 independent accumulators for loop unrolling
            var acc0 = CreateIdentityVector256<T>(op);
            var acc1 = CreateIdentityVector256<T>(op);
            var acc2 = CreateIdentityVector256<T>(op);
            var acc3 = CreateIdentityVector256<T>(op);

            int unrollStep = vectorCount * 4;
            int unrollEnd = size - unrollStep;

            int i = 0;

            // 4x unrolled loop - process 4 vectors per iteration
            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);
                acc0 = CombineVectors256(acc0, v0, op);
                acc1 = CombineVectors256(acc1, v1, op);
                acc2 = CombineVectors256(acc2, v2, op);
                acc3 = CombineVectors256(acc3, v3, op);
            }

            // Tree reduction: 4 -> 2 -> 1
            var acc01 = CombineVectors256(acc0, acc1, op);
            var acc23 = CombineVectors256(acc2, acc3, op);
            var accumVec = CombineVectors256(acc01, acc23, op);

            // Remainder loop (0-3 vectors)
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec = CombineVectors256(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce256(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using Vector128 SIMD with 4x unrolling.
        /// Uses 4 independent accumulators to break dependency chains.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd128<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            int vectorEnd = size - vectorCount;

            // Initialize 4 independent accumulators for loop unrolling
            var acc0 = CreateIdentityVector128<T>(op);
            var acc1 = CreateIdentityVector128<T>(op);
            var acc2 = CreateIdentityVector128<T>(op);
            var acc3 = CreateIdentityVector128<T>(op);

            int unrollStep = vectorCount * 4;
            int unrollEnd = size - unrollStep;

            int i = 0;

            // 4x unrolled loop - process 4 vectors per iteration
            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector128.Load(data + i);
                var v1 = Vector128.Load(data + i + vectorCount);
                var v2 = Vector128.Load(data + i + vectorCount * 2);
                var v3 = Vector128.Load(data + i + vectorCount * 3);
                acc0 = CombineVectors128(acc0, v0, op);
                acc1 = CombineVectors128(acc1, v1, op);
                acc2 = CombineVectors128(acc2, v2, op);
                acc3 = CombineVectors128(acc3, v3, op);
            }

            // Tree reduction: 4 -> 2 -> 1
            var acc01 = CombineVectors128(acc0, acc1, op);
            var acc23 = CombineVectors128(acc2, acc3, op);
            var accumVec = CombineVectors128(acc01, acc23, op);

            // Remainder loop (0-3 vectors)
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                accumVec = CombineVectors128(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce128(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using scalar loop.
        /// </summary>
        private static unsafe T ReduceContiguousAxisScalar<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            for (int i = 0; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce a strided axis (non-contiguous).
        /// </summary>
        private static unsafe T ReduceStridedAxis<T>(T* data, int size, int stride, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            for (int i = 0; i < size; i++)
            {
                result = CombineScalars(result, data[i * stride], op);
            }

            return result;
        }

        /// <summary>
        /// Get the identity value for a reduction operation.
        /// </summary>
        private static T GetIdentityValue<T>(ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float val = op switch
                {
                    ReductionOp.Sum => 0f,
                    ReductionOp.Prod => 1f,
                    ReductionOp.Min => float.PositiveInfinity,
                    ReductionOp.Max => float.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(double))
            {
                double val = op switch
                {
                    ReductionOp.Sum => 0.0,
                    ReductionOp.Prod => 1.0,
                    ReductionOp.Min => double.PositiveInfinity,
                    ReductionOp.Max => double.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(int))
            {
                int val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => int.MaxValue,
                    ReductionOp.Max => int.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(long))
            {
                long val = op switch
                {
                    ReductionOp.Sum => 0L,
                    ReductionOp.Prod => 1L,
                    ReductionOp.Min => long.MaxValue,
                    ReductionOp.Max => long.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(byte))
            {
                byte val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => byte.MaxValue,
                    ReductionOp.Max => byte.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(short))
            {
                short val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => short.MaxValue,
                    ReductionOp.Max => short.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => ushort.MaxValue,
                    ReductionOp.Max => ushort.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(uint))
            {
                uint val = op switch
                {
                    ReductionOp.Sum => 0u,
                    ReductionOp.Prod => 1u,
                    ReductionOp.Min => uint.MaxValue,
                    ReductionOp.Max => uint.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong val = op switch
                {
                    ReductionOp.Sum => 0UL,
                    ReductionOp.Prod => 1UL,
                    ReductionOp.Min => ulong.MaxValue,
                    ReductionOp.Max => ulong.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported for axis reduction");
        }

        /// <summary>
        /// Create identity Vector256 for reduction operation.
        /// </summary>
        private static Vector256<T> CreateIdentityVector256<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector256.Create(identity);
        }

        /// <summary>
        /// Create identity Vector128 for reduction operation.
        /// </summary>
        private static Vector128<T> CreateIdentityVector128<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector128.Create(identity);
        }

        /// <summary>
        /// Combine two Vector256 values using reduction operation.
        /// </summary>
        private static Vector256<T> CombineVectors256<T>(Vector256<T> a, Vector256<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector256.Add(a, b),
                ReductionOp.Prod => Vector256.Multiply(a, b),
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Combine two Vector128 values using reduction operation.
        /// </summary>
        private static Vector128<T> CombineVectors128<T>(Vector128<T> a, Vector128<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector128.Add(a, b),
                ReductionOp.Prod => Vector128.Multiply(a, b),
                ReductionOp.Min => Vector128.Min(a, b),
                ReductionOp.Max => Vector128.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Horizontal reduce Vector256 to scalar.
        /// </summary>
        private static T HorizontalReduce256<T>(Vector256<T> vec, ReductionOp op) where T : unmanaged
        {
            // First reduce to Vector128
            var lower = vec.GetLower();
            var upper = vec.GetUpper();
            var combined = CombineVectors128(lower, upper, op);

            return HorizontalReduce128(combined, op);
        }

        /// <summary>
        /// Horizontal reduce Vector128 to scalar.
        /// </summary>
        private static T HorizontalReduce128<T>(Vector128<T> vec, ReductionOp op) where T : unmanaged
        {
            int count = Vector128<T>.Count;
            T result = vec.GetElement(0);

            for (int i = 1; i < count; i++)
            {
                result = CombineScalars(result, vec.GetElement(i), op);
            }

            return result;
        }

        /// <summary>
        /// Combine two scalar values using reduction operation.
        /// </summary>
        private static T CombineScalars<T>(T a, T b, ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float fa = (float)(object)a;
                float fb = (float)(object)b;
                float result = op switch
                {
                    ReductionOp.Sum => fa + fb,
                    ReductionOp.Prod => fa * fb,
                    ReductionOp.Min => Math.Min(fa, fb),
                    ReductionOp.Max => Math.Max(fa, fb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double da = (double)(object)a;
                double db = (double)(object)b;
                double result = op switch
                {
                    ReductionOp.Sum => da + db,
                    ReductionOp.Prod => da * db,
                    ReductionOp.Min => Math.Min(da, db),
                    ReductionOp.Max => Math.Max(da, db),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                int ia = (int)(object)a;
                int ib = (int)(object)b;
                int result = op switch
                {
                    ReductionOp.Sum => ia + ib,
                    ReductionOp.Prod => ia * ib,
                    ReductionOp.Min => Math.Min(ia, ib),
                    ReductionOp.Max => Math.Max(ia, ib),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long la = (long)(object)a;
                long lb = (long)(object)b;
                long result = op switch
                {
                    ReductionOp.Sum => la + lb,
                    ReductionOp.Prod => la * lb,
                    ReductionOp.Min => Math.Min(la, lb),
                    ReductionOp.Max => Math.Max(la, lb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                int ba = (byte)(object)a;
                int bb = (byte)(object)b;
                byte result = op switch
                {
                    ReductionOp.Sum => (byte)(ba + bb),
                    ReductionOp.Prod => (byte)(ba * bb),
                    ReductionOp.Min => (byte)Math.Min(ba, bb),
                    ReductionOp.Max => (byte)Math.Max(ba, bb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                int sa = (short)(object)a;
                int sb = (short)(object)b;
                short result = op switch
                {
                    ReductionOp.Sum => (short)(sa + sb),
                    ReductionOp.Prod => (short)(sa * sb),
                    ReductionOp.Min => (short)Math.Min(sa, sb),
                    ReductionOp.Max => (short)Math.Max(sa, sb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                int usa = (ushort)(object)a;
                int usb = (ushort)(object)b;
                ushort result = op switch
                {
                    ReductionOp.Sum => (ushort)(usa + usb),
                    ReductionOp.Prod => (ushort)(usa * usb),
                    ReductionOp.Min => (ushort)Math.Min(usa, usb),
                    ReductionOp.Max => (ushort)Math.Max(usa, usb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint ua = (uint)(object)a;
                uint ub = (uint)(object)b;
                uint result = op switch
                {
                    ReductionOp.Sum => ua + ub,
                    ReductionOp.Prod => ua * ub,
                    ReductionOp.Min => Math.Min(ua, ub),
                    ReductionOp.Max => Math.Max(ua, ub),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong ula = (ulong)(object)a;
                ulong ulb = (ulong)(object)b;
                ulong result = op switch
                {
                    ReductionOp.Sum => ula + ulb,
                    ReductionOp.Prod => ula * ulb,
                    ReductionOp.Min => Math.Min(ula, ulb),
                    ReductionOp.Max => Math.Max(ula, ulb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        #endregion

        #region IKernelProvider SIMD Helper Interface Implementation

        /// <inheritdoc />
        unsafe bool IKernelProvider.All<T>(T* data, int size)
        {
            return AllSimdHelper<T>(data, size);
        }

        /// <inheritdoc />
        unsafe bool IKernelProvider.Any<T>(T* data, int size)
        {
            return AnySimdHelper<T>(data, size);
        }

        /// <inheritdoc />
        unsafe void IKernelProvider.FindNonZero<T>(T* data, int size, System.Collections.Generic.List<int> indices)
        {
            NonZeroSimdHelper(data, size, indices);
        }

        /// <inheritdoc />
        NumSharp.Generic.NDArray<int>[] IKernelProvider.ConvertFlatToCoordinates(System.Collections.Generic.List<int> flatIndices, int[] shape)
        {
            return ConvertFlatIndicesToCoordinates(flatIndices, shape);
        }

        /// <inheritdoc />
        unsafe NumSharp.Generic.NDArray<int>[] IKernelProvider.FindNonZeroStrided<T>(T* data, int[] shape, int[] strides, int offset)
        {
            return FindNonZeroStridedHelper(data, shape, strides, offset);
        }

        /// <inheritdoc />
        unsafe int IKernelProvider.CountTrue(bool* data, int size)
        {
            return CountTrueSimdHelper(data, size);
        }

        /// <inheritdoc />
        unsafe int IKernelProvider.CopyMasked<T>(T* src, bool* mask, T* dest, int size)
        {
            return CopyMaskedElementsHelper(src, mask, dest, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.Variance<T>(T* data, int size, int ddof)
        {
            return VarSimdHelper(data, size, ddof);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.StandardDeviation<T>(T* data, int size, int ddof)
        {
            return StdSimdHelper(data, size, ddof);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanSumFloat(float* data, int size)
        {
            return NanSumSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanSumDouble(double* data, int size)
        {
            return NanSumSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanProdFloat(float* data, int size)
        {
            return NanProdSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanProdDouble(double* data, int size)
        {
            return NanProdSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanMinFloat(float* data, int size)
        {
            return NanMinSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanMinDouble(double* data, int size)
        {
            return NanMinSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanMaxFloat(float* data, int size)
        {
            return NanMaxSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanMaxDouble(double* data, int size)
        {
            return NanMaxSimdHelperDouble(data, size);
        }

        #endregion

        #region Var/Std Axis Reduction

        /// <summary>
        /// Create an axis Var/Std reduction kernel.
        /// Uses two-pass algorithm: first compute mean along axis, then sum of squared differences.
        /// </summary>
        private static AxisReductionKernel CreateAxisVarStdReductionKernel(AxisReductionKernelKey key)
        {
            // Dispatch based on input type - output is always double for accuracy
            return key.InputType switch
            {
                NPTypeCode.Byte => CreateAxisVarStdKernelTyped<byte>(key),
                NPTypeCode.Int16 => CreateAxisVarStdKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisVarStdKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisVarStdKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisVarStdKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisVarStdKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisVarStdKernelTyped<ulong>(key),
                NPTypeCode.Single => CreateAxisVarStdKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisVarStdKernelTyped<double>(key),
                NPTypeCode.Decimal => CreateAxisVarStdKernelTypedDecimal(key),
                _ => CreateAxisVarStdKernelGeneral(key) // Fallback for Boolean, Char
            };
        }

        /// <summary>
        /// Create a typed axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelTyped<TInput>(AxisReductionKernelKey key)
            where TInput : unmanaged
        {
            bool isStd = key.Op == ReductionOp.Std;
            // ddof is not part of kernel key - will be passed as 0 for now
            // For proper ddof support, we'd need an extended kernel signature
            // The DefaultEngine handles ddof at a higher level

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdSimdHelper<TInput>(
                    (TInput*)input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    isStd, ddof: 0);
            };
        }

        /// <summary>
        /// Create a decimal axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelTypedDecimal(AxisReductionKernelKey key)
        {
            bool isStd = key.Op == ReductionOp.Std;

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdDecimalHelper(
                    (decimal*)input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    isStd, ddof: 0);
            };
        }

        /// <summary>
        /// Create a general (fallback) axis Var/Std kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisVarStdKernelGeneral(AxisReductionKernelKey key)
        {
            bool isStd = key.Op == ReductionOp.Std;

            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisVarStdGeneralHelper(
                    input, (double*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.InputType, isStd, ddof: 0);
            };
        }

        /// <summary>
        /// SIMD helper for axis Var/Std reduction.
        /// Uses two-pass algorithm: first compute mean, then sum of squared differences.
        /// </summary>
        internal static unsafe void AxisVarStdSimdHelper<TInput>(
            TInput* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            bool computeStd, int ddof)
            where TInput : unmanaged
        {
            int axisStride = inputStrides[axis];
            bool axisContiguous = axisStride == 1;

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

            // Divisor for variance (size - ddof)
            double divisor = axisSize - ddof;
            if (divisor <= 0)
                divisor = 1; // Prevent division by zero; will produce NaN behavior

            // Iterate over all output elements
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

                TInput* axisStart = input + inputBaseOffset;

                // Pass 1: Compute mean along axis
                double sum = 0;
                if (axisContiguous)
                {
                    sum = SumContiguousAxisDouble(axisStart, axisSize);
                }
                else
                {
                    for (int i = 0; i < axisSize; i++)
                        sum += ConvertToDouble(axisStart[i * axisStride]);
                }
                double mean = sum / axisSize;

                // Pass 2: Compute sum of squared differences
                double sqDiffSum = 0;
                if (axisContiguous)
                {
                    sqDiffSum = SumSquaredDiffContiguous(axisStart, axisSize, mean);
                }
                else
                {
                    for (int i = 0; i < axisSize; i++)
                    {
                        double val = ConvertToDouble(axisStart[i * axisStride]);
                        double diff = val - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                double variance = sqDiffSum / divisor;
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// Sum contiguous axis as double (for mean computation in Var/Std).
        /// </summary>
        private static unsafe double SumContiguousAxisDouble<T>(T* data, int size)
            where T : unmanaged
        {
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(int))
            {
                int* p = (int*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<int>.IsSupported && size >= Vector256<int>.Count)
                {
                    // Process int vectors, widening to long for accumulation to avoid overflow
                    int vectorCount = Vector256<int>.Count;  // 8 ints per vector
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var intVec = Vector256.Load(p + i);
                        // Sum all 8 ints - Vector256.Sum returns int, then add to long
                        totalSum += Vector256.Sum(intVec);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else if (typeof(T) == typeof(long))
            {
                long* p = (long*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<long>.IsSupported && size >= Vector256<long>.Count)
                {
                    int vectorCount = Vector256<long>.Count;  // 4 longs per vector
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<long>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    sum = Vector256.Sum(sumVec);
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }
            }
            else if (typeof(T) == typeof(short))
            {
                short* p = (short*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<short>.IsSupported && size >= Vector256<short>.Count)
                {
                    // Process short vectors - 16 shorts per vector
                    int vectorCount = Vector256<short>.Count;
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var shortVec = Vector256.Load(p + i);
                        // Convert to ints, then sum (avoids overflow)
                        var (lower, upper) = Vector256.Widen(shortVec);
                        totalSum += Vector256.Sum(lower) + Vector256.Sum(upper);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else if (typeof(T) == typeof(byte))
            {
                byte* p = (byte*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
                {
                    // Process byte vectors - 32 bytes per vector
                    int vectorCount = Vector256<byte>.Count;
                    int vectorEnd = size - vectorCount;
                    long totalSum = 0;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var byteVec = Vector256.Load(p + i);
                        // Widen byte->ushort->uint and sum
                        var (lower16, upper16) = Vector256.Widen(byteVec);
                        var (ll32, lu32) = Vector256.Widen(lower16);
                        var (ul32, uu32) = Vector256.Widen(upper16);
                        totalSum += Vector256.Sum(ll32) + Vector256.Sum(lu32) +
                                    Vector256.Sum(ul32) + Vector256.Sum(uu32);
                    }
                    sum = totalSum;
                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    long totalSum = 0;
                    for (int i = 0; i < size; i++)
                        totalSum += p[i];
                    sum = totalSum;
                }
            }
            else
            {
                // For other types (ushort, uint, ulong, char), use scalar loop
                for (int i = 0; i < size; i++)
                    sum += ConvertToDouble(data[i]);
            }

            return sum;
        }

        /// <summary>
        /// Sum squared differences from mean for contiguous axis.
        /// </summary>
        private static unsafe double SumSquaredDiffContiguous<T>(T* data, int size, double mean)
            where T : unmanaged
        {
            double sqDiffSum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)data;
                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }
                    sqDiffSum = Vector256.Sum(sqDiffVec);
                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)data;
                float fMean = (float)mean;
                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(fMean);
                    var sqDiffVec = Vector256<float>.Zero;
                    int i = 0;
                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }
                    sqDiffSum = Vector256.Sum(sqDiffVec);
                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (typeof(T) == typeof(int))
            {
                int* p = (int*)(void*)data;
                // 4x loop unrolling for better performance
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(long))
            {
                long* p = (long*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(short))
            {
                short* p = (short*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else if (typeof(T) == typeof(byte))
            {
                byte* p = (byte*)(void*)data;
                int unrollEnd = size - 4;
                int i = 0;
                for (; i <= unrollEnd; i += 4)
                {
                    double d0 = p[i] - mean;
                    double d1 = p[i + 1] - mean;
                    double d2 = p[i + 2] - mean;
                    double d3 = p[i + 3] - mean;
                    sqDiffSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }
                for (; i < size; i++)
                {
                    double diff = p[i] - mean;
                    sqDiffSum += diff * diff;
                }
            }
            else
            {
                // For other types (ushort, uint, ulong, char)
                for (int i = 0; i < size; i++)
                {
                    double diff = ConvertToDouble(data[i]) - mean;
                    sqDiffSum += diff * diff;
                }
            }

            return sqDiffSum;
        }

        /// <summary>
        /// Helper for axis Var/Std with decimal input.
        /// </summary>
        internal static unsafe void AxisVarStdDecimalHelper(
            decimal* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            bool computeStd, int ddof)
        {
            int axisStride = inputStrides[axis];

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

            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                decimal* axisStart = input + inputBaseOffset;

                // Pass 1: Compute mean
                decimal sum = 0;
                for (int i = 0; i < axisSize; i++)
                    sum += axisStart[i * axisStride];
                decimal mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                decimal sqDiffSum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    decimal diff = axisStart[i * axisStride] - mean;
                    sqDiffSum += diff * diff;
                }

                double variance = (double)(sqDiffSum / (decimal)divisor);
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        /// <summary>
        /// General helper for axis Var/Std with runtime type dispatch.
        /// </summary>
        internal static unsafe void AxisVarStdGeneralHelper(
            void* input, double* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            NPTypeCode inputType, bool computeStd, int ddof)
        {
            int axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();

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
            double divisor = axisSize - ddof;
            if (divisor <= 0) divisor = 1;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
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

                // Pass 1: Compute mean
                double sum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
                    sum += ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);
                }
                double mean = sum / axisSize;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
                    double val = ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);
                    double diff = val - mean;
                    sqDiffSum += diff * diff;
                }

                double variance = sqDiffSum / divisor;
                output[outputOffset] = computeStd ? Math.Sqrt(variance) : variance;
            }
        }

        #endregion
    }
}
