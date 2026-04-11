using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.Arg.cs - ArgMax/ArgMin Axis Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisArgReductionKernel - ArgMax/ArgMin dispatcher
//   - CreateAxisArgReductionKernelTyped<T> - typed kernel
//   - AxisArgReductionHelper<T> - SIMD helper
//   - ArgReduceAxis variants (float NaN, double NaN, bool, numeric)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region ArgMax/ArgMin Axis Reduction
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
            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
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
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op)
            where T : unmanaged
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

                // Find argmax or argmin along axis
                T* axisStart = input + inputBaseOffset;
                long resultIndex = ArgReduceAxis(axisStart, axisSize, axisStride, op);

                output[outputOffset] = resultIndex;
            }
        }

        /// <summary>
        /// Find the index of the max or min value along an axis.
        /// </summary>
        private static unsafe long ArgReduceAxis<T>(T* data, long size, long stride, ReductionOp op)
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
        private static unsafe long ArgReduceAxisFloatNaN(float* data, long size, long stride, ReductionOp op)
        {
            float extreme = data[0];
            long extremeIdx = 0;

            for (long i = 1; i < size; i++)
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
        private static unsafe long ArgReduceAxisDoubleNaN(double* data, long size, long stride, ReductionOp op)
        {
            double extreme = data[0];
            long extremeIdx = 0;

            for (long i = 1; i < size; i++)
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
        private static unsafe long ArgReduceAxisBool(bool* data, long size, long stride, ReductionOp op)
        {
            bool extreme = data[0];
            long extremeIdx = 0;

            for (long i = 1; i < size; i++)
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
        private static unsafe long ArgReduceAxisNumeric<T>(T* data, long size, long stride, ReductionOp op)
            where T : unmanaged
        {
            // Use IComparer to compare values
            T extreme = data[0];
            long extremeIdx = 0;

            for (long i = 1; i < size; i++)
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

        #endregion
    }
}
