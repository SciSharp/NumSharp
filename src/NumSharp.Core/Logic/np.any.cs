using System;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if any element evaluates to True (non-zero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html
        public static bool any(NDArray a)
        {
            return a.TensorEngine.Any(a);
        }

        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="nd">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis or axes along which a logical OR reduction is performed. The default (axis = None) is to perform a logical OR over all the dimensions of the input array. axis may be negative, in which case it counts from the last to the first axis.</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html
        public static NDArray<bool> any(NDArray nd, int axis, bool keepdims = false)
        {
            if (nd == null)
            {
                throw new ArgumentNullException(nameof(nd), "Can't operate with null array");
            }
            if (nd.ndim == 0)
            {
                throw new ArgumentException("Can't operate with zero-dimensional array");
            }
            if (axis < 0)
                axis = nd.ndim + axis;
            if (axis < 0 || axis >= nd.ndim)
            {
                throw new ArgumentOutOfRangeException(nameof(axis));
            }

            long[] inputShape = nd.shape;
            long[] outputShape = new long[keepdims ? inputShape.Length : inputShape.Length - 1];
            int outputIndex = 0;
            for (int i = 0; i < inputShape.Length; i++)
            {
                if (i != axis)
                {
                    outputShape[outputIndex++] = inputShape[i];
                }
                else if (keepdims)
                {
                    outputShape[outputIndex++] = 1;
                }
            }

            NDArray<bool> resultArray = zeros<bool>(outputShape).MakeGeneric<bool>();
            Span<bool> resultSpan = resultArray.GetData().AsSpan<bool>();

            long axisSize = inputShape[axis];

            long postAxisStride = 1;
            for (int i = axis + 1; i < inputShape.Length; i++)
            {
                postAxisStride *= inputShape[i];
            }

            // Dispatch by type
            bool success = nd.typecode switch
            {
                NPTypeCode.Boolean => ComputeAnyPerAxis(nd.MakeGeneric<bool>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Byte => ComputeAnyPerAxis(nd.MakeGeneric<byte>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Int16 => ComputeAnyPerAxis(nd.MakeGeneric<short>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.UInt16 => ComputeAnyPerAxis(nd.MakeGeneric<ushort>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Int32 => ComputeAnyPerAxis(nd.MakeGeneric<int>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.UInt32 => ComputeAnyPerAxis(nd.MakeGeneric<uint>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Int64 => ComputeAnyPerAxis(nd.MakeGeneric<long>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.UInt64 => ComputeAnyPerAxis(nd.MakeGeneric<ulong>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Char => ComputeAnyPerAxis(nd.MakeGeneric<char>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Double => ComputeAnyPerAxis(nd.MakeGeneric<double>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Single => ComputeAnyPerAxis(nd.MakeGeneric<float>(), axisSize, postAxisStride, resultSpan),
                NPTypeCode.Decimal => ComputeAnyPerAxis(nd.MakeGeneric<decimal>(), axisSize, postAxisStride, resultSpan),
                _ => throw new NotSupportedException($"Type {nd.typecode} is not supported")
            };

            if (!success)
            {
                throw new InvalidOperationException("Failed to compute any() along the specified axis");
            }

            return resultArray;
        }

        private static bool ComputeAnyPerAxis<T>(NDArray<T> nd, long axisSize, long postAxisStride, Span<bool> resultSpan) where T : unmanaged
        {
            Span<T> inputSpan = nd.GetData().AsSpan<T>();

            for (int o = 0; o < resultSpan.Length; o++)
            {
                long blockIndex = o / postAxisStride;
                long inBlockIndex = o % postAxisStride;
                long inputStartIndex = blockIndex * axisSize * postAxisStride + inBlockIndex;

                bool currentResult = false;
                for (long a = 0; a < axisSize; a++)
                {
                    // Span indexing requires int, safe here as we're iterating through the span
                    int inputIndex = (int)(inputStartIndex + a * postAxisStride);
                    if (!inputSpan[inputIndex].Equals(default(T)))
                    {
                        currentResult = true;
                        break;
                    }
                }
                resultSpan[o] = currentResult;
            }

            return true;
        }
    }
}
