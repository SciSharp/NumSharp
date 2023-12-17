using System;
using NumSharp.Generic;

namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>A new boolean or ndarray is returned unless out is specified, in which case a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html</remarks>
        public static bool all(NDArray a)
        {
#if _REGEN
            #region Compute
		    switch (a.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return _all_linear<#2>(a.MakeGeneric<#2>());
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (a.typecode)
            {
                case NPTypeCode.Boolean: return _all_linear<bool>(a.MakeGeneric<bool>());
                case NPTypeCode.Byte: return _all_linear<byte>(a.MakeGeneric<byte>());
                case NPTypeCode.Int16: return _all_linear<short>(a.MakeGeneric<short>());
                case NPTypeCode.UInt16: return _all_linear<ushort>(a.MakeGeneric<ushort>());
                case NPTypeCode.Int32: return _all_linear<int>(a.MakeGeneric<int>());
                case NPTypeCode.UInt32: return _all_linear<uint>(a.MakeGeneric<uint>());
                case NPTypeCode.Int64: return _all_linear<long>(a.MakeGeneric<long>());
                case NPTypeCode.UInt64: return _all_linear<ulong>(a.MakeGeneric<ulong>());
                case NPTypeCode.Char: return _all_linear<char>(a.MakeGeneric<char>());
                case NPTypeCode.Double: return _all_linear<double>(a.MakeGeneric<double>());
                case NPTypeCode.Single: return _all_linear<float>(a.MakeGeneric<float>());
                case NPTypeCode.Decimal: return _all_linear<decimal>(a.MakeGeneric<decimal>());
                default:
                    throw new NotSupportedException();
            }
            #endregion
#endif
        }

        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis or axes along which a logical AND reduction is performed. The default (axis = None) is to perform a logical OR over all the dimensions of the input array. axis may be negative, in which case it counts from the last to the first axis.</param>
        /// <returns>A new boolean or ndarray is returned unless out is specified, in which case a reference to out is returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html</remarks>
        public static NDArray<bool> all(NDArray nd, int axis, bool keepdims = false)
        {
            if (axis < 0)
                axis = nd.ndim + axis;
            if (axis < 0 || axis >= nd.ndim)
            {
                throw new ArgumentOutOfRangeException(nameof(axis));
            }
            if (nd.ndim == 0)
            {
                throw new ArgumentException("Can't operate with zero array");
            }
            if (nd == null)
            {
                throw new ArgumentException("Can't operate with null array");
            }

            int[] inputShape = nd.shape;
            int[] outputShape = new int[keepdims ? inputShape.Length : inputShape.Length - 1];
            int outputIndex = 0;
            for (int i = 0; i < inputShape.Length; i++)
            {
                if (i != axis)
                {
                    outputShape[outputIndex++] = inputShape[i];
                }
                else if (keepdims)
                {
                    outputShape[outputIndex++] = 1; // 保留轴，但长度为1
                }
            }
            
            NDArray<bool> resultArray = (NDArray<bool>)zeros<bool>(outputShape);
            Span<bool> resultSpan = resultArray.GetData().AsSpan<bool>(); 

            int axisSize = inputShape[axis];

            // It help to build an index
            int preAxisStride = 1;
            for (int i = 0; i < axis; i++)
            {
                preAxisStride *= inputShape[i];
            }

            int postAxisStride = 1;
            for (int i = axis + 1; i < inputShape.Length; i++)
            {
                postAxisStride *= inputShape[i];
            }

           
            // Operate different logic by TypeCode
            bool computationSuccess = false;
            switch (nd.typecode)
            {
                case NPTypeCode.Boolean: computationSuccess = ComputeAllPerAxis<bool>(nd.MakeGeneric<bool>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Byte: computationSuccess = ComputeAllPerAxis<byte>(nd.MakeGeneric<byte>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Int16: computationSuccess = ComputeAllPerAxis<short>(nd.MakeGeneric<short>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.UInt16: computationSuccess = ComputeAllPerAxis<ushort>(nd.MakeGeneric<ushort>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Int32: computationSuccess = ComputeAllPerAxis<int>(nd.MakeGeneric<int>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.UInt32: computationSuccess = ComputeAllPerAxis<uint>(nd.MakeGeneric<uint>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Int64: computationSuccess = ComputeAllPerAxis<long>(nd.MakeGeneric<long>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.UInt64: computationSuccess = ComputeAllPerAxis<ulong>(nd.MakeGeneric<ulong>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Char: computationSuccess = ComputeAllPerAxis<char>(nd.MakeGeneric<char>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Double: computationSuccess = ComputeAllPerAxis<double>(nd.MakeGeneric<double>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Single: computationSuccess = ComputeAllPerAxis<float>(nd.MakeGeneric<float>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                case NPTypeCode.Decimal: computationSuccess = ComputeAllPerAxis<decimal>(nd.MakeGeneric<decimal>(), axis, preAxisStride, postAxisStride, axisSize, resultSpan); break;
                default:
                    throw new NotSupportedException($"Type {nd.typecode} is not supported");
            }

            if (!computationSuccess)
            {
                throw new InvalidOperationException("Failed to compute all() along the specified axis");
            }

            return resultArray;
        }

        private static bool ComputeAllPerAxis<T>(NDArray<T> nd, int axis, int preAxisStride, int postAxisStride, int axisSize, Span<bool> resultSpan) where T : unmanaged
        { 
            Span<T> inputSpan = nd.GetData().AsSpan<T>();

            
            for (int o = 0; o < resultSpan.Length; o++)
            {
                int blockIndex = o / postAxisStride;
                int inBlockIndex = o % postAxisStride;
                int inputStartIndex = blockIndex * axisSize * postAxisStride + inBlockIndex;

                bool currentResult = true;
                for (int a = 0; a < axisSize; a++)
                {
                    int inputIndex = inputStartIndex + a * postAxisStride;
                    if (inputSpan[inputIndex].Equals(default(T)))
                    {
                        currentResult = false;
                        break; 
                    }
                }
                resultSpan[o] = currentResult;
            }

            return true;
        }

        private static bool _all_linear<T>(NDArray<T> nd) where T : unmanaged
        {
            if (nd.Shape.IsContiguous)
            {
                unsafe
                {
                    var addr = nd.Address;
                    var len = nd.size;
                    for (int i = 0; i < len; i++)
                    {
                        if (addr[i].Equals(default(T))) //if (lhs != 0/false/0f)
                            return false;
                    }

                    return true;
                }
            }
            else
            {
                using (var incr = new NDIterator<T>(nd))
                {
                    var next = incr.MoveNext;
                    var hasnext = incr.HasNext;

                    while (hasnext())
                    {
                        if (next().Equals(default(T))) //if (lhs != 0/false/0f)
                            return false;
                    }

                    return true;
                }
            }
        }
    }
}
