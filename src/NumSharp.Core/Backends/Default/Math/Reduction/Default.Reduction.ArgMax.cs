using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;

            if (shape.IsEmpty)
                throw new ArgumentException("attempt to get argmax of an empty sequence");

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.argmax(np.zeros((0,3)), axis=0) raises ValueError (reducing along zero-size axis)
            // NumPy: np.argmax(np.zeros((0,3)), axis=1) returns array([], dtype=int64) with shape (0,)
            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Reducing along non-zero axis - return empty Int64 array with reduced shape
                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var result = np.empty(new Shape(resultShape), NPTypeCode.Int64);

                if (keepdims)
                {
                    var keepdimsShape = new int[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++)
                    {
                        if (d == emptyAxis)
                            keepdimsShape[d] = 1;
                        else
                            keepdimsShape[d] = resultShape[sd++];
                    }
                    result.Storage.Reshape(new Shape(keepdimsShape));
                }

                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var r = NDArray.Scalar(0L);  // Int64 for NumPy 2.x alignment
                if (keepdims)
                {
                    var keepdimsShape = new int[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            if (axis_ == null)
            {
                // Use IL-generated kernels for element-wise reduction
                var r = NDArray.Scalar(argmax_elementwise_il(arr));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new int[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1, result is all zeros
                if (keepdims)
                {
                    // Keep the axis but reduce to size 1 (it's already 1)
                    return np.zeros(shape.dimensions, NPTypeCode.Int64);
                }
                return np.squeeze_fast(np.zeros(shape.dimensions, NPTypeCode.Int64), axis);
            }

            //handle keepdims - prepare output shape
            Shape axisedShape = Shape.GetAxis(shape, axis);
            Shape outputShape = axisedShape;
            if (keepdims)
            {
                // Insert a 1 at the axis position
                var keepdimsShapeDims = new int[arr.ndim];
                int srcIdx = 0;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (i == axis)
                        keepdimsShapeDims[i] = 1;
                    else
                        keepdimsShapeDims[i] = axisedShape.dimensions[srcIdx++];
                }
                outputShape = new Shape(keepdimsShapeDims);
            }

            //prepare ret
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<#2>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        |#2 max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (arr.GetTypeCode)
		    {
			    case NPTypeCode.Boolean:
                {
                    // Boolean: True=1, False=0, so argmax finds first True
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<bool>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        bool max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            // For argmax: True > False, so if val is True and max is False
                            if (val && !max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Byte:
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<byte>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        byte max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Int16: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<short>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        short max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.UInt16: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<ushort>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        ushort max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Int32: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<int>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        int max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.UInt32: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<uint>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        uint max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Int64: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<long>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        long max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.UInt64: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<ulong>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        ulong max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Char: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<char>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        char max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Double:
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<double>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        double max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            // NumPy: first NaN always wins
                            if (val > max || (double.IsNaN(val) && !double.IsNaN(max)))
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Single:
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<float>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        float max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            // NumPy: first NaN always wins
                            if (val > max || (float.IsNaN(val) && !float.IsNaN(max)))
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    case NPTypeCode.Decimal: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<decimal>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        decimal max = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val > max)
                            {
                                max = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt64(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif

            // Apply keepdims reshape if needed
            if (keepdims)
            {
                ret.Storage.Reshape(outputShape);
            }

            return ret;
        }

        public int ArgMaxElementwise(NDArray arr)
        {
            return Converts.ToInt32(argmax_elementwise(arr));
        }

        protected object argmax_elementwise(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return NDArray.Scalar(0);

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = arr.AsIterator<#2>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    |#2 max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (arr.GetTypeCode)
		    {
			    case NPTypeCode.Boolean:
                {
                    // Boolean: True=1, False=0, so argmax finds first True
                    var iter = arr.AsIterator<bool>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    bool max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        // For argmax: True > False
                        if (val && !max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Byte:
                {
                    var iter = arr.AsIterator<byte>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    byte max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Int16: 
                {
                    var iter = arr.AsIterator<short>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    short max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.UInt16: 
                {
                    var iter = arr.AsIterator<ushort>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    ushort max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Int32: 
                {
                    var iter = arr.AsIterator<int>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    int max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.UInt32: 
                {
                    var iter = arr.AsIterator<uint>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    uint max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Int64: 
                {
                    var iter = arr.AsIterator<long>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    long max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.UInt64: 
                {
                    var iter = arr.AsIterator<ulong>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    ulong max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Char: 
                {
                    var iter = arr.AsIterator<char>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    char max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    double max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        // NumPy: first NaN always wins
                        if (val > max || (double.IsNaN(val) && !double.IsNaN(max)))
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    float max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        // NumPy: first NaN always wins
                        if (val > max || (float.IsNaN(val) && !float.IsNaN(max)))
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    case NPTypeCode.Decimal: 
                {
                    var iter = arr.AsIterator<decimal>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    decimal max = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val > max)
                        {
                            max = val;
                            maxAt = idx;
                        }

                        idx++;
                    }

                    return maxAt;
                }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }
    }
}
