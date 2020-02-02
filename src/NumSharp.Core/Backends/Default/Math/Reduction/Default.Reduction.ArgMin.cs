using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceArgMin(NDArray arr, int? axis_)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return NDArray.Scalar(0);

            if (axis_ == null)
                return NDArray.Scalar(argmin_elementwise(arr));

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));
            if (shape[axis] == 1)
            {
                //if the given div axis is 1 and can be squeezed out.
                return np.squeeze_fast(arr, axis);
            }

            //handle keepdims
            Shape axisedShape = Shape.GetAxis(shape, axis);

            //prepare ret
            var ret = new NDArray(NPTypeCode.Int32, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new NDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

#if _REGEN1
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
                        |#2 min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
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
			    case NPTypeCode.Byte: 
                {
                    int at;
                    do
                    {
                        var iter = arr[slices].AsIterator<byte>();
                        var moveNext = iter.MoveNext;
                        var hasNext = iter.HasNext;
                        int idx = 1, maxAt = 0;
                        byte min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
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
                        int min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
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
                        long min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
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
                        float min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
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
                        double min = moveNext();
                        while (hasNext())
                        {
                            var val = moveNext();
                            if (val < min)
                            {
                                min = val;
                                maxAt = idx;
                            }

                            idx++;
                        }

                        ret.SetInt32(maxAt, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
                }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif

            return ret;
        }

        public int ArgMinElementwise(NDArray arr)
        {
            return Converts.ToInt32(argmin_elementwise(arr));
        }

        protected object argmin_elementwise(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return NDArray.Scalar(0);
#if _REGEN1
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
                    |#2 min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
			    case NPTypeCode.Byte: 
                {
                    var iter = arr.AsIterator<byte>();
                    var moveNext = iter.MoveNext;
                    var hasNext = iter.HasNext;
                    int idx = 1, maxAt = 0;
                    byte min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
                    int min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
                    long min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
                    float min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
                    double min = moveNext();
                    while (hasNext())
                    {
                        var val = moveNext();
                        if (val < min)
                        {
                            min = val;
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
