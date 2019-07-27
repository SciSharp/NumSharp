using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])

            if (axis_ == null)
                return NDArray.Scalar(sum_elementwise(arr));

            var axis = axis_.Value;
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.NDim == 1 || shape.IsScalar)
                return arr;

            if (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1) //if the given div axis is 1 and can be squeezed out.
                return np.squeeze_fast(arr, axis);

            //handle keepdims
            Shape axisedShape = Shape.GetAxis(shape, axis);

            //prepare ret
            var ret = new NDArray(arr.dtype, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new NDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

#if _REGEN
            #region Compute
		    switch (ret.GetTypeCode)
		    {
			    %foreach supported_currently_supported,supported_currently_supported_lowercase,supported_currently_supported_defaultvals%
			    case NPTypeCode.#1:
                    do
                    {
                        |#2 sum = #3;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<#2>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.Set#1(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
		    switch (ret.GetTypeCode)
		    {
			    case NPTypeCode.Boolean:
                    do
                    {
                        bool sum = false;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<bool>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum |= moveNext();

                        ret.SetBoolean(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Byte:
                    do
                    {
                        byte sum = 0;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<byte>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetByte(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Int16:
                    do
                    {
                        short sum = 0;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<short>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetInt16(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.UInt16:
                    do
                    {
                        ushort sum = 0;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<ushort>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetUInt16(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Int32:
                    do
                    {
                        int sum = 0;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<int>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetInt32(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.UInt32:
                    do
                    {
                        uint sum = 0u;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<uint>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetUInt32(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Int64:
                    do
                    {
                        long sum = 0L;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<long>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetInt64(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.UInt64:
                    do
                    {
                        ulong sum = 0UL;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<ulong>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetUInt64(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Char:
                    do
                    {
                        char sum = '\0';
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<char>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetChar(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Double:
                    do
                    {
                        double sum = 0d;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<double>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetDouble(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Single:
                    do
                    {
                        float sum = 0f;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<float>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetSingle(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    case NPTypeCode.Decimal:
                    do
                    {
                        decimal sum = 0m;
                        var iter = arr[slices].AsIterator();
                        var moveNext = iter.MoveNext<decimal>();
                        var hasNext = iter.HasNext;

                        while (hasNext())
                            sum += moveNext();

                        ret.SetDecimal(sum, iterIndex);
                    } while (iterAxis.Next() != null && iterRet.Next() != null);
                    break;
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif

            if (keepdims)
                ret.reshape(np.broadcast_to(ret.Shape, arr.Shape));

            return ret;
        }

        protected object sum_elementwise(NDArray arr)
        {
#if _REGEN
            #region Compute
		    switch (arr.GetTypeCode)
		    {
			    %foreach supported_currently_supported,supported_currently_supported_lowercase,supported_currently_supported_defaultvals%
			    case NPTypeCode.#1:
			    {
                    |#2 ret = #3;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<#2>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
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
                    bool ret = false;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<bool>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret |= moveNext();

                    return ret;
                }

                case NPTypeCode.Byte:
                {
                    byte ret = 0;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<byte>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Int16:
                {
                    short ret = 0;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<short>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.UInt16:
                {
                    ushort ret = 0;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<ushort>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Int32:
                {
                    int ret = 0;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<int>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.UInt32:
                {
                    uint ret = 0u;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<uint>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Int64:
                {
                    long ret = 0L;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<long>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.UInt64:
                {
                    ulong ret = 0UL;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<ulong>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Char:
                {
                    char ret = '\0';
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<char>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Double:
                {
                    double ret = 0d;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<double>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Single:
                {
                    float ret = 0f;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<float>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                case NPTypeCode.Decimal:
                {
                    decimal ret = 0m;
                    var iter = arr.AsIterator();
                    var moveNext = iter.MoveNext<decimal>();
                    var hasNext = iter.HasNext;

                    while (hasNext())
                        ret += moveNext();

                    return ret;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }
    }
}
