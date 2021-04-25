using System;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var r = NDArray.Scalar(0);
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(std_elementwise(arr, typeCode, ddof));
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }
            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 and can be squeezed out.
                if (keepdims)
                    return new NDArray(arr.Storage.Alias());
                return np.squeeze_fast(arr, axis);
            }

            //handle keepdims
            Shape axisedShape = Shape.GetAxis(shape, axis);
            var retType = typeCode ?? (arr.GetTypeCode.GetComputingType());

            //prepare ret
            var ret = new NDArray(retType, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueNDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            //resolve the accumulator type

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach except(supported_numericals, "Decimal"), except(supported_numericals_lowercase, "decimal")%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_onevales%
			            case NPTypeCode.#101: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<#2>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.Set#101(Converts.To#101(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<#2>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.Set#101(Converts.To#101(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    %
                %foreach ["Decimal"], ["decimal"]%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_onevales%
			            case NPTypeCode.#101: 
                        {                            
                            if (ddof.HasValue) {
                                var _ddof = (decimal) ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<#2>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.Set#101(Converts.To#101(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<#2>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.Set#101(Converts.To#101(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            break;
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
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
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<short>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ushort>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<uint>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<ulong>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<char>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / (slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(Math.Sqrt(sum / slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);

                                break;
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt16(Converts.ToInt16(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt16(Converts.ToUInt16(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt32(Converts.ToUInt32(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetUInt64(Converts.ToUInt64(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetChar(Converts.ToChar(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(DecimalEx.Sqrt(sum / ((decimal)slice.size - _ddof))), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }
                            else
                            {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<decimal>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<decimal>(slice, NPTypeCode.Double);

                                    decimal sum = 0;
                                    while (hasNext())
                                    {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDecimal(Converts.ToDecimal(DecimalEx.Sqrt(sum / (decimal)slice.size)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                            }

                            break;
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif

            if (keepdims)
                ret.Storage.ExpandDimension(axis);

            return ret;
        }

        public T StdElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return Converts.ChangeType<T>(std_elementwise(arr, typeCode, ddof));
        }

        protected object std_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return NDArray.Scalar(0);

            var retType = typeCode ?? (arr.GetTypeCode).GetComputingType();
#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach except(supported_numericals, "Decimal"), except(supported_numericals_lowercase, "decimal")%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_onevales%
			            case NPTypeCode.#101: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (#102) Math.Sqrt(sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (#102) Math.Sqrt(sum / arr.size);
                            }
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    %
                %foreach ["Decimal"], ["decimal"]%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_onevales%
			            case NPTypeCode.#101: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (#102) DecimalEx.Sqrt(sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (#102) DecimalEx.Sqrt(sum / (decimal) arr.size);
                            }
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
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
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)Math.Sqrt(sum / arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / (arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)Math.Sqrt(sum / arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    switch (retType)
                    {
                        case NPTypeCode.Byte:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Int16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (short)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.UInt16:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ushort)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Int32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.UInt32:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (uint)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Int64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.UInt64:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (ulong)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Char:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (char)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Double:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Single:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        case NPTypeCode.Decimal:
                        {
                            if (ddof.HasValue)
                            {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)DecimalEx.Sqrt(sum / ((decimal)arr.size - _ddof));
                            }
                            else
                            {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext())
                                {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (decimal)DecimalEx.Sqrt(sum / (decimal)arr.size);
                            }
                        }

                        default:
                            throw new NotSupportedException();
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }
    }
}
