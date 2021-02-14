using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null)
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
                var r = NDArray.Scalar(var_elementwise(arr, typeCode, ddof));
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
            var iterRet = new NDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            //resolve the accumulator type

#if _REGEN1
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

                                    ret.Set#101(Converts.To#101(sum / (slice.size - _ddof)), iterIndex);
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

                                    ret.Set#101(Converts.To#101(sum / slice.size), iterIndex);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<byte>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / slice.size), iterIndex);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<int>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / slice.size), iterIndex);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<long>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / slice.size), iterIndex);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<float>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / slice.size), iterIndex);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetByte(Converts.ToByte(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt32(Converts.ToInt32(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetInt64(Converts.ToInt64(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetSingle(Converts.ToSingle(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / (slice.size - _ddof)), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
                            } else {
                                do
                                {
                                    var slice = arr[slices];
                                    var iter = slice.AsIterator<double>();
                                    var moveNext = iter.MoveNext;
                                    var hasNext = iter.HasNext;
                                    var xmean = MeanElementwise<double>(slice, NPTypeCode.Double);

                                    double sum = 0;
                                    while (hasNext()) {
                                        var a = moveNext() - xmean;
                                        sum += a * a;
                                    }

                                    ret.SetDouble(Converts.ToDouble(sum / slice.size), iterIndex);
                                } while (iterAxis.Next() != null && iterRet.Next() != null);
                                break;
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

            if (keepdims)
                ret.Storage.ExpandDimension(axis);

            return ret;
        }

        public T VarElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return (T)Converts.ChangeType(var_elementwise(arr, typeCode, ddof), InfoOf<T>.NPTypeCode);
        }

        protected object var_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return NDArray.Scalar(0);

            var retType = typeCode ?? (arr.GetTypeCode).GetComputingType();
#if _REGEN1
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

                                return (#102) (sum / (arr.size - _ddof));
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

                                return (#102) (sum / arr.size);
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
                                return (#102) (sum / ((decimal) arr.size - _ddof));
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
                                return (#102) (sum / (decimal) arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (byte) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (int) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (long) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (float) (sum / arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = ddof.Value;
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / (arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<double>(arr, NPTypeCode.Double);

                                double sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }

                                return (double) (sum / arr.size);
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
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (byte) (sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (byte) (sum / (decimal) arr.size);
                            }
                        }
			            case NPTypeCode.Int32: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (int) (sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (int) (sum / (decimal) arr.size);
                            }
                        }
			            case NPTypeCode.Int64: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (long) (sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (long) (sum / (decimal) arr.size);
                            }
                        }
			            case NPTypeCode.Single: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (float) (sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (float) (sum / (decimal) arr.size);
                            }
                        }
			            case NPTypeCode.Double: 
                        {
                            if (ddof.HasValue) {
                                var _ddof = (decimal)ddof.Value;
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (double) (sum / ((decimal) arr.size - _ddof));
                            } else {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var xmean = MeanElementwise<decimal>(arr, NPTypeCode.Double);

                                decimal sum = 0;
                                while (hasNext()) {
                                    var a = moveNext() - xmean;
                                    sum += a * a;
                                }
                                return (double) (sum / (decimal) arr.size);
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
