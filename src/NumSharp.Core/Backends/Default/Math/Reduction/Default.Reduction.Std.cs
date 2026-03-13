using System;
using DecimalMath;
using NumSharp.Backends.Kernels;
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

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.std(np.zeros((0,3)), axis=0) returns array([nan, nan, nan]) (reducing along zero-size axis)
            // NumPy: np.std(np.zeros((0,3)), axis=1) returns array([]) with shape (0,) (reducing along non-zero axis)
            if (arr.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - return NaN scalar
                    var r = NDArray.Scalar(double.NaN);
                    if (keepdims)
                    {
                        var keepdimsShape = new int[arr.ndim];
                        for (int i = 0; i < arr.ndim; i++)
                            keepdimsShape[i] = 1;
                        r.Storage.Reshape(new Shape(keepdimsShape));
                    }
                    return r;
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var emptyOutputType = typeCode ?? NPTypeCode.Double;

                NDArray result;
                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - return NaN filled array
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                    for (int i = 0; i < result.size; i++)
                        result.SetAtIndex(double.NaN, i);
                }
                else
                {
                    // Reducing along non-zero axis - return empty array with reduced shape
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                }

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
                // NumPy: std of single element is 0.0 (always returns float64 for std)
                var r = NDArray.Scalar(0.0);
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new int[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(std_elementwise(arr, typeCode, ddof));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new int[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
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
                //if the given div axis is 1 - std of a single element is 0
                //Return zeros with the appropriate shape (NumPy behavior)
                if (keepdims)
                {
                    var keepdimsShapeDims = new int[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShapeDims[i] = (i == axis) ? 1 : shape[i];
                    return np.zeros(keepdimsShapeDims, typeCode ?? arr.GetTypeCode.GetComputingType());
                }
                return np.zeros(Shape.GetAxis(shape, axis), typeCode ?? arr.GetTypeCode.GetComputingType());
            }

            // IL-generated axis reduction fast path
            if (ILKernelGenerator.Enabled)
            {
                var ilResult = ExecuteAxisStdReductionIL(arr, axis, keepdims, typeCode ?? NPTypeCode.Double, ddof ?? 0);
                if (ilResult != null)
                    return ilResult;
            }

            //handle keepdims
            Shape axisedShape = Shape.GetAxis(shape, axis);
            var retType = typeCode ?? (arr.GetTypeCode.GetComputingType());

            //prepare ret
            var ret = new NDArray(retType, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
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

            // SIMD fast-path for contiguous arrays
            if (ILKernelGenerator.Enabled && arr.Shape.IsContiguous)
            {
                int _ddof = ddof ?? 0;
                double std;

                unsafe
                {
                    switch (arr.GetTypeCode)
                    {
                        case NPTypeCode.Single:
                            std = ILKernelGenerator.StdSimdHelper((float*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Double:
                            std = ILKernelGenerator.StdSimdHelper((double*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Byte:
                            std = ILKernelGenerator.StdSimdHelper((byte*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int16:
                            std = ILKernelGenerator.StdSimdHelper((short*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt16:
                            std = ILKernelGenerator.StdSimdHelper((ushort*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int32:
                            std = ILKernelGenerator.StdSimdHelper((int*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt32:
                            std = ILKernelGenerator.StdSimdHelper((uint*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int64:
                            std = ILKernelGenerator.StdSimdHelper((long*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt64:
                            std = ILKernelGenerator.StdSimdHelper((ulong*)arr.Address, arr.size, _ddof);
                            break;
                        default:
                            goto fallback;
                    }
                }

                // Convert to requested return type
                return Converts.ChangeType(std, retType);

                fallback:;
            }
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

        /// <summary>
        /// IL-generated axis standard deviation reduction. Returns null if kernel not available.
        /// </summary>
        private unsafe NDArray ExecuteAxisStdReductionIL(in NDArray arr, int axis, bool keepdims, NPTypeCode outputType, int ddof)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // Std axis reduction always outputs double for accuracy
            var key = new AxisReductionKernelKey(inputType, NPTypeCode.Double, ReductionOp.Std, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = ILKernelGenerator.TryGetAxisReductionKernel(key);

            if (kernel == null)
                return null;

            var outputDims = new int[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Double, outputShape, false);

            int axisSize = shape.dimensions[axis];
            int outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (int* inputStrides = shape.strides)
            fixed (int* inputDims = shape.dimensions)
            fixed (int* outputStrides = result.Shape.strides)
            {
                // The kernel computes std with ddof=0 by default
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);

                // For ddof != 0, adjust: std_ddof = std_0 * sqrt(n / (n - ddof))
                if (ddof != 0)
                {
                    double* resultPtr = (double*)result.Address;
                    double adjustment = Math.Sqrt((double)axisSize / (axisSize - ddof));
                    for (int i = 0; i < outputSize; i++)
                        resultPtr[i] *= adjustment;
                }
            }

            // Convert to requested output type if different from double
            if (outputType != NPTypeCode.Double)
            {
                result = Cast(result, outputType, copy: true);
            }

            if (keepdims)
            {
                var ks = new int[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }
    }
}
