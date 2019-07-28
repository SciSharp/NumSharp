using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])

            if (axis_ == null)
            {
                var r = NDArray.Scalar(mean_elementwise(arr, typeCode));
                return keepdims ? r.reshape(np.broadcast_to(r.Shape, arr.Shape)) : r;
            }

            var axis = axis_.Value;
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.NDim == 1 || shape.IsScalar)
                return arr;

            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1) //if the given div axis is 1 and can be squeezed out.
                return np.squeeze_fast(arr, axis);

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

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_accumulatingType_defaultvals,supported_numericals_accumulatingType%
			            case NPTypeCode.#101: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                |#104 sum = #103;
                                while (hasNext())
                                    sum += (#104) moveNext();

                                ret.Set#1(Convert.To#1(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetByte(Convert.ToByte(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetByte(Convert.ToByte(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetInt16(Convert.ToInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/(ulong) slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetInt32(Convert.ToInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/(ulong) slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetInt64(Convert.ToInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetChar(Convert.ToChar(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetChar(Convert.ToChar(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetDouble(Convert.ToDouble(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetSingle(Convert.ToSingle(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
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
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/ (ulong)slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var slice = arr[slices];
                                var iter = slice.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum/slice.size), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                ret.reshape(np.broadcast_to(ret.Shape, arr.Shape));

            return ret;
        }

        protected object mean_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            var retType = typeCode ?? (arr.GetTypeCode.GetComputingType());
#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_accumulatingType_defaultvals,supported_numericals_accumulatingType%
			            case NPTypeCode.#101: 
                        {
                                var iter = arr.AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                |#104 sum = #103;
                                while (hasNext())
                                    sum += (#104) moveNext();

                                return Convert.To#1(sum/arr.size);
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
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToByte(sum/(ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToByte(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToInt16(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToInt16(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToUInt16(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToUInt16(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToInt32(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToInt32(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToUInt32(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToUInt32(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToInt64(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToInt64(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToUInt64(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToUInt64(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToChar(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToChar(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToDouble(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToDouble(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToSingle(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToSingle(sum/arr.size);
                            break;
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
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int32 sum = 0;
                                while (hasNext())
                                    sum += (Int32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = 0u;
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Int64 sum = 0L;
                                while (hasNext())
                                    sum += (Int64) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt64 sum = 0UL;
                                while (hasNext())
                                    sum += (UInt64) moveNext();

                                return Convert.ToDecimal(sum/ (ulong)arr.size);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                UInt32 sum = '\0';
                                while (hasNext())
                                    sum += (UInt32) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Double sum = 0d;
                                while (hasNext())
                                    sum += (Double) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Single sum = 0f;
                                while (hasNext())
                                    sum += (Single) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                                var iter = arr.AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                Decimal sum = 0m;
                                while (hasNext())
                                    sum += (Decimal) moveNext();

                                return Convert.ToDecimal(sum/arr.size);
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
        }
    }
}
