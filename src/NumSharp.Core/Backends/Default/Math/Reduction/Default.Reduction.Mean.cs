using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceMean(in NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(mean_elementwise(arr, typeCode));
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                return r;
            }

            var axis = axis_.Value;
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

                                ret.Set#101(Convert.To#101(sum/slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetByte(Convert.ToByte(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt16(Convert.ToInt16(sum / slice.size), iterIndex);
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
                                uint sum = 0;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt16(Convert.ToUInt16(sum / slice.size), iterIndex);
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
                                int sum = 0;
                                while (hasNext())
                                    sum += (int)moveNext();

                                ret.SetInt32(Convert.ToInt32(sum / slice.size), iterIndex);
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
                                uint sum = 0u;
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetUInt32(Convert.ToUInt32(sum / slice.size), iterIndex);
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
                                long sum = 0L;
                                while (hasNext())
                                    sum += (long)moveNext();

                                ret.SetInt64(Convert.ToInt64(sum / slice.size), iterIndex);
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
                                ulong sum = 0UL;
                                while (hasNext())
                                    sum += (ulong)moveNext();

                                ret.SetUInt64(Convert.ToUInt64(sum / (ulong)slice.size), iterIndex);
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
                                uint sum = '\0';
                                while (hasNext())
                                    sum += (uint)moveNext();

                                ret.SetChar(Convert.ToChar(sum / slice.size), iterIndex);
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
                                double sum = 0d;
                                while (hasNext())
                                    sum += (double)moveNext();

                                ret.SetDouble(Convert.ToDouble(sum / slice.size), iterIndex);
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
                                float sum = 0f;
                                while (hasNext())
                                    sum += (float)moveNext();

                                ret.SetSingle(Convert.ToSingle(sum / slice.size), iterIndex);
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
                                decimal sum = 0m;
                                while (hasNext())
                                    sum += (decimal)moveNext();

                                ret.SetDecimal(Convert.ToDecimal(sum / slice.size), iterIndex);
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
                ret.Shape.ExpandDimension(axis);

            return ret;
        }

        public T MeanElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            return (T)Convert.ChangeType(mean_elementwise(arr, typeCode), typeof(T));
        }

        protected object mean_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

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

                            return Convert.To#101(sum/arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToByte(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int16:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt16:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt16(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int32:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int sum = 0;
                            while (hasNext())
                                sum += (int)moveNext();

                            return Convert.ToInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt32:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = 0u;
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToUInt32(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Int64:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long sum = 0L;
                            while (hasNext())
                                sum += (long)moveNext();

                            return Convert.ToInt64(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.UInt64:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong sum = 0UL;
                            while (hasNext())
                                sum += (ulong)moveNext();

                            return Convert.ToUInt64(sum / (ulong)arr.size);
                            break;
                        }

                        case NPTypeCode.Char:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint sum = '\0';
                            while (hasNext())
                                sum += (uint)moveNext();

                            return Convert.ToChar(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Double:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double sum = 0d;
                            while (hasNext())
                                sum += (double)moveNext();

                            return Convert.ToDouble(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Single:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float sum = 0f;
                            while (hasNext())
                                sum += (float)moveNext();

                            return Convert.ToSingle(sum / arr.size);
                            break;
                        }

                        case NPTypeCode.Decimal:
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal sum = 0m;
                            while (hasNext())
                                sum += (decimal)moveNext();

                            return Convert.ToDecimal(sum / arr.size);
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
