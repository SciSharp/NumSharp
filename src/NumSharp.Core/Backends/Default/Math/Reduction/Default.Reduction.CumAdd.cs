using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override unsafe NDArray ReduceCumAdd(in NDArray arr, int? axis_, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || shape.size == 1 && shape.dimensions.Length == 1)
            {
                var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
                return r;
            }

            if (axis_ == null)
            {
                var r = cumsum_elementwise(arr, typeCode);
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
                return np.squeeze_fast(arr, axis);
            }

            //prepare ret
            var ret = new NDArray(typeCode ?? (arr.GetTypeCode.GetAccumulatingType()), shape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var slices = iterAxis.Slices;

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (ret.GetTypeCode)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_accumulatingType_defaultvals,supported_numericals_accumulatingType%
			            case NPTypeCode.#101: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<#2>();
                                var iterAxedRet = ret[slices].AsIterator<#102>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                |#102 sum = default;
                                while (hasNext())
                                {
                                    sum += (#102) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int16: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.UInt16: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int32: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.UInt32: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int64: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.UInt64: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Char: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Double: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Single: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Decimal: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                short sum = default;
                                while (hasNext())
                                {
                                    sum += (short) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ushort sum = default;
                                while (hasNext())
                                {
                                    sum += (ushort) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                uint sum = default;
                                while (hasNext())
                                {
                                    sum += (uint) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                ulong sum = default;
                                while (hasNext())
                                {
                                    sum += (ulong) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                char sum = default;
                                while (hasNext())
                                {
                                    sum += (char) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var iterAxedRet = ret[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                decimal sum = default;
                                while (hasNext())
                                {
                                    sum += (decimal) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
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

            return ret;
        }

        public NDArray CumSumElementwise<T>(in NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var ret = cumsum_elementwise(arr, typeCode);
            return typeCode.HasValue && typeCode.Value != ret.typecode ? ret.astype(typeCode.Value, true) : ret;
        }

        protected unsafe NDArray cumsum_elementwise(in NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retType, Shape.Vector(arr.size));

#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase%
			            case NPTypeCode.#101: 
                        {
                            var iter = arr.AsIterator<#2>();
                            var addr = (#102*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            |#102 sum = default;
                            while (hasNext())
                            {
                                sum += (#102) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<short>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<char>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (short*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            short sum = default;
                            while (hasNext())
                            {
                                sum += (short) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (ushort*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ushort sum = default;
                            while (hasNext())
                            {
                                sum += (ushort) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (uint*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            uint sum = default;
                            while (hasNext())
                            {
                                sum += (uint) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (ulong*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            ulong sum = default;
                            while (hasNext())
                            {
                                sum += (ulong) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (char*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            char sum = default;
                            while (hasNext())
                            {
                                sum += (char) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var addr = (decimal*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            decimal sum = default;
                            while (hasNext())
                            {
                                sum += (decimal) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
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
