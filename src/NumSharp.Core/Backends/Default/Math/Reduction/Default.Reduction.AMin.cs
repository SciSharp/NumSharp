using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])

            if (axis_ == null)
            {
                var r = NDArray.Scalar(amin_elementwise(arr, typeCode));
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

            //prepare ret
            var ret = new NDArray(typeCode ?? arr.GetTypeCode, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new NDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
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
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals%
			            case NPTypeCode.#101: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                |#102 min = (#102)moveNext();
                                while (hasNext())
                                    min = (#102) Math.Min((#102)moveNext(), min);

                                ret.Set#1(Convert.To#1(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetByte(Convert.ToByte(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<short>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetInt16(Convert.ToInt16(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ushort>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetUInt16(Convert.ToUInt16(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetInt32(Convert.ToInt32(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<uint>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetUInt32(Convert.ToUInt32(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetInt64(Convert.ToInt64(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<ulong>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetUInt64(Convert.ToUInt64(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<char>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetChar(Convert.ToChar(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetDouble(Convert.ToDouble(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetSingle(Convert.ToSingle(min), iterIndex);
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
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte min = (byte)moveNext();
                                while (hasNext())
                                    min = (byte) Math.Min((byte)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                short min = (short)moveNext();
                                while (hasNext())
                                    min = (short) Math.Min((short)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt16: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ushort min = (ushort)moveNext();
                                while (hasNext())
                                    min = (ushort) Math.Min((ushort)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int min = (int)moveNext();
                                while (hasNext())
                                    min = (int) Math.Min((int)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                uint min = (uint)moveNext();
                                while (hasNext())
                                    min = (uint) Math.Min((uint)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long min = (long)moveNext();
                                while (hasNext())
                                    min = (long) Math.Min((long)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.UInt64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                ulong min = (ulong)moveNext();
                                while (hasNext())
                                    min = (ulong) Math.Min((ulong)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Char: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                char min = (char)moveNext();
                                while (hasNext())
                                    min = (char) Math.Min((char)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double min = (double)moveNext();
                                while (hasNext())
                                    min = (double) Math.Min((double)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float min = (float)moveNext();
                                while (hasNext())
                                    min = (float) Math.Min((float)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Decimal: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<decimal>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                decimal min = (decimal)moveNext();
                                while (hasNext())
                                    min = (decimal) Math.Min((decimal)moveNext(), min);

                                ret.SetDecimal(Convert.ToDecimal(min), iterIndex);
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

        protected object amin_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            var retType = typeCode ?? arr.GetTypeCode;
#if _REGEN
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals%
			            case NPTypeCode.#101: 
                        {
                            var iter = arr.AsIterator<#2>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            |#102 min = (#102)moveNext();
                            while (hasNext())
                                min = (#102) Math.Min((#102)moveNext(), min);

                            return Convert.To#1(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToByte(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToByte(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToInt16(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<short>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToInt16(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToUInt16(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<ushort>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToUInt16(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToInt32(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToInt32(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToUInt32(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<uint>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToUInt32(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToInt64(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToInt64(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToUInt64(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<ulong>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToUInt64(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToChar(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<char>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToChar(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToDouble(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToDouble(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToSingle(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToSingle(min);
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
                            byte min = (byte)moveNext();
                            while (hasNext())
                                min = (byte) Math.Min((byte)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Int16: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            short min = (short)moveNext();
                            while (hasNext())
                                min = (short) Math.Min((short)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.UInt16: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ushort min = (ushort)moveNext();
                            while (hasNext())
                                min = (ushort) Math.Min((ushort)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int min = (int)moveNext();
                            while (hasNext())
                                min = (int) Math.Min((int)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.UInt32: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            uint min = (uint)moveNext();
                            while (hasNext())
                                min = (uint) Math.Min((uint)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long min = (long)moveNext();
                            while (hasNext())
                                min = (long) Math.Min((long)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.UInt64: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            ulong min = (ulong)moveNext();
                            while (hasNext())
                                min = (ulong) Math.Min((ulong)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Char: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            char min = (char)moveNext();
                            while (hasNext())
                                min = (char) Math.Min((char)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double min = (double)moveNext();
                            while (hasNext())
                                min = (double) Math.Min((double)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float min = (float)moveNext();
                            while (hasNext())
                                min = (float) Math.Min((float)moveNext(), min);

                            return Convert.ToDecimal(min);
                        }
			            case NPTypeCode.Decimal: 
                        {
                            var iter = arr.AsIterator<decimal>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            decimal min = (decimal)moveNext();
                            while (hasNext())
                                min = (decimal) Math.Min((decimal)moveNext(), min);

                            return Convert.ToDecimal(min);
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
