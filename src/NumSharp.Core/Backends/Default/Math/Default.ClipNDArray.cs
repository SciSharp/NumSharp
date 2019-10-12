using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;
using System.Linq;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // TODO! create an overload because np.power also allows to pass an array of exponents for every entry in the array

        public override NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, Type dtype, NDArray @out = null) => ClipNDArray(lhs, min, max, dtype?.GetTypeCode(), @out);

        public override NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();
            var broadcasted = np.broadcast_arrays(new NDArray[] {lhs, min, max}.Where(nd => !(nd is null)).ToArray());
            var _min = min is null ? null : np.broadcast_to(min, lhs.Shape);
            var _max = max is null ? null : np.broadcast_to(max, lhs.Shape);

            if (@out is null)
                @out = Cast(lhs, typeCode ?? lhs.typecode, copy: true);
            else if (@out.Shape != lhs.Shape)
                throw new ArgumentException($"@out's shape ({@out.Shape}) must match lhs's shape ({lhs.Shape}).'");
                
            var len = @out.size;
            if (!(min is null) && !(max is null))
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                        %foreach except(supported_numericals),except(supported_numericals_lowercase)%
	                    case NPTypeCode.#1:
	                    {
                            var out_addr = (#2*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.To#1(_max.GetAtIndex(i));
                                var minval = Converts.To#1(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
	                    }
	                    %
	                    default:
		                    throw new NotSupportedException();
#else
                        case NPTypeCode.Byte:
                        {
                            var out_addr = (byte*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToByte(_max.GetAtIndex(i));
                                var minval = Converts.ToByte(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int16:
                        {
                            var out_addr = (short*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt16(_max.GetAtIndex(i));
                                var minval = Converts.ToInt16(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt16:
                        {
                            var out_addr = (ushort*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt16(_max.GetAtIndex(i));
                                var minval = Converts.ToUInt16(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int32:
                        {
                            var out_addr = (int*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt32(_max.GetAtIndex(i));
                                var minval = Converts.ToInt32(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt32:
                        {
                            var out_addr = (uint*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt32(_max.GetAtIndex(i));
                                var minval = Converts.ToUInt32(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int64:
                        {
                            var out_addr = (long*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt64(_max.GetAtIndex(i));
                                var minval = Converts.ToInt64(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt64:
                        {
                            var out_addr = (ulong*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt64(_max.GetAtIndex(i));
                                var minval = Converts.ToUInt64(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Char:
                        {
                            var out_addr = (char*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToChar(_max.GetAtIndex(i));
                                var minval = Converts.ToChar(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Double:
                        {
                            var out_addr = (double*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToDouble(_max.GetAtIndex(i));
                                var minval = Converts.ToDouble(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Single:
                        {
                            var out_addr = (float*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToSingle(_max.GetAtIndex(i));
                                var minval = Converts.ToSingle(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Decimal:
                        {
                            var out_addr = (decimal*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToDecimal(_max.GetAtIndex(i));
                                var minval = Converts.ToDecimal(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                else if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }

                        default:
                            throw new NotSupportedException();
#endif
                    }
                }
            else if (!(min is null))
            {
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                        %foreach except(supported_numericals),except(supported_numericals_lowercase)%
	                    case NPTypeCode.#1:
	                    {
                            var out_addr = (#2*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.To#1(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
	                    }
	                    %
	                    default:
		                    throw new NotSupportedException();
#else

                        case NPTypeCode.Byte:
                        {
                            var out_addr = (byte*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToByte(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int16:
                        {
                            var out_addr = (short*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToInt16(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt16:
                        {
                            var out_addr = (ushort*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToUInt16(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int32:
                        {
                            var out_addr = (int*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToInt32(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt32:
                        {
                            var out_addr = (uint*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToUInt32(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int64:
                        {
                            var out_addr = (long*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToInt64(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt64:
                        {
                            var out_addr = (ulong*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToUInt64(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Char:
                        {
                            var out_addr = (char*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToChar(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Double:
                        {
                            var out_addr = (double*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToDouble(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Single:
                        {
                            var out_addr = (float*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToSingle(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Decimal:
                        {
                            var out_addr = (decimal*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var minval = Converts.ToDecimal(_min.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val < minval)
                                    val = minval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        default:
                            throw new NotSupportedException();
#endif
                    }
                }
            }
            else if (!(max is null))
            {
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                        %foreach except(supported_numericals),except(supported_numericals_lowercase)%
	                    case NPTypeCode.#1:
	                    {
                            var out_addr = (#2*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.To#1(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
	                    }
	                    %
	                    default:
		                    throw new NotSupportedException();
#else

                        case NPTypeCode.Byte:
                        {
                            var out_addr = (byte*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToByte(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int16:
                        {
                            var out_addr = (short*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt16(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt16:
                        {
                            var out_addr = (ushort*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt16(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int32:
                        {
                            var out_addr = (int*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt32(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt32:
                        {
                            var out_addr = (uint*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt32(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Int64:
                        {
                            var out_addr = (long*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToInt64(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.UInt64:
                        {
                            var out_addr = (ulong*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToUInt64(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Char:
                        {
                            var out_addr = (char*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToChar(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Double:
                        {
                            var out_addr = (double*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToDouble(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Single:
                        {
                            var out_addr = (float*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToSingle(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        case NPTypeCode.Decimal:
                        {
                            var out_addr = (decimal*)@out.Address;
                            Parallel.For(0, len, i =>
                            {
                                var maxval = Converts.ToDecimal(_max.GetAtIndex(i));
                                var val = *(out_addr + i);
                                if (val > maxval)
                                    val = maxval;
                                *(out_addr + i) = val;
                            });
                            return @out;
                        }
                        default:
                            throw new NotSupportedException();
#endif
                    }
                }
            }

            throw new ArgumentException("Both a_min and a_max are null");
        }
    }
}
