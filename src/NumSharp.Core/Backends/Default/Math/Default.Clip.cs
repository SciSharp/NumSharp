using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // TODO! create an overload because np.power also allows to pass an array of exponents for every entry in the array

        public override NDArray Clip(in NDArray lhs, in ValueType min, in ValueType max, Type dtype) => Clip(lhs, min, max, dtype?.GetTypeCode());

        public override NDArray Clip(in NDArray lhs, in ValueType min, in ValueType max, NPTypeCode? typeCode = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();

            var @out = Cast(lhs, typeCode ?? lhs.typecode, copy: true);
            var len = @out.size;
            if (min != null && max != null)
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var minval = Converts.To#1(min);
                        var maxval = Converts.To#1(max);
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
                    {
                        var minval = Converts.ToDecimal(min);
                        var maxval = Converts.ToDecimal(max);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var minval = Converts.ToByte(min);
                        var maxval = Converts.ToByte(max);
                        var out_addr = (byte*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var minval = Converts.ToInt16(min);
                        var maxval = Converts.ToInt16(max);
                        var out_addr = (short*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var minval = Converts.ToUInt16(min);
                        var maxval = Converts.ToUInt16(max);
                        var out_addr = (ushort*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var minval = Converts.ToInt32(min);
                        var maxval = Converts.ToInt32(max);
                        var out_addr = (int*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var minval = Converts.ToUInt32(min);
                        var maxval = Converts.ToUInt32(max);
                        var out_addr = (uint*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var minval = Converts.ToInt64(min);
                        var maxval = Converts.ToInt64(max);
                        var out_addr = (long*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var minval = Converts.ToUInt64(min);
                        var maxval = Converts.ToUInt64(max);
                        var out_addr = (ulong*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var minval = Converts.ToChar(min);
                        var maxval = Converts.ToChar(max);
                        var out_addr = (char*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var minval = Converts.ToDouble(min);
                        var maxval = Converts.ToDouble(max);
                        var out_addr = (double*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var minval = Converts.ToSingle(min);
                        var maxval = Converts.ToSingle(max);
                        var out_addr = (float*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
                    case NPTypeCode.Decimal:
                    {
                        var minval = Converts.ToDecimal(min);
                        var maxval = Converts.ToDecimal(max);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            else if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                    }
                }
            else if (min != null)
            {
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var minval = Converts.To#1(min);
                        var out_addr = (#2*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
                    {
                        var minval = Converts.ToDecimal(min);
                        var out_addr = (decimal*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var minval = Converts.ToByte(min);
                        var out_addr = (byte*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var minval = Converts.ToInt16(min);
                        var out_addr = (short*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var minval = Converts.ToUInt16(min);
                        var out_addr = (ushort*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var minval = Converts.ToInt32(min);
                        var out_addr = (int*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var minval = Converts.ToUInt32(min);
                        var out_addr = (uint*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var minval = Converts.ToInt64(min);
                        var out_addr = (long*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var minval = Converts.ToUInt64(min);
                        var out_addr = (ulong*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var minval = Converts.ToChar(min);
                        var out_addr = (char*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var minval = Converts.ToDouble(min);
                        var out_addr = (double*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var minval = Converts.ToSingle(min);
                        var out_addr = (float*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
                    case NPTypeCode.Decimal:
                    {
                        var minval = Converts.ToDecimal(min);
                        var out_addr = (decimal*)@out.Address;
                         for (int i = 0; i < len; i++) 
                        {
                            var val = out_addr[i];
                            if (val < minval)
                                val = minval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                    }
                }
            }
            else if (max != null)
            {
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
#if _REGEN
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var maxval = Converts.To#1(max);
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
                    {
                        var maxval = Converts.ToDecimal(max);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var maxval = Converts.ToByte(max);
                        var out_addr = (byte*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var maxval = Converts.ToInt16(max);
                        var out_addr = (short*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var maxval = Converts.ToUInt16(max);
                        var out_addr = (ushort*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var maxval = Converts.ToInt32(max);
                        var out_addr = (int*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var maxval = Converts.ToUInt32(max);
                        var out_addr = (uint*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var maxval = Converts.ToInt64(max);
                        var out_addr = (long*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var maxval = Converts.ToUInt64(max);
                        var out_addr = (ulong*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var maxval = Converts.ToChar(max);
                        var out_addr = (char*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var maxval = Converts.ToDouble(max);
                        var out_addr = (double*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var maxval = Converts.ToSingle(max);
                        var out_addr = (float*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
                    case NPTypeCode.Decimal:
                    {
                        var maxval = Converts.ToDecimal(max);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var val = out_addr[i];
                            if (val > maxval)
                                val = maxval;
                            out_addr[i] = val;
                        }
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                    }
                }
            }

            return null;
        }
    }
}
