using System;
using DecimalMath;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // TODO! create an overload because np.power also allows to pass an array of exponents for every entry in the array

        public override NDArray Power(in NDArray lhs, in ValueType rhs, Type dtype) => Power(lhs, rhs, dtype?.GetTypeCode());

        public override NDArray Power(in NDArray lhs, in ValueType rhs, NPTypeCode? typeCode = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();

            var @out = Cast(lhs, typeCode ?? lhs.typecode, copy: true);
            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
#if _REGEN

                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.To#1(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
                    {
                        var right = (decimal) Convert.ToDecimal(rhs);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = DecimalEx.Pow(*(out_addr + i), right);

                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (byte*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToByte(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (short*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt16(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (ushort*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt16(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (int*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt32(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (uint*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt32(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (long*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt64(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (ulong*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt64(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (char*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToChar(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (double*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToDouble(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var right = Convert.ToDouble(rhs);
                        var out_addr = (float*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToSingle(Math.Pow(*(out_addr + i), right));

                        return @out;
	                }
                    case NPTypeCode.Decimal:
                    {
                        var right = (decimal) Convert.ToDecimal(rhs);
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = DecimalEx.Pow(*(out_addr + i), right);

                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
