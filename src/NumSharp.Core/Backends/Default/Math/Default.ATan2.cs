using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ATan2(in NDArray y, in NDArray x, Type dtype) => ATan2(y, x, dtype?.GetTypeCode());

        public override NDArray ATan2(in NDArray y, in NDArray x, NPTypeCode? typeCode = null)
        {
            if (y.size == 0)
                return y.Clone();

            var out_y = Cast(y, ResolveUnaryReturnType(y, typeCode), copy: true);
            var out_x = Cast(x, ResolveUnaryReturnType(x, typeCode), copy: true);
            var len = out_y.size;

            unsafe
            {
                switch (out_y.GetTypeCode)
                {
#if _REGEN
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++) out_addr[i] = Converts.To#1(Math.Atan2(out_addr[i]));
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++) out_addr[i] = (DecimalEx.Tan(out_addr[i]));
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var out_addr = (byte*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;

                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToByte(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Int16:
	                {
                        var out_addr = (short*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToInt16(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var out_addr = (ushort*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToUInt16(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToInt32(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var out_addr = (uint*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToUInt32(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToInt64(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var out_addr = (ulong*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToUInt64(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Char:
	                {
                        var out_addr = (char*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToChar(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToDouble(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = Converts.ToSingle(Math.Atan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
                    case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)out_y.Address;
                            var out_addr_x = (byte*)out_x.Address;
                            for (int i = 0; i < len; i++) out_addr[i] = (DecimalEx.ATan2(out_addr[i], out_addr_x[i]));
                        return out_y;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
