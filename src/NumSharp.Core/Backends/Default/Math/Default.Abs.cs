using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Abs(in NDArray nd, Type dtype) => Abs(nd, dtype?.GetTypeCode());

        public override NDArray Abs(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = Cast(nd, ResolveUnaryReturnType(nd, typeCode), copy: true);
            if (@out.typecode.IsUnsigned())
                return @out;

            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
#if _REGEN
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.To#1(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = (DecimalEx.Abs(*(out_addr + i))));
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var out_addr = (short*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt16(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt32(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt64(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var out_addr = (char*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToChar(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToDouble(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToSingle(Math.Abs(*(out_addr + i))));
                        return @out;
	                }
                    case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = (DecimalEx.Abs(*(out_addr + i))));
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
