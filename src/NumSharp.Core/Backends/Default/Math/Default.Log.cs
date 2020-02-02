using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log(in NDArray nd, Type dtype) => Log(nd, dtype?.GetTypeCode());

        public override NDArray Log(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone(); 

            var @out = Cast(nd, ResolveUnaryReturnType(nd, typeCode), copy: true);
            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
#if _REGEN1
                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.To#1(Math.Log(*(out_addr + i))));
                        return @out;
	                }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var out_addr = (byte*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToByte(Math.Log(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt32(Math.Log(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt64(Math.Log(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToSingle(Math.Log(*(out_addr + i))));
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToDouble(Math.Log(*(out_addr + i))));
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
