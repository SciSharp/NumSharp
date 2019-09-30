using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Frac(in NDArray nd, Type dtype) => Frac(nd, dtype?.GetTypeCode());

        public override NDArray Frac(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = Cast(nd, ResolveUnaryReturnType(nd, typeCode), copy: true);
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
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.To#1(Math.Frac(*(out_addr + i))));
                        return @out;
	                }
	                %
                    case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = (DecimalEx.Frac(*(out_addr + i))));
                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#else
                    case NPTypeCode.Double:
                        {
                            var out_addr = (double*)@out.Address;
                            Parallel.For(0, len, i => *(out_addr + i) = Converts.ToDouble(*(out_addr + i) - Math.Truncate(*(out_addr + i))));
                            return @out;
                        }
                    case NPTypeCode.Single:
                        {
                            var out_addr = (float*)@out.Address;
                            Parallel.For(0, len, i => *(out_addr + i) = Converts.ToSingle(*(out_addr + i) - Math.Truncate(*(out_addr + i))));
                            return @out;
                        }
                    case NPTypeCode.Decimal:
                        {
                            var out_addr = (decimal*)@out.Address;
                            Parallel.For(0, len, i => *(out_addr + i) = (*(out_addr + i) - Math.Truncate(*(out_addr + i))));
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
