using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, Type dtype) => ModF(nd, dtype?.GetTypeCode());

        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, NPTypeCode? typeCode = null)
        {
            var @out = Cast(nd, typeCode ?? nd.typecode, copy: true);
            var @out1 = Cast(nd, typeCode ?? nd.typecode, copy: true);
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
	                default:
		                throw new NotSupportedException();
#else
                    case NPTypeCode.Double:
                        {
                            var out_addr = (double*)@out.Address;
                            var out1_addr = (double*)@out1.Address;
                            Parallel.For(0, len, (i) =>
                            {
                                var trunc = Math.Truncate(*(out_addr + i));
                                *(out_addr + i) = Converts.ToDouble(*(out_addr + i) - trunc);
                                *(out1_addr + i) = trunc;
                            });

                            return (@out, @out1);
                        }
                    case NPTypeCode.Single:
                        {
                            var out_addr = (float*)@out.Address;
                            var out1_addr = (float*)@out1.Address;
                            Parallel.For(0, len, (i) =>
                            {
                                var trunc = Math.Truncate(*(out_addr + i));
                                *(out_addr + i) = Converts.ToSingle(*(out_addr + i) - trunc);
                                *(out1_addr + i) = Convert.ToSingle(trunc);
                            });

                            return (@out, @out1);
                        }
                    default:
                        throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
