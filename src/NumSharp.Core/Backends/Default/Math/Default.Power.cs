using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;

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
#if _REGEN1

                    %foreach except(supported_numericals, "Decimal"),except(supported_numericals_lowercase, "decimal")%
	                case NPTypeCode.#1:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (#2*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.To#1(Math.Pow(*(out_addr + i), right)));
                        return @out;
	                }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (byte*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToByte(Math.Pow(*(out_addr + i), right)));
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (int*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt32(Math.Pow(*(out_addr + i), right)));
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (long*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToInt64(Math.Pow(*(out_addr + i), right)));
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (float*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToSingle(Math.Pow(*(out_addr + i), right)));
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var right = Converts.ToDouble(rhs);
                        var out_addr = (double*)@out.Address;
                        Parallel.For(0, len, i => *(out_addr + i) = Converts.ToDouble(Math.Pow(*(out_addr + i), right)));
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
