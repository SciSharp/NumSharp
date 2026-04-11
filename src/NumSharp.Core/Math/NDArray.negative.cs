using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Numerical negative, element-wise.
        /// Returns -x for each element (negates ALL values, not just positive).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.negative.html</remarks>
        public NDArray negative()
        {
            if (this.size == 0)
                return this.Clone();

            var @out = TensorEngine.Cast(this, dtype ?? this.dtype, copy: true);

            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Boolean:
                    {
                        // For booleans, negative is logical NOT (same as NumPy)
                        var out_addr = (bool*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = !out_addr[i];
                        return @out;
                    }
#if _REGEN
                    %foreach supported_numericals_signed,supported_numericals_signed_lowercase%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++)
                            out_addr[i] = (#2)(-out_addr[i]);
                        return @out;
	                }
	                %
                    %foreach supported_numericals_unsigned,supported_numericals_unsigned_lowercase,supported_numericals_unsigned_defaultvals
	                case NPTypeCode.#1:
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Int16:
	                {
                        var out_addr = (short*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = (short)(-out_addr[i]);
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Byte:
	                case NPTypeCode.UInt16:
	                case NPTypeCode.UInt32:
	                case NPTypeCode.UInt64:
	                case NPTypeCode.Char:
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
