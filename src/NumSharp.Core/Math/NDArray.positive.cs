using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Positives all negative values.
        /// </summary>
        public NDArray positive()
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
                        var out_addr = (bool*)@out.Address;
                        var addr = (bool*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = !*(addr + i);
                        return @out;
                    }
#if _REGEN1
                    %foreach supported_numericals_signed,supported_numericals_signed_lowercase%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;

                        for (int i = 0; i < len; i++) {
                            var val = *(out_addr + i);
                            if (val < 0)
                                *(out_addr + i) = -val;
                        }
                        return @out;
	                }
	                %
                    %foreach supported_numericals_unsigned,supported_numericals_unsigned_lowercase,supported_numericals_unsigned_defaultvals
	                case NPTypeCode.#1:
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;

                        for (int i = 0; i < len; i++) {
                            var val = *(out_addr + i);
                            if (val < 0)
                                *(out_addr + i) = -val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;

                        for (int i = 0; i < len; i++) {
                            var val = *(out_addr + i);
                            if (val < 0)
                                *(out_addr + i) = -val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;

                        for (int i = 0; i < len; i++) {
                            var val = *(out_addr + i);
                            if (val < 0)
                                *(out_addr + i) = -val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;

                        for (int i = 0; i < len; i++) {
                            var val = *(out_addr + i);
                            if (val < 0)
                                *(out_addr + i) = -val;
                        }
                        return @out;
	                }
	                case NPTypeCode.Byte:
	                case NPTypeCode.UInt16:
	                case NPTypeCode.UInt32:
	                case NPTypeCode.UInt64:
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
