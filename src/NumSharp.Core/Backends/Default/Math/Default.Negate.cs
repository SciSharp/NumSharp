using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Negate(in NDArray nd)
        {
            if (nd.size == 0)
                return nd.Clone(); //return new to maintain immutability.

            var @out = new NDArray(nd.dtype, nd.Shape, false);
            unsafe
            {
                switch (nd.GetTypeCode)
                {
                    case NPTypeCode.Boolean:
                    {
                        var out_addr = (bool*)@out.Address;
                        var addr = (bool*)nd.Address;
                        var len = nd.size;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = !*(addr + i);
                        return @out;
                    }
#if _REGEN1
	                %foreach supported_numericals_unsigned,supported_numericals_unsigned_lowercase,supported_numericals_unsigned_defaultvals%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        var addr = (#2*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == #3)
                                *(out_addr + i) = #3;
                            else
                                *(out_addr + i) = Converts.To#1(~val+1);
                        }

                        return @out;
	                }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var out_addr = (byte*)@out.Address;
                        var addr = (byte*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == 0)
                                *(out_addr + i) = 0;
                            else
                                *(out_addr + i) = Converts.ToByte(~val+1);
                        }

                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var out_addr = (ushort*)@out.Address;
                        var addr = (ushort*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == 0)
                                *(out_addr + i) = 0;
                            else
                                *(out_addr + i) = Converts.ToUInt16(~val+1);
                        }

                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var out_addr = (uint*)@out.Address;
                        var addr = (uint*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == 0U)
                                *(out_addr + i) = 0U;
                            else
                                *(out_addr + i) = Converts.ToUInt32(~val+1);
                        }

                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var out_addr = (ulong*)@out.Address;
                        var addr = (ulong*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == 0UL)
                                *(out_addr + i) = 0UL;
                            else
                                *(out_addr + i) = Converts.ToUInt64(~val+1);
                        }

                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                }
            }


            return @out;
        }
    }
}
