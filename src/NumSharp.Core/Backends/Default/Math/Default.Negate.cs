using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;

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
#if _REGEN
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
                                *(out_addr + i) =
 #3;
                            else
                                *(out_addr + i) = Convert.To#1(~val+1);
                        }

                        return @out;
	                }
	                %
                    %foreach supported_numericals_signed,supported_numericals_signed_lowercase%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        var addr = (#2*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.To#1(-*(addr + i));

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
                                *(out_addr + i) = Convert.ToByte(~val + 1);
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
                                *(out_addr + i) = Convert.ToUInt16(~val + 1);
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
                                *(out_addr + i) = Convert.ToUInt32(~val + 1);
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
                                *(out_addr + i) = Convert.ToUInt64(~val + 1);
                        }

                        return @out;
                    }

                    case NPTypeCode.Char:
                    {
                        var out_addr = (char*)@out.Address;
                        var addr = (char*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                        {
                            var val = *(addr + i);
                            if (val == '\0')
                                *(out_addr + i) = '\0';
                            else
                                *(out_addr + i) = Convert.ToChar(~val + 1);
                        }

                        return @out;
                    }

                    case NPTypeCode.Int16:
                    {
                        var out_addr = (short*)@out.Address;
                        var addr = (short*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt16(-*(addr + i));

                        return @out;
                    }

                    case NPTypeCode.Int32:
                    {
                        var out_addr = (int*)@out.Address;
                        var addr = (int*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt32(-*(addr + i));

                        return @out;
                    }

                    case NPTypeCode.Int64:
                    {
                        var out_addr = (long*)@out.Address;
                        var addr = (long*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt64(-*(addr + i));

                        return @out;
                    }

                    case NPTypeCode.Double:
                    {
                        var out_addr = (double*)@out.Address;
                        var addr = (double*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToDouble(-*(addr + i));

                        return @out;
                    }

                    case NPTypeCode.Single:
                    {
                        var out_addr = (float*)@out.Address;
                        var addr = (float*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToSingle(-*(addr + i));

                        return @out;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var out_addr = (decimal*)@out.Address;
                        var addr = (decimal*)nd.Address;
                        var len = nd.size;

                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToDecimal(-*(addr + i));

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
