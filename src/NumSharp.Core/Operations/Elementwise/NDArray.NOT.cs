using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        public static unsafe NDArray<bool> operator !(NDArray self)
        {
            var result = new NDArray(typeof(bool), self.shape);
            switch (self.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                {
                    var from = (bool*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = !*(from + len);

                    return result.MakeGeneric<bool>();
                }
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: 
                {
                    var from = (#2*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            %
	            default:
		            throw new NotSupportedException();
#else

	            case NPTypeCode.Byte: 
                {
                    var from = (byte*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Int16: 
                {
                    var from = (short*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.UInt16: 
                {
                    var from = (ushort*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Int32: 
                {
                    var from = (int*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.UInt32: 
                {
                    var from = (uint*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Int64: 
                {
                    var from = (long*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.UInt64: 
                {
                    var from = (ulong*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Char: 
                {
                    var from = (char*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Double: 
                {
                    var from = (double*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Single: 
                {
                    var from = (float*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            case NPTypeCode.Decimal: 
                {
                    var from = (decimal*)self.Address;
                    var to = (bool*)result.Address;

                    var len = result.len;
                    for (int i = 0; i < len; i++)
                        *(to + i) = *(from + len) == 0; //if val is 0 then write true

                    return result.MakeGeneric<bool>();
                }
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
