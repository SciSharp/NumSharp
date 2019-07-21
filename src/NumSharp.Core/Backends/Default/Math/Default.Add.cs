#if _REGEN_TEMPLATE
%template "./Add/Default.Add.#1.cs" for every supported_currently_supported, supported_currently_supported_lowercase
#endif

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Add(in NDArray x, in NDArray y)
        {
            switch (x.GetTypeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1: return Add#1(x,y);
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return AddBoolean(x, y);
                case NPTypeCode.Byte: return AddByte(x, y);
                case NPTypeCode.Int16: return AddInt16(x, y);
                case NPTypeCode.UInt16: return AddUInt16(x, y);
                case NPTypeCode.Int32: return AddInt32(x, y);
                case NPTypeCode.UInt32: return AddUInt32(x, y);
                case NPTypeCode.Int64: return AddInt64(x, y);
                case NPTypeCode.UInt64: return AddUInt64(x, y);
                case NPTypeCode.Char: return AddChar(x, y);
                case NPTypeCode.Double: return AddDouble(x, y);
                case NPTypeCode.Single: return AddSingle(x, y);
                case NPTypeCode.Decimal: return AddDecimal(x, y);
                default:
                    throw new NotSupportedException();
#endif
            }
        }
    }
}
