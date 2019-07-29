#if _REGEN_TEMPLATE
%template "./Add/Default.Add.#1.cs" for every supported_dtypes, supported_dtypes_lowercase
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
        public override NDArray Add(in NDArray lhs, in NDArray rhs)
        {
            switch (lhs.GetTypeCode)
            {
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Add#1(lhs, rhs);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return AddBoolean(lhs, rhs);
	            case NPTypeCode.Byte: return AddByte(lhs, rhs);
	            case NPTypeCode.Int16: return AddInt16(lhs, rhs);
	            case NPTypeCode.UInt16: return AddUInt16(lhs, rhs);
	            case NPTypeCode.Int32: return AddInt32(lhs, rhs);
	            case NPTypeCode.UInt32: return AddUInt32(lhs, rhs);
	            case NPTypeCode.Int64: return AddInt64(lhs, rhs);
	            case NPTypeCode.UInt64: return AddUInt64(lhs, rhs);
	            case NPTypeCode.Char: return AddChar(lhs, rhs);
	            case NPTypeCode.Double: return AddDouble(lhs, rhs);
	            case NPTypeCode.Single: return AddSingle(lhs, rhs);
	            case NPTypeCode.Decimal: return AddDecimal(lhs, rhs);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
