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
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray<bool> Compare(in NDArray x, in NDArray y)
        {
            switch (x.GetTypeCode)
            {
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Equals#1(x,y);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return EqualsBoolean(x,y);
	            case NPTypeCode.Byte: return EqualsByte(x,y);
	            case NPTypeCode.Int16: return EqualsInt16(x,y);
	            case NPTypeCode.UInt16: return EqualsUInt16(x,y);
	            case NPTypeCode.Int32: return EqualsInt32(x,y);
	            case NPTypeCode.UInt32: return EqualsUInt32(x,y);
	            case NPTypeCode.Int64: return EqualsInt64(x,y);
	            case NPTypeCode.UInt64: return EqualsUInt64(x,y);
	            case NPTypeCode.Char: return EqualsChar(x,y);
	            case NPTypeCode.Double: return EqualsDouble(x,y);
	            case NPTypeCode.Single: return EqualsSingle(x,y);
	            case NPTypeCode.Decimal: return EqualsDecimal(x,y);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
