#if _REGEN_TEMPLATE
%template "./Add/Default.Add.#1.cs" for every supported_dtypes, supported_dtypes_lowercase
#endif

using System;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray<bool> Compare(in NDArray x, in NDArray y)
        {
            switch (x.GetTypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Equals#1(x,y);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return EqualsBoolean(x,y);
	            case NPTypeCode.Byte: return EqualsByte(x,y);
	            case NPTypeCode.Int32: return EqualsInt32(x,y);
	            case NPTypeCode.Int64: return EqualsInt64(x,y);
	            case NPTypeCode.Single: return EqualsSingle(x,y);
	            case NPTypeCode.Double: return EqualsDouble(x,y);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
