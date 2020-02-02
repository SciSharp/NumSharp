using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Divide(in NDArray lhs, in NDArray rhs)
        {
            switch (lhs.GetTypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Divide#1(lhs, rhs);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return DivideBoolean(lhs, rhs);
	            case NPTypeCode.Byte: return DivideByte(lhs, rhs);
	            case NPTypeCode.Int32: return DivideInt32(lhs, rhs);
	            case NPTypeCode.Int64: return DivideInt64(lhs, rhs);
	            case NPTypeCode.Single: return DivideSingle(lhs, rhs);
	            case NPTypeCode.Double: return DivideDouble(lhs, rhs);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
