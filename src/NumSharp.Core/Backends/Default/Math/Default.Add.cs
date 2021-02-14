using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Add(in NDArray lhs, in NDArray rhs)
        {
            switch (lhs.GetTypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Add#1(lhs, rhs);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return AddBoolean(lhs, rhs);
	            case NPTypeCode.Byte: return AddByte(lhs, rhs);
	            case NPTypeCode.Int32: return AddInt32(lhs, rhs);
	            case NPTypeCode.Int64: return AddInt64(lhs, rhs);
	            case NPTypeCode.Single: return AddSingle(lhs, rhs);
	            case NPTypeCode.Double: return AddDouble(lhs, rhs);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
