using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Mod(in NDArray lhs, in NDArray rhs)
        {
            switch (lhs.GetTypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Mod#1(lhs, rhs);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return ModBoolean(lhs, rhs);
	            case NPTypeCode.Byte: return ModByte(lhs, rhs);
	            case NPTypeCode.Int32: return ModInt32(lhs, rhs);
	            case NPTypeCode.Int64: return ModInt64(lhs, rhs);
	            case NPTypeCode.Single: return ModSingle(lhs, rhs);
	            case NPTypeCode.Double: return ModDouble(lhs, rhs);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
