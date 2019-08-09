using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Mod(in NDArray lhs, in NDArray rhs)
        {
            switch (lhs.GetTypeCode)
            {
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return Mod#1(lhs, rhs);
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return ModBoolean(lhs, rhs);
	            case NPTypeCode.Byte: return ModByte(lhs, rhs);
	            case NPTypeCode.Int16: return ModInt16(lhs, rhs);
	            case NPTypeCode.UInt16: return ModUInt16(lhs, rhs);
	            case NPTypeCode.Int32: return ModInt32(lhs, rhs);
	            case NPTypeCode.UInt32: return ModUInt32(lhs, rhs);
	            case NPTypeCode.Int64: return ModInt64(lhs, rhs);
	            case NPTypeCode.UInt64: return ModUInt64(lhs, rhs);
	            case NPTypeCode.Char: return ModChar(lhs, rhs);
	            case NPTypeCode.Double: return ModDouble(lhs, rhs);
	            case NPTypeCode.Single: return ModSingle(lhs, rhs);
	            case NPTypeCode.Decimal: return ModDecimal(lhs, rhs);
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
