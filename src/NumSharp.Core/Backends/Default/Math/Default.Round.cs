using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Round(in NDArray nd, Type dtype) => Round(nd, dtype?.GetTypeCode());

        public override NDArray Round(in NDArray nd, int decimals, Type dtype) => Round(nd, decimals, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise round using IL-generated kernels.
        /// </summary>
        public override NDArray Round(in NDArray nd, NPTypeCode? typeCode = null)
        {
            return ExecuteUnaryOp(in nd, UnaryOp.Round, ResolveUnaryReturnType(nd, typeCode));
        }

        /// <summary>
        /// Element-wise round with specified decimal places.
        /// Note: This overload uses traditional loop implementation for precision control.
        /// </summary>
        public override NDArray Round(in NDArray nd, int decimals, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = Cast(nd, ResolveUnaryReturnType(nd, typeCode), copy: true);
            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Double:
                    {
                        var out_addr = (double*)@out.Address;
                        for (int i = 0; i < len; i++) out_addr[i] = Math.Round(out_addr[i], decimals);
                        return @out;
                    }
                    case NPTypeCode.Single:
                    {
                        var out_addr = (float*)@out.Address;
                        for (int i = 0; i < len; i++) out_addr[i] = (float)Math.Round(out_addr[i], decimals);
                        return @out;
                    }
                    case NPTypeCode.Decimal:
                    {
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++) out_addr[i] = decimal.Round(out_addr[i], decimals);
                        return @out;
                    }
                    default:
                        // For integer types, rounding with decimals has no effect
                        return @out;
                }
            }
        }
    }
}
