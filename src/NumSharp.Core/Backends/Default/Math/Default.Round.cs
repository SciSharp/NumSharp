using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Round(NDArray nd, Type dtype) => Round(nd, dtype?.GetTypeCode());

        public override NDArray Round(NDArray nd, int decimals, Type dtype) => Round(nd, decimals, dtype?.GetTypeCode());

        /// <summary>
        /// np.round with decimals==0 is the 'rint' ufunc; integer inputs take
        /// NumPy's identity-copy path (probed: round_(i4) -> int32 unchanged,
        /// while np.rint(i4) would be float64 -- np.round is NOT a rint alias
        /// for ints). np.round itself accepts out= only (no where=/dtype=
        /// kwargs -- it is a function, not a ufunc; probed 2.4.2).
        /// </summary>

        /// <summary>
        /// Element-wise round using IL-generated kernels.
        /// </summary>
        public override NDArray Round(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);

            var inputType = nd.GetTypeCode;
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "rint");

            bool isInt = inputType.IsInteger();
            if (!typeCode.HasValue && isInt && @out is null && where is null)
                return Cast(nd, inputType, copy: true); // np.round int path = identity copy

            var loopType = typeCode ?? (isInt ? inputType : ResolveUnaryReturnType(nd, null));
            return ExecuteUnaryOp(nd, UnaryOp.Round, loopType, @out, where);
        }

        /// <summary>
        /// Element-wise round with specified decimal places.
        /// Note: This overload uses traditional loop implementation for precision control.
        /// </summary>
        public override NDArray Round(NDArray nd, int decimals, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            // decimals==0 routes to the rint ufunc path (out-cast errors there
            // name 'rint'); decimals!=0 is a multiply -> rint -> divide
            // COMPOSITION in NumPy, proven by the probed cast error naming
            // ufunc 'multiply' (round(f8, 1, out=i4) -> "Cannot cast ufunc
            // 'multiply' output from ..."). The hand loop below stays the
            // compute path; a provided out gets the composition's validation
            // then a masked identity copy through the shared Into machinery.
            if (decimals == 0)
                return Round(nd, typeCode, @out, null);

            // NumPy's integer path: round(int_array, decimals >= 0) is an
            // identity COPY (probed: round(i4, 2) -> int32 unchanged,
            // round(i4, 1, out=i4) -> identity ints). Negative decimals do
            // real banker's work on ints in NumPy -- a pre-existing NumSharp
            // gap the hand loop below keeps (it no-ops for ints either way).
            if (decimals >= 0 && !typeCode.HasValue && nd.GetTypeCode.IsInteger())
            {
                if (@out is null)
                    return Cast(nd, nd.GetTypeCode, copy: true);
                return ExecuteUnaryOp(nd, UnaryOp.Positive, nd.GetTypeCode, @out, null);
            }

            if (@out is not null)
            {
                var tmpOut = RoundDecimalsCore(nd, decimals, typeCode);
                ValidateOutCast(tmpOut.GetTypeCode, @out.typecode, "multiply");
                return ExecuteUnaryOp(tmpOut, UnaryOp.Positive, tmpOut.GetTypeCode, @out, null);
            }

            return RoundDecimalsCore(nd, decimals, typeCode);
        }

        private NDArray RoundDecimalsCore(NDArray nd, int decimals, NPTypeCode? typeCode)
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
                        for (long i = 0; i < len; i++) out_addr[i] = Math.Round(out_addr[i], decimals);
                        return @out;
                    }
                    case NPTypeCode.Single:
                    {
                        var out_addr = (float*)@out.Address;
                        for (long i = 0; i < len; i++) out_addr[i] = (float)Math.Round(out_addr[i], decimals);
                        return @out;
                    }
                    case NPTypeCode.Decimal:
                    {
                        var out_addr = (decimal*)@out.Address;
                        for (long i = 0; i < len; i++) out_addr[i] = decimal.Round(out_addr[i], decimals);
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
