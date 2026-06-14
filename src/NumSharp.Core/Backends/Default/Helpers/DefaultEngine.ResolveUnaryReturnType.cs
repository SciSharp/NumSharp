using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        [MethodImpl(OptimizeAndInline)]
        public NPTypeCode ResolveUnaryReturnType(NDArray nd, Type @override) => ResolveUnaryReturnType(nd, @override?.GetTypeCode());

        [MethodImpl(OptimizeAndInline)]
        public NPTypeCode ResolveUnaryReturnType(NDArray nd, NPTypeCode? @override, string ufunc = "sin")
        {
            if (!@override.HasValue)
                return nd.GetTypeCode.GetComputingType();

            var over = @override.Value;
            if (over < NPTypeCode.Single)
                throw new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {ufunc}");

            return over;
        }

        /// <summary>
        ///     Resolve the result dtype for a float-producing unary ufunc (sqrt/cbrt/exp/log/trig/…)
        ///     using NumPy's <b>width-based</b> promotion (NEP50), rather than always widening to
        ///     float64. NumPy picks the narrowest float that fits the input's integer width:
        ///     <list type="bullet">
        ///       <item>bool / int8 / uint8 → float16</item>
        ///       <item>int16 / uint16 / char → float32</item>
        ///       <item>int32 / uint32 / int64 / uint64 → float64</item>
        ///       <item>float16/float32/float64/decimal/complex → preserved</item>
        ///     </list>
        ///     An explicit <paramref name="override"/> dtype is honored (must be a float/complex
        ///     loop, matching NumPy's "no loop matching signature" error for integer targets;
        ///     <paramref name="ufunc"/> is the NumPy ufunc name quoted in that error).
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public NPTypeCode ResolveUnaryFloatReturnType(NDArray nd, NPTypeCode? @override, string ufunc = "sin")
        {
            if (@override.HasValue)
            {
                var over = @override.Value;
                if (over < NPTypeCode.Single)
                    throw new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
                // The input must reach the requested loop dtype by a same_kind cast (int->float is
                // allowed, complex->real is not). Without this guard a complex input + real-float
                // dtype= (e.g. np.exp(complex128, dtype=float64)) selected a real output buffer half
                // the width of the complex the kernel writes — a buffer overflow / segfault. NumPy
                // raises "Cannot cast ufunc '<name>' input from complex128 to float64 ..." instead.
                ValidateUnaryInputCast(nd.GetTypeCode, over, ufunc);
                return over;
            }

            switch (nd.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                    return NPTypeCode.Half;       // float16
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    return NPTypeCode.Single;     // float32
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return NPTypeCode.Double;     // float64
                default:
                    // Half/Single/Double/Decimal/Complex preserve their dtype.
                    return nd.GetTypeCode;
            }
        }
    }
}
