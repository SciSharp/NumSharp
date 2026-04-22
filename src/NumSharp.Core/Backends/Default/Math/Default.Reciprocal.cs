using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Reciprocal(NDArray nd, Type dtype) => Reciprocal(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise reciprocal (1/x) using IL-generated kernels.
        /// NumPy: for integer dtypes, the result preserves the input dtype with
        /// C-style truncated integer division (so 1/x is 0 for |x| >= 2, and 0
        /// for x == 0 per NumPy seterr=ignore semantics).
        /// </summary>
        public override NDArray Reciprocal(NDArray nd, NPTypeCode? typeCode = null)
        {
            if (!typeCode.HasValue && nd.GetTypeCode.IsInteger())
                return ReciprocalInteger(nd);
            return ExecuteUnaryOp(nd, UnaryOp.Reciprocal, ResolveUnaryReturnType(nd, typeCode));
        }

        private static NDArray ReciprocalInteger(NDArray nd)
        {
            // NumPy: 1/x with C truncating integer division, returning 0 when x == 0.
            var tc = nd.GetTypeCode;
            var result = new NDArray(tc, new Shape((long[])nd.shape.Clone()), false);
            long n = nd.size;
            unsafe
            {
                switch (tc)
                {
                    case NPTypeCode.SByte:
                    {
                        var src = (sbyte*)nd.Unsafe.Address;
                        var dst = (sbyte*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? (sbyte)0 : (sbyte)(1 / src[i]);
                        break;
                    }
                    case NPTypeCode.Byte:
                    {
                        var src = (byte*)nd.Unsafe.Address;
                        var dst = (byte*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? (byte)0 : (byte)(1 / src[i]);
                        break;
                    }
                    case NPTypeCode.Int16:
                    {
                        var src = (short*)nd.Unsafe.Address;
                        var dst = (short*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? (short)0 : (short)(1 / src[i]);
                        break;
                    }
                    case NPTypeCode.UInt16:
                    {
                        var src = (ushort*)nd.Unsafe.Address;
                        var dst = (ushort*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? (ushort)0 : (ushort)(1 / src[i]);
                        break;
                    }
                    case NPTypeCode.Int32:
                    {
                        var src = (int*)nd.Unsafe.Address;
                        var dst = (int*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? 0 : 1 / src[i];
                        break;
                    }
                    case NPTypeCode.UInt32:
                    {
                        var src = (uint*)nd.Unsafe.Address;
                        var dst = (uint*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? 0u : 1u / src[i];
                        break;
                    }
                    case NPTypeCode.Int64:
                    {
                        var src = (long*)nd.Unsafe.Address;
                        var dst = (long*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? 0L : 1L / src[i];
                        break;
                    }
                    case NPTypeCode.UInt64:
                    {
                        var src = (ulong*)nd.Unsafe.Address;
                        var dst = (ulong*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) dst[i] = src[i] == 0 ? 0UL : 1UL / src[i];
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Integer reciprocal not supported for {tc}");
                }
            }
            return result;
        }
    }
}
