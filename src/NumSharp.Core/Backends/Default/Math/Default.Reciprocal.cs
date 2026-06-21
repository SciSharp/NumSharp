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
        public override NDArray Reciprocal(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);

            // dtype= selects the loop; the input must reach it same_kind
            // (probed: reciprocal(f8, dtype=i4) raises the input cast error;
            // reciprocal(i4, dtype=f8) -> [1., .5, .333..., .25]).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(nd.GetTypeCode, typeCode.Value, "reciprocal");

            var loopType = typeCode ?? nd.GetTypeCode;
            if (loopType.IsInteger())
            {
                // Integer loop: C-truncating 1/x with the probed 1/0 ->
                // MinValue semantic. The hand loop is the compute path; a
                // provided out/where gets a masked identity copy through the
                // shared Into machinery (plan §4.3.3 temp+copy route).
                var tmp = ReciprocalInteger(typeCode.HasValue && typeCode.Value != nd.GetTypeCode
                    ? Cast(nd, typeCode.Value, copy: true)
                    : nd);
                if (@out is null && where is null)
                    return tmp;
                return ExecuteUnaryOp(tmp, UnaryOp.Positive, tmp.GetTypeCode, @out, where);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Reciprocal, ResolveUnaryReturnType(nd, typeCode), @out, where);
        }

        private static unsafe NDArray ReciprocalInteger(NDArray nd)
        {
            // NumPy: 1/x with C truncating integer division. 1/0 produces the
            // signed MinValue with a RuntimeWarning in NumPy 2.4.2 (probed:
            // reciprocal(i4 [1,2,-3,0]) -> [1, 0, 0, -2147483648]); unsigned
            // zero stays 0. The input is read through its strides (FlatStrideOffset),
            // so sliced / strided / transposed / broadcast (stride=0) views are consumed
            // in place — no materializing copy — while the result is freshly C-contiguous.
            var tc = nd.GetTypeCode;
            var result = new NDArray(tc, new Shape((long[])nd.shape.Clone()), false);
            long n = nd.size;
            bool contig = nd.Shape.IsContiguous;
            var dims = nd.shape;
            var strides = nd.strides;
            int ndim = nd.ndim;
            byte* basePtr = (byte*)nd.Address + nd.Shape.offset * nd.dtypesize;
            switch (tc)
            {
                case NPTypeCode.SByte:
                {
                    var src = (sbyte*)basePtr;
                    var dst = (sbyte*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? sbyte.MinValue : (sbyte)(1 / x); }
                    break;
                }
                case NPTypeCode.Byte:
                {
                    var src = (byte*)basePtr;
                    var dst = (byte*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (byte)0 : (byte)(1 / x); }
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var src = (short*)basePtr;
                    var dst = (short*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? short.MinValue : (short)(1 / x); }
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var src = (ushort*)basePtr;
                    var dst = (ushort*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (ushort)0 : (ushort)(1 / x); }
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var src = (int*)basePtr;
                    var dst = (int*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? int.MinValue : 1 / x; }
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var src = (uint*)basePtr;
                    var dst = (uint*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? 0u : 1u / x; }
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var src = (long*)basePtr;
                    var dst = (long*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? long.MinValue : 1L / x; }
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var src = (ulong*)basePtr;
                    var dst = (ulong*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? 0UL : 1UL / x; }
                    break;
                }
                default:
                    throw new NotSupportedException($"Integer reciprocal not supported for {tc}");
            }
            return result;
        }
    }
}
