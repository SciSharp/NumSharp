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
            // NumPy reciprocal(bool) takes the int8 loop (True -> 1/1 = 1, False -> 1/0 = 0),
            // not a float loop: probed np.reciprocal([True,False]) -> int8 [1, 0].
            if (loopType == NPTypeCode.Boolean)
                loopType = NPTypeCode.SByte;

            // Char is a 16-bit unsigned integer (NumSharp's uint16 masquerade): NumPy
            // reciprocal(uint16) takes the integer loop (1//x, 0 for |x|>=2), NOT a float
            // loop. IsInteger() deliberately excludes Char, so admit it explicitly here.
            if (loopType.IsInteger() || loopType == NPTypeCode.Char)
            {
                // Integer loop: C-truncating 1/x with the probed per-type 1/0 sentinel.
                // The hand loop is the compute path; a provided out/where gets a masked
                // identity copy through the shared Into machinery (plan §4.3.3 temp+copy route).
                // The input is cast to the loop dtype first (e.g. bool -> int8, or an
                // explicit dtype=) so ReciprocalInteger always sees a supported integer type.
                var tmp = ReciprocalInteger(loopType != nd.GetTypeCode
                    ? Cast(nd, loopType, copy: true)
                    : nd);
                if (@out is null && where is null)
                    return tmp;
                return ExecuteUnaryOp(tmp, UnaryOp.Positive, tmp.GetTypeCode, @out, where);
            }

            return ExecuteUnaryOp(nd, UnaryOp.Reciprocal, ResolveUnaryReturnType(nd, typeCode), @out, where);
        }

        private static unsafe NDArray ReciprocalInteger(NDArray nd)
        {
            // NumPy: 1/x with C truncating integer division (so |x| >= 2 -> 0, 1 -> 1,
            // -1 -> -1). The 1/0 result is per-type and was probed bit-exact against
            // NumPy 2.4.2 (RuntimeWarning, deterministic across array sizes / lane
            // positions): the sign-bit sentinel 0x80..0 ONLY for int32 / int64 / uint64
            // (int32 -> int.MinValue, int64 -> long.MinValue, uint64 -> 2^63), and 0 for
            // every narrower type — int8, int16, uint8, uint16, AND uint32.
            // The input is read through its strides (FlatStrideOffset), so sliced /
            // strided / transposed / broadcast (stride=0) views are consumed in place —
            // no materializing copy — while the result is freshly C-contiguous.
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
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (sbyte)0 : (sbyte)(1 / x); }
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
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (short)0 : (short)(1 / x); }
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var src = (ushort*)basePtr;
                    var dst = (ushort*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (ushort)0 : (ushort)(1 / x); }
                    break;
                }
                case NPTypeCode.Char:
                {
                    // Char == 16-bit unsigned: same narrow-type 1/0 -> 0 sentinel as UInt16.
                    var src = (char*)basePtr;
                    var dst = (char*)result.Address;
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? (char)0 : (char)(1 / x); }
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
                    for (long i = 0; i < n; i++) { var x = src[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]; dst[i] = x == 0 ? 0x8000000000000000UL : 1UL / x; }
                    break;
                }
                default:
                    throw new NotSupportedException($"Integer reciprocal not supported for {tc}");
            }
            return result;
        }
    }
}
