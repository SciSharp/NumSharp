using System;
using NumSharp;
using NumSharp.Backends;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    public static unsafe partial class OpenBlasEngine
    {
        // The CBLAS product family NumPy routes through cblas beyond dot/matmul —
        // np.inner / np.vdot / np.vecdot / np.matvec / np.vecmat. Ported route-for-route from NumPy
        // 2.4.2, because the compositions the engine falls back to (Multiply+ReduceAdd for vecdot,
        // conj+dot for vdot/vecmat) reassociate the sum differently than NumPy's cblas dot and so are
        // NOT byte-identical for complex. The two dot flavours are the whole story here:
        //   • UNCONJUGATED  (?dotu / @name@_dot)  — np.inner, np.matvec, and dot/matmul's row·column.
        //   • CONJUGATING   (?dotc / CDOUBLE_vdot / @name@_dotc) — np.vdot, np.vecdot, np.vecmat.
        // For REAL dtypes the two coincide (conjugation is a no-op), which is why IBlasType<T>.Dotc
        // delegates to Dot there and only the complex struct calls zdotc.
        //
        // Each gufunc (vecdot/matvec/vecmat) is its own outer loop over the BROADCAST of the operands'
        // leading axes, invoking the per-core dot/gemv/gemm exactly as NumPy's @TYPE@_vecdot /
        // @TYPE@_matvec / @TYPE@_vecmat do — which is what makes the result bit-identical, since each
        // output element is one independent cblas call on the same operand bytes.

        /// <summary>
        ///     Parity entry point for <c>np.inner</c>: NumPy's <c>PyArray_InnerProduct</c> swaps
        ///     <paramref name="b"/>'s last two axes (when <c>a.ndim &gt;= 1</c> and <c>b.ndim &gt;= 2</c>)
        ///     and hands the pair to the very same <c>PyArray_MatrixProduct2</c> behind <c>np.dot</c> —
        ///     so this is <see cref="TryDot"/> on the swapped operand, byte-identical by construction.
        /// </summary>
        internal static bool TryInner(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLoaded)
                return false;

            var common = np._FindCommonArrayType(a.GetTypeCode, b.GetTypeCode);
            if (!IsSupported(common))
                return false;

            var right = (a.ndim >= 1 && b.ndim >= 2) ? np.swapaxes(b, -1, -2) : b;
            return TryDot(a, right, out result);
        }

        /// <summary>
        ///     Parity entry point for <c>np.vdot</c>: NumPy's <c>array_vdot</c> casts to the common
        ///     dtype, flattens BOTH operands to 1-D C-order, and calls the CONJUGATING dot
        ///     (<c>CDOUBLE_vdot</c> for complex, <c>@name@_dot</c> for real — conjugation is a no-op
        ///     there). Always a 0-d result.
        /// </summary>
        internal static bool TryVdot(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLoaded)
                return false;

            var common = np._FindCommonArrayType(a.GetTypeCode, b.GetTypeCode);
            if (!IsSupported(common))
                return false;

            // A length mismatch is NumPy's reshape error, reproduced by the managed Vdot composition —
            // decline so it raises rather than silently dotting a truncated length.
            if (a.size != b.size)
                return false;

            // FromAny(common) + Newshape({-1}, CORDER): a fresh C-contiguous 1-D read in logical order.
            // A C-contiguous operand's C-order flatten IS (Address+offset, stride 1, size) — precisely
            // what Vdot1D reads through strides[ndim-1]==1, so ravel is needed ONLY to materialise a
            // non-contiguous operand into that layout (or to give a 0-d one a last axis to stride).
            // Skipping the ravel view-allocation on the common contiguous path keeps this byte-identical
            // (same bytes, same C-order, same unit stride) while dropping ~0.5µs of the wrapper.
            var ac = AsCommon(a, common);
            var bc = AsCommon(b, common);
            var af = ac.ndim > 0 && ac.Shape.IsContiguous ? ac : np.ravel(ac);
            var bf = bc.ndim > 0 && bc.Shape.IsContiguous ? bc : np.ravel(bc);
            long n = af.size;

            // fillZeros:false throughout this file: every product's dot/gemv loop writes all output cells.
            var outBuf = new NDArray(common, Shape.Scalar, fillZeros: false);
            result = outBuf;

            if (common == NPTypeCode.Single)
                Vdot1D<float, SingleBlas>(af, bf, outBuf, n);
            else if (common == NPTypeCode.Double)
                Vdot1D<double, DoubleBlas>(af, bf, outBuf, n);
            else
                Vdot1D<Complex, ComplexBlas>(af, bf, outBuf, n);

            return true;
        }

        private static void Vdot1D<T, TOps>(NDArray af, NDArray bf, NDArray outBuf, long n)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);
            T* pa = (T*)af.Address + af.Shape.offset;
            T* pb = (T*)bf.Address + bf.Shape.offset;
            T* po = (T*)outBuf.Address + outBuf.Shape.offset;
            // Both are C-contiguous 1-D (stride 1 element) — NumPy's raveled stride == itemsize.
            ops.Dotc(pa, af.Shape.strides[af.ndim - 1], pb, bf.Shape.strides[bf.ndim - 1], po, n);
        }

        /// <summary>
        ///     Parity entry point for the <c>np.vecdot</c> gufunc <c>(n),(n)-&gt;()</c> — the CONJUGATING
        ///     dot per element of the broadcast of the operands' leading axes (NumPy's
        ///     <c>@TYPE@_vecdot</c>, <c>#DOT = dotc</c> for complex).
        /// </summary>
        internal static bool TryVecdot(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLoaded)
                return false;

            var common = np._FindCommonArrayType(x1.GetTypeCode, x2.GetTypeCode);
            if (!IsSupported(common))
                return false;

            var a = AsCommon(x1, common);
            var b = AsCommon(x2, common);

            result = common == NPTypeCode.Single ? Vecdot<float, SingleBlas>(a, b, common)
                : common == NPTypeCode.Double ? Vecdot<double, DoubleBlas>(a, b, common)
                : Vecdot<Complex, ComplexBlas>(a, b, common);

            return result is not null;
        }

        private static NDArray Vecdot<T, TOps>(NDArray a, NDArray b, NPTypeCode common)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            if (!TryLeadingBroadcast(a, 1, b, 1, out var bshape, out var sa, out var sb))
                return null;

            var ops = default(TOps);
            long n = a.Shape.dimensions[a.ndim - 1];
            long isA = a.Shape.strides[a.ndim - 1];
            long isB = b.Shape.strides[b.ndim - 1];

            var outShape = bshape.Length == 0 ? Shape.Scalar : new Shape(bshape);
            var outBuf = new NDArray(common, outShape, fillZeros: false);

            T* pa = (T*)a.Address + a.Shape.offset;
            T* pb = (T*)b.Address + b.Shape.offset;
            T* po = (T*)outBuf.Address + outBuf.Shape.offset;

            long count = ProductOf(bshape);
            var coord = bshape.Length == 0 ? Array.Empty<long>() : new long[bshape.Length];
            for (long e = 0; e < count; e++)
            {
                long oa = 0, ob = 0;
                for (int i = 0; i < coord.Length; i++)
                {
                    oa += coord[i] * sa[i];
                    ob += coord[i] * sb[i];
                }

                ops.Dotc(pa + oa, isA, pb + ob, isB, po + e, n);
                Advance(coord, bshape);
            }

            return outBuf;
        }

        /// <summary>
        ///     Parity entry point for the <c>np.matvec</c> gufunc <c>(m,n),(n)-&gt;(m)</c> — NumPy's
        ///     <c>@TYPE@_matvec</c>: a <c>?gemv</c> per broadcast element when the matrix is blasable
        ///     (and <c>dm,dn &gt; 1</c>), otherwise the UNCONJUGATED dot per row. No conjugation — this
        ///     is the linear transform, not the inner product.
        /// </summary>
        internal static bool TryMatvec(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLoaded)
                return false;

            var common = np._FindCommonArrayType(x1.GetTypeCode, x2.GetTypeCode);
            if (!IsSupported(common))
                return false;

            var a = AsCommon(x1, common);
            var b = AsCommon(x2, common);

            result = common == NPTypeCode.Single ? Matvec<float, SingleBlas>(a, b, common)
                : common == NPTypeCode.Double ? Matvec<double, DoubleBlas>(a, b, common)
                : Matvec<Complex, ComplexBlas>(a, b, common);

            return result is not null;
        }

        private static NDArray Matvec<T, TOps>(NDArray a, NDArray b, NPTypeCode common)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            if (!TryLeadingBroadcast(a, 2, b, 1, out var bshape, out var sa, out var sb))
                return null;

            var ops = default(TOps);
            long dm = a.Shape.dimensions[a.ndim - 2];
            long dn = a.Shape.dimensions[a.ndim - 1];
            long is1M = a.Shape.strides[a.ndim - 2];
            long is1N = a.Shape.strides[a.ndim - 1];
            long is2N = b.Shape.strides[b.ndim - 1];
            const long osM = 1; // the result's core (m) axis is contiguous within each batch element

            var outBuf = new NDArray(common, LeadingPlusCore(bshape, dm), fillZeros: false);

            long maxSize = OpenBlasNative.BlasMaxSize;
            bool tooBig = dm > maxSize || dn > maxSize;
            bool i1Blasable = IsBlasable2d(is1M, is1N, dm, dn) || IsBlasable2d(is1N, is1M, dn, dm);
            bool i2Blasable = IsBlasable2d(is2N, 1, dn, 1);
            bool blasable = i1Blasable && i2Blasable && !tooBig && dn > 1 && dm > 1;

            T* pa = (T*)a.Address + a.Shape.offset;
            T* pb = (T*)b.Address + b.Shape.offset;
            T* po = (T*)outBuf.Address + outBuf.Shape.offset;

            long count = ProductOf(bshape);
            var coord = bshape.Length == 0 ? Array.Empty<long>() : new long[bshape.Length];
            for (long e = 0; e < count; e++)
            {
                long oa = 0, ob = 0;
                for (int i = 0; i < coord.Length; i++)
                {
                    oa += coord[i] * sa[i];
                    ob += coord[i] * sb[i];
                }

                T* op = po + e * dm;
                if (blasable)
                    Gemv<T, TOps>(pa + oa, is1M, is1N, pb + ob, is2N, op, osM, dm, dn);
                else
                    for (long j = 0; j < dm; j++)
                        ops.Dot(pa + oa + j * is1M, is1N, pb + ob, is2N, op + j * osM, dn);

                Advance(coord, bshape);
            }

            return outBuf;
        }

        /// <summary>
        ///     Parity entry point for the <c>np.vecmat</c> gufunc <c>(n),(n,m)-&gt;(m)</c> — NumPy's
        ///     <c>@TYPE@_vecmat</c>, conjugating the vector. When blasable it is a <c>?gemm</c> with a
        ///     <c>CblasConjTrans</c> vector for COMPLEX (gemv cannot conjugate) and a plain <c>?gemv</c>
        ///     for REAL; otherwise the CONJUGATING dot per column.
        /// </summary>
        internal static bool TryVecmat(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            if (!OpenBlasNative.IsLoaded)
                return false;

            var common = np._FindCommonArrayType(x1.GetTypeCode, x2.GetTypeCode);
            if (!IsSupported(common))
                return false;

            var a = AsCommon(x1, common);
            var b = AsCommon(x2, common);

            result = common == NPTypeCode.Single ? Vecmat<float, SingleBlas>(a, b, common)
                : common == NPTypeCode.Double ? Vecmat<double, DoubleBlas>(a, b, common)
                : Vecmat<Complex, ComplexBlas>(a, b, common);

            return result is not null;
        }

        private static NDArray Vecmat<T, TOps>(NDArray a, NDArray b, NPTypeCode common)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            if (!TryLeadingBroadcast(a, 1, b, 2, out var bshape, out var sa, out var sb))
                return null;

            var ops = default(TOps);
            long dn = a.Shape.dimensions[a.ndim - 1];
            long dm = b.Shape.dimensions[b.ndim - 1];
            long is1N = a.Shape.strides[a.ndim - 1];
            long is2N = b.Shape.strides[b.ndim - 2];
            long is2M = b.Shape.strides[b.ndim - 1];
            const long osM = 1;

            var outBuf = new NDArray(common, LeadingPlusCore(bshape, dm), fillZeros: false);

            long maxSize = OpenBlasNative.BlasMaxSize;
            bool tooBig = dm > maxSize || dn > maxSize;
            bool i1Blasable = IsBlasable2d(is1N, 1, dn, 1);
            bool i2Blasable = IsBlasable2d(is2N, is2M, dn, dm) || IsBlasable2d(is2M, is2N, dm, dn);
            bool blasable = i1Blasable && i2Blasable && !tooBig && dn > 1 && dm > 1;
            bool complex = IsComplex<T>();

            T* pa = (T*)a.Address + a.Shape.offset;
            T* pb = (T*)b.Address + b.Shape.offset;
            T* po = (T*)outBuf.Address + outBuf.Shape.offset;

            long count = ProductOf(bshape);
            var coord = bshape.Length == 0 ? Array.Empty<long>() : new long[bshape.Length];
            for (long e = 0; e < count; e++)
            {
                long oa = 0, ob = 0;
                for (int i = 0; i < coord.Length; i++)
                {
                    oa += coord[i] * sa[i];
                    ob += coord[i] * sb[i];
                }

                T* op = po + e * dm;
                if (blasable)
                {
                    if (complex)
                        VecmatViaGemm<T, TOps>(pa + oa, is1N, pb + ob, is2N, is2M, op, dn, dm);
                    else
                        Gemv<T, TOps>(pb + ob, is2M, is2N, pa + oa, is1N, op, osM, dm, dn);
                }
                else
                {
                    for (long j = 0; j < dm; j++)
                        ops.Dotc(pa + oa, is1N, pb + ob + j * is2M, is2N, op + j * osM, dn);
                }

                Advance(coord, bshape);
            }

            return outBuf;
        }

        /// <summary>
        ///     Port of <c>@name@_vecmat_via_gemm</c> (matmul.c.src): complex <c>vecmat</c> cannot use
        ///     <c>gemv</c> because the vector must be conjugated, so it is a <c>1×M×N</c> <c>gemm</c>
        ///     with <c>CblasConjTrans</c> on the vector.
        /// </summary>
        private static void VecmatViaGemm<T, TOps>(T* ip1, long is1N, T* ip2, long is2N, long is2M,
            T* op, long n, long m)
            where T : unmanaged
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);
            const CBlasOrder order = CBlasOrder.RowMajor;
            long ldc = m; // os_m == 1 element, so ldc is m

            const CBlasTranspose trans1 = CBlasTranspose.ConjTrans;
            long lda = is1N;

            CBlasTranspose trans2;
            long ldb;
            if (IsBlasable2d(is2N, is2M, n, m))
            {
                trans2 = CBlasTranspose.NoTrans;
                ldb = is2N;
            }
            else
            {
                trans2 = CBlasTranspose.Trans;
                ldb = is2M;
            }

            ops.Gemm(order, trans1, trans2, 1, m, n, ip1, lda, ip2, ldb, op, ldc);
        }

        /// <summary>True only for the complex128 instantiation — constant-folded per generic type.</summary>
        private static bool IsComplex<T>() => typeof(T) == typeof(Complex);

        /// <summary>
        ///     Broadcasts the LEADING (loop) axes of two operands against each other — the axes before
        ///     the trailing <paramref name="cra"/>/<paramref name="crb"/> core dimensions — into a
        ///     shared shape, returning it plus each operand's per-axis element strides (0 where a
        ///     dimension is 1 or absent, the read-only stretch). Mirrors a gufunc's loop broadcast.
        /// </summary>
        /// <returns>False when the leading axes are not broadcast-compatible; the caller then declines
        /// and the managed composition raises NumPy's broadcast error.</returns>
        private static bool TryLeadingBroadcast(NDArray a, int cra, NDArray b, int crb,
            out long[] bshape, out long[] sa, out long[] sb)
        {
            bshape = null;
            sa = null;
            sb = null;

            var ad = a.Shape.dimensions;
            var astr = a.Shape.strides;
            var bd = b.Shape.dimensions;
            var bstr = b.Shape.strides;
            int na = a.ndim - cra;
            int nb = b.ndim - crb;
            int nl = Math.Max(na, nb);

            // The common no-batch case (a bare vector·matrix / matrix·vector, e.g. np.vecmat((n),(n,m))):
            // no leading axes, so the broadcast is one element. Hand back the cached empty arrays rather
            // than allocating three zero-length ones per call — the callers' loops key off `.Length` so
            // the shared instances are only ever read (never advanced), and `count == 1` runs the single
            // gemv/dot with zero coordinate offsets. Byte-identical, just alloc-free.
            if (nl == 0)
            {
                bshape = sa = sb = Array.Empty<long>();
                return true;
            }

            var shape = new long[nl];
            var stra = new long[nl];
            var strb = new long[nl];
            for (int i = 0; i < nl; i++)
            {
                int ia = na - nl + i;
                int ib = nb - nl + i;
                long da = ia >= 0 ? ad[ia] : 1;
                long db = ib >= 0 ? bd[ib] : 1;

                long dd;
                if (da == db) dd = da;
                else if (da == 1) dd = db;
                else if (db == 1) dd = da;
                else return false; // leading axes do not broadcast

                shape[i] = dd;
                stra[i] = (ia >= 0 && da != 1) ? astr[ia] : 0;
                strb[i] = (ib >= 0 && db != 1) ? bstr[ib] : 0;
            }

            bshape = shape;
            sa = stra;
            sb = strb;
            return true;
        }

        /// <summary>Appends a single core dimension to a leading shape (the <c>(…,m)</c> result shape).</summary>
        private static Shape LeadingPlusCore(long[] bshape, long m)
        {
            var dims = new long[bshape.Length + 1];
            Array.Copy(bshape, dims, bshape.Length);
            dims[bshape.Length] = m;
            return new Shape(dims);
        }

        private static long ProductOf(long[] dims)
        {
            long p = 1;
            for (int i = 0; i < dims.Length; i++)
                p *= dims[i];
            return p;
        }

        /// <summary>C-order odometer over <paramref name="shape"/>, last axis fastest.</summary>
        private static void Advance(long[] coord, long[] shape)
        {
            for (int i = coord.Length - 1; i >= 0; i--)
            {
                if (++coord[i] < shape[i])
                    break;
                coord[i] = 0;
            }
        }
    }
}
