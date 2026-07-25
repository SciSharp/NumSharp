using System;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels.Blas
{
    internal static unsafe partial class BlasParity
    {
        /// <summary>
        ///     Parity entry point for the 2-D core of <c>np.matmul</c> (and of every batch element of
        ///     a stacked matmul): mirrors <c>@TYPE@_matmul</c> and writes into <paramref name="result"/>.
        /// </summary>
        /// <returns>False when the parity backend is off or cannot service this input.</returns>
        internal static bool TryMatmul2D(NDArray left, NDArray right, NDArray result)
        {
            if (!Enabled || !CBlasNative.IsLoaded)
                return false;

            var typeCode = result.GetTypeCode;
            if (!IsSupported(typeCode))
                return false;

            if (left.Shape.NDim != 2 || right.Shape.NDim != 2 || result.Shape.NDim != 2)
                return false;

            left = AsCommon(left, typeCode);
            right = AsCommon(right, typeCode);

            if (typeCode == NPTypeCode.Single)
                Matmul2D<float, SingleBlas>(left, right, result);
            else
                Matmul2D<double, DoubleBlas>(left, right, result);

            return true;
        }

        private static void Matmul2D<T, TOps>(NDArray left, NDArray right, NDArray result)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            var a = AsMat<T>(left);
            var b = AsMat<T>(right);
            var c = AsMat<T>(result);

            long dm = a.D0, dn = a.D1, dp = b.D1;
            var plan = BuildMatmulPlan<T>(a.S0, a.S1, b.S0, b.S1, c.S0, c.S1, dm, dn, dp);
            try
            {
                MatmulCore<T, TOps>(ref plan, a.Data, a.S0, a.S1, b.Data, b.S0, b.S1,
                    c.Data, c.S0, c.S1, dm, dn, dp);
            }
            finally
            {
                FreeMatmulPlan(ref plan);
            }
        }

        /// <summary>
        ///     Parity entry point for <c>np.dot</c>: mirrors <c>PyArray_MatrixProduct2</c> — the
        ///     cblas dispatcher for float/double operands of at most 2 dimensions, and the
        ///     <c>dotfunc</c>-per-output-element iterator route for anything higher.
        /// </summary>
        /// <returns>False when the parity backend is off or cannot service this input.</returns>
        internal static bool TryDot(NDArray left, NDArray right, out NDArray result)
        {
            result = null;
            if (!Enabled || !CBlasNative.IsLoaded)
                return false;

            var typeCode = np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode);
            if (!IsSupported(typeCode))
                return false;

            left = AsCommon(left, typeCode);
            right = AsCommon(right, typeCode);

            // HAVE_CBLAS branch of PyArray_MatrixProduct2 — only for ndim <= 2.
            if (left.Shape.NDim <= 2 && right.Shape.NDim <= 2)
            {
                return typeCode == NPTypeCode.Single
                    ? TryDot2D<float, SingleBlas>(left, right, out result)
                    : TryDot2D<double, DoubleBlas>(left, right, out result);
            }

            if (left.Shape.NDim == 0 || right.Shape.NDim == 0)
                return false; // np.multiply — exact either way, leave it to the engine

            return typeCode == NPTypeCode.Single
                ? TryDotND<float, SingleBlas>(left, right, out result)
                : TryDotND<double, DoubleBlas>(left, right, out result);
        }

        /// <summary>
        ///     Port of the generic (non-cblas) tail of <c>PyArray_MatrixProduct2</c>
        ///     (multiarraymodule.c): for operands above 2-D, NumPy does NOT call gemm — it walks
        ///     every (row of a, column-plane of b) pair and calls the dtype's <c>dotfunc</c>, i.e.
        ///     the double-accumulating chunked <c>cblas_?dot</c>. Reproducing that is the only way
        ///     the N-D <c>np.dot</c> matches, since gemm and <c>?dot</c> round differently.
        /// </summary>
        private static bool TryDotND<T, TOps>(NDArray ap1, NDArray ap2, out NDArray result)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            result = null;
            var ops = default(TOps);

            var s1 = ap1.Shape;
            var s2 = ap2.Shape;
            int nd1 = s1.NDim, nd2 = s2.NDim;
            long l = s1.dimensions[nd1 - 1];
            int matchDim = nd2 > 1 ? nd2 - 2 : 0;
            if (s2.dimensions[matchDim] != l)
                return false;

            int nd = nd1 + nd2 - 2;
            var dimensions = new long[Math.Max(nd, 1)];
            int j = 0;
            for (int i = 0; i < nd1 - 1; i++)
                dimensions[j++] = s1.dimensions[i];
            for (int i = 0; i < nd2 - 2; i++)
                dimensions[j++] = s2.dimensions[i];
            if (nd2 > 1)
                dimensions[j++] = s2.dimensions[nd2 - 1];

            long is1 = s1.strides[nd1 - 1];
            long is2 = s2.strides[matchDim];

            var outShape = nd == 0 ? Shape.Scalar : new Shape(SubArray(dimensions, nd));
            var outBuf = new NDArray(InfoOf<T>.NPTypeCode, outShape);
            result = outBuf;

            T* op = (T*)outBuf.Address + outBuf.Shape.offset;
            if (outShape.size == 0)
                return true;

            if (s1.size == 0 && s2.size == 0)
            {
                Zero(op, outShape.size);
                return true;
            }

            T* p1 = (T*)ap1.Address + s1.offset;
            T* p2 = (T*)ap2.Address + s2.offset;

            // PyArray_IterAllButAxis: every coordinate of the operand except the contracted axis,
            // in C order of the remaining axes.
            var it1 = new AxisSkippingIterator(s1, nd1 - 1);
            var it2 = new AxisSkippingIterator(s2, matchDim);

            for (long i = 0; i < it1.Count; i++)
            {
                long off1 = it1.OffsetAt(i);
                for (long k = 0; k < it2.Count; k++, op++)
                    ops.Dot(p1 + off1, is1, p2 + it2.OffsetAt(k), is2, op, l);
            }

            return true;
        }

        /// <summary>
        ///     Walks every coordinate of a shape except one axis, yielding the element offset of each
        ///     — NumPy's <c>PyArray_IterAllButAxis</c>, resolved by index so the two nested walks stay
        ///     independent (the inner one restarts for every outer step).
        /// </summary>
        private readonly struct AxisSkippingIterator
        {
            private readonly long[] _dims;
            private readonly long[] _strides;

            public readonly long Count;

            public AxisSkippingIterator(Shape shape, int skipAxis)
            {
                int nd = shape.NDim;
                int n = Math.Max(nd - 1, 0);
                _dims = new long[n];
                _strides = new long[n];
                long count = 1;
                int j = 0;
                for (int i = 0; i < nd; i++)
                {
                    if (i == skipAxis)
                        continue;

                    _dims[j] = shape.dimensions[i];
                    _strides[j] = shape.strides[i];
                    count *= _dims[j];
                    j++;
                }

                Count = count;
            }

            /// <summary>The element offset of the <paramref name="index"/>-th coordinate (C order).</summary>
            public long OffsetAt(long index)
            {
                long offset = 0;
                for (int i = _dims.Length - 1; i >= 0; i--)
                {
                    long dim = _dims[i];
                    if (dim == 0)
                        return 0;

                    long coord = index % dim;
                    index /= dim;
                    offset += coord * _strides[i];
                }

                return offset;
            }
        }
    }
}
