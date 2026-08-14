using System;
using NumSharp;
using NumSharp.Backends;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Interop.OpenBLAS
{
    public static unsafe partial class OpenBlasEngine
    {
        /// <summary>NumPy's <c>MatrixShape</c> (cblasfuncs.c) — how <c>np.dot</c> classifies an operand.</summary>
        private enum MatrixShape
        {
            Scalar,
            Column,
            Row,
            Matrix
        }

        /// <summary>Port of <c>_select_matrix_shape</c> (cblasfuncs.c).</summary>
        private static MatrixShape SelectMatrixShape(NDArray array)
        {
            switch (array.ndim)
            {
                case 0:
                    return MatrixShape.Scalar;
                case 1:
                    return array.Shape.dimensions[0] > 1 ? MatrixShape.Column : MatrixShape.Scalar;
                case 2:
                    if (array.Shape.dimensions[0] > 1)
                        return array.Shape.dimensions[1] == 1 ? MatrixShape.Column : MatrixShape.Matrix;

                    return array.Shape.dimensions[1] == 1 ? MatrixShape.Scalar : MatrixShape.Row;
            }

            return MatrixShape.Matrix;
        }

        /// <summary>
        ///     Port of <c>_bad_strides</c> (cblasfuncs.c): negative, non-element-aligned or
        ///     broadcast (zero with dim &gt; 1) strides force <c>np.dot</c> to copy the operand up
        ///     front. The itemsize-alignment tests are vacuous in element units.
        /// </summary>
        private static bool BadStrides(NDArray array)
        {
            var shape = array.Shape;
            for (int i = 0; i < shape.NDim; i++)
            {
                long stride = shape.strides[i];
                if (stride < 0)
                    return true;
                if (stride == 0 && shape.dimensions[i] > 1)
                    return true;
            }

            return false;
        }

        /// <summary>
        ///     <c>PyArray_NewCopy(ap, NPY_ANYORDER)</c> — Fortran order only when the source is
        ///     F-contiguous (and not C-contiguous), C order otherwise.
        /// </summary>
        private static NDArray CopyAnyOrder(NDArray array)
            => array.Shape.IsFContiguous && !array.Shape.IsContiguous
                ? np.asfortranarray(array)
                : np.ascontiguousarray(array.Shape.IsContiguous ? array.copy() : array);

        /// <summary><c>PyArray_Copy(ap)</c> — always a fresh C-order copy.</summary>
        private static NDArray CopyCOrder(NDArray array)
            => array.Shape.IsContiguous ? array.copy() : np.ascontiguousarray(array);

        /// <summary>A single memory segment — NumPy's <c>PyArray_ISONESEGMENT</c>.</summary>
        private static bool IsOneSegment(NDArray array)
            => array.Shape.IsContiguous || array.Shape.IsFContiguous;

        /// <summary>
        ///     Port of <c>cblas_matrixproduct</c> (numpy/_core/src/common/cblasfuncs.c) — the
        ///     dispatcher behind <c>np.dot</c> for float/double operands of at most 2 dimensions.
        ///     It is NOT the same dispatcher as <c>np.matmul</c>'s: the two agree bit-for-bit on
        ///     nearly every input, but pick different routes when an operand is not blasable (e.g.
        ///     a strided matrix times a vector — gemv-on-a-copy here, the portable loop there).
        /// </summary>
        /// <returns>False when this input is outside the ported surface; the caller then falls back.</returns>
        internal static bool TryDot2D<T, TOps>(NDArray left, NDArray right, out NDArray result)
            where T : unmanaged, INumberBase<T>
            where TOps : struct, IBlasType<T>
        {
            result = null;
            var ops = default(TOps);
            var typeCode = InfoOf<T>.NPTypeCode;

            NDArray ap1 = left, ap2 = right;
            if (BadStrides(ap1)) ap1 = CopyAnyOrder(ap1);
            if (BadStrides(ap2)) ap2 = CopyAnyOrder(ap2);

            NDArray oap1 = ap1, oap2 = ap2;
            var ap1shape = SelectMatrixShape(ap1);
            var ap2shape = SelectMatrixShape(ap2);

            long l;
            int nd;
            long ap1stride = 0;
            var dimensions = new long[2];

            if (ap1shape == MatrixShape.Scalar || ap2shape == MatrixShape.Scalar)
            {
                // One of ap1 or ap2 is a scalar — make ap2 the scalar.
                if (ap1shape == MatrixShape.Scalar)
                {
                    var t = ap1;
                    ap1 = ap2;
                    ap2 = t;
                    ap1shape = ap2shape;
                    ap2shape = MatrixShape.Scalar;
                }

                if (ap1shape == MatrixShape.Row)
                    ap1stride = ap1.Shape.strides[1];
                else if (ap1.ndim > 0)
                    ap1stride = ap1.Shape.strides[0];

                if (oap1.ndim == 0 || oap2.ndim == 0)
                {
                    var thisArray = oap1.ndim == 0 ? oap2 : oap1;
                    nd = thisArray.ndim;
                    l = 1;
                    for (int j = 0; j < nd; j++)
                    {
                        dimensions[j] = thisArray.Shape.dimensions[j];
                        l *= dimensions[j];
                    }
                }
                else
                {
                    l = oap1.Shape.dimensions[oap1.ndim - 1];
                    if (oap2.Shape.dimensions[0] != l)
                        return false; // misaligned — let NumSharp raise its own error

                    nd = ap1.ndim + ap2.ndim - 2;
                    if (nd == 1)
                    {
                        // Either ap1 or ap2 is 1-D and the other 2-D. Fix it so that
                        // dot(shape=(N,1), shape=(1,)) and dot(shape=(1,), shape=(1,N)) both
                        // return an (N,) array (but use the fast scalar code).
                        dimensions[0] = oap1.ndim == 2 ? oap1.Shape.dimensions[0] : oap2.Shape.dimensions[1];
                        l = dimensions[0];
                    }
                    else if (nd == 2)
                    {
                        dimensions[0] = oap1.Shape.dimensions[0];
                        dimensions[1] = oap2.Shape.dimensions[1];
                        // dot(shape=(1,1), shape=(1,N)) and dot(shape=(N,1), shape=(1,1)) must use
                        // scalar multiplication appropriately.
                        l = ap1shape == MatrixShape.Row ? dimensions[1] : dimensions[0];
                    }

                    // Check if the summation dimension is 0-sized.
                    if (oap1.Shape.dimensions[oap1.ndim - 1] == 0)
                        l = 0;
                }
            }
            else
            {
                // Both ap1 and ap2 are vectors or matrices.
                l = ap1.Shape.dimensions[ap1.ndim - 1];
                if (ap2.Shape.dimensions[0] != l)
                    return false;

                nd = ap1.ndim + ap2.ndim - 2;
                if (nd == 1)
                    dimensions[0] = ap1.ndim == 2 ? ap1.Shape.dimensions[0] : ap2.Shape.dimensions[1];
                else if (nd == 2)
                {
                    dimensions[0] = ap1.Shape.dimensions[0];
                    dimensions[1] = ap2.Shape.dimensions[1];
                }
            }

            var outShape = nd == 0 ? Shape.Scalar : new Shape(SubArray(dimensions, nd));
            var outBuf = new NDArray(typeCode, outShape);
            result = outBuf;

            T* op = (T*)outBuf.Address + outBuf.Shape.offset;
            long numElements = outShape.size;
            Zero(op, numElements);
            if (numElements == 0 || l == 0)
                return true;

            if (ap2shape == MatrixShape.Scalar)
            {
                // Multiplication by a scalar -- Level 1 BLAS. If ap1 is a matrix and not contiguous
                // we cannot blast through it with a single striding factor.
                T* p1 = (T*)ap1.Address + ap1.Shape.offset;
                T* p2 = (T*)ap2.Address + ap2.Shape.offset;
                if (l == 1)
                {
                    *op = *p2 * *p1;
                }
                else if (ap1shape != MatrixShape.Matrix)
                {
                    ops.Axpy(l, *p2, p1, ap1stride, op, 1);
                }
                else
                {
                    int maxind = ap1.Shape.dimensions[0] >= ap1.Shape.dimensions[1] ? 0 : 1;
                    int oind = 1 - maxind;
                    T* ptr = p1;
                    T* optr = op;
                    long len = ap1.Shape.dimensions[maxind];
                    T val = *p2;
                    long a1s = ap1.Shape.strides[maxind];
                    long outs = outBuf.Shape.strides[maxind];
                    for (long i = 0; i < ap1.Shape.dimensions[oind]; i++)
                    {
                        ops.Axpy(len, val, ptr, a1s, optr, outs);
                        ptr += ap1.Shape.strides[oind];
                        optr += outBuf.Shape.strides[oind];
                    }
                }

                return true;
            }

            if (ap2shape == MatrixShape.Column && ap1shape != MatrixShape.Matrix)
            {
                // Dot product between two vectors -- Level 1 BLAS.
                T* p1 = (T*)ap1.Address + ap1.Shape.offset;
                T* p2 = (T*)ap2.Address + ap2.Shape.offset;
                ops.Dot(p1, ap1.Shape.strides[ap1shape == MatrixShape.Row ? 1 : 0],
                    p2, ap2.Shape.strides[0], op, l);
                return true;
            }

            if (ap1shape == MatrixShape.Matrix && ap2shape != MatrixShape.Matrix)
            {
                // Matrix vector multiplication -- Level 2 BLAS. lda must be MAX(M,1).
                if (!IsOneSegment(ap1))
                    ap1 = CopyCOrder(ap1);

                CBlasOrder order;
                long lda;
                if (ap1.Shape.IsContiguous)
                {
                    order = CBlasOrder.RowMajor;
                    lda = ap1.Shape.dimensions[1] > 1 ? ap1.Shape.dimensions[1] : 1;
                }
                else
                {
                    order = CBlasOrder.ColMajor;
                    lda = ap1.Shape.dimensions[0] > 1 ? ap1.Shape.dimensions[0] : 1;
                }

                long ap2s = ap2.Shape.strides[0];
                ops.Gemv(order, CBlasTranspose.NoTrans,
                    ap1.Shape.dimensions[0], ap1.Shape.dimensions[1],
                    (T*)ap1.Address + ap1.Shape.offset, lda,
                    (T*)ap2.Address + ap2.Shape.offset, ap2s, op, 1);
                return true;
            }

            if (ap1shape != MatrixShape.Matrix && ap2shape == MatrixShape.Matrix)
            {
                // Vector matrix multiplication -- Level 2 BLAS.
                if (!IsOneSegment(ap2))
                    ap2 = CopyCOrder(ap2);

                CBlasOrder order;
                long lda;
                if (ap2.Shape.IsContiguous)
                {
                    order = CBlasOrder.RowMajor;
                    lda = ap2.Shape.dimensions[1] > 1 ? ap2.Shape.dimensions[1] : 1;
                }
                else
                {
                    order = CBlasOrder.ColMajor;
                    lda = ap2.Shape.dimensions[0] > 1 ? ap2.Shape.dimensions[0] : 1;
                }

                long ap1s = ap1.Shape.strides[ap1shape == MatrixShape.Row ? 1 : 0];
                ops.Gemv(order, CBlasTranspose.Trans,
                    ap2.Shape.dimensions[0], ap2.Shape.dimensions[1],
                    (T*)ap2.Address + ap2.Shape.offset, lda,
                    (T*)ap1.Address + ap1.Shape.offset, ap1s, op, 1);
                return true;
            }

            {
                // Matrix matrix multiplication -- Level 3 BLAS. L x M multiplied by M x N.
                if (!ap2.Shape.IsContiguous && !ap2.Shape.IsFContiguous)
                    ap2 = CopyCOrder(ap2);
                if (!ap1.Shape.IsContiguous && !ap1.Shape.IsFContiguous)
                    ap1 = CopyCOrder(ap1);

                const CBlasOrder order = CBlasOrder.RowMajor;
                var trans1 = CBlasTranspose.NoTrans;
                var trans2 = CBlasTranspose.NoTrans;
                long bigL = ap1.Shape.dimensions[0];
                long n = ap2.Shape.dimensions[1];
                long m = ap2.Shape.dimensions[0];
                long lda = ap1.Shape.dimensions[1] > 1 ? ap1.Shape.dimensions[1] : 1;
                long ldb = ap2.Shape.dimensions[1] > 1 ? ap2.Shape.dimensions[1] : 1;

                // Avoid temporary copies for arrays in Fortran order.
                if (ap1.Shape.IsFContiguous)
                {
                    trans1 = CBlasTranspose.Trans;
                    lda = ap1.Shape.dimensions[0] > 1 ? ap1.Shape.dimensions[0] : 1;
                }

                if (ap2.Shape.IsFContiguous)
                {
                    trans2 = CBlasTranspose.Trans;
                    ldb = ap2.Shape.dimensions[0] > 1 ? ap2.Shape.dimensions[0] : 1;
                }

                T* p1 = (T*)ap1.Address + ap1.Shape.offset;
                T* p2 = (T*)ap2.Address + ap2.Shape.offset;
                long ldc = outBuf.Shape.dimensions[1] > 1 ? outBuf.Shape.dimensions[1] : 1;

                // Use syrk if we have a case of a matrix times its transpose. Otherwise, use gemm.
                if (p1 == p2 &&
                    ap1.Shape.dimensions[0] == ap2.Shape.dimensions[1] &&
                    ap1.Shape.dimensions[1] == ap2.Shape.dimensions[0] &&
                    ap1.Shape.strides[0] == ap2.Shape.strides[1] &&
                    ap1.Shape.strides[1] == ap2.Shape.strides[0] &&
                    (trans1 == CBlasTranspose.Trans) ^ (trans2 == CBlasTranspose.Trans) &&
                    (trans1 == CBlasTranspose.NoTrans) ^ (trans2 == CBlasTranspose.NoTrans))
                {
                    if (trans1 == CBlasTranspose.NoTrans)
                        Syrk<T, TOps>(order, trans1, n, m, p1, lda, op, ldc);
                    else
                        Syrk<T, TOps>(order, trans1, n, m, p2, ldb, op, ldc);
                }
                else
                {
                    ops.Gemm(order, trans1, trans2, bigL, n, m, p1, lda, p2, ldb, op, ldc);
                }

                return true;
            }
        }

        /// <summary>
        ///     Port of the <c>syrk</c> helper in cblasfuncs.c: computes the upper triangle of
        ///     <c>A·Aᵀ</c> and mirrors it into the lower one.
        /// </summary>
        private static void Syrk<T, TOps>(CBlasOrder order, CBlasTranspose trans, long n, long k,
            T* a, long lda, T* c, long ldc)
            where T : unmanaged
            where TOps : struct, IBlasType<T>
        {
            var ops = default(TOps);
            ops.Syrk(order, CBlasUpLo.Upper, trans, n, k, a, lda, c, ldc);

            for (long i = 0; i < n; i++)
                for (long j = i + 1; j < n; j++)
                    c[j * ldc + i] = c[i * ldc + j];
        }

        private static long[] SubArray(long[] source, int count)
        {
            var dst = new long[count];
            Array.Copy(source, dst, count);
            return dst;
        }
    }
}
