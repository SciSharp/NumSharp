using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

// =============================================================================
// The 1-D inner product (numpy.dot of two 1-D arrays, without conjugation).
// =============================================================================
//
// NumPy's PyArray_MatrixProduct2 converts both operands to their common dtype first. For float32 / float64 / complex128
// it then runs cblas_matrixproduct (cblasfuncs.c): two length-1 vectors are the plain product; an operand with a
// negative stride, or a zero one over more than one element, is copied to a contiguous array (_bad_strides); and the
// dtype's dot function computes the rest - FLOAT_dot / DOUBLE_dot / CDOUBLE_dot over scipy-openblas' sdot / ddot /
// zdotu, NDDot's exact models. A copied operand is read in place in its logical order where the model allows it (the
// result is the copy's), and a mixed pair is cast into scratch with the cast kernels instead of through arrays.
// Every other dtype runs its own dot function:
//   ints / Decimal -> an INumber<T> accumulator in the dtype (NumPy's products wrap in-dtype:
//                     int8 [100,100]·[100,100] -> 32)
//   float16        -> HALF_dot's float32 sum, rounded once (NaN order through NDHalfDot)
//   bool           -> OR over k of (a[k] AND b[k])
// The result keeps the operands' (common) dtype - int32·int32 is int32, not np.sum's int64 - and an empty dot is 0 of
// that dtype.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's np.dot of two 1-D arrays (an inner product, no conjugation): both operands converted to their
        ///     common dtype, then that dtype's 1-D route - float32 / float64 / complex128 through
        ///     <c>cblas_matrixproduct</c>'s rules (<see cref="Dot1DSingle"/>, <see cref="Dot1DDouble"/>,
        ///     <see cref="Dot1DComplex"/>), every other dtype through its dot function.
        /// </summary>
        /// <param name="left">The first vector (NumPy's <c>ap1</c>).</param>
        /// <param name="right">The second vector (<c>ap2</c>).</param>
        /// <returns>A fresh 0-d result of the common dtype.</returns>
        /// <exception cref="IncorrectShapeException">The vectors' lengths differ.</exception>
        /// <exception cref="DTypePromotionError">The operands have no common dtype.</exception>
        private unsafe NDArray DotInner1D(NDArray left, NDArray right)
        {
            long n = left.Shape.dimensions[0];
            if (right.Shape.dimensions[0] != n)
                throw new IncorrectShapeException(
                    $"shapes ({left.Shape.dimensions[0]},) and ({right.Shape.dimensions[0]},) not aligned: " +
                    $"{left.Shape.dimensions[0]} (dim 0) != {right.Shape.dimensions[0]} (dim 0)");

            // PyArray_MatrixProduct2 converts BOTH operands to their common dtype (PyArray_FromAny) before choosing a
            // route, so a mixed pair is that dtype's dot of the converted values; the former route multiplied with
            // promotion and then reduced the products, a pairwise sum where NumPy's dot sums in its own order.
            if (left.typecode != right.typecode)
                return DotInner1DMixed(left, right, n);

            var tc = left.typecode;

            // numpy: empty dot → scalar 0 of the INPUT dtype (not the widened sum dtype).
            if (n == 0)
                return NewScalar(tc.GetDefaultValue(), tc);

            long sa = left.Shape.strides[0];
            long sb = right.Shape.strides[0];

            switch (tc)
            {
                case NPTypeCode.Double:
                    return NewScalar(Dot1DDouble((double*)left.Address + left.Shape.offset, sa,
                        (double*)right.Address + right.Shape.offset, sb, n));
                case NPTypeCode.Single:
                    return NewScalar(Dot1DSingle((float*)left.Address + left.Shape.offset, sa,
                        (float*)right.Address + right.Shape.offset, sb, n));
                case NPTypeCode.Complex:
                    return NewScalar(Dot1DComplex((Complex*)left.Address + left.Shape.offset, sa,
                        (Complex*)right.Address + right.Shape.offset, sb, n));
                default:
                    return OtherDotFunction1D(left, right, sa, sb, n);
            }
        }

        /// <summary>
        ///     The dtype's dot function itself over two 1-D operands of one length (NumPy's <c>PyArray_ArrFuncs.dotfunc</c>),
        ///     without np.dot's <c>cblas_matrixproduct</c> rules: a length-1 pair is summed from +0 like any other, and a
        ///     negative or zero stride runs the dot function's own loop instead of being copied. np.vdot ends here
        ///     (NumPy's <c>array_vdot</c> calls the dot function directly; <see cref="TensorEngine.Vdot"/>).
        /// </summary>
        /// <param name="left">The first vector (<c>ip1</c>), already conjugated by np.vdot when complex.</param>
        /// <param name="right">The second vector (<c>ip2</c>), the same length.</param>
        /// <returns>A fresh 0-d result of the operands' common dtype.</returns>
        /// <exception cref="DTypePromotionError">The operands have no common dtype.</exception>
        /// <remarks>
        ///     <c>CDOUBLE_vdot</c> conjugates inside its sums (<c>cblas_zdotc_sub</c> / its own loop); np.vdot conjugates
        ///     the first operand and runs <c>CDOUBLE_dot</c>, the same sums: negating every imaginary part of x negates
        ///     the two sums that use it exactly (round-to-nearest is symmetric), so <c>re = Σxr·yr + Σxi·yi</c> and
        ///     <c>im = Σxr·yi − Σxi·yr</c> come out as zdotc's.
        /// </remarks>
        internal override unsafe NDArray DotFunction1D(NDArray left, NDArray right)
        {
            long n = left.Shape.dimensions[0];
            if (right.Shape.dimensions[0] != n)
                throw new IncorrectShapeException(
                    $"shapes ({left.Shape.dimensions[0]},) and ({right.Shape.dimensions[0]},) not aligned: " +
                    $"{left.Shape.dimensions[0]} (dim 0) != {right.Shape.dimensions[0]} (dim 0)");

            // array_vdot converts both operands to their common dtype first (PyArray_FromAny), as np.dot does.
            if (left.typecode != right.typecode)
                return DotFunction1DMixed(left, right);

            var tc = left.typecode;
            if (n == 0)
                return NewScalar(tc.GetDefaultValue(), tc);

            long sa = left.Shape.strides[0];
            long sb = right.Shape.strides[0];
            switch (tc)
            {
                case NPTypeCode.Double:
                    return NewScalar(NDDot.DoubleDot((double*)left.Address + left.Shape.offset, sa,
                        (double*)right.Address + right.Shape.offset, sb, n));
                case NPTypeCode.Single:
                    return NewScalar(NDDot.FloatDot((float*)left.Address + left.Shape.offset, sa,
                        (float*)right.Address + right.Shape.offset, sb, n));
                case NPTypeCode.Complex:
                    return NewScalar(NDDot.ComplexDot((Complex*)left.Address + left.Shape.offset, sa,
                        (Complex*)right.Address + right.Shape.offset, sb, n));
                default:
                    return OtherDotFunction1D(left, right, sa, sb, n);
            }
        }

        /// <summary>
        ///     <see cref="DotFunction1D"/> for two operands of different dtypes: both converted to their common dtype through
        ///     astype, then the dot function of that dtype.
        /// </summary>
        /// <param name="left">The first vector.</param>
        /// <param name="right">The second vector (the same length).</param>
        /// <returns>A fresh 0-d result of the common dtype.</returns>
        /// <exception cref="DTypePromotionError">The operands have no common dtype.</exception>
        /// <remarks>A method of its own so that <see cref="DotFunction1D"/> carries no exception-handling region.</remarks>
        private NDArray DotFunction1DMixed(NDArray left, NDArray right)
        {
            var common = DTypePromotion.PromoteTypes(left.dtype, right.dtype);
            NDArray ap1 = left.astype(common, false), ap2 = right.astype(common, false);
            try
            {
                return DotFunction1D(ap1, ap2);
            }
            finally
            {
                // astype(copy: false) is the operand itself when it already has the common dtype; a converted copy is
                // this call's intermediate, released here because the engine may run outside any NDScope.
                if (!ReferenceEquals(ap1, left))
                    ap1.Dispose();
                if (!ReferenceEquals(ap2, right))
                    ap2.Dispose();
            }
        }

        /// <summary>
        ///     The dot functions of the dtypes cblas does not take, shared by np.dot's 1-D route and
        ///     <see cref="DotFunction1D"/> (for them the two are one): the integers and decimal sum in the dtype
        ///     (<see cref="DotGeneric{T}"/>), float16 is HALF_dot (<see cref="DotHalf"/>), bool is BOOL_dot
        ///     (<see cref="DotBool"/>), and char takes the promoted product and its sum.
        /// </summary>
        /// <param name="left">The first vector.</param>
        /// <param name="right">The second vector (the same dtype and length).</param>
        /// <param name="sa">The first vector's element stride.</param>
        /// <param name="sb">The second vector's element stride.</param>
        /// <param name="n">The length (at least 1).</param>
        /// <returns>A fresh 0-d result of the dtype.</returns>
        private NDArray OtherDotFunction1D(NDArray left, NDArray right, long sa, long sb, long n)
        {
            switch (left.typecode)
            {
                case NPTypeCode.Boolean: return NewScalar(DotBool(left, right, sa, sb, n));
                case NPTypeCode.Byte:    return NewScalar(DotGeneric<byte>(left, right, sa, sb, n));
                case NPTypeCode.SByte:   return NewScalar(DotGeneric<sbyte>(left, right, sa, sb, n));
                case NPTypeCode.Int16:   return NewScalar(DotGeneric<short>(left, right, sa, sb, n));
                case NPTypeCode.UInt16:  return NewScalar(DotGeneric<ushort>(left, right, sa, sb, n));
                case NPTypeCode.Int32:   return NewScalar(DotGeneric<int>(left, right, sa, sb, n));
                case NPTypeCode.UInt32:  return NewScalar(DotGeneric<uint>(left, right, sa, sb, n));
                case NPTypeCode.Int64:   return NewScalar(DotGeneric<long>(left, right, sa, sb, n));
                case NPTypeCode.UInt64:  return NewScalar(DotGeneric<ulong>(left, right, sa, sb, n));
                case NPTypeCode.Half:    return NewScalar(DotHalf(left, right, sa, sb, n));
                case NPTypeCode.Decimal: return NewScalar(DotGeneric<decimal>(left, right, sa, sb, n));
                default:
                    // Char (no INumber<char>) or anything unforeseen → existing path.
                    var product = left * right;
                    var fallback = ReduceAdd(product, null, false, dtype: product.GetTypeCode.AsType());
                    product.Dispose();
                    return fallback;
            }
        }

        /// <summary>
        ///     <see cref="DotInner1D"/> for two operands of different dtypes: both converted to their common dtype, as
        ///     PyArray_MatrixProduct2 converts them, then that dtype's route. A BLAS dtype casts into scratch
        ///     (<see cref="ConvertedDot1D{T}"/>); any other converts through astype and runs <see cref="DotInner1D"/>.
        /// </summary>
        /// <param name="left">The first vector.</param>
        /// <param name="right">The second vector (the same length).</param>
        /// <param name="n">The length.</param>
        /// <returns>A fresh 0-d result of the common dtype.</returns>
        /// <exception cref="DTypePromotionError">The operands have no common dtype.</exception>
        /// <remarks>A method of its own so that <see cref="DotInner1D"/>, which every 1-D dot runs, carries no
        ///     exception-handling region: one in it slowed every short dot.</remarks>
        private unsafe NDArray DotInner1DMixed(NDArray left, NDArray right, long n)
        {
            var common = DTypePromotion.PromoteTypes(left.dtype, right.dtype);
            switch (common.GetTypeCode())
            {
                // The BLAS dtypes: the operands cast into scratch, no intermediate arrays (n == 0 has nothing to cast).
                case NPTypeCode.Double when n > 0 && CastableTo(left, right, NPTypeCode.Double):
                    return NewScalar(ConvertedDot1D<double>(left, right, n, NPTypeCode.Double, &Dot1DDouble));
                case NPTypeCode.Single when n > 0 && CastableTo(left, right, NPTypeCode.Single):
                    return NewScalar(ConvertedDot1D<float>(left, right, n, NPTypeCode.Single, &Dot1DSingle));
                case NPTypeCode.Complex when n > 0 && CastableTo(left, right, NPTypeCode.Complex):
                    return NewScalar(ConvertedDot1D<Complex>(left, right, n, NPTypeCode.Complex, &Dot1DComplex));
            }

            NDArray ap1 = left.astype(common, false), ap2 = right.astype(common, false);
            try
            {
                return DotInner1D(ap1, ap2);
            }
            finally
            {
                // astype(copy: false) is the operand itself when it already has the common dtype; a converted copy is
                // this call's intermediate, released here because the engine may run outside any NDScope.
                if (!ReferenceEquals(ap1, left))
                    ap1.Dispose();
                if (!ReferenceEquals(ap2, right))
                    ap2.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's 1-D · 1-D float64 route (<c>cblas_matrixproduct</c>, cblasfuncs.c): two length-1 vectors are both
        ///     <c>_scalar</c>, and their dot is the plain product; otherwise an operand with a negative stride, or a zero
        ///     stride over more than one element (<c>_bad_strides</c>), is copied to a contiguous buffer first, and
        ///     <c>DOUBLE_dot</c> runs over the strides that leaves (<see cref="NDDot.DoubleDot"/>).
        /// </summary>
        /// <param name="a">The first vector's element 0 (NumPy's <c>ap1</c>).</param>
        /// <param name="sa">Its element stride (any sign).</param>
        /// <param name="b">The second vector's element 0 (<c>ap2</c>).</param>
        /// <param name="sb">Its element stride (any sign).</param>
        /// <param name="n">The length (at least 1).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        /// <remarks>
        ///     The plain product keeps a -0.0 (<c>[-0.0]·[1.0]</c> is -0.0), where a dot function's sum from +0 would not.
        ///     A reversed or broadcast operand therefore takes the unit-stride kernel over its values in order, as NumPy's
        ///     copy does. With np.multithreading on, two unit strides and a long enough vector, the product is NumSharp's
        ///     opt-in chunked sum (<see cref="DotParallelF64"/>), whose partial sums are not NumPy's order.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double Dot1DDouble(double* a, long sa, double* b, long sb, long n)
        {
            if (n == 1)
                return a[0] * b[0];   // the l == 1 branch: `*ap2 * *ap1` after cblas_matrixproduct's swap, ap1 * ap2

            if (sa > 0 && sb > 0)
            {
                int p = sa == 1 && sb == 1 ? MultiThread.DegreeOfParallelism(n) : 1;
                return p > 1 ? DotParallelF64(a, b, n, p)
                    : sa == 1 && sb == 1 ? NDDot.DoubleDotUnit(a, 1, b, 1, n) : NDDot.DoubleDotStrided(a, sa, b, sb, n);
            }

            // _bad_strides: NumPy copies such an operand (PyArray_NewCopy) and its dot function then sees a unit stride
            // there. The arithmetic depends only on the values in order, so the copy is skipped where the run can be
            // read in place: cblas' strided loop (the other operand stepped) reads any stride, the unit kernels a
            // reversed or broadcast run.
            long ea = sa > 0 ? sa : 1, eb = sb > 0 ? sb : 1;
            if (ea != 1 || eb != 1)
                return NDDot.DoubleDotStrided(a, sa, b, sb, n);
            if (NDDot.UnitReadable(sa) && NDDot.UnitReadable(sb))
                return NDDot.DoubleDotUnitInOrder(a, sa, b, sb, n);
            return Dot1DDoubleCopied(a, sa, b, sb, n);
        }

        /// <summary>
        ///     The rest of <see cref="Dot1DDouble"/>'s unit-stride class: an operand whose stride is negative but not -1 (a
        ///     <c>[::-2]</c> run) is copied in order to pooled scratch, as NumPy copies it, and the unit model reads the copy.
        /// </summary>
        /// <param name="a">The first vector's element 0.</param>
        /// <param name="sa">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="b">The second vector's element 0.</param>
        /// <param name="sb">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="n">The length (at least 2).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        /// <remarks>A method of its own so that the routes above carry no exception-handling region.</remarks>
        private static unsafe double Dot1DDoubleCopied(double* a, long sa, double* b, long sb, long n)
        {
            // Pooled scratch (the allocation chokepoint), returned with the size it was taken with.
            long bytes = n * sizeof(double);
            double* ca = null, cb = null;
            try
            {
                if (sa != 1)
                {
                    ca = (double*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        ca[i] = a[i * sa];
                }
                if (sb != 1)
                {
                    cb = (double*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        cb[i] = b[i * sb];
                }
                return NDDot.DoubleDotUnitInOrder(ca != null ? ca : a, 1, cb != null ? cb : b, 1, n);
            }
            finally
            {
                if (ca != null)
                    SizeBucketedBufferPool.Return((IntPtr)ca, bytes);
                if (cb != null)
                    SizeBucketedBufferPool.Return((IntPtr)cb, bytes);
            }
        }

        /// <summary>
        ///     NumPy's 1-D · 1-D float32 route (<c>cblas_matrixproduct</c>): as <see cref="Dot1DDouble"/>, with
        ///     <c>FLOAT_dot</c> (<see cref="NDDot.FloatDot"/>) and a float32 plain product for two length-1 vectors.
        /// </summary>
        /// <param name="a">The first vector's element 0 (NumPy's <c>ap1</c>).</param>
        /// <param name="sa">Its element stride (any sign).</param>
        /// <param name="b">The second vector's element 0 (<c>ap2</c>).</param>
        /// <param name="sb">Its element stride (any sign).</param>
        /// <param name="n">The length (at least 1).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        /// <remarks>With np.multithreading on, the chunked opt-in is <see cref="DotParallelF32"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float Dot1DSingle(float* a, long sa, float* b, long sb, long n)
        {
            if (n == 1)
                return a[0] * b[0];   // the l == 1 branch: a float32 product

            if (sa > 0 && sb > 0)
            {
                int p = sa == 1 && sb == 1 ? MultiThread.DegreeOfParallelism(n) : 1;
                return p > 1 ? DotParallelF32(a, b, n, p)
                    : sa == 1 && sb == 1 ? NDDot.FloatDotUnit(a, 1, b, 1, n) : NDDot.FloatDotStrided(a, sa, b, sb, n);
            }

            // _bad_strides: NumPy copies such an operand (PyArray_NewCopy) and its dot function then sees a unit stride
            // there. The arithmetic depends only on the values in order, so the copy is skipped where the run can be
            // read in place: cblas' strided loop (the other operand stepped) reads any stride, the unit kernels a
            // reversed or broadcast run.
            long ea = sa > 0 ? sa : 1, eb = sb > 0 ? sb : 1;
            if (ea != 1 || eb != 1)
                return NDDot.FloatDotStrided(a, sa, b, sb, n);
            if (NDDot.UnitReadable(sa) && NDDot.UnitReadable(sb))
                return NDDot.FloatDotUnitInOrder(a, sa, b, sb, n);
            return Dot1DSingleCopied(a, sa, b, sb, n);
        }

        /// <summary>
        ///     The rest of <see cref="Dot1DSingle"/>'s unit-stride class: an operand whose stride is negative but not -1 (a
        ///     <c>[::-2]</c> run) is copied in order to pooled scratch, as NumPy copies it, and the unit model reads the copy.
        /// </summary>
        /// <param name="a">The first vector's element 0.</param>
        /// <param name="sa">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="b">The second vector's element 0.</param>
        /// <param name="sb">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="n">The length (at least 2).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        /// <remarks>A method of its own so that the routes above carry no exception-handling region.</remarks>
        private static unsafe float Dot1DSingleCopied(float* a, long sa, float* b, long sb, long n)
        {
            // Pooled scratch (the allocation chokepoint), returned with the size it was taken with.
            long bytes = n * sizeof(float);
            float* ca = null, cb = null;
            try
            {
                if (sa != 1)
                {
                    ca = (float*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        ca[i] = a[i * sa];
                }
                if (sb != 1)
                {
                    cb = (float*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        cb[i] = b[i * sb];
                }
                return NDDot.FloatDotUnitInOrder(ca != null ? ca : a, 1, cb != null ? cb : b, 1, n);
            }
            finally
            {
                if (ca != null)
                    SizeBucketedBufferPool.Return((IntPtr)ca, bytes);
                if (cb != null)
                    SizeBucketedBufferPool.Return((IntPtr)cb, bytes);
            }
        }

        /// <summary>
        ///     NumPy's 1-D · 1-D complex128 route (<c>cblas_matrixproduct</c>): as <see cref="Dot1DDouble"/>, with
        ///     <c>CDOUBLE_dot</c> (<see cref="NDDot.ComplexDot"/>) and, for two length-1 vectors, the plain complex product
        ///     <c>(ar·br − ai·bi) + (ar·bi + ai·br)i</c>.
        /// </summary>
        /// <param name="a">The first vector's element 0 (NumPy's <c>ap1</c>).</param>
        /// <param name="sa">Its element stride (any sign).</param>
        /// <param name="b">The second vector's element 0 (<c>ap2</c>, not conjugated).</param>
        /// <param name="sb">Its element stride (any sign).</param>
        /// <param name="n">The length (at least 1).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe Complex Dot1DComplex(Complex* a, long sa, Complex* b, long sb, long n)
        {
            if (n == 1)
            {
                // The l == 1 branch writes the parts itself (no C99 complex and no zeroed sum): -0 parts survive.
                double ar = ((double*)a)[0], ai = ((double*)a)[1], br = ((double*)b)[0], bi = ((double*)b)[1];
                return new Complex(ar * br - ai * bi, ar * bi + ai * br);
            }

            if (sa > 0 && sb > 0)
                return sa == 1 && sb == 1 ? NDDot.ComplexDotUnit(a, 1, b, 1, n) : NDDot.ComplexDotStrided(a, sa, b, sb, n);

            // _bad_strides, read in place where possible (see Dot1DDouble).
            long ea = sa > 0 ? sa : 1, eb = sb > 0 ? sb : 1;
            if (ea != 1 || eb != 1)
                return NDDot.ComplexDotStrided(a, sa, b, sb, n);
            if (NDDot.UnitReadable(sa) && NDDot.UnitReadable(sb))
                return NDDot.ComplexDotUnitInOrder(a, sa, b, sb, n);
            return Dot1DComplexCopied(a, sa, b, sb, n);
        }

        /// <summary>
        ///     The rest of <see cref="Dot1DComplex"/>'s unit-stride class: an operand whose stride is negative but not -1 is
        ///     copied in order to pooled scratch, as NumPy copies it, and the unit model reads the copy.
        /// </summary>
        /// <param name="a">The first vector's element 0.</param>
        /// <param name="sa">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="b">The second vector's element 0 (not conjugated).</param>
        /// <param name="sb">Its element stride (1, or negative / zero when copied).</param>
        /// <param name="n">The length (at least 2).</param>
        /// <returns>The dot product NumPy's 1-D route writes.</returns>
        /// <remarks>A method of its own so that the route above carries no exception-handling region.</remarks>
        private static unsafe Complex Dot1DComplexCopied(Complex* a, long sa, Complex* b, long sb, long n)
        {
            long bytes = n * sizeof(Complex);
            Complex* ca = null, cb = null;
            try
            {
                if (sa != 1)
                {
                    ca = (Complex*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        ca[i] = a[i * sa];
                }
                if (sb != 1)
                {
                    cb = (Complex*)SizeBucketedBufferPool.Take(bytes);
                    for (long i = 0; i < n; i++)
                        cb[i] = b[i * sb];
                }
                return NDDot.ComplexDotUnitInOrder(ca != null ? ca : a, 1, cb != null ? cb : b, 1, n);
            }
            finally
            {
                if (ca != null)
                    SizeBucketedBufferPool.Return((IntPtr)ca, bytes);
                if (cb != null)
                    SizeBucketedBufferPool.Return((IntPtr)cb, bytes);
            }
        }

        /// <summary>
        ///     Whether both 1-D operands can be cast to <paramref name="to"/> by the cast kernels (the conversions astype
        ///     runs, NumPy's casts): the condition of the scratch route of a mixed pair.
        /// </summary>
        /// <param name="left">The first vector.</param>
        /// <param name="right">The second vector.</param>
        /// <param name="to">The common dtype (float32, float64 or complex128).</param>
        /// <returns>True when each operand already has <paramref name="to"/> or has a cast kernel to it.</returns>
        private static bool CastableTo(NDArray left, NDArray right, NPTypeCode to)
        {
            static bool One(NDArray x, NPTypeCode to) =>
                x.typecode == to
                || (x.Shape.strides[0] == 1 && DirectILKernelGenerator.TryGetCastKernel(x.typecode, to) != null)
                || DirectILKernelGenerator.TryGetStridedCastKernel(x.typecode, to) != null;
            return One(left, to) && One(right, to);
        }

        /// <summary>
        ///     A mixed pair through a BLAS dtype's 1-D route: PyArray_MatrixProduct2 converts both operands to the common
        ///     dtype (contiguous copies, NumPy's casts) and runs that dtype's route; here an operand that needs it is cast
        ///     into scratch with the cast kernels (on the stack up to 2 KB, pooled above), and one that already has the
        ///     dtype is read where it is.
        /// </summary>
        /// <typeparam name="T">The common dtype's element type.</typeparam>
        /// <param name="left">The first vector.</param>
        /// <param name="right">The second vector.</param>
        /// <param name="n">The length (at least 1).</param>
        /// <param name="to">The common dtype.</param>
        /// <param name="dot1d">The dtype's 1-D route (<see cref="Dot1DDouble"/>, <see cref="Dot1DSingle"/>,
        ///     <see cref="Dot1DComplex"/>).</param>
        /// <returns>The dot product NumPy writes.</returns>
        /// <remarks>Why: the former route built a promoted product array and reduced it, and converting through astype
        ///     builds an array per operand; for a short vector those allocations are most of the call.
        ///     <see cref="CastableTo"/> guards it.</remarks>
        [SkipLocalsInit]
        private static unsafe T ConvertedDot1D<T>(NDArray left, NDArray right, long n, NPTypeCode to,
            delegate*<T*, long, T*, long, long, T> dot1d) where T : unmanaged
        {
            const int StackBytes = 2048;
            long bytes = n * sizeof(T);
            byte* stackA = stackalloc byte[StackBytes];
            byte* stackB = stackalloc byte[StackBytes];
            byte* poolA = null, poolB = null;
            try
            {
                T* pa;
                long sa;
                if (left.typecode == to)
                {
                    pa = (T*)left.Address + left.Shape.offset;
                    sa = left.Shape.strides[0];
                }
                else
                {
                    byte* dst = bytes <= StackBytes ? stackA : (poolA = (byte*)SizeBucketedBufferPool.Take(bytes));
                    CastRun(left, to, dst, n);
                    pa = (T*)dst;
                    sa = 1;
                }

                T* pb;
                long sb;
                if (right.typecode == to)
                {
                    pb = (T*)right.Address + right.Shape.offset;
                    sb = right.Shape.strides[0];
                }
                else
                {
                    byte* dst = bytes <= StackBytes ? stackB : (poolB = (byte*)SizeBucketedBufferPool.Take(bytes));
                    CastRun(right, to, dst, n);
                    pb = (T*)dst;
                    sb = 1;
                }

                return dot1d(pa, sa, pb, sb, n);
            }
            finally
            {
                if (poolA != null)
                    SizeBucketedBufferPool.Return((IntPtr)poolA, bytes);
                if (poolB != null)
                    SizeBucketedBufferPool.Return((IntPtr)poolB, bytes);
            }
        }

        /// <summary>
        ///     Casts a 1-D operand's values, in order, to <paramref name="to"/> into a contiguous buffer with the cast kernels
        ///     (the contiguous one for a unit stride, the strided one for any other stride, a reversed one included).
        /// </summary>
        /// <param name="src">The operand (its dtype has a cast kernel to <paramref name="to"/>: <see cref="CastableTo"/>).</param>
        /// <param name="to">The target dtype.</param>
        /// <param name="dst">The buffer: room for <paramref name="n"/> elements of <paramref name="to"/>.</param>
        /// <param name="n">The operand's length.</param>
        /// <exception cref="InvalidOperationException">No cast kernel (excluded by <see cref="CastableTo"/>).</exception>
        private static unsafe void CastRun(NDArray src, NPTypeCode to, byte* dst, long n)
        {
            byte* p = (byte*)src.Address + src.Shape.offset * src.dtypesize;
            long stride = src.Shape.strides[0];
            if (stride == 1)
            {
                var contiguous = DirectILKernelGenerator.TryGetCastKernel(src.typecode, to);
                if (contiguous != null)
                {
                    contiguous(p, dst, n);
                    return;
                }
            }

            var strided = DirectILKernelGenerator.TryGetStridedCastKernel(src.typecode, to)
                          ?? throw new InvalidOperationException($"no cast kernel from {src.typecode} to {to}");
            long srcStride = stride, dstStride = 1, shape = n;
            strided(p, dst, &srcStride, &dstStride, &shape, 1);
        }

        /// <summary>
        ///     np.multithreading's opt-in for a long contiguous float64 dot (not NumPy's order): [0, n) split into
        ///     <paramref name="p"/> contiguous chunks, each summed on its own thread by <see cref="SimdDot.DotDouble"/>, the
        ///     partial sums added in chunk order (deterministic for a given p). Partials are padded to a cache line to
        ///     avoid false sharing.
        /// </summary>
        /// <param name="a">The first vector (unit stride).</param>
        /// <param name="b">The second vector (unit stride).</param>
        /// <param name="n">The length.</param>
        /// <param name="p">The number of chunks (MultiThread.DegreeOfParallelism, above 1).</param>
        /// <returns>The chunked dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DotParallelF64(double* a, double* b, long n, int p)
        {
            nint na = (nint)a, nb = (nint)b;
            var partials = new double[p * 8]; // 8 doubles = 64B per slot
            long chunk = (n + p - 1) / p;
            Parallel.For(0, p, t =>
            {
                long start = (long)t * chunk;
                long len = Math.Min(chunk, n - start);
                if (len > 0)
                    unsafe { partials[t * 8] = SimdDot.DotDouble((double*)na + start, (double*)nb + start, len); }
            });
            double s = 0;
            for (int t = 0; t < p; t++) s += partials[t * 8];
            return s;
        }

        /// <summary>
        ///     np.multithreading's opt-in for a long contiguous float32 dot: as <see cref="DotParallelF64"/>, the chunks
        ///     summed by <see cref="SimdDot.DotFloat"/> and the partials added in float32.
        /// </summary>
        /// <param name="a">The first vector (unit stride).</param>
        /// <param name="b">The second vector (unit stride).</param>
        /// <param name="n">The length.</param>
        /// <param name="p">The number of chunks (above 1).</param>
        /// <returns>The chunked dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotParallelF32(float* a, float* b, long n, int p)
        {
            nint na = (nint)a, nb = (nint)b;
            var partials = new float[p * 16]; // 16 floats = 64B per slot
            long chunk = (n + p - 1) / p;
            Parallel.For(0, p, t =>
            {
                long start = (long)t * chunk;
                long len = Math.Min(chunk, n - start);
                if (len > 0)
                    unsafe { partials[t * 16] = SimdDot.DotFloat((float*)na + start, (float*)nb + start, len); }
            });
            float s = 0;
            for (int t = 0; t < p; t++) s += partials[t * 16];
            return s;
        }

        /// <summary>
        /// Same-type scalar fused dot for the INumber&lt;T&gt; dtypes (ints, Half, Decimal).
        /// The contiguous branch is a tight <c>acc += a[i] * b[i]</c> the JIT can
        /// auto-vectorize for primitive T; the strided branch indexes by element stride.
        /// Arithmetic is in T, so integer products wrap in-dtype like NumPy.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe T DotGeneric<T>(NDArray left, NDArray right, long sa, long sb, long n)
            where T : unmanaged, INumber<T>
        {
            T* a = (T*)left.Address + left.Shape.offset;
            T* b = (T*)right.Address + right.Shape.offset;
            T acc = T.Zero;
            if (sa == 1 && sb == 1)
                for (long i = 0; i < n; i++) acc += a[i] * b[i];
            else
                for (long i = 0; i < n; i++) acc += a[i * sa] * b[i * sb];
            return acc;
        }

        /// <summary>
        /// Half inner product with a FLOAT32 accumulator. NOT <see cref="DotGeneric{T}"/>: NumPy's
        /// HALF_dot declares <c>float tmp = 0.</c>, accumulates every product there, and narrows
        /// once with <c>npy_float_to_half(tmp)</c>. Accumulating in Half saturates instead —
        /// <c>dot(ones(4096), ones(4096))</c> stalls at 2048, since half cannot represent 2049.
        /// A NaN result carries NumPy's NaN: it is recomputed by <see cref="NDHalfDot.Sum"/>, which
        /// encodes HALF_dot's operand order (the fast loop's order is RyuJIT's).
        /// </summary>
        /// <param name="left">The first vector (NumPy's <c>ip1</c>).</param>
        /// <param name="right">The second vector (<c>ip2</c>).</param>
        /// <param name="sa">The first vector's element stride.</param>
        /// <param name="sb">The second vector's element stride.</param>
        /// <param name="n">The number of terms.</param>
        /// <returns>The dot product, narrowed to Half once.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe Half DotHalf(NDArray left, NDArray right, long sa, long sb, long n)
        {
            Half* a = (Half*)left.Address + left.Shape.offset;
            Half* b = (Half*)right.Address + right.Shape.offset;
            float acc = 0f;
            if (sa == 1 && sb == 1)
                for (long i = 0; i < n; i++) acc += (float)a[i] * (float)b[i];
            else
                for (long i = 0; i < n; i++) acc += (float)a[i * sa] * (float)b[i * sb];
            // Only a NaN depends on the operand order (a non-NaN float32 multiply and add are commutative).
            if (float.IsNaN(acc))
                acc = NDHalfDot.Sum(a, sa, b, sb, n);
            return (Half)acc;
        }

        /// <summary>numpy bool dot: <c>OR</c> over k of <c>(a[k] AND b[k])</c>; short-circuits on first hit.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool DotBool(NDArray left, NDArray right, long sa, long sb, long n)
        {
            bool* a = (bool*)left.Address + left.Shape.offset;
            bool* b = (bool*)right.Address + right.Shape.offset;
            for (long i = 0; i < n; i++)
                if (a[i * sa] && b[i * sb]) return true;
            return false;
        }

    }
}
