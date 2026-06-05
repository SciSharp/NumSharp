using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

// =============================================================================
// Fused 1-D inner product (numpy.dot of two 1-D arrays, without conjugation).
// =============================================================================
//
// numpy.dot(a, b) for two vectors is sum(a[i] * b[i]). The previous NumSharp
// implementation evaluated `left * right` (a full n-element temporary array)
// and then ReduceAdd'd it — two passes over the data plus an allocation that
// shows up as GC pressure under repeated calls.
//
// This fused path computes the inner product in a single pass with no temp:
//   float / double  → SimdDot SIMD multiply-accumulate (contiguous fast path)
//   ints/Half/Decimal → INumber<T> scalar accumulator (JIT auto-vectorizes the
//                       contiguous loop; preserves NumPy's wrap-in-dtype result)
//   bool            → OR over k of (a[k] AND b[k]) — matches numpy bool dot
//   Complex         → Complex accumulator (no conjugation)
//
// Dtype rules (verified against numpy 2.4.2):
//   - same-type result PRESERVES the input dtype (int32·int32 -> int32, NOT the
//     widened int64 that np.sum would give; float16·float16 -> float16);
//   - integer products wrap in the element dtype before accumulating
//     (int8 [100,100]·[100,100] -> 32), which INumber<T> arithmetic reproduces;
//   - empty -> scalar 0 of the input dtype;
//   - mixed dtype -> NEP50 promotion, handled by the left*right + ReduceAdd
//     fallback (which already promotes correctly).
//
// All kernels are stride-aware (read strides[0] and offset), so sliced, reversed
// (`a[::-1]`) and stepped (`a[::2]`) 1-D views are consumed in place — no copy.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Inner product of two 1-D arrays (numpy.dot vector·vector semantics).
        /// Same-dtype operands take a fused single-pass kernel; mixed dtypes fall
        /// through to the promotion-aware <c>left * right</c> + <c>ReduceAdd</c> path.
        /// </summary>
        private unsafe NDArray DotInner1D(NDArray left, NDArray right)
        {
            long n = left.shape[0];
            if (right.shape[0] != n)
                throw new IncorrectShapeException(
                    $"shapes ({left.shape[0]},) and ({right.shape[0]},) not aligned: " +
                    $"{left.shape[0]} (dim 0) != {right.shape[0]} (dim 0)");

            var tc = left.typecode;

            // Mixed dtype → existing NEP50 promotion path (left*right promotes, then reduce).
            if (tc != right.typecode)
            {
                var product = left * right;
                return ReduceAdd(product, null, false, typeCode: product.GetTypeCode);
            }

            // numpy: empty dot → scalar 0 of the INPUT dtype (not the widened sum dtype).
            if (n == 0)
                return NDArray.Scalar(tc.GetDefaultValue(), tc);

            long sa = left.Shape.strides[0];
            long sb = right.Shape.strides[0];
            bool contig = sa == 1 && sb == 1;

            switch (tc)
            {
                case NPTypeCode.Double:
                {
                    double* a = (double*)left.Address + left.Shape.offset;
                    double* b = (double*)right.Address + right.Shape.offset;
                    double r = contig ? SimdDot.DotDouble(a, b, n) : DotStridedF64(a, sa, b, sb, n);
                    return NDArray.Scalar(r);
                }
                case NPTypeCode.Single:
                {
                    float* a = (float*)left.Address + left.Shape.offset;
                    float* b = (float*)right.Address + right.Shape.offset;
                    float r = contig ? SimdDot.DotFloat(a, b, n) : DotStridedF32(a, sa, b, sb, n);
                    return NDArray.Scalar(r);
                }
                case NPTypeCode.Boolean: return NDArray.Scalar(DotBool(left, right, sa, sb, n));
                case NPTypeCode.Complex: return NDArray.Scalar(DotComplex(left, right, sa, sb, n));
                case NPTypeCode.Byte:    return NDArray.Scalar(DotGeneric<byte>(left, right, sa, sb, n));
                case NPTypeCode.SByte:   return NDArray.Scalar(DotGeneric<sbyte>(left, right, sa, sb, n));
                case NPTypeCode.Int16:   return NDArray.Scalar(DotGeneric<short>(left, right, sa, sb, n));
                case NPTypeCode.UInt16:  return NDArray.Scalar(DotGeneric<ushort>(left, right, sa, sb, n));
                case NPTypeCode.Int32:   return NDArray.Scalar(DotGeneric<int>(left, right, sa, sb, n));
                case NPTypeCode.UInt32:  return NDArray.Scalar(DotGeneric<uint>(left, right, sa, sb, n));
                case NPTypeCode.Int64:   return NDArray.Scalar(DotGeneric<long>(left, right, sa, sb, n));
                case NPTypeCode.UInt64:  return NDArray.Scalar(DotGeneric<ulong>(left, right, sa, sb, n));
                case NPTypeCode.Half:    return NDArray.Scalar(DotGeneric<Half>(left, right, sa, sb, n));
                case NPTypeCode.Decimal: return NDArray.Scalar(DotGeneric<decimal>(left, right, sa, sb, n));
                default:
                    // Char (no INumber<char>) or anything unforeseen → existing path.
                    var product = left * right;
                    return ReduceAdd(product, null, false, typeCode: product.GetTypeCode);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DotStridedF64(double* a, long sa, double* b, long sb, long n)
        {
            double s = 0;
            for (long i = 0; i < n; i++) s += a[i * sa] * b[i * sb];
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotStridedF32(float* a, long sa, float* b, long sb, long n)
        {
            float s = 0;
            for (long i = 0; i < n; i++) s += a[i * sa] * b[i * sb];
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

        /// <summary>Complex inner product without conjugation (matches numpy.dot for complex vectors).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe Complex DotComplex(NDArray left, NDArray right, long sa, long sb, long n)
        {
            Complex* a = (Complex*)left.Address + left.Shape.offset;
            Complex* b = (Complex*)right.Address + right.Shape.offset;
            Complex acc = Complex.Zero;
            for (long i = 0; i < n; i++) acc += a[i * sa] * b[i * sb];
            return acc;
        }
    }
}
