using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Cast.FromBool.cs - Boolean-source contig cast kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - TryGetBoolSourceKernel — normalized SIMD widen/convert for bool -> {int, float}
//
// A bool's numeric value is its TRUTH VALUE (0 or 1), never its raw storage byte:
// np.frombuffer is a zero-copy view, so a bool buffer can hold 0x80, and NumPy's
// bool casts normalize (probed 2.4.2: frombuffer([0x80]).astype(int8) == 1). That
// is why TryGetCastKernel bails every bool source away from the generic
// reinterpreting SIMD kernels — which left bool -> float64 on the scalar
// ConvertValue path at 0.37x NumPy (33.7 ms vs 12.6 ms at 10M). These kernels
// normalize FIRST — one unsigned-byte Min against 1 (min(v,1): 0 -> 0, nonzero
// -> 1, NumPy's byte_to_true) — then widen/convert 0/1 lanes, which is exact in
// every destination dtype. Scalar tails use the same `!= 0` test, so body and
// tail agree on every byte value.
//
// Plain C# V256 helpers (the CountTrueSimdHelper / ArgMaxBoolHelper pattern) —
// the CastKernel delegate binds them directly, no IL emission needed. Hosts
// without V256 return null and keep the correct scalar path.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Boolean-Source Cast Kernels

        /// <summary>
        /// Contig cast kernel for a Boolean SOURCE: normalized widen/convert to the integer and
        /// float dtypes. Returns null for other destinations (Half/Complex/Decimal keep the
        /// scalar normalize path) and on hosts without V256.
        /// </summary>
        internal static CastKernel TryGetBoolSourceKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType != NPTypeCode.Boolean || !Vector256.IsHardwareAccelerated)
                return null;

            unsafe
            {
                return dstType switch
                {
                    NPTypeCode.Byte or NPTypeCode.SByte => BoolCast1B,
                    NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char => BoolCast2B,
                    NPTypeCode.Int32 or NPTypeCode.UInt32 => BoolCast4B,
                    NPTypeCode.Int64 or NPTypeCode.UInt64 => BoolCast8B,
                    NPTypeCode.Single => BoolCastF32,
                    NPTypeCode.Double => BoolCastF64,
                    _ => (CastKernel)null
                };
            }
        }

        /// <summary>bool -> int8/uint8: the normalize itself (min(v,1)) stored as-is.</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCast1B(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            byte* d = (byte*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            for (long n = count - 32; i <= n; i += 32)
                Vector256.Min(Vector256.Load(s + i), ones).Store(d + i);
            for (; i < count; i++)
                d[i] = (byte)(s[i] != 0 ? 1 : 0);
        }

        /// <summary>bool -> int16/uint16/char: normalize then one Widen level (element-order API).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCast2B(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            ushort* d = (ushort*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            for (long n = count - 32; i <= n; i += 32)
            {
                var (lo, hi) = Vector256.Widen(Vector256.Min(Vector256.Load(s + i), ones));
                lo.Store(d + i);
                hi.Store(d + i + 16);
            }
            for (; i < count; i++)
                d[i] = (ushort)(s[i] != 0 ? 1 : 0);
        }

        /// <summary>bool -> int32/uint32: normalize then two Widen levels.</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCast4B(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            uint* d = (uint*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            for (long n = count - 32; i <= n; i += 32)
            {
                var (lo16, hi16) = Vector256.Widen(Vector256.Min(Vector256.Load(s + i), ones));
                var (w0, w1) = Vector256.Widen(lo16);
                var (w2, w3) = Vector256.Widen(hi16);
                w0.Store(d + i);
                w1.Store(d + i + 8);
                w2.Store(d + i + 16);
                w3.Store(d + i + 24);
            }
            for (; i < count; i++)
                d[i] = (uint)(s[i] != 0 ? 1 : 0);
        }

        /// <summary>bool -> int64/uint64: normalize then three Widen levels.</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCast8B(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            ulong* d = (ulong*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            for (long n = count - 32; i <= n; i += 32)
            {
                var (lo16, hi16) = Vector256.Widen(Vector256.Min(Vector256.Load(s + i), ones));
                var (u0, u1) = Vector256.Widen(lo16);
                var (u2, u3) = Vector256.Widen(hi16);
                var (q0, q1) = Vector256.Widen(u0);
                var (q2, q3) = Vector256.Widen(u1);
                var (q4, q5) = Vector256.Widen(u2);
                var (q6, q7) = Vector256.Widen(u3);
                q0.Store(d + i);
                q1.Store(d + i + 4);
                q2.Store(d + i + 8);
                q3.Store(d + i + 12);
                q4.Store(d + i + 16);
                q5.Store(d + i + 20);
                q6.Store(d + i + 24);
                q7.Store(d + i + 28);
            }
            for (; i < count; i++)
                d[i] = (ulong)(s[i] != 0 ? 1 : 0);
        }

        /// <summary>bool -> float32: widen to int32 lanes then vcvtdq2ps (0/1 are exact).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCastF32(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            float* d = (float*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            for (long n = count - 32; i <= n; i += 32)
            {
                var (lo16, hi16) = Vector256.Widen(Vector256.Min(Vector256.Load(s + i), ones));
                var (w0, w1) = Vector256.Widen(lo16);
                var (w2, w3) = Vector256.Widen(hi16);
                Vector256.ConvertToSingle(w0.AsInt32()).Store(d + i);
                Vector256.ConvertToSingle(w1.AsInt32()).Store(d + i + 8);
                Vector256.ConvertToSingle(w2.AsInt32()).Store(d + i + 16);
                Vector256.ConvertToSingle(w3.AsInt32()).Store(d + i + 24);
            }
            for (; i < count; i++)
                d[i] = s[i] != 0 ? 1f : 0f;
        }

        /// <summary>bool -> float64: widen to int32 lanes then vcvtdq2pd per 128-bit half (0/1 exact).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        private static unsafe void BoolCastF64(void* src, void* dst, long count)
        {
            byte* s = (byte*)src;
            double* d = (double*)dst;
            var ones = Vector256.Create((byte)1);
            long i = 0;
            if (Avx.IsSupported)
            {
                for (long n = count - 32; i <= n; i += 32)
                {
                    var (lo16, hi16) = Vector256.Widen(Vector256.Min(Vector256.Load(s + i), ones));
                    var (w0, w1) = Vector256.Widen(lo16);
                    var (w2, w3) = Vector256.Widen(hi16);
                    Avx.ConvertToVector256Double(w0.AsInt32().GetLower()).Store(d + i);
                    Avx.ConvertToVector256Double(w0.AsInt32().GetUpper()).Store(d + i + 4);
                    Avx.ConvertToVector256Double(w1.AsInt32().GetLower()).Store(d + i + 8);
                    Avx.ConvertToVector256Double(w1.AsInt32().GetUpper()).Store(d + i + 12);
                    Avx.ConvertToVector256Double(w2.AsInt32().GetLower()).Store(d + i + 16);
                    Avx.ConvertToVector256Double(w2.AsInt32().GetUpper()).Store(d + i + 20);
                    Avx.ConvertToVector256Double(w3.AsInt32().GetLower()).Store(d + i + 24);
                    Avx.ConvertToVector256Double(w3.AsInt32().GetUpper()).Store(d + i + 28);
                }
            }
            for (; i < count; i++)
                d[i] = s[i] != 0 ? 1d : 0d;
        }

        #endregion
    }
}
