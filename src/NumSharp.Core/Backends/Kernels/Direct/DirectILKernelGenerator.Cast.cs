using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // DirectILKernelGenerator.Cast.cs
    //   OWNERSHIP: Cross-dtype copy kernels (contig and strided/broadcast).
    //
    //   TWO PUBLIC ENTRY POINTS:
    //     - CastKernel        : (src, dst, count)  - flat contig src+dst
    //     - StridedCastKernel : (src, dst, srcStrides, dstStrides, shape, ndim)
    //                          - handles arbitrary strides, broadcast (stride=0),
    //                            outer-strided/inner-contig, fully strided
    //
    //   RESPONSIBILITY:
    //     - One IL-generated DynamicMethod per (src, dst) NPTypeCode pair, per shape.
    //     - Strategy selection: MemoryCopy / Widen / Narrow / Convert / Scalar.
    //     - Width selection: V512 / V256 / V128 picked from <see cref="VectorBits"/>;
    //       Vector256 used when the source is wide enough (4+ bytes), otherwise V128.
    //     - Strided kernels detect "inner unit stride for both src and dst" at runtime
    //       and use the full SIMD body for the innermost axis, walking outer dims via
    //       incremental coord advance. When inner is also strided, fall back to scalar.
    //
    //   PARITY WITH NumPy:
    //     - Cross-dtype contig: 1-3x of NumPy (within noise of NumPy's bulk dtype loops).
    //     - Broadcast (stride=0 outer): outer loop reuses src ptr, inner SIMD body per
    //       outer row -> NumPy parity for (1,N)->(M,N) cases.
    //     - Outer-strided/inner-contig (e.g. arr[::2,:]): outer coord walk + inner SIMD.
    //
    //   CALLERS:
    //     - NpyIter.Copy when src.TypeCode != dst.TypeCode (contig => CastKernel,
    //       general => StridedCastKernel).
    // =============================================================================
    public static partial class DirectILKernelGenerator
    {
        // -----------------------------------------------------------------
        // Public delegates
        // -----------------------------------------------------------------

        /// <summary>
        /// Cross-dtype contiguous copy kernel.
        /// Both <paramref name="src"/> and <paramref name="dst"/> must be contiguous.
        /// </summary>
        public unsafe delegate void CastKernel(void* src, void* dst, long count);

        /// <summary>
        /// Cross-dtype strided/broadcast copy kernel.
        /// </summary>
        /// <param name="src">Source base pointer (already offset by Shape.offset).</param>
        /// <param name="dst">Destination base pointer.</param>
        /// <param name="srcStrides">Source strides in elements (length = ndim).</param>
        /// <param name="dstStrides">Destination strides in elements (length = ndim).</param>
        /// <param name="shape">Shape in elements (length = ndim).</param>
        /// <param name="ndim">Number of dimensions; coalesced when possible by NpyIter.</param>
        public unsafe delegate void StridedCastKernel(
            void* src, void* dst,
            long* srcStrides, long* dstStrides,
            long* shape, int ndim);

        // -----------------------------------------------------------------
        // Caches
        // -----------------------------------------------------------------

        private readonly struct CastKernelKey : IEquatable<CastKernelKey>
        {
            public readonly NPTypeCode Src;
            public readonly NPTypeCode Dst;

            public CastKernelKey(NPTypeCode src, NPTypeCode dst) { Src = src; Dst = dst; }

            public bool Equals(CastKernelKey other) => Src == other.Src && Dst == other.Dst;
            public override bool Equals(object obj) => obj is CastKernelKey k && Equals(k);
            public override int GetHashCode() => ((int)Src << 8) | (int)Dst;
            public override string ToString() => $"{Src}To{Dst}";
        }

        private static readonly ConcurrentDictionary<CastKernelKey, CastKernel> _castCache = new();
        private static readonly ConcurrentDictionary<CastKernelKey, byte> _castUnsupported = new();

        private static readonly ConcurrentDictionary<CastKernelKey, StridedCastKernel> _stridedCastCache = new();
        private static readonly ConcurrentDictionary<CastKernelKey, byte> _stridedCastUnsupported = new();

        /// <summary>Number of cached contig cast kernels (diagnostics).</summary>
        public static int CastCachedCount => _castCache.Count;

        /// <summary>Number of cached strided cast kernels (diagnostics).</summary>
        public static int StridedCastCachedCount => _stridedCastCache.Count;

        // -----------------------------------------------------------------
        // Public API
        // -----------------------------------------------------------------

        /// <summary>
        /// Get or generate a contig cast kernel for the given pair.
        /// Returns <c>null</c> for unsupported pairs (Boolean/Char/Half/Complex/Decimal involved).
        /// </summary>
        public static CastKernel TryGetCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Enabled) return null;
            // NumPy-faithful cvtt fast path for the common float->int32 downcast (beats NumPy).
            var floatToInt32 = TryGetFloatToInt32Kernel(srcType, dstType);
            if (floatToInt32 != null) return floatToInt32;
            // NumPy-faithful cvtt+truncating-Narrow for float->{i8,u8,i16,u16,char} (Phase-0's
            // worst cells: f32->i8 was 0.09x). Bit-exact with Converts.To{Narrow}(double).
            var floatToNarrow = TryGetFloatToNarrowIntKernel(srcType, dstType);
            if (floatToNarrow != null) return floatToNarrow;
            // Complex -> int via real-part deinterleave + cvtt (c128->i8 was 0.61x). Drops
            // imaginary (NumPy ComplexWarning); bit-exact with Converts.To{X}(Complex).
            var complexToInt = TryGetComplexToIntKernel(srcType, dstType);
            if (complexToInt != null) return complexToInt;
            // Half -> int via Giesen bit-fiddle widen + cvtt (no F16C in this .NET; f16->i32
            // was ~0.69x). Bit-exact with Converts.To{X}(Half).
            var halfToX = TryGetHalfToXKernel(srcType, dstType);
            if (halfToX != null) return halfToX;
            // {int,float,half,char} -> bool via SIMD `!= 0` compare (Phase-0's worst dst column).
            var toBool = TryGetToBoolKernel(srcType, dstType);
            if (toBool != null) return toBool;
            if (DivergesFromNumpyCast(srcType, dstType)) return null;

            var key = new CastKernelKey(srcType, dstType);
            if (_castCache.TryGetValue(key, out var existing)) return existing;
            if (_castUnsupported.ContainsKey(key)) return null;

            try
            {
                var kernel = GenerateCastKernel(key);
                if (kernel == null)
                {
                    _castUnsupported.TryAdd(key, 0);
                    return null;
                }
                return _castCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetCastKernel({srcType}, {dstType}): {ex.GetType().Name}: {ex.Message}");
                _castUnsupported.TryAdd(key, 0);
                return null;
            }
        }

        /// <summary>
        /// Get or generate a strided/broadcast cast kernel for the given pair.
        /// Returns <c>null</c> for unsupported pairs.
        /// </summary>
        public static StridedCastKernel TryGetStridedCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Enabled) return null;
            // NumPy-faithful cvtt strided fast path for double->int32 (unit / reversed / gathered inner).
            var d2iStrided = TryGetDoubleToInt32StridedKernel(srcType, dstType);
            if (d2iStrided != null) return d2iStrided;
            // f32->i32 strided: whole-array fused VPGATHERDD+cvtt (no Narrow). Same 4->4 same-width
            // path the contig f32->i32 kernel covers, for strided rows. (u32 stays scalar — its NumPy
            // modular-wrap semantics need a float->i64 convert / AVX512 to vectorize faithfully.)
            var fwiStrided = TryGetFloatToWideIntStridedKernel(srcType, dstType);
            if (fwiStrided != null) return fwiStrided;
            // cvtt+truncating-Narrow strided for float->{i8,u8,i16,u16,char} (incl. char, which the
            // generic strided emitter rejects). Inner-contig rows run the Bulk; strided rows stage to
            // a contig buffer then vectorize. Bit-exact with the contiguous float->narrow kernels.
            var fnStrided = TryGetFloatToNarrowIntStridedKernel(srcType, dstType);
            if (fnStrided != null) return fnStrided;
            // {i32,u32,i64,u64}->narrower int strided: whole-array fused VPGATHER + truncating
            // Vector.Narrow (no cvtt). ss==1 rows run contig Load+Narrow; ss!=1 rows gather. Keyed
            // by (srcSize,dstSize); strictly-narrowing only (src==dst returns null, so same-type
            // copy that also calls this resolver is unaffected).
            var inStrided = TryGetIntToNarrowStridedKernel(srcType, dstType);
            if (inStrided != null) return inStrided;
            // Complex->int strided: contiguous-inner rows run the deinterleave Bulk, else scalar.
            var complexStrided = TryGetComplexToIntStridedKernel(srcType, dstType);
            if (complexStrided != null) return complexStrided;
            // Half->int strided (Giesen widen): reuses StridedNarrowDriver (srcSize=2).
            var halfStrided = TryGetHalfToXStridedKernel(srcType, dstType);
            if (halfStrided != null) return halfStrided;
            // {int,float,half,char}->bool strided (!= 0 compare): reuses StridedNarrowDriver (dstSize=1).
            var toBoolStrided = TryGetToBoolStridedKernel(srcType, dstType);
            if (toBoolStrided != null) return toBoolStrided;
            if (DivergesFromNumpyCast(srcType, dstType)) return null;

            var key = new CastKernelKey(srcType, dstType);
            if (_stridedCastCache.TryGetValue(key, out var existing)) return existing;
            if (_stridedCastUnsupported.ContainsKey(key)) return null;

            try
            {
                var kernel = GenerateStridedCastKernel(key);
                if (kernel == null)
                {
                    _stridedCastUnsupported.TryAdd(key, 0);
                    return null;
                }
                return _stridedCastCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetStridedCastKernel({srcType}, {dstType}): {ex.GetType().Name}: {ex.Message}");
                _stridedCastUnsupported.TryAdd(key, 0);
                return null;
            }
        }

        // -----------------------------------------------------------------
        // Strategy selection
        // -----------------------------------------------------------------

        private enum CastStrategy
        {
            None,
            MemoryCopy,
            ScalarOnly,
            WidenInt,
            NarrowInt,
            WidenIntChain2,
            Int32ToSingle,
            Int32ToDouble,
            SingleToInt32,
            SmallIntToSingle,
            SingleToDouble,
            DoubleToSingle,
        }

        /// <summary>
        /// True for (src,dst) pairs whose emitted SIMD kernel body still diverges from NumPy on
        /// out-of-range / NaN inputs, so the caller declines it and falls back to the Converts-backed
        /// scalar path (<see cref="Iteration.NpyIterCasting.ConvertValue"/>), bit-exact with NumPy.
        ///
        /// Now narrowed to ONE family: signed-narrow -> UInt64 (SByte/Int16/Int32), whose vectorized
        /// widen sign-extends only to 32 bits. (float->int is now NumPy-faithful everywhere: the
        /// scalar path via EmitConvertTo -> Converts.*, the float->int32 contig path via the cvtt
        /// helpers in TryGetCastKernel, and the remaining float->int pairs via the ScalarOnly
        /// strategy which has no SIMD body.)
        /// </summary>
        internal static bool DivergesFromNumpyCast(NPTypeCode src, NPTypeCode dst)
        {
            if (dst == NPTypeCode.UInt64 &&
                (src == NPTypeCode.SByte || src == NPTypeCode.Int16 || src == NPTypeCode.Int32))
                return true;
            return false;
        }

        /// <summary>
        /// Returns the NumPy-faithful contig <see cref="CastKernel"/> for float-&gt;int32, or null.
        /// The hardware truncating-convert (VCVTTPD2DQ / VCVTTPS2DQ) yields exactly NumPy's result
        /// — truncate toward zero, int.MinValue on NaN/overflow — so no saturation fixup is needed.
        /// The IL generator's width-matched emitter can't express the double-&gt;int32 lane halving
        /// cleanly, so these two pairs use a width-adaptive helper (the "tidy helper -&gt; single Call"
        /// pattern) instead of generated IL.
        /// </summary>
        internal static unsafe CastKernel TryGetFloatToInt32Kernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (dstType != NPTypeCode.Int32) return null;
            if (srcType == NPTypeCode.Double) return CastDoubleToInt32Contig;
            if (srcType == NPTypeCode.Single) return CastSingleToInt32Contig;
            return null;
        }

        // 4M f64->i32 (warm): ~1.48 ms vs NumPy 1.87 ms; vs 5.17 ms for the scalar Converts loop.
        private static unsafe void CastDoubleToInt32Contig(void* srcV, void* dstV, long count)
        {
            double* src = (double*)srcV;
            int* dst = (int*)dstV;
            long i = 0;
            if (Avx512F.IsSupported)
            {
                for (; i + 8 <= count; i += 8)
                    Vector256.Store(Avx512F.ConvertToVector256Int32WithTruncation(Vector512.Load(src + i)), dst + i);
            }
            else if (Avx.IsSupported)
            {
                for (; i + 4 <= count; i += 4)
                    Vector128.Store(Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(src + i)), dst + i);
            }
            else if (Sse2.IsSupported)
            {
                for (; i + 2 <= count; i += 2)
                    *(long*)(dst + i) = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i)).AsInt64().ToScalar();
            }
            for (; i < count; i++) dst[i] = Converts.ToInt32(src[i]);
        }

        private static unsafe void CastSingleToInt32Contig(void* srcV, void* dstV, long count)
        {
            float* src = (float*)srcV;
            int* dst = (int*)dstV;
            long i = 0;
            if (Avx512F.IsSupported)
            {
                for (; i + 16 <= count; i += 16)
                    Vector512.Store(Avx512F.ConvertToVector512Int32WithTruncation(Vector512.Load(src + i)), dst + i);
            }
            else if (Avx.IsSupported)
            {
                for (; i + 8 <= count; i += 8)
                    Vector256.Store(Avx.ConvertToVector256Int32WithTruncation(Vector256.Load(src + i)), dst + i);
            }
            else if (Sse2.IsSupported)
            {
                for (; i + 4 <= count; i += 4)
                    Vector128.Store(Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(src + i)), dst + i);
            }
            for (; i < count; i++) dst[i] = Converts.ToInt32(src[i]);
        }

        /// <summary>NumPy-faithful cvtt strided <see cref="StridedCastKernel"/> for double-&gt;int32, or null.</summary>
        internal static unsafe StridedCastKernel TryGetDoubleToInt32StridedKernel(NPTypeCode srcType, NPTypeCode dstType)
            => (srcType == NPTypeCode.Double && dstType == NPTypeCode.Int32) ? CastDoubleToInt32Strided : null;

        // Strided/broadcast double->int32 cast. The innermost axis is vectorized with VCVTTPD2DQ
        // for the three common dst-contiguous run shapes — unit-stride source, reversed source
        // ([::-1]), and positive-strided source (a[:,::2] etc., via AVX2 gather) — and falls back
        // to the NumPy-faithful scalar Converts path for everything else. Outer dims are walked
        // with an incremental-offset odometer (element strides, matching the IL strided kernel).
        private static unsafe void CastDoubleToInt32Strided(
            void* srcV, void* dstV, long* srcStrides, long* dstStrides, long* shape, int ndim)
        {
            double* src = (double*)srcV;
            int* dst = (int*)dstV;
            if (ndim == 0) { dst[0] = Converts.ToInt32(src[0]); return; }

            int outer = ndim - 1;
            long innerN = shape[outer];
            long ss = srcStrides[outer];
            long ds = dstStrides[outer];

            long outerCount = 1;
            for (int a = 0; a < outer; a++) outerCount *= shape[a];

            long* coord = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++) coord[a] = 0;

            long srcOff = 0, dstOff = 0;
            for (long o = 0; o < outerCount; o++)
            {
                InnerCastDoubleToInt32(src + srcOff, dst + dstOff, innerN, ss, ds);

                for (int ax = outer - 1; ax >= 0; ax--)
                {
                    coord[ax]++; srcOff += srcStrides[ax]; dstOff += dstStrides[ax];
                    if (coord[ax] < shape[ax]) break;
                    coord[ax] = 0; srcOff -= srcStrides[ax] * shape[ax]; dstOff -= dstStrides[ax] * shape[ax];
                }
            }
        }

        private static unsafe void InnerCastDoubleToInt32(double* s, int* d, long n, long ss, long ds)
        {
            long i = 0;
            if (ds == 1)
            {
                if (ss == 1)
                {
                    if (Avx.IsSupported)
                        for (; i + 4 <= n; i += 4)
                            Vector128.Store(Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(s + i)), d + i);
                    else if (Sse2.IsSupported)
                        for (; i + 2 <= n; i += 2)
                            *(long*)(d + i) = Sse2.ConvertToVector128Int32WithTruncation(Vector128.Load(s + i)).AsInt64().ToScalar();
                }
                else if (ss == -1 && Avx.IsSupported)
                {
                    // logical s[i..i+3] live at memory [s-i-3 .. s-i]; load forward, cvtt, reverse lanes.
                    var rev = Vector128.Create(3, 2, 1, 0);
                    for (; i + 4 <= n; i += 4)
                    {
                        var iv = Avx.ConvertToVector128Int32WithTruncation(Vector256.Load(s - i - 3));
                        Vector128.Store(Vector128.Shuffle(iv, rev), d + i);
                    }
                }
                else if (ss > 1 && ss <= int.MaxValue / 4 && Avx2.IsSupported)
                {
                    var idx = Vector128.Create(0, (int)ss, (int)(2 * ss), (int)(3 * ss));
                    for (; i + 4 <= n; i += 4)
                        Vector128.Store(Avx.ConvertToVector128Int32WithTruncation(Avx2.GatherVector256(s + i * ss, idx, 8)), d + i);
                }
            }
            for (; i < n; i++) d[i * ds] = Converts.ToInt32(s[i * ss]);
        }

        private static bool IsIntegerCast(NPTypeCode t) =>
               t == NPTypeCode.Byte || t == NPTypeCode.SByte
            || t == NPTypeCode.Int16 || t == NPTypeCode.UInt16
            || t == NPTypeCode.Int32 || t == NPTypeCode.UInt32
            || t == NPTypeCode.Int64 || t == NPTypeCode.UInt64;

        private static bool IsFloatCast(NPTypeCode t) =>
               t == NPTypeCode.Single || t == NPTypeCode.Double;

        private static (CastStrategy strategy, int simdBits, int vstep) ResolveStrategy(NPTypeCode src, NPTypeCode dst)
        {
            if (!(IsIntegerCast(src) || IsFloatCast(src)) || !(IsIntegerCast(dst) || IsFloatCast(dst)))
                return (CastStrategy.None, 0, 0);

            if (src == dst) return (CastStrategy.MemoryCopy, 0, 0);

            int srcSize = GetTypeSize(src);
            int dstSize = GetTypeSize(dst);
            bool srcInt = IsIntegerCast(src);
            bool dstInt = IsIntegerCast(dst);
            bool srcFloat = IsFloatCast(src);
            bool dstFloat = IsFloatCast(dst);

            if (srcSize == dstSize)
            {
                if (srcInt && dstInt) return (CastStrategy.MemoryCopy, 0, 0);
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            if (srcInt && dstInt)
            {
                if (dstSize > srcSize)
                {
                    int ratio = dstSize / srcSize;
                    if (ratio == 2)
                    {
                        int simdBits = (srcSize >= 4 && VectorBits >= 256) ? 256
                                     : (VectorBits >= 128) ? 128 : 0;
                        int vstep = simdBits == 0 ? 0 : (simdBits / 8) / srcSize;
                        return (CastStrategy.WidenInt, simdBits, vstep);
                    }
                    if (ratio == 4)
                    {
                        int simdBits = VectorBits >= 128 ? 128 : 0;
                        int vstep = simdBits == 0 ? 0 : 16 / srcSize;
                        return (CastStrategy.WidenIntChain2, simdBits, vstep);
                    }
                    return (CastStrategy.ScalarOnly, 0, 0);
                }
                else
                {
                    int ratio = srcSize / dstSize;
                    if (ratio == 2)
                    {
                        int simdBits = VectorBits >= 128 ? 128 : 0;
                        int vstep = simdBits == 0 ? 0 : (16 / srcSize) * 2;
                        return (CastStrategy.NarrowInt, simdBits, vstep);
                    }
                    return (CastStrategy.ScalarOnly, 0, 0);
                }
            }

            if (srcInt && dstFloat)
            {
                if (src == NPTypeCode.Int32 && dst == NPTypeCode.Single)
                {
                    int simdBits = VectorBits >= 256 ? 256 : (VectorBits >= 128 ? 128 : 0);
                    int vstep = simdBits == 0 ? 0 : (simdBits / 8) / srcSize;
                    return (CastStrategy.Int32ToSingle, simdBits, vstep);
                }
                if (src == NPTypeCode.Int32 && dst == NPTypeCode.Double)
                {
                    int simdBits = VectorBits >= 128 ? 128 : 0;
                    int vstep = simdBits == 0 ? 0 : 16 / srcSize;
                    return (CastStrategy.Int32ToDouble, simdBits, vstep);
                }
                if ((src == NPTypeCode.Int16 || src == NPTypeCode.UInt16) && dst == NPTypeCode.Single)
                {
                    int simdBits = VectorBits >= 128 ? 128 : 0;
                    int vstep = simdBits == 0 ? 0 : 16 / srcSize;
                    return (CastStrategy.SmallIntToSingle, simdBits, vstep);
                }
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            if (srcFloat && dstInt)
            {
                if (src == NPTypeCode.Single && dst == NPTypeCode.Int32)
                {
                    int simdBits = VectorBits >= 256 ? 256 : (VectorBits >= 128 ? 128 : 0);
                    int vstep = simdBits == 0 ? 0 : (simdBits / 8) / srcSize;
                    return (CastStrategy.SingleToInt32, simdBits, vstep);
                }
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            if (srcFloat && dstFloat)
            {
                if (src == NPTypeCode.Single && dst == NPTypeCode.Double)
                {
                    int simdBits = VectorBits >= 128 ? 128 : 0;
                    int vstep = simdBits == 0 ? 0 : 16 / srcSize;
                    return (CastStrategy.SingleToDouble, simdBits, vstep);
                }
                if (src == NPTypeCode.Double && dst == NPTypeCode.Single)
                {
                    int simdBits = VectorBits >= 128 ? 128 : 0;
                    int vstep = simdBits == 0 ? 0 : (16 / srcSize) * 2;
                    return (CastStrategy.DoubleToSingle, simdBits, vstep);
                }
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            return (CastStrategy.None, 0, 0);
        }

        // =================================================================
        // Contig kernel generation
        // =================================================================

        private static CastKernel GenerateCastKernel(CastKernelKey key)
        {
            var (strategy, simdBits, vstep) = ResolveStrategy(key.Src, key.Dst);
            if (strategy == CastStrategy.None)
                return null;

            var dm = new DynamicMethod(
                name: $"Cast_{key}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            EmitContigCastBody(dm.GetILGenerator(), key, strategy, simdBits, vstep);
            return dm.CreateDelegate<CastKernel>();
        }

        /// <summary>
        /// Args: 0 = src void*, 1 = dst void*, 2 = count long.
        /// </summary>
        private static void EmitContigCastBody(ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep)
        {
            int srcSize = GetTypeSize(key.Src);

            if (strategy == CastStrategy.MemoryCopy)
            {
                EmitMemoryCopyInline(il, srcSize, /*pushSrc*/() => il.Emit(OpCodes.Ldarg_0),
                                                  /*pushDst*/() => il.Emit(OpCodes.Ldarg_1),
                                                  /*pushCount*/() => il.Emit(OpCodes.Ldarg_2));
                il.Emit(OpCodes.Ret);
                return;
            }

            // Source / dest base pointers come straight from args.
            EmitSimdLoopAndTail(il, key, strategy, simdBits, vstep,
                pushSrcBase: () => il.Emit(OpCodes.Ldarg_0),
                pushDstBase: () => il.Emit(OpCodes.Ldarg_1),
                pushCount:  () => il.Emit(OpCodes.Ldarg_2));

            il.Emit(OpCodes.Ret);
        }

        // =================================================================
        // Strided kernel generation
        // =================================================================

        private static StridedCastKernel GenerateStridedCastKernel(CastKernelKey key)
        {
            var (strategy, simdBits, vstep) = ResolveStrategy(key.Src, key.Dst);
            if (strategy == CastStrategy.None)
                return null;

            var dm = new DynamicMethod(
                name: $"StridedCast_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*),  // src
                    typeof(void*),  // dst
                    typeof(long*),  // srcStrides
                    typeof(long*),  // dstStrides
                    typeof(long*),  // shape
                    typeof(int),    // ndim
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            EmitStridedCastBody(dm.GetILGenerator(), key, strategy, simdBits, vstep);
            return dm.CreateDelegate<StridedCastKernel>();
        }

        /// <summary>
        /// Strided/broadcast cast body. Args:
        ///   0 = src void*, 1 = dst void*, 2 = srcStrides long*, 3 = dstStrides long*,
        ///   4 = shape long*, 5 = ndim int.
        ///
        /// Emitted IL structure:
        ///   if (ndim == 0): single scalar conv; return.
        ///
        ///   Walk outer dims with coord/offset arrays (localloc).
        ///   At each outer position, test:
        ///     if (srcStrides[innerAxis] == 1 && dstStrides[innerAxis] == 1):
        ///         <inner SIMD loop + scalar tail for shape[innerAxis] elements>
        ///     else:
        ///         <scalar strided inner loop>
        ///   Advance outer coords (innermost-first, incremental offset update).
        /// </summary>
        private static void EmitStridedCastBody(ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);

            // ---- Locals ----
            // Inner axis info (shape, strides).
            var locInnerN          = il.DeclareLocal(typeof(long));
            var locInnerSrcStride  = il.DeclareLocal(typeof(long));
            var locInnerDstStride  = il.DeclareLocal(typeof(long));

            // Per-outer-iter offsets (in elements; multiplied by elem size when used).
            var locOuterSrcOffset  = il.DeclareLocal(typeof(long));
            var locOuterDstOffset  = il.DeclareLocal(typeof(long));

            // Coord array (long*) — outerNdim entries — stackalloc via localloc.
            var locCoords          = il.DeclareLocal(typeof(long*));
            var locOuterNdim       = il.DeclareLocal(typeof(int));

            // For inner loops.
            var locI               = il.DeclareLocal(typeof(long));

            // ---- Labels ----
            var lblScalar0DStart   = il.DefineLabel();   // ndim==0 branch
            var lblOuterLoopHead   = il.DefineLabel();
            var lblOuterLoopBody   = il.DefineLabel();
            var lblInnerContigPath = il.DefineLabel();
            var lblInnerScalarPath = il.DefineLabel();
            var lblAfterInner      = il.DefineLabel();
            var lblAdvanceOuter    = il.DefineLabel();
            var lblRet             = il.DefineLabel();

            // ---- ndim == 0: do single scalar conv ----
            il.Emit(OpCodes.Ldarg_S, (byte)5);   // ndim
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bne_Un, lblScalar0DStart);

            // Single-element scalar: dst = (TDst) src
            // dst ptr on stack
            il.Emit(OpCodes.Ldarg_1);
            // load src val
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, key.Src);
            EmitConvertTo(il, key.Src, key.Dst);
            EmitStoreIndirect(il, key.Dst);
            il.Emit(OpCodes.Br, lblRet);

            il.MarkLabel(lblScalar0DStart);

            // ---- innerN = shape[ndim-1]; innerSrcStride = srcStrides[ndim-1]; innerDstStride = dstStrides[ndim-1] ----
            // ndim-1
            il.Emit(OpCodes.Ldarg_S, (byte)5);   // ndim
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Conv_I);
            // shape[ndim-1]
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_S, (byte)4);   // shape
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerN);
            // (innerIdx is still on stack as native int)
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_2);            // srcStrides
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerSrcStride);
            // innerIdx still on stack
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_3);            // dstStrides
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerDstStride);

            // ---- outerNdim = ndim - 1 ----
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuterNdim);

            // ---- Broadcast fast path: detect "outer dims fully broadcast (srcStrides==0) AND inner contig" ----
            //
            // When all outer src strides are 0, every outer row of dst gets the same converted source data.
            // Convert the source row ONCE into dst[0..innerN], then memcpy that row into each remaining
            // outer dst position. Saves N-1 SIMD conversion passes for an N-row broadcast.
            //
            // This is the NumPy strategy for (1,N)→(M,N) broadcast casts; without it we'd re-convert the
            // same src row M times when one conversion + (M-1) memcpys suffices.
            {
                var lblNotBroadcast = il.DefineLabel();
                var lblBroadcastFastPath = il.DefineLabel();

                // outerNdim > 0 (must have outer dims to broadcast across)
                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ble, lblNotBroadcast);

                // innerSrcStride == 1 && innerDstStride == 1
                il.Emit(OpCodes.Ldloc, locInnerSrcStride);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Bne_Un, lblNotBroadcast);
                il.Emit(OpCodes.Ldloc, locInnerDstStride);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Bne_Un, lblNotBroadcast);

                // All srcStrides[0..outerNdim-1] == 0
                var locCheckIdx = il.DeclareLocal(typeof(int));
                var checkHead = il.DefineLabel();
                var checkBody = il.DefineLabel();

                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locCheckIdx);
                il.MarkLabel(checkHead);
                il.Emit(OpCodes.Ldloc, locCheckIdx);
                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Blt, checkBody);
                il.Emit(OpCodes.Br, lblBroadcastFastPath);

                il.MarkLabel(checkBody);
                il.Emit(OpCodes.Ldarg_2);   // srcStrides
                il.Emit(OpCodes.Ldloc, locCheckIdx);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblNotBroadcast);

                il.Emit(OpCodes.Ldloc, locCheckIdx);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locCheckIdx);
                il.Emit(OpCodes.Br, checkHead);

                // ---- Broadcast fast path body ----
                il.MarkLabel(lblBroadcastFastPath);
                EmitBroadcastConvertThenMemcpy(il, key, strategy, simdBits, vstep,
                    srcSize, dstSize, locInnerN, locOuterNdim);
                il.Emit(OpCodes.Br, lblRet);

                il.MarkLabel(lblNotBroadcast);
            }

            // ---- coords[] = stackalloc long[outerNdim] (or 1 if outerNdim==0 to avoid 0-byte localloc) ----
            // localloc expects unsigned native int count of bytes.
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            // size_bytes = max(outerNdim, 1) * 8
            var lblSizeReady = il.DefineLabel();
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bgt, lblSizeReady);
            il.Emit(OpCodes.Pop);
            il.Emit(OpCodes.Ldc_I4_1);
            il.MarkLabel(lblSizeReady);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);

            // ---- Zero coords[] ----
            // for k=0; k<outerNdim; k++ coords[k] = 0
            {
                var locK = il.DeclareLocal(typeof(int));
                var zeroHead = il.DefineLabel();
                var zeroBody = il.DefineLabel();

                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locK);
                il.MarkLabel(zeroHead);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Blt, zeroBody);
                il.Emit(OpCodes.Br, lblOuterLoopHead);

                il.MarkLabel(zeroBody);
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);

                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, zeroHead);
            }

            // ---- outerSrcOffset = 0; outerDstOffset = 0 ----
            il.MarkLabel(lblOuterLoopHead);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterSrcOffset);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterDstOffset);

            // (Note: after first iteration, offsets are updated via incremental advance,
            //  so we reset only on entry — actually no, we don't reset, the outer
            //  advance maintains them. Move the zeroing before the loop.)
            // Restructure: just enter the body.
            il.MarkLabel(lblOuterLoopBody);

            // ---- Branch on inner contig ----
            il.Emit(OpCodes.Ldloc, locInnerSrcStride);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblInnerScalarPath);
            il.Emit(OpCodes.Ldloc, locInnerDstStride);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblInnerScalarPath);

            // ---- Inner contig: SIMD body for innerN elements at (src+outerSrcOffset*srcSize, dst+outerDstOffset*dstSize) ----
            il.MarkLabel(lblInnerContigPath);
            Action pushSrcInnerBase = () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
                il.Emit(OpCodes.Ldc_I8, (long)srcSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            };
            Action pushDstInnerBase = () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldc_I8, (long)dstSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            };
            Action pushInnerCount = () => il.Emit(OpCodes.Ldloc, locInnerN);

            if (strategy == CastStrategy.MemoryCopy)
            {
                // Same-type inner row -> Buffer.MemoryCopy(srcRow, dstRow, innerN*elemSize, innerN*elemSize)
                EmitMemoryCopyInline(il, srcSize, pushSrcInnerBase, pushDstInnerBase, pushInnerCount);
            }
            else
            {
                EmitSimdLoopAndTail(il, key, strategy, simdBits, vstep,
                    pushSrcBase: pushSrcInnerBase,
                    pushDstBase: pushDstInnerBase,
                    pushCount: pushInnerCount);
            }
            il.Emit(OpCodes.Br, lblAfterInner);

            // ---- Inner scalar (innerStride != 1 for either): per-element strided ----
            il.MarkLabel(lblInnerScalarPath);
            {
                var lblScalarHead = il.DefineLabel();
                var lblScalarEnd  = il.DefineLabel();
                var locSi         = il.DeclareLocal(typeof(long));   // scalar inner-index
                var locInnerSrcOff = il.DeclareLocal(typeof(long));
                var locInnerDstOff = il.DeclareLocal(typeof(long));

                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locSi);
                il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
                il.Emit(OpCodes.Stloc, locInnerSrcOff);
                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Stloc, locInnerDstOff);

                il.MarkLabel(lblScalarHead);
                il.Emit(OpCodes.Ldloc, locSi);
                il.Emit(OpCodes.Ldloc, locInnerN);
                il.Emit(OpCodes.Bge, lblScalarEnd);

                // dst[innerDstOff] = (TDst) src[innerSrcOff]
                // dst ptr
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locInnerDstOff);
                il.Emit(OpCodes.Ldc_I8, (long)dstSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                // load src val
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locInnerSrcOff);
                il.Emit(OpCodes.Ldc_I8, (long)srcSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, key.Src);
                EmitConvertTo(il, key.Src, key.Dst);
                EmitStoreIndirect(il, key.Dst);

                // Advance offsets and si
                il.Emit(OpCodes.Ldloc, locInnerSrcOff);
                il.Emit(OpCodes.Ldloc, locInnerSrcStride);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locInnerSrcOff);

                il.Emit(OpCodes.Ldloc, locInnerDstOff);
                il.Emit(OpCodes.Ldloc, locInnerDstStride);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locInnerDstOff);

                il.Emit(OpCodes.Ldloc, locSi);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locSi);
                il.Emit(OpCodes.Br, lblScalarHead);

                il.MarkLabel(lblScalarEnd);
            }

            il.MarkLabel(lblAfterInner);

            // ---- Advance outer coords. Innermost outer axis = outerNdim-1, walks first. ----
            // If outerNdim == 0: we're done after the single inner pass.
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ble, lblRet);

            il.MarkLabel(lblAdvanceOuter);
            {
                // for (int axis = outerNdim - 1; axis >= 0; axis--) {
                //     coords[axis]++;
                //     outerSrcOffset += srcStrides[axis];
                //     outerDstOffset += dstStrides[axis];
                //     if (coords[axis] < shape[axis]) goto outerLoopBody;
                //     coords[axis] = 0;
                //     outerSrcOffset -= srcStrides[axis] * shape[axis];
                //     outerDstOffset -= dstStrides[axis] * shape[axis];
                // }
                // // fell through all axes => done
                // goto lblRet;
                var locAxis = il.DeclareLocal(typeof(int));
                var advHead = il.DefineLabel();
                var advBody = il.DefineLabel();
                var advNext = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locAxis);

                il.MarkLabel(advHead);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Bge, advBody);
                il.Emit(OpCodes.Br, lblRet);    // overflowed all axes

                il.MarkLabel(advBody);

                // coords[axis]++
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stind_I8);

                // outerSrcOffset += srcStrides[axis]
                il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
                il.Emit(OpCodes.Ldarg_2);   // srcStrides
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locOuterSrcOffset);

                // outerDstOffset += dstStrides[axis]
                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldarg_3);   // dstStrides
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locOuterDstOffset);

                // if (coords[axis] < shape[axis]) goto lblOuterLoopBody
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_S, (byte)4);   // shape
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Blt, lblOuterLoopBody);

                // coords[axis] = 0; outerSrcOffset -= srcStrides[axis] * shape[axis]; outerDstOffset -= dstStrides[axis] * shape[axis]
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);

                il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locOuterSrcOffset);

                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locOuterDstOffset);

                il.MarkLabel(advNext);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locAxis);
                il.Emit(OpCodes.Br, advHead);
            }

            il.MarkLabel(lblRet);
            il.Emit(OpCodes.Ret);
        }

        // =================================================================
        // Broadcast fast path helper
        // =================================================================

        /// <summary>
        /// Emit the "convert once then memcpy the rest" body for the broadcast fast path.
        ///
        /// Step 1: Convert one source row (innerN elements) into dst[0..innerN].
        ///         - Same-type (MemoryCopy): Buffer.MemoryCopy(src, dst, innerN*srcSize, innerN*srcSize).
        ///         - Cross-type: SIMD loop + scalar tail for innerN elements (reuses EmitSimdLoopAndTail).
        ///
        /// Step 2: For each remaining outer position, Buffer.MemoryCopy from dst[0..innerN] into
        ///         dst[outerDstOffset..outerDstOffset+innerN]. Outer coord advance is incremental
        ///         (same pattern as the regular outer loop, no mod/div).
        /// </summary>
        private static void EmitBroadcastConvertThenMemcpy(
            ILGenerator il,
            CastKernelKey key,
            CastStrategy strategy,
            int simdBits,
            int vstep,
            int srcSize,
            int dstSize,
            LocalBuilder locInnerN,
            LocalBuilder locOuterNdim)
        {
            // ---- Step 1: Convert src row → dst row 0 ----
            if (strategy == CastStrategy.MemoryCopy)
            {
                // Same-type: Buffer.MemoryCopy(src, dst, innerN*srcSize, innerN*srcSize)
                EmitMemoryCopyInline(il, srcSize,
                    pushSrc:   () => il.Emit(OpCodes.Ldarg_0),
                    pushDst:   () => il.Emit(OpCodes.Ldarg_1),
                    pushCount: () => il.Emit(OpCodes.Ldloc, locInnerN));
            }
            else
            {
                // Cross-type: SIMD loop + scalar tail for innerN elements.
                EmitSimdLoopAndTail(il, key, strategy, simdBits, vstep,
                    pushSrcBase: () => il.Emit(OpCodes.Ldarg_0),
                    pushDstBase: () => il.Emit(OpCodes.Ldarg_1),
                    pushCount:   () => il.Emit(OpCodes.Ldloc, locInnerN));
            }

            // ---- Step 2: Memcpy dst[0..innerN] into each remaining outer dst row ----
            //
            // Layout of the rest of the body:
            //   - Allocate coords[outerNdim] via localloc; zero it.
            //   - outerDstOffset = 0.
            //   - Advance coords once before the first memcpy (we've already written row 0).
            //   - Loop:
            //       memcpy(dst, dst + outerDstOffset * dstSize, innerN * dstSize, ...)
            //       advance coords → outerDstOffset
            //       if overflow → done.
            //   The advance happens at top so the loop exits cleanly when overflowing all axes.

            var locCoords         = il.DeclareLocal(typeof(long*));
            var locOuterDstOffset = il.DeclareLocal(typeof(long));

            // Allocate coords array
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);

            // Zero coords[]
            {
                var locK = il.DeclareLocal(typeof(int));
                var zeroHead = il.DefineLabel();
                var zeroBody = il.DefineLabel();

                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locK);
                il.MarkLabel(zeroHead);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldloc, locOuterNdim);
                var zeroDone = il.DefineLabel();
                il.Emit(OpCodes.Bge, zeroDone);

                il.MarkLabel(zeroBody);
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);

                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, zeroHead);

                il.MarkLabel(zeroDone);
            }

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterDstOffset);

            var lblBroadcastLoopHead = il.DefineLabel();
            var lblBroadcastLoopBody = il.DefineLabel();
            var lblBroadcastDone     = il.DefineLabel();

            il.Emit(OpCodes.Br, lblBroadcastLoopHead);

            // ---- Memcpy body ----
            il.MarkLabel(lblBroadcastLoopBody);

            // Buffer.MemoryCopy(dst + 0, dst + outerDstOffset * dstSize, innerN * dstSize, innerN * dstSize)
            var memCopy = typeof(Buffer).GetMethod(
                nameof(Buffer.MemoryCopy),
                new[] { typeof(void*), typeof(void*), typeof(long), typeof(long) })!;

            il.Emit(OpCodes.Ldarg_1);   // src = dst row 0

            il.Emit(OpCodes.Ldarg_1);   // dst = dst + outerDstOffset * dstSize
            il.Emit(OpCodes.Ldloc, locOuterDstOffset);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locInnerN);   // sizeInBytes
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);

            il.Emit(OpCodes.Ldloc, locInnerN);   // bytesToCopy
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);

            il.EmitCall(OpCodes.Call, memCopy, null);

            // ---- Advance coords (innermost first) ----
            il.MarkLabel(lblBroadcastLoopHead);
            {
                var locAxis = il.DeclareLocal(typeof(int));
                var advHead = il.DefineLabel();
                var advBody = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locAxis);

                il.MarkLabel(advHead);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Bge, advBody);
                il.Emit(OpCodes.Br, lblBroadcastDone);   // overflowed all axes

                il.MarkLabel(advBody);

                // coords[axis]++
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stind_I8);

                // outerDstOffset += dstStrides[axis]
                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locOuterDstOffset);

                // if (coords[axis] < shape[axis]) goto loopBody
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_S, (byte)4);   // shape
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Blt, lblBroadcastLoopBody);

                // Coords overflow on this axis: reset coord, subtract stride*shape, move outward.
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);

                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locOuterDstOffset);

                il.Emit(OpCodes.Ldloc, locAxis);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locAxis);
                il.Emit(OpCodes.Br, advHead);
            }

            il.MarkLabel(lblBroadcastDone);
        }

        // =================================================================
        // SIMD loop + scalar tail (shared by contig and strided kernels)
        // =================================================================

        /// <summary>
        /// Emit: { long i=0; while(i+vstep<=count) { simd_body(i); i+=vstep; } while(i<count) { scalar(i); i++; } }
        /// using <paramref name="pushSrcBase"/> / <paramref name="pushDstBase"/> for base ptrs and
        /// <paramref name="pushCount"/> for the count.
        /// </summary>
        private static void EmitSimdLoopAndTail(
            ILGenerator il,
            CastKernelKey key,
            CastStrategy strategy,
            int simdBits,
            int vstep,
            Action pushSrcBase,
            Action pushDstBase,
            Action pushCount)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);

            var locI = il.DeclareLocal(typeof(long));

            var lblSimdHead   = il.DefineLabel();
            var lblSimdEnd    = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblRet        = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ---- SIMD loop ----
            if (strategy != CastStrategy.ScalarOnly && simdBits > 0 && vstep > 0)
            {
                il.MarkLabel(lblSimdHead);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vstep);
                il.Emit(OpCodes.Add);
                pushCount();
                il.Emit(OpCodes.Bgt, lblSimdEnd);

                EmitSimdIteration(il, key, strategy, simdBits, vstep, locI, pushSrcBase, pushDstBase);

                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vstep);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblSimdHead);

                il.MarkLabel(lblSimdEnd);
            }

            // ---- Scalar tail ----
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            pushCount();
            il.Emit(OpCodes.Bge, lblRet);

            EmitScalarStore(il, key.Src, key.Dst, srcSize, dstSize, locI, pushSrcBase, pushDstBase);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblRet);
        }

        // -----------------------------------------------------------------
        // Scalar element store: dst[i] = (TDst) src[i]
        // -----------------------------------------------------------------
        private static void EmitScalarStore(ILGenerator il, NPTypeCode srcType, NPTypeCode dstType,
                                            int srcSize, int dstSize, LocalBuilder locI,
                                            Action pushSrcBase, Action pushDstBase)
        {
            // dst ptr
            pushDstBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // load src
            pushSrcBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, srcType);

            EmitConvertTo(il, srcType, dstType);
            EmitStoreIndirect(il, dstType);
        }

        // -----------------------------------------------------------------
        // SIMD iteration dispatch
        // -----------------------------------------------------------------
        private static void EmitSimdIteration(ILGenerator il, CastKernelKey key, CastStrategy strategy,
                                              int simdBits, int vstep, LocalBuilder locI,
                                              Action pushSrcBase, Action pushDstBase)
        {
            switch (strategy)
            {
                case CastStrategy.WidenInt:
                    EmitWidenInt(il, key, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.NarrowInt:
                    EmitNarrowInt(il, key, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.WidenIntChain2:
                    EmitWidenIntChain2(il, key, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.Int32ToSingle:
                    EmitInt32ToSingle(il, simdBits, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.Int32ToDouble:
                    EmitInt32ToDouble(il, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.SingleToInt32:
                    EmitSingleToInt32(il, simdBits, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.SmallIntToSingle:
                    EmitSmallIntToSingle(il, key.Src, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.SingleToDouble:
                    EmitSingleToDouble(il, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                case CastStrategy.DoubleToSingle:
                    EmitDoubleToSingle(il, simdBits, vstep, locI, pushSrcBase, pushDstBase);
                    break;
                default:
                    throw new InvalidOperationException($"SIMD emission not implemented for {strategy}");
            }
        }

        // =================================================================
        // Per-strategy SIMD body emitters
        //   Each emits the loop body for one vector iteration.
        //   They consume `pushSrcBase` / `pushDstBase` (push base ptr on stack)
        //   and `locI` (current element index in the inner loop).
        // =================================================================

        private static void EmitWidenInt(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI,
                                        Action pushSrcBase, Action pushDstBase)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            var srcElem = GetClrType(key.Src);
            var dstElem = WidenSrcSignedness(key.Src, key.Dst);

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, dstSize);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, vstep / 2, dstSize);
            });
        }

        private static void EmitNarrowInt(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI,
                                         Action pushSrcBase, Action pushDstBase)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            var srcElem = GetClrType(key.Src);
            var dstElem = NarrowDstSignedness(key.Src, key.Dst);

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, vstep / 2, srcSize);
            });

            il.EmitCall(OpCodes.Call, GetNarrowMethod(simdBits, srcElem), null);

            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, dstSize);
            });
        }

        private static void EmitWidenIntChain2(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI,
                                              Action pushSrcBase, Action pushDstBase)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            int outPerStore = vstep / 4;

            var srcElem = GetClrType(key.Src);
            var midElem = key.Src == NPTypeCode.Byte ? typeof(ushort)
                        : key.Src == NPTypeCode.SByte ? typeof(short)
                        : key.Src == NPTypeCode.UInt16 ? typeof(uint)
                        : typeof(int);
            var dstElem = WidenSrcSignedness(key.Src, key.Dst);

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            var locS0 = il.DeclareLocal(VType(simdBits, midElem));
            var locS1 = il.DeclareLocal(VType(simdBits, midElem));

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            il.Emit(OpCodes.Stloc, locS0);
            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            il.Emit(OpCodes.Stloc, locS1);

            EmitWidenAndStore(il, locS0, midElem, dstElem, simdBits, locI, 0 * outPerStore, dstSize, isUpper: false, pushDstBase);
            EmitWidenAndStore(il, locS0, midElem, dstElem, simdBits, locI, 1 * outPerStore, dstSize, isUpper: true,  pushDstBase);
            EmitWidenAndStore(il, locS1, midElem, dstElem, simdBits, locI, 2 * outPerStore, dstSize, isUpper: false, pushDstBase);
            EmitWidenAndStore(il, locS1, midElem, dstElem, simdBits, locI, 3 * outPerStore, dstSize, isUpper: true,  pushDstBase);
        }

        private static void EmitWidenAndStore(ILGenerator il, LocalBuilder src, Type midElem, Type dstElem,
                                              int simdBits, LocalBuilder locI, int elemOffset, int dstSize, bool isUpper,
                                              Action pushDstBase)
        {
            il.Emit(OpCodes.Ldloc, src);
            var widen = isUpper
                ? GetWidenUpperMethod(simdBits, midElem)
                : GetWidenLowerMethod(simdBits, midElem);
            il.EmitCall(OpCodes.Call, widen, null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, elemOffset, dstSize);
            });
        }

        private static void EmitInt32ToSingle(ILGenerator il, int simdBits, LocalBuilder locI,
                                              Action pushSrcBase, Action pushDstBase)
        {
            EmitLoadVector(il, typeof(int), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        private static void EmitInt32ToDouble(ILGenerator il, int simdBits, int vstep, LocalBuilder locI,
                                              Action pushSrcBase, Action pushDstBase)
        {
            EmitLoadVector(il, typeof(int), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToDoubleFromInt64Method(simdBits), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 8);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToDoubleFromInt64Method(simdBits), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });
        }

        private static void EmitSingleToInt32(ILGenerator il, int simdBits, LocalBuilder locI,
                                              Action pushSrcBase, Action pushDstBase)
        {
            EmitLoadVector(il, typeof(float), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });
            il.EmitCall(OpCodes.Call, GetConvertToInt32FromSingleMethod(simdBits), null);
            EmitStoreVector(il, typeof(int), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        private static void EmitSmallIntToSingle(ILGenerator il, NPTypeCode src, int simdBits, int vstep, LocalBuilder locI,
                                                 Action pushSrcBase, Action pushDstBase)
        {
            var srcElem = GetClrType(src);
            bool isUnsigned = src == NPTypeCode.UInt16;

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 2);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            if (isUnsigned) il.EmitCall(OpCodes.Call, GetAsMethod(simdBits, typeof(uint), typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            if (isUnsigned) il.EmitCall(OpCodes.Call, GetAsMethod(simdBits, typeof(uint), typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, vstep / 2, 4);
            });
        }

        private static void EmitSingleToDouble(ILGenerator il, int simdBits, int vstep, LocalBuilder locI,
                                               Action pushSrcBase, Action pushDstBase)
        {
            EmitLoadVector(il, typeof(float), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, typeof(float)), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 8);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, typeof(float)), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });
        }

        private static void EmitDoubleToSingle(ILGenerator il, int simdBits, int vstep, LocalBuilder locI,
                                               Action pushSrcBase, Action pushDstBase)
        {
            EmitLoadVector(il, typeof(double), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, 0, 8);
            });
            EmitLoadVector(il, typeof(double), simdBits, () =>
            {
                pushSrcBase();
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });

            il.EmitCall(OpCodes.Call, GetNarrowMethod(simdBits, typeof(double)), null);

            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        // =================================================================
        // MemoryCopy inline helper
        // =================================================================

        private static void EmitMemoryCopyInline(ILGenerator il, int elemSize,
                                                 Action pushSrc, Action pushDst, Action pushCount)
        {
            var memCopy = typeof(Buffer).GetMethod(
                nameof(Buffer.MemoryCopy),
                new[] { typeof(void*), typeof(void*), typeof(long), typeof(long) })
                ?? throw new MissingMethodException(typeof(Buffer).FullName, nameof(Buffer.MemoryCopy));

            pushSrc();
            pushDst();
            pushCount();
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            pushCount();
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.EmitCall(OpCodes.Call, memCopy, null);
        }

        // =================================================================
        // IL helpers (offset/vector load/store/method lookup)
        // =================================================================

        private static void EmitOffsetExpr(ILGenerator il, LocalBuilder locI, int elemOffset, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, locI);
            if (elemOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, (long)elemOffset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        private static void EmitLoadVector(ILGenerator il, Type elementType, int simdBits, Action pushPtr)
        {
            pushPtr();
            il.EmitCall(OpCodes.Call, GetVectorLoadMethod(simdBits, elementType), null);
        }

        private static void EmitStoreVector(ILGenerator il, Type elementType, int simdBits, Action pushPtr)
        {
            pushPtr();
            il.EmitCall(OpCodes.Call, GetVectorStoreMethod(simdBits, elementType), null);
        }

        // Thin file-local aliases for VectorMethodCache so the existing call sites in this
        // file (EmitStridedCastBody, EmitBroadcastConvertThenMemcpy, etc.) read unchanged.
        // Cast.Masked.cs reuses the same aliases via the partial-class scope.
        private static Type VType(int simdBits, Type elem) => VectorMethodCache.V(simdBits, elem);
        [Obsolete("Unused alias. Call VectorMethodCache.Container directly.", error: true)]
        private static Type ContainerType(int simdBits) => VectorMethodCache.Container(simdBits);

        private static MethodInfo GetVectorLoadMethod(int simdBits, Type elementType)
            => VectorMethodCache.Load(simdBits, elementType);

        private static MethodInfo GetVectorStoreMethod(int simdBits, Type elementType)
            => VectorMethodCache.Store(simdBits, elementType);

        private static MethodInfo GetWidenLowerMethod(int simdBits, Type sourceElementType)
            => VectorMethodCache.WidenLower(simdBits, sourceElementType);

        private static MethodInfo GetWidenUpperMethod(int simdBits, Type sourceElementType)
            => VectorMethodCache.WidenUpper(simdBits, sourceElementType);

        private static MethodInfo GetNarrowMethod(int simdBits, Type sourceElementType)
            => VectorMethodCache.Narrow(simdBits, sourceElementType);

        private static MethodInfo GetConvertToSingleFromInt32Method(int simdBits)
            => VectorMethodCache.ConvertToSingleFromInt32(simdBits);

        private static MethodInfo GetConvertToDoubleFromInt64Method(int simdBits)
            => VectorMethodCache.ConvertToDoubleFromInt64(simdBits);

        private static MethodInfo GetConvertToInt32FromSingleMethod(int simdBits)
            => VectorMethodCache.ConvertToInt32FromSingle(simdBits);

        private static MethodInfo GetAsMethod(int simdBits, Type fromElem, Type toElem)
            => VectorMethodCache.As(simdBits, fromElem, toElem);

        // =================================================================
        // Signedness helpers
        // =================================================================

        private static Type WidenSrcSignedness(NPTypeCode src, NPTypeCode dst)
        {
            switch (src)
            {
                case NPTypeCode.SByte:  return typeof(short);
                case NPTypeCode.Byte:   return typeof(ushort);
                case NPTypeCode.Int16:  return typeof(int);
                case NPTypeCode.UInt16: return typeof(uint);
                case NPTypeCode.Int32:  return typeof(long);
                case NPTypeCode.UInt32: return typeof(ulong);
                case NPTypeCode.Single: return typeof(double);
                default:                return GetClrType(dst);
            }
        }

        private static Type NarrowDstSignedness(NPTypeCode src, NPTypeCode dst)
        {
            switch (src)
            {
                case NPTypeCode.Int16:  return typeof(sbyte);
                case NPTypeCode.UInt16: return typeof(byte);
                case NPTypeCode.Int32:  return typeof(short);
                case NPTypeCode.UInt32: return typeof(ushort);
                case NPTypeCode.Int64:  return typeof(int);
                case NPTypeCode.UInt64: return typeof(uint);
                case NPTypeCode.Double: return typeof(float);
                default:                return GetClrType(dst);
            }
        }
    }
}
