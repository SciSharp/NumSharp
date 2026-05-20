using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // ILKernelGenerator.Cast.cs
    //   OWNERSHIP: Cross-dtype copy kernels (contiguous src → contiguous dst).
    //   RESPONSIBILITY:
    //     - One IL-generated DynamicMethod per (src, dst) NPTypeCode pair.
    //     - Strategy selection: MemoryCopy / Widen / Narrow / Convert / Scalar.
    //     - Width selection: V512 / V256 / V128 picked from <see cref="VectorBits"/>;
    //       Vector256 used when the source is wide enough (4+ bytes), otherwise V128.
    //     - Scalar tail loop appended unconditionally (count not vector-multiple).
    //   PARITY WITH SimdCast.cs (replaced):
    //     - Same instruction sequences (WidenLower/WidenUpper, Narrow, ConvertToSingle,
    //       ConvertToDouble, ConvertToInt32) — emitted via IL instead of C#.
    //     - Same AVX-2-only conversions emit SIMD; AVX-512-only pairs
    //       (UInt32↔Float, UInt64↔Float, Long↔Double, Single→UInt32/Int64,
    //        Double→Int32/Int64/UInt64) fall through to scalar tail.
    //     - 8× widen (byte→long / sbyte→long) emitted as scalar — same as old code
    //       which delegated to JIT auto-vectorization.
    //   CALLER: NpyIter.Copy when state.IsContiguousCopy and src.TypeCode != dst.TypeCode.
    // =============================================================================
    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Cross-dtype copy kernel delegate.
        /// Both <paramref name="src"/> and <paramref name="dst"/> must be contiguous.
        /// </summary>
        /// <param name="src">Pointer to source buffer.</param>
        /// <param name="dst">Pointer to destination buffer.</param>
        /// <param name="count">Number of elements to copy.</param>
        public unsafe delegate void CastKernel(void* src, void* dst, long count);

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
        // Pairs we've already decided are unsupported (so we don't repeatedly retry IL gen).
        private static readonly ConcurrentDictionary<CastKernelKey, byte> _castUnsupported = new();

        /// <summary>
        /// Number of cached cast kernels (diagnostics).
        /// </summary>
        public static int CastCachedCount => _castCache.Count;

        /// <summary>
        /// Get or generate a cast kernel for the given (<paramref name="srcType"/> → <paramref name="dstType"/>) pair.
        /// Returns <c>null</c> when the pair is unsupported (Boolean/Char/Half/Complex/Decimal involved).
        /// </summary>
        public static CastKernel TryGetCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Enabled) return null;

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

        // ===================================================
        // Strategy selection
        // ===================================================

        /// <summary>
        /// Cast strategy — picks the IL emission template for the (src, dst) pair.
        /// </summary>
        private enum CastStrategy
        {
            /// <summary>Unsupported (Boolean/Char/Half/Complex/Decimal involved).</summary>
            None,
            /// <summary>Same size — bit-identical Buffer.MemoryCopy.</summary>
            MemoryCopy,
            /// <summary>Per-element scalar loop only (no SIMD path emitted).</summary>
            ScalarOnly,
            /// <summary>1-step int→int widen: V.WidenLower/Upper.</summary>
            WidenInt,
            /// <summary>1-step int→int narrow: V.Narrow.</summary>
            NarrowInt,
            /// <summary>2-step int→int widen: WidenLower/Upper of WidenLower/Upper.</summary>
            WidenIntChain2,
            /// <summary>V.ConvertToSingle on V&lt;int&gt;.</summary>
            Int32ToSingle,
            /// <summary>V128.WidenLower/Upper(V&lt;int&gt;) → ConvertToDouble.</summary>
            Int32ToDouble,
            /// <summary>V.ConvertToInt32 on V&lt;float&gt;.</summary>
            SingleToInt32,
            /// <summary>V128.WidenLower/Upper(V&lt;short/ushort&gt;) → ConvertToSingle.</summary>
            SmallIntToSingle,
            /// <summary>V128.WidenLower/Upper on V&lt;float&gt;.</summary>
            SingleToDouble,
            /// <summary>V128.Narrow(V&lt;double&gt;, V&lt;double&gt;).</summary>
            DoubleToSingle,
        }

        private static bool IsIntegerCast(NPTypeCode t) =>
               t == NPTypeCode.Byte || t == NPTypeCode.SByte
            || t == NPTypeCode.Int16 || t == NPTypeCode.UInt16
            || t == NPTypeCode.Int32 || t == NPTypeCode.UInt32
            || t == NPTypeCode.Int64 || t == NPTypeCode.UInt64;

        private static bool IsFloatCast(NPTypeCode t) =>
               t == NPTypeCode.Single || t == NPTypeCode.Double;

        /// <summary>
        /// Map (src, dst) to a strategy, the SIMD container width to use, and the per-iteration source step.
        /// </summary>
        private static (CastStrategy strategy, int simdBits, int vstep) ResolveStrategy(NPTypeCode src, NPTypeCode dst)
        {
            // Unsupported on either side.
            if (!(IsIntegerCast(src) || IsFloatCast(src)) || !(IsIntegerCast(dst) || IsFloatCast(dst)))
                return (CastStrategy.None, 0, 0);

            // Identical type → straight memcpy.
            if (src == dst) return (CastStrategy.MemoryCopy, 0, 0);

            int srcSize = GetTypeSize(src);
            int dstSize = GetTypeSize(dst);
            bool srcInt = IsIntegerCast(src);
            bool dstInt = IsIntegerCast(dst);
            bool srcFloat = IsFloatCast(src);
            bool dstFloat = IsFloatCast(dst);

            // Same byte-size, int↔int — only the metadata differs, bits are identical.
            // (Int32↔UInt32, Int64↔UInt64, etc.) Single↔Int32/Double↔Int64 are NOT bit-identical
            // semantically, so route those through scalar.
            if (srcSize == dstSize)
            {
                if (srcInt && dstInt) return (CastStrategy.MemoryCopy, 0, 0);
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            // ---- int → int ----
            if (srcInt && dstInt)
            {
                if (dstSize > srcSize)
                {
                    int ratio = dstSize / srcSize;
                    if (ratio == 2)
                    {
                        // V256 for int32→int64 etc. when available (large per-iter throughput);
                        // V128 for smaller source elements (matches old SimdCast width choices).
                        int simdBits = (srcSize >= 4 && VectorBits >= 256) ? 256
                                     : (VectorBits >= 128) ? 128
                                     : 0;
                        int vstep = simdBits == 0 ? 0 : (simdBits / 8) / srcSize;
                        return (CastStrategy.WidenInt, simdBits, vstep);
                    }
                    if (ratio == 4)
                    {
                        // 2-step widen — old code used V128.
                        int simdBits = VectorBits >= 128 ? 128 : 0;
                        int vstep = simdBits == 0 ? 0 : 16 / srcSize;
                        return (CastStrategy.WidenIntChain2, simdBits, vstep);
                    }
                    // 8× (byte→long): scalar — old code let JIT auto-vectorize.
                    return (CastStrategy.ScalarOnly, 0, 0);
                }
                else
                {
                    int ratio = srcSize / dstSize;
                    if (ratio == 2)
                    {
                        int simdBits = VectorBits >= 128 ? 128 : 0;
                        // Narrow consumes 2× V128 loads per V128 store → vstep = (V128 src elements) * 2.
                        int vstep = simdBits == 0 ? 0 : (16 / srcSize) * 2;
                        return (CastStrategy.NarrowInt, simdBits, vstep);
                    }
                    return (CastStrategy.ScalarOnly, 0, 0);
                }
            }

            // ---- int → float ----
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
                    // Limited to V128 — Vector256.ConvertToDouble(V256<long>) needs AVX-512DQ.
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
                // Everything else (uint32→float, uint64→float, int64→double, etc.): scalar.
                return (CastStrategy.ScalarOnly, 0, 0);
            }

            // ---- float → int ----
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

            // ---- float → float ----
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

        // ===================================================
        // Kernel generation
        // ===================================================

        private static CastKernel GenerateCastKernel(CastKernelKey key)
        {
            var (strategy, simdBits, vstep) = ResolveStrategy(key.Src, key.Dst);
            if (strategy == CastStrategy.None)
                return null;

            var dm = new DynamicMethod(
                name: $"Cast_{key}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(void*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            EmitCastBody(dm.GetILGenerator(), key, strategy, simdBits, vstep);
            return dm.CreateDelegate<CastKernel>();
        }

        private static void EmitCastBody(ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);

            // ---- MemoryCopy fast path ----
            if (strategy == CastStrategy.MemoryCopy)
            {
                // Buffer.MemoryCopy(src, dst, count*size, count*size); return;
                var memCopy = typeof(Buffer).GetMethod(
                    nameof(Buffer.MemoryCopy),
                    new[] { typeof(void*), typeof(void*), typeof(long), typeof(long) })
                    ?? throw new MissingMethodException(typeof(Buffer).FullName, nameof(Buffer.MemoryCopy));

                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, (long)srcSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, (long)srcSize);
                il.Emit(OpCodes.Mul);
                il.EmitCall(OpCodes.Call, memCopy, null);
                il.Emit(OpCodes.Ret);
                return;
            }

            // ---- General template: SIMD loop (optional) + scalar tail ----
            var locI = il.DeclareLocal(typeof(long));   // current element index

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblRet = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ---- SIMD loop (when strategy supports it and platform has the needed width) ----
            if (strategy != CastStrategy.ScalarOnly && simdBits > 0 && vstep > 0)
            {
                il.MarkLabel(lblSimdHead);

                // if ((i + vstep) > count) goto simd_end;
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vstep);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Bgt, lblSimdEnd);

                EmitSimdIteration(il, key, strategy, simdBits, vstep, locI);

                // i += vstep
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vstep);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblSimdHead);

                il.MarkLabel(lblSimdEnd);
            }

            // ---- Scalar tail: for (; i < count; i++) dst[i] = (TDst)src[i]; ----
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblRet);

            EmitScalarStore(il, key.Src, key.Dst, srcSize, dstSize, locI);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblRet);
            il.Emit(OpCodes.Ret);
        }

        // ===================================================
        // Scalar element store
        //   Emits: dst[i] = (TDst)src[i]
        // ===================================================
        private static void EmitScalarStore(ILGenerator il, NPTypeCode srcType, NPTypeCode dstType,
                                            int srcSize, int dstSize, LocalBuilder locI)
        {
            // Push dst pointer: arg1 + i * dstSize
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load src[i]: *(TSrc*)(arg0 + i * srcSize)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, srcType);

            // (TDst) — uses the existing centralized conversion helper which handles
            // signed/unsigned promotion (Conv_R_Un for unsigned → float).
            EmitConvertTo(il, srcType, dstType);

            // Store at dst[i]
            EmitStoreIndirect(il, dstType);
        }

        // ===================================================
        // SIMD iteration body — dispatches to per-strategy emitter
        // ===================================================
        private static void EmitSimdIteration(ILGenerator il, CastKernelKey key, CastStrategy strategy,
                                              int simdBits, int vstep, LocalBuilder locI)
        {
            switch (strategy)
            {
                case CastStrategy.WidenInt:
                    EmitWidenInt(il, key, simdBits, vstep, locI);
                    break;
                case CastStrategy.NarrowInt:
                    EmitNarrowInt(il, key, simdBits, vstep, locI);
                    break;
                case CastStrategy.WidenIntChain2:
                    EmitWidenIntChain2(il, key, simdBits, vstep, locI);
                    break;
                case CastStrategy.Int32ToSingle:
                    EmitInt32ToSingle(il, simdBits, locI);
                    break;
                case CastStrategy.Int32ToDouble:
                    EmitInt32ToDouble(il, simdBits, vstep, locI);
                    break;
                case CastStrategy.SingleToInt32:
                    EmitSingleToInt32(il, simdBits, locI);
                    break;
                case CastStrategy.SmallIntToSingle:
                    EmitSmallIntToSingle(il, key.Src, simdBits, vstep, locI);
                    break;
                case CastStrategy.SingleToDouble:
                    EmitSingleToDouble(il, simdBits, vstep, locI);
                    break;
                case CastStrategy.DoubleToSingle:
                    EmitDoubleToSingle(il, simdBits, vstep, locI);
                    break;
                default:
                    throw new InvalidOperationException($"SIMD emission not implemented for {strategy}");
            }
        }

        // ---- Per-strategy SIMD emitters ----

        /// <summary>
        /// 1-step integer widen (e.g. Int32 → Int64, Int16 → Int32, Byte → UInt16).
        ///   v   = V.Load(src + i*srcSize)
        ///   lo  = V.WidenLower(v)  → store at dst + i*dstSize
        ///   hi  = V.WidenUpper(v)  → store at dst + (i + vstep/2)*dstSize
        /// </summary>
        private static void EmitWidenInt(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            var srcElem = GetClrType(key.Src);
            var dstElem = WidenSrcSignedness(key.Src, key.Dst);

            // Load V<srcElem>(arg0 + i*srcSize)
            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            // dup → WidenLower → store at (i, dstSize)
            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, dstSize);
            });

            // WidenUpper → store at (i + vstep/2, dstSize)
            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, vstep / 2, dstSize);
            });
        }

        /// <summary>
        /// 1-step integer narrow (e.g. Int32 → Int16, Int16 → Byte, Int64 → Int32).
        ///   lo = V128.Load(src + i*srcSize)
        ///   hi = V128.Load(src + (i + vstep/2)*srcSize)
        ///   V128.Narrow(lo, hi) → store at dst + i*dstSize
        /// </summary>
        private static void EmitNarrowInt(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            var srcElem = GetClrType(key.Src);
            var dstElem = NarrowDstSignedness(key.Src, key.Dst);

            // Push lo
            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            // Push hi
            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, vstep / 2, srcSize);
            });

            // V128.Narrow(lo, hi)
            il.EmitCall(OpCodes.Call, GetNarrowMethod(simdBits, srcElem), null);

            // Store at dst + i*dstSize
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, dstSize);
            });
        }

        /// <summary>
        /// 2-step integer widen (Byte → UInt32, SByte → Int32, UInt16 → UInt64, Int16 → Int64).
        ///   v  = V128.Load(src + i*srcSize)
        ///   s0 = WidenLower(v); s1 = WidenUpper(v)        (mid-width vectors)
        ///   WidenLower(s0).Store(dst + i*dstSize)
        ///   WidenUpper(s0).Store(dst + (i + 1*q)*dstSize)
        ///   WidenLower(s1).Store(dst + (i + 2*q)*dstSize)
        ///   WidenUpper(s1).Store(dst + (i + 3*q)*dstSize)
        ///     where q = output elements per stored vector = vstep / 4
        /// </summary>
        private static void EmitWidenIntChain2(ILGenerator il, CastKernelKey key, int simdBits, int vstep, LocalBuilder locI)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            int outPerStore = vstep / 4;

            var srcElem = GetClrType(key.Src);                          // byte or sbyte
            var midElem = key.Src == NPTypeCode.Byte ? typeof(ushort)
                        : key.Src == NPTypeCode.SByte ? typeof(short)
                        : key.Src == NPTypeCode.UInt16 ? typeof(uint)
                        : typeof(int);  // Int16 → mid=int
            var dstElem = WidenSrcSignedness(key.Src, key.Dst);

            // Load v
            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, srcSize);
            });

            // Locals to hold s0, s1 (mid vectors).
            var locS0 = il.DeclareLocal(VType(simdBits, midElem));
            var locS1 = il.DeclareLocal(VType(simdBits, midElem));

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            il.Emit(OpCodes.Stloc, locS0);
            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            il.Emit(OpCodes.Stloc, locS1);

            // Stores at offset 0, q, 2q, 3q
            EmitWidenAndStore(il, locS0, midElem, dstElem, simdBits, locI, 0 * outPerStore, dstSize, isUpper: false);
            EmitWidenAndStore(il, locS0, midElem, dstElem, simdBits, locI, 1 * outPerStore, dstSize, isUpper: true);
            EmitWidenAndStore(il, locS1, midElem, dstElem, simdBits, locI, 2 * outPerStore, dstSize, isUpper: false);
            EmitWidenAndStore(il, locS1, midElem, dstElem, simdBits, locI, 3 * outPerStore, dstSize, isUpper: true);
        }

        private static void EmitWidenAndStore(ILGenerator il, LocalBuilder src, Type midElem, Type dstElem,
                                              int simdBits, LocalBuilder locI, int elemOffset, int dstSize, bool isUpper)
        {
            il.Emit(OpCodes.Ldloc, src);
            var widen = isUpper
                ? GetWidenUpperMethod(simdBits, midElem)
                : GetWidenLowerMethod(simdBits, midElem);
            il.EmitCall(OpCodes.Call, widen, null);
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, elemOffset, dstSize);
            });
        }

        /// <summary>
        /// V&lt;int&gt; → V&lt;float&gt; via Vector.ConvertToSingle(V&lt;int&gt;).
        /// </summary>
        private static void EmitInt32ToSingle(ILGenerator il, int simdBits, LocalBuilder locI)
        {
            // v = V.Load(src + i*4)
            EmitLoadVector(il, typeof(int), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 4);
            });
            // V.ConvertToSingle(v)
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            // Store at dst + i*4
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        /// <summary>
        /// V128&lt;int&gt; → V128&lt;double&gt; via WidenLower/Upper then ConvertToDouble.
        ///   v = V128.Load(src + i*4)
        ///   lo = V128.WidenLower(v) → V128&lt;long&gt;
        ///   V128.ConvertToDouble(lo) → store at dst + i*8
        ///   hi = V128.WidenUpper(v) → V128&lt;long&gt;
        ///   V128.ConvertToDouble(hi) → store at dst + (i+2)*8
        /// </summary>
        private static void EmitInt32ToDouble(ILGenerator il, int simdBits, int vstep, LocalBuilder locI)
        {
            // V128 only (vstep == 4 ints).
            EmitLoadVector(il, typeof(int), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToDoubleFromInt64Method(simdBits), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 8);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToDoubleFromInt64Method(simdBits), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });
        }

        /// <summary>
        /// V&lt;float&gt; → V&lt;int&gt; via Vector.ConvertToInt32(V&lt;float&gt;).
        /// </summary>
        private static void EmitSingleToInt32(ILGenerator il, int simdBits, LocalBuilder locI)
        {
            EmitLoadVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 4);
            });
            il.EmitCall(OpCodes.Call, GetConvertToInt32FromSingleMethod(simdBits), null);
            EmitStoreVector(il, typeof(int), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        /// <summary>
        /// V128&lt;short/ushort&gt; → V128&lt;float&gt; via WidenLower/Upper then ConvertToSingle.
        /// </summary>
        private static void EmitSmallIntToSingle(ILGenerator il, NPTypeCode src, int simdBits, int vstep, LocalBuilder locI)
        {
            var srcElem = GetClrType(src);  // short or ushort
            // After WidenLower, ushort → uint, short → int. ConvertToSingle requires V<int>.
            // For ushort, we reinterpret V<uint> as V<int> via .As<>; bit pattern fine since values
            // fit in [0, 65535] — no sign-bit issue.
            bool isUnsigned = src == NPTypeCode.UInt16;

            EmitLoadVector(il, srcElem, simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 2);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, srcElem), null);
            if (isUnsigned) il.EmitCall(OpCodes.Call, GetAsMethod(simdBits, typeof(uint), typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, srcElem), null);
            if (isUnsigned) il.EmitCall(OpCodes.Call, GetAsMethod(simdBits, typeof(uint), typeof(int)), null);
            il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, vstep / 2, 4);
            });
        }

        /// <summary>
        /// V128&lt;float&gt; → V128&lt;double&gt; via V128.WidenLower/Upper(V128&lt;float&gt;).
        /// </summary>
        private static void EmitSingleToDouble(ILGenerator il, int simdBits, int vstep, LocalBuilder locI)
        {
            EmitLoadVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 4);
            });

            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, GetWidenLowerMethod(simdBits, typeof(float)), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 8);
            });

            il.EmitCall(OpCodes.Call, GetWidenUpperMethod(simdBits, typeof(float)), null);
            EmitStoreVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });
        }

        /// <summary>
        /// V128&lt;double&gt; → V128&lt;float&gt; via V128.Narrow(V128&lt;double&gt;, V128&lt;double&gt;).
        /// </summary>
        private static void EmitDoubleToSingle(ILGenerator il, int simdBits, int vstep, LocalBuilder locI)
        {
            EmitLoadVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, 0, 8);
            });
            EmitLoadVector(il, typeof(double), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                EmitOffsetExpr(il, locI, vstep / 2, 8);
            });

            il.EmitCall(OpCodes.Call, GetNarrowMethod(simdBits, typeof(double)), null);

            EmitStoreVector(il, typeof(float), simdBits, () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                EmitOffsetExpr(il, locI, 0, 4);
            });
        }

        // ===================================================
        // IL helpers
        // ===================================================

        /// <summary>
        /// Emit: i + elemOffset (or just i if 0), multiplied by elemSize, converted to native int.
        /// Resulting native int is then added to whatever pointer was on the stack before this call.
        /// </summary>
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

        /// <summary>
        /// Emit a call to Vector{simdBits}.Load&lt;T&gt;(T*). The action pushes the T* pointer onto the stack.
        /// </summary>
        private static void EmitLoadVector(ILGenerator il, Type elementType, int simdBits, Action pushPtr)
        {
            pushPtr();
            il.EmitCall(OpCodes.Call, GetVectorLoadMethod(simdBits, elementType), null);
        }

        /// <summary>
        /// Stack must already have V&lt;T&gt; on top. Emit: pushPtr → Vector{simdBits}.Store(this V&lt;T&gt;, T*).
        /// </summary>
        private static void EmitStoreVector(ILGenerator il, Type elementType, int simdBits, Action pushPtr)
        {
            pushPtr();
            il.EmitCall(OpCodes.Call, GetVectorStoreMethod(simdBits, elementType), null);
        }

        // ===================================================
        // Vector method lookup
        // ===================================================

        private static Type VType(int simdBits, Type elem) => simdBits switch
        {
            128 => typeof(Vector128<>).MakeGenericType(elem),
            256 => typeof(Vector256<>).MakeGenericType(elem),
            512 => typeof(Vector512<>).MakeGenericType(elem),
            _ => throw new NotSupportedException($"SIMD width {simdBits} not supported")
        };

        private static Type ContainerType(int simdBits) => simdBits switch
        {
            128 => typeof(Vector128),
            256 => typeof(Vector256),
            512 => typeof(Vector512),
            _ => throw new NotSupportedException($"SIMD width {simdBits} not supported")
        };

        private static MethodInfo GetVectorLoadMethod(int simdBits, Type elementType)
        {
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Load" && m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType.IsPointer)
                .MakeGenericMethod(elementType);
        }

        private static MethodInfo GetVectorStoreMethod(int simdBits, Type elementType)
        {
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Store" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetParameters()[0].ParameterType.IsGenericType)
                .MakeGenericMethod(elementType);
        }

        private static MethodInfo GetWidenLowerMethod(int simdBits, Type sourceElementType)
        {
            var paramVec = VType(simdBits, sourceElementType);
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "WidenLower" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType == paramVec);
        }

        private static MethodInfo GetWidenUpperMethod(int simdBits, Type sourceElementType)
        {
            var paramVec = VType(simdBits, sourceElementType);
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "WidenUpper" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType == paramVec);
        }

        private static MethodInfo GetNarrowMethod(int simdBits, Type sourceElementType)
        {
            var paramVec = VType(simdBits, sourceElementType);
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Narrow" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetParameters()[0].ParameterType == paramVec &&
                            m.GetParameters()[1].ParameterType == paramVec);
        }

        private static MethodInfo GetConvertToSingleFromInt32Method(int simdBits)
        {
            var paramVec = VType(simdBits, typeof(int));
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "ConvertToSingle" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType == paramVec);
        }

        private static MethodInfo GetConvertToDoubleFromInt64Method(int simdBits)
        {
            var paramVec = VType(simdBits, typeof(long));
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "ConvertToDouble" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType == paramVec);
        }

        private static MethodInfo GetConvertToInt32FromSingleMethod(int simdBits)
        {
            var paramVec = VType(simdBits, typeof(float));
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "ConvertToInt32" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType == paramVec);
        }

        /// <summary>
        /// Get Vector{N}.As&lt;TFrom, TTo&gt;(V&lt;TFrom&gt;) for bit-reinterpretation.
        /// Used by SmallIntToSingle for V&lt;uint&gt; → V&lt;int&gt; bit-cast before ConvertToSingle.
        /// </summary>
        private static MethodInfo GetAsMethod(int simdBits, Type fromElem, Type toElem)
        {
            string name = "As" + toElem.Name;
            // Try named overload first (AsInt32, AsSingle, etc.)
            var named = ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(m => m.Name == name && m.IsGenericMethod &&
                                     m.GetParameters().Length == 1 &&
                                     m.GetGenericArguments().Length == 1);
            if (named != null) return named.MakeGenericMethod(fromElem);

            // Fallback: generic As<TFrom, TTo>
            var generic = ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "As" && m.IsGenericMethod &&
                            m.GetGenericArguments().Length == 2 &&
                            m.GetParameters().Length == 1);
            return generic.MakeGenericMethod(fromElem, toElem);
        }

        // ===================================================
        // Signedness helpers
        //
        // Widening preserves the sign of the source element type (sbyte → short
        // sign-extends; byte → ushort zero-extends). The dst NPTypeCode's signedness
        // doesn't affect the bits — Int16 and UInt16 share storage layout. We pick
        // the canonical destination CLR type that matches the widen behaviour.
        //
        // Narrowing is sign-agnostic — Narrow(V<int>, V<int>) → V<short> drops the
        // upper 16 bits regardless of sign; same bits regardless of how dst is
        // declared. We pick V<short> when narrowing from signed, V<ushort> when
        // unsigned, purely to match the Narrow overload signature.
        // ===================================================

        private static Type WidenSrcSignedness(NPTypeCode src, NPTypeCode dst)
        {
            // For int → int widening: choose dst CLR type that matches src's sign convention.
            //   sbyte → short (signed)        byte → ushort (unsigned)
            //   short → int   (signed)        ushort → uint  (unsigned)
            //   int   → long  (signed)        uint   → ulong (unsigned)
            // For float (Single → Double), use double.
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
            // For int → int narrowing: dst CLR type that matches src's sign.
            //   short → sbyte (signed)   ushort → byte (unsigned)
            //   int   → short (signed)   uint   → ushort (unsigned)
            //   long  → int   (signed)   ulong  → uint  (unsigned)
            // For float (Double → Single), use float.
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
