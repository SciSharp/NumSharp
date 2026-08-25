using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Unary.Predicate.cs - Predicate IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitIsFiniteCall - float.IsFinite/double.IsFinite
//   - EmitIsNanCall - float.IsNaN/double.IsNaN
//   - EmitIsInfCall - float.IsInfinity/double.IsInfinity
//   - Integer types always return true/false appropriately
//   - GetPredicateContiguousKernel - dedicated SIMD float->bool whole-array
//     kernels for IsNan/IsInf/IsFinite (the port of NumPy's
//     loops_unary_fp_le.dispatch.c.src vectorized predicates). The general
//     unary route treats these as cross-dtype (float in, bool out) and so
//     falls to a scalar per-element walk — 10x slower than NumPy's SIMD
//     loop at 100K float32. The predicate mask is pure lane logic:
//       isnan(v)    = NOT (v == v)                  (npyv_notnan)
//       isinf(v)    = |v| == +inf                   (exponent-all-set, mantissa 0)
//       isfinite(v) = |v| <  +inf                   (NOT exponent-all-set)
//     followed by the comparison kernels' mask->bool store (MSB extract +
//     BMI2 PDEP -> one 8-byte store per vector).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Unary Predicate IL Emission
        /// <summary>
        /// Emit IsFinite check.
        /// For float/double: calls float.IsFinite/double.IsFinite
        /// For integer types: always true (integers cannot be infinite or NaN)
        /// </summary>
        private static void EmitIsFiniteCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsFinite(x)
                var method = typeof(float).GetMethod("IsFinite", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsFinite(x)
                var method = typeof(double).GetMethod("IsFinite", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always true
                // Pop the value from stack and push true (1)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_1);
            }
        }

        /// <summary>
        /// Emit IsNaN check.
        /// For float/double: calls float.IsNaN/double.IsNaN
        /// For integer types: always false (integers cannot be NaN)
        /// </summary>
        private static void EmitIsNanCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsNaN(x)
                var method = typeof(float).GetMethod("IsNaN", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsNaN(x)
                var method = typeof(double).GetMethod("IsNaN", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always false
                // Pop the value from stack and push false (0)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_0);
            }
        }

        /// <summary>
        /// Emit IsInfinity check.
        /// For float/double: calls float.IsInfinity/double.IsInfinity
        /// For integer types: always false (integers cannot be infinite)
        /// </summary>
        private static void EmitIsInfCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsInfinity(x)
                var method = typeof(float).GetMethod("IsInfinity", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsInfinity(x)
                var method = typeof(double).GetMethod("IsInfinity", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always false
                // Pop the value from stack and push false (0)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_0);
            }
        }

        #endregion

        #region SIMD Predicate Kernels (float/double -> bool)

        // Cache keyed (op, inputType) — the output is always Boolean and the
        // kernel is contiguous-only (linear input[i] -> output[i]); strided
        // sources reach it through the buffered gather tile in
        // DefaultEngine.TryBufferedStridedUnaryOp, exactly as the SIMD-capable
        // same-dtype unary ops do.
        private static readonly ConcurrentDictionary<(UnaryOp Op, NPTypeCode Input), UnaryKernel?> _predicateContigKernels = new();

        /// <summary>
        ///     Dedicated SIMD contiguous kernel for the float-classification
        ///     predicates (<see cref="UnaryOp.IsNan"/> / <see cref="UnaryOp.IsInf"/> /
        ///     <see cref="UnaryOp.IsFinite"/>) over <see cref="NPTypeCode.Single"/> /
        ///     <see cref="NPTypeCode.Double"/> input — the port of NumPy's vectorized
        ///     <c>loops_unary_fp_le.dispatch.c.src</c> loops (npyv_pack_isnan/isinf/isfinite).
        ///
        ///     The general unary route cannot vectorize these: its SIMD gate
        ///     (<see cref="CanUseUnarySimd"/>) requires input dtype == output dtype and a
        ///     predicate is inherently float→bool, so IsNan/IsInf/IsFinite fell to a
        ///     per-element scalar walk (measured 40.6 µs vs NumPy's 4.1 µs on 100K
        ///     float32 isnan). Lane logic per vector, all in the float domain so no
        ///     reinterpret casts are needed:
        ///       isnan:    NOT Equals(v, v)          — NaN is the only self-unequal value
        ///       isinf:    Equals(Abs(v), +inf)      — Abs(NaN) is NaN, NaN == inf is false
        ///       isfinite: LessThan(Abs(v), +inf)    — NaN &lt; inf is false, inf &lt; inf is false
        ///     then the comparison kernels' mask→bool store (ExtractMostSignificantBits +
        ///     BMI2 PDEP → one sized store per vector, per-lane fallback without BMI2).
        ///
        ///     Returns null when the (op, dtype) pair is outside this kernel's scope —
        ///     integer/Half/Complex inputs keep the existing scalar routes (NumPy's own
        ///     HALF/CFLOAT predicate loops are scalar too), and callers fall back to
        ///     <see cref="GetUnaryKernel"/>.
        /// </summary>
        public static UnaryKernel? GetPredicateContiguousKernel(UnaryOp op, NPTypeCode inputType)
        {
            if (op != UnaryOp.IsNan && op != UnaryOp.IsInf && op != UnaryOp.IsFinite)
                return null;
            if (inputType != NPTypeCode.Single && inputType != NPTypeCode.Double)
                return null;
            if (!Vector128.IsHardwareAccelerated)
                return null;

            return _predicateContigKernels.GetOrAdd((op, inputType),
                static key => GeneratePredicateContiguousKernel(key.Op, key.Input));
        }

        private static UnaryKernel GeneratePredicateContiguousKernel(UnaryOp op, NPTypeCode inputType)
        {
            var dm = new DynamicMethod(
                name: $"Predicate_Contig_{op}_{inputType}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*),   // input, output(bool*)
                    typeof(long*), typeof(long*),   // strides, shape (unused — contiguous)
                    typeof(int), typeof(long)       // ndim (unused), totalSize
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();
            EmitPredicateSimdLoop(il, op, inputType);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<UnaryKernel>();
        }

        /// <summary>
        ///     The unary edition of <c>EmitComparisonSimdLoop</c>: 4×-unrolled SIMD body
        ///     + single-vector remainder + scalar tail. Args follow the
        ///     <see cref="UnaryKernel"/> signature — input arg0, output (bool*) arg1,
        ///     totalSize arg5.
        /// </summary>
        private static void EmitPredicateSimdLoop(ILGenerator il, UnaryOp op, NPTypeCode inputType)
        {
            long vectorCount = GetVectorCount(inputType);
            const int unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;
            var clrType = GetClrType(inputType);
            var vectorType = GetVectorType(clrType);
            int elemSize = GetTypeSize(inputType);

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var locMask0 = il.DeclareLocal(vectorType);
            var locMask1 = il.DeclareLocal(vectorType);
            var locMask2 = il.DeclareLocal(vectorType);
            var locMask3 = il.DeclareLocal(vectorType);
            var maskLocals = new[] { locMask0, locMask1, locMask2, locMask3 };

            // Hoisted +inf broadcast for the Abs-compare predicates (scalar
            // pre-broadcast pattern). IsNan needs no constant.
            LocalBuilder? locInf = null;
            if (op != UnaryOp.IsNan)
            {
                locInf = il.DeclareLocal(vectorType);
                if (inputType == NPTypeCode.Single)
                    il.Emit(OpCodes.Ldc_R4, float.PositiveInfinity);
                else
                    il.Emit(OpCodes.Ldc_R8, double.PositiveInfinity);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(VectorBits, clrType), null);
                il.Emit(OpCodes.Stloc, locInf);
            }

            // unrollEnd = totalSize - unrollStep + 1
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Ldc_I8, unrollStep - 1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = totalSize - vectorCount + 1
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Ldc_I8, vectorCount - 1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnrollLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int n = 0; n < unrollFactor; n++)
            {
                long offset = n * vectorCount;

                // Load input vector at (i + offset) * elemSize
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, inputType);

                EmitVectorPredicateMask(il, op, clrType, locInf, VectorBits);
                il.Emit(OpCodes.Stloc, maskLocals[n]);
            }

            for (int n = 0; n < unrollFactor; n++)
            {
                long offset = n * vectorCount;

                var locIOffset = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Stloc, locIOffset);

                EmitPredicateMaskToBool(il, inputType, (int)vectorCount, locIOffset, maskLocals[n], outputArg: 1, VectorBits);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnrollLoop);

            // === REMAINDER SIMD LOOP (0-3 vectors) ===
            il.MarkLabel(lblUnrollEnd);
            il.MarkLabel(lblRemainderLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemainderEnd);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, inputType);

            EmitVectorPredicateMask(il, op, clrType, locInf, VectorBits);
            il.Emit(OpCodes.Stloc, locMask0);

            EmitPredicateMaskToBool(il, inputType, (int)vectorCount, locI, locMask0, outputArg: 1, VectorBits);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRemainderLoop);

            // === SCALAR TAIL LOOP ===
            il.MarkLabel(lblRemainderEnd);
            il.MarkLabel(lblTailLoop);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblTailEnd);

            // output[i] = predicate(input[i])
            il.Emit(OpCodes.Ldarg_1); // output (bool*)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, inputType);

            switch (op)
            {
                case UnaryOp.IsNan: EmitIsNanCall(il, inputType); break;
                case UnaryOp.IsInf: EmitIsInfCall(il, inputType); break;
                default: EmitIsFiniteCall(il, inputType); break;
            }

            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTailLoop);

            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        ///     Emit the predicate's lane mask. Stack has [v] (a float/double vector);
        ///     result is a full-lane mask vector (all-ones = true) of the same
        ///     element type, ready for MSB extraction.
        /// </summary>
        private static void EmitVectorPredicateMask(ILGenerator il, UnaryOp op, Type clrType, LocalBuilder? locInf, int bits)
        {
            switch (op)
            {
                case UnaryOp.IsNan:
                    // NOT Equals(v, v) — NaN lanes are the only self-unequal ones.
                    il.Emit(OpCodes.Dup);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(bits, clrType), null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(bits, clrType), null);
                    break;

                case UnaryOp.IsInf:
                    // Equals(Abs(v), +inf) — Abs(NaN) stays NaN and NaN == inf is false.
                    il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(bits, "Abs", clrType, paramCount: 1), null);
                    il.Emit(OpCodes.Ldloc, locInf!);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(bits, clrType), null);
                    break;

                default: // IsFinite
                    // LessThan(Abs(v), +inf) — false for ±inf (inf < inf) and NaN (unordered).
                    il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(bits, "Abs", clrType, paramCount: 1), null);
                    il.Emit(OpCodes.Ldloc, locInf!);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.LessThan(bits, clrType), null);
                    break;
            }
        }

        /// <summary>
        ///     Mask → bool store for the predicate kernels — the same
        ///     ExtractMostSignificantBits + BMI2 PDEP + sized-store scheme as the
        ///     comparison kernels' <c>EmitMaskToBoolExtraction</c>, except the bool
        ///     output arg slot is caller-supplied: arg1 in the <see cref="UnaryKernel"/>
        ///     signature (contiguous kernel), arg2 in <see cref="StridedUnaryKernel"/>
        ///     (the comparison variant hardcodes its own arg2 result slot).
        /// </summary>
        private static void EmitPredicateMaskToBool(ILGenerator il, NPTypeCode type,
            int vectorCount, LocalBuilder locI, LocalBuilder locMask, byte outputArg, int bits)
        {
            var clrType = GetClrType(type);

            il.Emit(OpCodes.Ldloc, locMask);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(bits, clrType), null);
            var locBits = il.DeclareLocal(typeof(uint));
            il.Emit(OpCodes.Stloc, locBits);

            // ── FAST PATH: BMI2 PDEP for ≤8 lanes ────────────────────────
            if (s_bmi2X64Pdep != null && vectorCount <= 8)
            {
                // packed = PDEP((ulong)bits, 0x0101010101010101)
                il.Emit(OpCodes.Ldloc, locBits);
                il.Emit(OpCodes.Conv_U8);
                il.Emit(OpCodes.Ldc_I8, unchecked((long)0x0101010101010101UL));
                il.EmitCall(OpCodes.Call, s_bmi2X64Pdep, null);
                var locPacked = il.DeclareLocal(typeof(ulong));
                il.Emit(OpCodes.Stloc, locPacked);

                il.Emit(OpCodes.Ldarg_S, outputArg);    // output (bool*)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locPacked);
                if (vectorCount == 8)
                {
                    il.Emit(OpCodes.Stind_I8);
                }
                else if (vectorCount == 4)
                {
                    il.Emit(OpCodes.Conv_U4);
                    il.Emit(OpCodes.Stind_I4);
                }
                else if (vectorCount == 2)
                {
                    il.Emit(OpCodes.Conv_U2);
                    il.Emit(OpCodes.Stind_I2);
                }
                else
                {
                    il.Emit(OpCodes.Pop);
                    il.Emit(OpCodes.Pop);
                    goto PerLane;
                }
                return;
            }

            PerLane:
            // ── FALLBACK: per-lane shift-mask-store ──────────────────────
            for (int j = 0; j < vectorCount; j++)
            {
                il.Emit(OpCodes.Ldarg_S, outputArg); // output (bool*)
                il.Emit(OpCodes.Ldloc, locI);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, (long)j);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Add);

                il.Emit(OpCodes.Ldloc, locBits);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.And);

                il.Emit(OpCodes.Stind_I1);
            }
        }

        // -----------------------------------------------------------------
        // Fused strided predicate kernels (1-D strided float/double source →
        // contiguous bool) — the port of NumPy's NCONTIG→CONTIG predicate
        // loops in loops_unary_fp_le (npyv_loadn strided vector build).
        // -----------------------------------------------------------------

        private static readonly ConcurrentDictionary<(UnaryOp Op, NPTypeCode Input), StridedUnaryKernel?> _predicateStridedKernels = new();

        /// <summary>
        ///     Fused strided kernel for IsNan/IsInf/IsFinite over a 1-D strided
        ///     (stepped / reversed / offset) Single/Double source writing a fresh
        ///     contiguous bool result — NumPy's <c>simd_unary_KIND_TYPE_NCONTIG_CONTIG</c>
        ///     shape: each vector is assembled from <c>lanes</c> strided scalar loads
        ///     (<see cref="VectorMethodCache.CreateElements"/>, the npyv_loadn
        ///     equivalent NumPy prefers over hardware gather for these cheap lane ops),
        ///     the predicate mask applied, and the bools stored via the PDEP pack.
        ///     Replaces the scalar gather-to-tile route, which paid ~1 ns/element on
        ///     the gather alone (70 µs vs NumPy's 4.9 µs on a 50K stride-2 isnan).
        ///     Same (op, dtype) gates as <see cref="GetPredicateContiguousKernel"/>.
        /// </summary>
        public static StridedUnaryKernel? GetPredicateStridedKernel(UnaryOp op, NPTypeCode inputType)
        {
            if (op != UnaryOp.IsNan && op != UnaryOp.IsInf && op != UnaryOp.IsFinite)
                return null;
            if (inputType != NPTypeCode.Single && inputType != NPTypeCode.Double)
                return null;
            if (!Vector128.IsHardwareAccelerated)
                return null;

            return _predicateStridedKernels.GetOrAdd((op, inputType),
                static key => GeneratePredicateStridedKernel(key.Op, key.Input));
        }

        private static StridedUnaryKernel GeneratePredicateStridedKernel(UnaryOp op, NPTypeCode inputType)
        {
            // StridedUnaryKernel signature: (void* src, long srcByteStride, void* dst, long count)
            var dm = new DynamicMethod(
                name: $"Predicate_Strided_{op}_{inputType}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(long), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();
            EmitPredicateStridedBody(il, op, inputType);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<StridedUnaryKernel>();
        }

        /// <summary>
        ///     Three-stage strided predicate loop (mirrors <c>EmitStridedUnaryBody</c>):
        ///     a 2×-unrolled SIMD body assembling vectors from strided scalar loads, a
        ///     1-vector remainder, and a scalar tail — with the predicate mask + PDEP
        ///     bool store in place of the unary op + vector store. Output is the
        ///     contiguous bool result at arg2; the running source pointer advances by
        ///     the (possibly negative) byte stride.
        /// </summary>
        private static void EmitPredicateStridedBody(ILGenerator il, UnaryOp op, NPTypeCode inputType)
        {
            // NumPy pins this file to 128-bit SIMD (NPY_SIMD_FORCE_128): the op is a
            // couple of lane instructions, so the cost is the strided vector ASSEMBLY,
            // and a 4-lane insert chain (movss + 3×insertps) is cheaper per element
            // than a cross-lane 8-lane build. Measured here too: Single at 256-bit
            // ran ~2x slower than NumPy; 128-bit closes it. Double keeps the global
            // width — its 4-lane V256 build is two 128-bit halves and already at
            // NumPy parity.
            int bits = inputType == NPTypeCode.Single ? 128 : VectorBits;
            int elemSize = GetTypeSize(inputType);
            int vstep = bits / (elemSize * 8);
            const int unroll = 2;
            long unrollStep = (long)vstep * unroll;
            var clrType = GetClrType(inputType);
            var vectorType = VectorMethodCache.V(bits, clrType);

            var locSrc = il.DeclareLocal(typeof(void*));
            var locStride = il.DeclareLocal(typeof(long));
            var locI = il.DeclareLocal(typeof(long));
            var locCount = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            var locMask = il.DeclareLocal(vectorType);

            LocalBuilder? locInf = null;
            if (op != UnaryOp.IsNan)
            {
                locInf = il.DeclareLocal(vectorType);
                if (inputType == NPTypeCode.Single)
                    il.Emit(OpCodes.Ldc_R4, float.PositiveInfinity);
                else
                    il.Emit(OpCodes.Ldc_R8, double.PositiveInfinity);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(bits, clrType), null);
                il.Emit(OpCodes.Stloc, locInf);
            }

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // locStride = srcByteStride; locCount = count; locSrc = src; locI = 0
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, locStride);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, locCount);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // unrollEnd = count - unrollStep; vectorEnd = count - vstep
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // === 2x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int n = 0; n < unroll; n++)
                EmitStridedPredicateVector(il, op, inputType, clrType, bits, vstep, locSrc, locStride, locI, locMask, locInf, (long)n * vstep);

            EmitAdvanceSrc(il, locSrc, locStride, unrollStep);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === 1-VECTOR REMAINDER ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            EmitStridedPredicateVector(il, op, inputType, clrType, bits, vstep, locSrc, locStride, locI, locMask, locInf, 0);

            EmitAdvanceSrc(il, locSrc, locStride, vstep);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Bge, lblTailEnd);

            // dst[i] = predicate(*(T*)src)
            il.Emit(OpCodes.Ldarg_2); // dst (bool*)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locSrc);
            EmitLoadIndirect(il, inputType);
            switch (op)
            {
                case UnaryOp.IsNan: EmitIsNanCall(il, inputType); break;
                case UnaryOp.IsInf: EmitIsInfCall(il, inputType); break;
                default: EmitIsFiniteCall(il, inputType); break;
            }
            il.Emit(OpCodes.Stind_I1);

            // src += stride; i++
            il.Emit(OpCodes.Ldloc, locSrc);
            il.Emit(OpCodes.Ldloc, locStride);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);

            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        ///     One fused strided predicate vector: assemble <c>vstep</c> strided scalar
        ///     loads into a vector (<see cref="VectorMethodCache.CreateElements"/>),
        ///     apply the predicate mask, and PDEP-store the bools at
        ///     <c>dst + i + elemBase</c>.
        /// </summary>
        private static void EmitStridedPredicateVector(
            ILGenerator il, UnaryOp op, NPTypeCode inputType, Type clrType, int bits, int vstep,
            LocalBuilder locSrc, LocalBuilder locStride, LocalBuilder locI,
            LocalBuilder locMask, LocalBuilder? locInf, long elemBase)
        {
            for (int k = 0; k < vstep; k++)
            {
                long laneIdx = elemBase + k;
                il.Emit(OpCodes.Ldloc, locSrc);
                if (laneIdx != 0)
                {
                    il.Emit(OpCodes.Ldc_I8, laneIdx);
                    il.Emit(OpCodes.Ldloc, locStride);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                EmitLoadIndirect(il, inputType);
            }
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateElements(bits, clrType), null);

            EmitVectorPredicateMask(il, op, clrType, locInf, bits);
            il.Emit(OpCodes.Stloc, locMask);

            var locIOffset = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldloc, locI);
            if (elemBase != 0)
            {
                il.Emit(OpCodes.Ldc_I8, elemBase);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Stloc, locIOffset);

            EmitPredicateMaskToBool(il, inputType, vstep, locIOffset, locMask, outputArg: 2, bits);
        }

        #endregion
    }
}
