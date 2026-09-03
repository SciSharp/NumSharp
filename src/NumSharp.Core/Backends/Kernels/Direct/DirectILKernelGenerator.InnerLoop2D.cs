using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.InnerLoop2D.cs — 2-D coalesced-block element-wise kernel
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A (rows, w) view that is CONTIGUOUS on the inner axis and STRIDED on the outer
// axis (the classic `m[:, :w]` column slice, or a broadcast-row operand) cannot
// coalesce to 1-D, so the ordinary Tier-3B route drives it under EXTERNAL_LOOP:
// the iterator calls the per-chunk kernel ONCE PER ROW, advancing the odometer
// (NDIter.ExternalLoopNext) between calls. For a narrow w that pays, per row:
//   • the odometer advance (~4.5 ns), and
//   • the per-chunk kernel's OWN prologue — snapshot ptrs, then a runtime
//     SIMD-viability dispatch (stride==elemSize checks per operand) — which
//     re-runs on every call even though the answer is identical for every row.
// After all that, only w elements are processed.
//
// Measured on 2M f64 as (rows, w): np.positive/np.sqrt on such a view sat at
// ~0.8-0.9x NumPy for small w, while a hand-written 2-D loop (one prologue, an
// inner SIMD run, a per-row pointer bump) is ~1.7-2.1x FASTER than NumPy — the
// whole gap is the per-row overhead, not the memory traffic (docs/NDITER_PERF_
// DISCOVERY.md §7 angle 2).
//
// THE 2-D CONTRACT
// ----------------
//   void(void** dataptrs, long innerCount, long* outerByteStrides, long outerCount)
//
// The kernel loops the OUTER axis itself: it runs the SIMD-viability dispatch and
// the loop-bound setup ONCE, then for each of `outerCount` rows it executes the
// 4x-unrolled SIMD + remainder + scalar-tail inner loop over `innerCount`
// contiguous elements (addressed as ptr + i*elemSize, exactly like the 1-D
// EmitSimdContigLoop) and advances each operand pointer by its own
// `outerByteStrides[op]`. So neither the odometer nor the prologue runs per row.
//
// SCOPE / GATING (enforced by NDIterRef.TryExecute2DElementwise before compiling)
// ------------------------------------------------------------------------------
//   • unbuffered, no where= mask, EXTERNAL_LOOP, post-coalesce NDim == 2
//   • every operand SIMD-capable and the SAME dtype (CanSimdAllOperands)
//   • the inner axis is contiguous (element stride 1) for EVERY operand
// Everything else (buffered cast, masked, strided/broadcast inner, non-SIMD
// dtypes, NDim != 2) keeps the per-chunk ForEach route unchanged. The outer
// stride is arbitrary per operand, so a broadcast-row operand (outer stride 0)
// is handled naturally — it re-reads the same row.
//
// The emitted body reuses the SAME scalarBody/vectorBody emit delegates the 1-D
// kernel uses, so the arithmetic is byte-identical to the ForEach path; only the
// loop structure differs, and element-wise ops carry no cross-element state.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// A whole-2-D-block element-wise kernel: it loops the outer (strided) axis
    /// itself, running an inner SIMD loop over <paramref name="innerCount"/>
    /// contiguous elements per row and advancing each operand pointer by
    /// <paramref name="outerByteStrides"/>[op]. One call covers the whole
    /// coalesced 2-D iteration.
    /// </summary>
    /// <param name="dataptrs">One byte-pointer per operand (inputs then output), at the block start.</param>
    /// <param name="innerCount">Elements per row (the contiguous inner axis length).</param>
    /// <param name="outerByteStrides">Per-operand outer-axis stride in BYTES (may be 0 for a broadcast operand, or negative).</param>
    /// <param name="outerCount">Number of rows (the outer axis length).</param>
    public unsafe delegate void ND2DElementwiseKernel(
        void** dataptrs, long innerCount, long* outerByteStrides, long outerCount);

    public static partial class DirectILKernelGenerator
    {
        #region 2-D coalesced-block kernel cache

        /// <summary>
        /// Packed-key cache for the 2-D block kernels. Separate from
        /// <see cref="_innerLoopKeyCache"/> because the delegate type differs
        /// (<see cref="ND2DElementwiseKernel"/> vs <see cref="NDInnerLoopFunc"/>).
        /// A (op, dtypes) key can appear in both — the same ufunc served either
        /// per-chunk (contiguous / 1-D) or as one 2-D block (strided narrow rows).
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, ND2DElementwiseKernel> _innerLoop2DCache = new();

        /// <summary>Look up a 2-D block kernel by its packed key without building a string.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryGet2DKernel(in InnerLoopKernelKey key, out ND2DElementwiseKernel kernel)
            => _innerLoop2DCache.TryGetValue(key, out kernel!);

        /// <summary>
        /// Serve the 2-D block kernel from the cache, else compile it under the
        /// equivalent (op, dtypes) identity and register it. The bodies must be
        /// the SAME ones the per-chunk kernel for this key would use, so the two
        /// routes stay byte-identical.
        /// </summary>
        internal static ND2DElementwiseKernel Compile2DElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator> vectorBody,
            in InnerLoopKernelKey key)
        {
            if (_innerLoop2DCache.TryGetValue(key, out var cached))
                return cached;

            var kernel = Generate2DElementwiseKernel(operandTypes, scalarBody, vectorBody, key.ToCacheKey());
            _innerLoop2DCache.TryAdd(key, kernel);
            return kernel;
        }

        private static ND2DElementwiseKernel Generate2DElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator> vectorBody,
            string cacheKey)
        {
            // ND2DElementwiseKernel signature:
            //   void(void** dataptrs, long innerCount, long* outerByteStrides, long outerCount)
            var dm = new DynamicMethod(
                name: $"ND2DLoop_{Sanitize(cacheKey)}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            var ptrLocals = new LocalBuilder[nOp];
            var outerStrideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }

            // ptrLocals[op] = (byte*)dataptrs[op]
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldarg_0);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * IntPtr.Size);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }

            // outerStrideLocals[op] = outerByteStrides[op]   (arg2 = long*)
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldarg_2);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * sizeof(long));
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, outerStrideLocals[op]);
            }

            EmitSimd2DContigLoop(il, operandTypes, ptrLocals, outerStrideLocals, vectorBody, scalarBody);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ND2DElementwiseKernel>();
        }

        /// <summary>
        /// Emit the outer (per-row) loop wrapping a 4x-unrolled SIMD + 1-vector
        /// remainder + scalar-tail inner loop. The inner-loop bound computations
        /// (unrollEnd, vectorEnd) are HOISTED above the outer loop because
        /// innerCount is the same for every row. The inner loop addresses each
        /// operand as <c>ptr + i*elemSize</c> and never mutates the base pointer;
        /// after a row the outer loop advances each base pointer by its own byte
        /// stride. Mirrors <see cref="EmitSimdContigLoop"/>'s inner shape so the
        /// two routes produce identical results.
        ///
        /// arg1 = innerCount, arg3 = outerCount.
        /// </summary>
        private static void EmitSimd2DContigLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] outerStrideLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long vectorCount = GetVectorCount(outType);
            long unrollStep = vectorCount * 4;

            var locUnrollEnd = il.DeclareLocal(typeof(long));  // innerCount - unrollStep (hoisted)
            var locVectorEnd = il.DeclareLocal(typeof(long));  // innerCount - vectorCount (hoisted)
            var locI = il.DeclareLocal(typeof(long));          // inner element index
            var locOuter = il.DeclareLocal(typeof(long));      // remaining rows

            // unrollEnd = innerCount - unrollStep
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = innerCount - vectorCount
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // locOuter = outerCount
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, locOuter);

            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // === OUTER (per-row) LOOP ===
            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblOuterEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // --- 4x UNROLLED SIMD ---
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int u = 0; u < 4; u++)
            {
                long offset = u * vectorCount;
                for (int op = 0; op < nIn; op++)
                {
                    EmitAddrIPlusOffset(il, ptrLocals[op], locI, offset, elemSize);
                    EmitVectorLoad(il, operandTypes[op]);
                }
                vectorBody(il);
                EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, offset, elemSize);
                EmitVectorStore(il, outType);
            }

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // --- REMAINDER SIMD (1 vector at a time) ---
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            for (int op = 0; op < nIn; op++)
            {
                EmitAddrIPlusOffset(il, ptrLocals[op], locI, 0, elemSize);
                EmitVectorLoad(il, operandTypes[op]);
            }
            vectorBody(il);
            EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, 0, elemSize);
            EmitVectorStore(il, outType);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // --- SCALAR TAIL (contiguous) ---
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);           // innerCount
            il.Emit(OpCodes.Bge, lblTailEnd);

            EmitScalarElement(il, operandTypes, ptrLocals, /*stridesInElems*/ null, locI, contig: true, scalarBody);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);

            // --- advance each operand pointer to the next row ---
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                il.Emit(OpCodes.Ldloc, outerStrideLocals[op]);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }

            // locOuter--
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuter);
            il.Emit(OpCodes.Br, lblOuter);
            il.MarkLabel(lblOuterEnd);
        }

        #endregion
    }
}
