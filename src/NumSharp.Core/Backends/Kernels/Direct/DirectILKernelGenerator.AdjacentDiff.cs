using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    // ============================ Adjacent-difference stencil ============================
    // A fused whole-array kernel for np.diff / np.ediff1d's single-difference core:
    //
    //     out[r, i] = a[r, i + 1] - a[r, i]        (per row, along the innermost axis)
    //
    // np.diff and np.ediff1d both reduce to `subtract(a[1:], a[:-1])` — two OVERLAPPING
    // contiguous views of one buffer. Driving that through the generic binary path
    // allocates the two slice views + an NDIter and reads the source as two operand
    // streams. Because the two operands are the SAME buffer offset by one element, the
    // whole thing is really an adjacent-difference *stencil*: one source, overlapping
    // SIMD loads, one output. This kernel expresses exactly that — no slice views, no
    // NDIter, one input stream — so it shaves the per-call overhead and the managed
    // allocation churn (the two view NDArrays + the iterator state) that the generic
    // route pays every call.
    //
    // Layout contract (enforced by the caller, np.DiffAdjacentContiguous):
    //   * src is C-contiguous; the diff runs along the innermost (contiguous) axis.
    //   * Row r begins at src + r*inLen and writes to dst + r*(inLen-1).
    //   * inLen >= 2, so every row produces at least one output.
    //
    // Numerics are bit-identical to the generic subtract: the SIMD body reuses the same
    // EmitVectorOperation(Subtract) the binary kernel emits, and the scalar tail / the
    // non-SIMD (Char / Half / Decimal / Complex) path reuse EmitScalarOperation(Subtract)
    // — the exact emitters np.diff's DiffSubtractViaNDIter uses today. Subtraction is
    // deterministic, so the driver never changes the result, only the overhead around it.
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Fused adjacent-difference kernel: for each of <c>rows</c> rows,
        ///     <c>dst[i] = src[i+1] - src[i]</c> over the row's <c>inLen</c> contiguous
        ///     elements (the output row has <c>inLen-1</c> elements). <paramref name="src"/>
        ///     and <paramref name="dst"/> are raw base pointers to C-contiguous buffers.
        /// </summary>
        public unsafe delegate void AdjacentDiffKernel(void* src, void* dst, long rows, long inLen);

        /// <summary>Cache of adjacent-difference kernels keyed by element dtype.</summary>
        internal static readonly ConcurrentDictionary<NPTypeCode, AdjacentDiffKernel> _adjacentDiffCache = new();

        /// <summary>
        ///     Get or generate the adjacent-difference (subtract) stencil kernel for
        ///     <paramref name="dt"/>. Returns <c>null</c> for Boolean (np.diff differences
        ///     booleans with not_equal, not subtract) and whenever IL generation is disabled
        ///     or the emitter rejects the dtype — the caller then falls back to the generic
        ///     view-subtract path.
        /// </summary>
        public static AdjacentDiffKernel GetAdjacentDiffKernel(NPTypeCode dt)
        {
            if (!Enabled)
                return null;
            if (dt == NPTypeCode.Boolean)
                return null; // boolean diff is not_equal, handled by the generic path

            try
            {
                return _adjacentDiffCache.GetOrAdd(dt, GenerateAdjacentDiffKernel);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[ILKernel] GetAdjacentDiffKernel({dt}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static AdjacentDiffKernel GenerateAdjacentDiffKernel(NPTypeCode dt)
        {
            var dm = new DynamicMethod(
                name: $"AdjacentDiff_{dt}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), // src
                    typeof(void*), // dst
                    typeof(long),  // rows
                    typeof(long)   // inLen (input row length; output row = inLen-1)
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            int elemSize = GetTypeSize(dt);
            bool simd = CanUseSimd(dt) && CanUseSimdForOp(BinaryOp.Subtract);
            int vectorCount = simd ? GetVectorCount(dt) : 0;

            var locN = il.DeclareLocal(typeof(long));   // outputs per row = inLen - 1
            var locR = il.DeclareLocal(typeof(long));   // row counter
            var locSp = il.DeclareLocal(typeof(byte*)); // current row's src base
            var locDp = il.DeclareLocal(typeof(byte*)); // current row's dst base
            var locI = il.DeclareLocal(typeof(long));   // stencil index within a row

            var lblRet = il.DefineLabel();

            // n = inLen - 1; if (n <= 0) return  (nothing to write for any row)
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locN);
            il.Emit(OpCodes.Ldloc, locN);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblRet);

            // r = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locR);

            var lblRowLoop = il.DefineLabel();
            var lblRowEnd = il.DefineLabel();

            il.MarkLabel(lblRowLoop);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldarg_2); // rows
            il.Emit(OpCodes.Bge, lblRowEnd);

            // sp = (byte*)src + r * inLen * elemSize
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSp);

            // dp = (byte*)dst + r * n * elemSize
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldloc, locN);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDp);

            EmitAdjacentDiffStencil(il, dt, elemSize, simd, vectorCount, locSp, locDp, locN, locI);

            // r++
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Br, lblRowLoop);
            il.MarkLabel(lblRowEnd);

            il.MarkLabel(lblRet);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<AdjacentDiffKernel>();
        }

        /// <summary>
        ///     Emit the per-row stencil <c>dst[i] = src[i+1] - src[i]</c> over
        ///     <c>locN</c> elements: a 4×-unrolled SIMD body + single-vector remainder +
        ///     scalar tail when <paramref name="simd"/>, else a pure scalar loop (used for
        ///     Char / Half / Decimal / Complex — matching the generic path's non-SIMD route).
        /// </summary>
        private static void EmitAdjacentDiffStencil(
            ILGenerator il, NPTypeCode dt, int elemSize, bool simd, int vectorCount,
            LocalBuilder locSp, LocalBuilder locDp, LocalBuilder locN, LocalBuilder locI)
        {
            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (simd)
            {
                long unrollStep = (long)vectorCount * 4;
                var locUnrollEnd = il.DeclareLocal(typeof(long));
                var locVecEnd = il.DeclareLocal(typeof(long));

                // unrollEnd = n - vectorCount*4 ; vecEnd = n - vectorCount
                il.Emit(OpCodes.Ldloc, locN);
                il.Emit(OpCodes.Ldc_I8, unrollStep);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locUnrollEnd);
                il.Emit(OpCodes.Ldloc, locN);
                il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVecEnd);

                var lblUnroll = il.DefineLabel();
                var lblUnrollEnd = il.DefineLabel();
                var lblRem = il.DefineLabel();
                var lblRemEnd = il.DefineLabel();

                // 4x unrolled SIMD
                il.MarkLabel(lblUnroll);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locUnrollEnd);
                il.Emit(OpCodes.Bgt, lblUnrollEnd);
                for (int u = 0; u < 4; u++)
                    EmitVectorStencilStep(il, dt, elemSize, locSp, locDp, locI, (long)vectorCount * u);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, unrollStep);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblUnroll);
                il.MarkLabel(lblUnrollEnd);

                // single-vector remainder
                il.MarkLabel(lblRem);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVecEnd);
                il.Emit(OpCodes.Bgt, lblRemEnd);
                EmitVectorStencilStep(il, dt, elemSize, locSp, locDp, locI, 0);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblRem);
                il.MarkLabel(lblRemEnd);
            }

            // scalar tail (all remaining elements; the whole row when !simd)
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locN);
            il.Emit(OpCodes.Bge, lblTailEnd);
            EmitScalarStencilStep(il, dt, elemSize, locSp, locDp, locI);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        // One SIMD lane-group: dst[i+off .. ] = load(sp+(i+off+1)) - load(sp+(i+off)).
        private static void EmitVectorStencilStep(
            ILGenerator il, NPTypeCode dt, int elemSize,
            LocalBuilder locSp, LocalBuilder locDp, LocalBuilder locI, long off)
        {
            // lhs = *(sp + (i + off + 1))
            il.Emit(OpCodes.Ldloc, locSp);
            EmitIndexTimesSize(il, locI, off + 1, elemSize);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, dt);

            // rhs = *(sp + (i + off))
            il.Emit(OpCodes.Ldloc, locSp);
            EmitIndexTimesSize(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, dt);

            EmitVectorOperation(il, BinaryOp.Subtract, dt); // lhs - rhs

            // store at dp + (i + off)   (stack: ..., vector, ptr)
            il.Emit(OpCodes.Ldloc, locDp);
            EmitIndexTimesSize(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, dt);
        }

        // One scalar element: *(dp + i) = *(sp + (i+1)) - *(sp + i).
        private static void EmitScalarStencilStep(
            ILGenerator il, NPTypeCode dt, int elemSize,
            LocalBuilder locSp, LocalBuilder locDp, LocalBuilder locI)
        {
            // store target ptr first (Stind pops [ptr, value])
            il.Emit(OpCodes.Ldloc, locDp);
            EmitIndexTimesSize(il, locI, 0, elemSize);
            il.Emit(OpCodes.Add);

            // value = *(sp + (i+1)) - *(sp + i)
            il.Emit(OpCodes.Ldloc, locSp);
            EmitIndexTimesSize(il, locI, 1, elemSize);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, dt);
            il.Emit(OpCodes.Ldloc, locSp);
            EmitIndexTimesSize(il, locI, 0, elemSize);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, dt);
            EmitScalarOperation(il, BinaryOp.Subtract, dt);

            EmitStoreIndirect(il, dt);
        }

        // Push (i + delta) * elemSize as a native int (byte offset from a row base).
        private static void EmitIndexTimesSize(ILGenerator il, LocalBuilder locI, long delta, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, locI);
            if (delta != 0)
            {
                il.Emit(OpCodes.Ldc_I8, delta);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
        }
    }
}
