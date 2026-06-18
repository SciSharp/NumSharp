using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Reduction.Pairwise.cs — IL-EMITTED SIMD pairwise sum reduce
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// NumPy sums floats with pairwise_sum (loops_utils.h.src): O(lg n) rounding error
// AND — because the 8-accumulator block auto-vectorizes — full SIMD throughput.
// The earlier C# port (PairwiseFold<T> in ILKernelGenerator.Reduction.cs) matched
// NumPy's summation ORDER bit-for-bit but was SCALAR: the .NET JIT does not
// auto-vectorize an 8-accumulator loop the way GCC does, so the contiguous (PINNED)
// reduce gave up SIMD and ran ~2-4.6x slower than a flat vector accumulator.
//
// This file emits, per dtype, a DynamicMethod that reproduces pairwise_sum EXACTLY
// (same recursion split, same 8-accumulator block, same tree-combine → bit-identical
// to np.add.reduce) while keeping the inner block in vector registers. The 8 NumPy
// accumulators r[0..7] map onto SIMD lanes: each r[k] still accumulates elements
// {k, k+8, k+16, …}, so the result is independent of the vector width — V128/V256/
// V512 all produce the same bits. We pick the x86 intrinsic load/add path
// (Avx.LoadVector256 / Avx.Add …) which the JIT compiles ~1.8x tighter than the
// cross-platform Vector256.* helpers (see VectorMethodCache).
//
// MEASURED (f64, axis=1, 1000x1000, AVX2 host): scalar pairwise 0.267 ms →
// emitted SIMD pairwise 0.123 ms (2.18x), beating NumPy 2.4.2 (0.207 ms) by 1.69x,
// and beating the old non-bit-exact flat accumulator (0.139 ms) too.
//
// MODEL — same NpyInnerLoopFunc per-chunk contract as the rest of
// ILKernelGenerator.Reduction.cs (PINNED outStride==0 vs SLAB outStride!=0). The
// emitted kernel folds a PINNED stripe with the recursive pairwise DynamicMethod and
// streams a SLAB stripe with a width-native vector add.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        // Recursive pairwise fold DynamicMethods, rooted here so cross-DynamicMethod
        // `call` sites (the kernel below, and the fold's own self-recursion) stay alive.
        private static readonly ConcurrentDictionary<NPTypeCode, DynamicMethod> _pwFolds = new();

        /// <summary>
        /// Try to build an IL-emitted SIMD pairwise Sum per-chunk kernel for
        /// <paramref name="tc"/> (same-type accumulation). Returns null when the dtype
        /// has no clean SIMD pairwise form (caller keeps the generic scalar fold).
        /// Currently emitted for the IEEE binary floats (Single/Double); the same
        /// emitter generalizes to any (clrType, elemSize) whose lane count divides 8.
        /// </summary>
        internal static unsafe NpyInnerLoopFunc TryEmitPairwiseSumKernel(NPTypeCode tc)
        {
            if (!DirectILKernelGenerator.Enabled) return null;
            if (tc != NPTypeCode.Double && tc != NPTypeCode.Single) return null; // float family only (NumPy pairwise dtypes)

            Type clrType = DirectILKernelGenerator.GetClrType(tc);
            int elemSize = DirectILKernelGenerator.GetTypeSize(tc);

            // Pick the widest SIMD register whose lane count divides 8 (NumPy's fixed
            // 8-accumulator block). double: V128→2, V256→4, V512→8. float: V128→4,
            // V256→8, V512→16 (capped to 256 so 8 accumulators still map 1:1).
            int simdBits = DirectILKernelGenerator.VectorBits;
            if (simdBits < 128) return null; // no SIMD on this host → keep generic scalar fold
            int laneCount = (simdBits / 8) / elemSize;
            while (laneCount > 8) { simdBits /= 2; laneCount = (simdBits / 8) / elemSize; }
            if (laneCount < 2 || (8 % laneCount) != 0) return null;

            try
            {
                DynamicMethod fold = _pwFolds.GetOrAdd(tc, t => EmitPairwiseFold(t, clrType, elemSize, simdBits, laneCount));
                return EmitPairwiseSumKernel(tc, clrType, elemSize, fold);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryEmitPairwiseSumKernel({tc}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        // ---- scalar element helpers (float vs double) -------------------------------
        private static OpCode LdindFloat(int elemSize) => elemSize == 8 ? OpCodes.Ldind_R8 : OpCodes.Ldind_R4;
        private static OpCode StindFloat(int elemSize) => elemSize == 8 ? OpCodes.Stind_R8 : OpCodes.Stind_R4;
        private static void EmitNegZero(ILGenerator il, int elemSize)
        {
            // -0.0 seed (NumPy: summing only -0 must stay -0).
            if (elemSize == 8) il.Emit(OpCodes.Ldc_R8, -0.0);
            else il.Emit(OpCodes.Ldc_R4, -0.0f);
        }

        /// <summary>
        /// Emit the recursive pairwise fold: <c>T fold(void* a, long n, long stride)</c>
        /// (stride in ELEMENTS). 1:1 with NumPy's pairwise_sum — n&lt;8 naive (seed -0),
        /// n≤128 the eight-accumulator unrolled block (SIMD when stride==1, scalar
        /// 8-accumulator otherwise) then the exact tree-combine, n&gt;128 split
        /// (kept a multiple of 8) and self-recurse. Bit-identical to np.add.reduce.
        /// </summary>
        private static DynamicMethod EmitPairwiseFold(NPTypeCode tc, Type clrType, int elemSize, int simdBits, int laneCount)
        {
            Type pV = typeof(void).MakePointerType();
            var dm = new DynamicMethod($"PwFold_{tc}", clrType, new[] { pV, typeof(long), typeof(long) },
                typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();

            int numAccVecs = 8 / laneCount;
            Type vecType = VectorMethodCache.V(simdBits, clrType);
            MethodInfo mLoad = VectorMethodCache.LoadX86(simdBits, clrType) ?? VectorMethodCache.Load(simdBits, clrType);
            MethodInfo mAdd = VectorMethodCache.BinaryX86(simdBits, "Add", clrType) ?? VectorMethodCache.Generic(simdBits, "Add", clrType, paramCount: 2);
            MethodInfo mGet = VectorMethodCache.GetElement(simdBits, clrType);
            OpCode ldT = LdindFloat(elemSize);

            var locI = il.DeclareLocal(typeof(long));
            var locRes = il.DeclareLocal(clrType);
            var locLim = il.DeclareLocal(typeof(long));
            var locN2 = il.DeclareLocal(typeof(long));
            var acc = new LocalBuilder[numAccVecs];
            for (int v = 0; v < numAccVecs; v++) acc[v] = il.DeclareLocal(vecType);
            // up to 8 scalar accumulators for the strided block
            var r = new LocalBuilder[8];
            for (int k = 0; k < 8; k++) r[k] = il.DeclareLocal(clrType);

            var LBASE = il.DefineLabel();
            var LLEAF = il.DefineLabel();
            var LSCALAR = il.DefineLabel();
            var LREC = il.DefineLabel();

            // ---- pointer math: push (a + elemIndex*stride*elemSize) -----------------
            // push (a + byteOffset) where byteOffset = idxExpr * elemSize, idxExpr already in elements×stride.
            void PtrAtElems(Action pushElemIndex, bool applyStride)
            {
                il.Emit(OpCodes.Ldarg_0);            // a
                pushElemIndex();                     // element index (long)
                if (applyStride) { il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Mul); } // * stride
                il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            // dispatch: n<8 → BASE ; n<=128 → LEAF ; else → REC
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Blt, LBASE);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 128L); il.Emit(OpCodes.Ble, LLEAF);
            il.Emit(OpCodes.Br, LREC);

            // ---- BASE: res=-0; for i in [0,n): res += a[i*stride] -------------------
            il.MarkLabel(LBASE);
            EmitNegZero(il, elemSize); il.Emit(OpCodes.Stloc, locRes);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            var bL = il.DefineLabel(); var bE = il.DefineLabel();
            il.MarkLabel(bL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Bge, bE);
            il.Emit(OpCodes.Ldloc, locRes);
            PtrAtElems(() => il.Emit(OpCodes.Ldloc, locI), applyStride: true); il.Emit(ldT);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, bL); il.MarkLabel(bE);
            il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Ret);

            // ---- LEAF: if stride!=1 → SCALAR else SIMD -----------------------------
            il.MarkLabel(LLEAF);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Bne_Un, LSCALAR);

            // SIMD block (contiguous). acc[v] = Load(a + v*laneCount); seeds a[0..7].
            for (int v = 0; v < numAccVecs; v++)
            {
                int seedElem = v * laneCount;
                il.Emit(OpCodes.Ldarg_0);
                if (seedElem != 0) { il.Emit(OpCodes.Ldc_I8, (long)seedElem * elemSize); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); }
                il.Emit(OpCodes.Call, mLoad); il.Emit(OpCodes.Stloc, acc[v]);
            }
            // lim = n - (n%8)
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locLim);
            // i = 8
            il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Stloc, locI);
            var lL = il.DefineLabel(); var lE = il.DefineLabel();
            il.MarkLabel(lL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locLim); il.Emit(OpCodes.Bge, lE);
            for (int v = 0; v < numAccVecs; v++)
            {
                int laneOff = v * laneCount;
                il.Emit(OpCodes.Ldloc, acc[v]);
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locI);
                if (laneOff != 0) { il.Emit(OpCodes.Ldc_I8, (long)laneOff); il.Emit(OpCodes.Add); }
                il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Call, mLoad);
                il.Emit(OpCodes.Call, mAdd); il.Emit(OpCodes.Stloc, acc[v]);
            }
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lL); il.MarkLabel(lE);
            // combine via GetElement: r[k] = acc[k/laneCount] lane (k%laneCount).
            // res = ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
            void LdR(int k) { il.Emit(OpCodes.Ldloc, acc[k / laneCount]); il.Emit(OpCodes.Ldc_I4, k % laneCount); il.Emit(OpCodes.Call, mGet); }
            LdR(0); LdR(1); il.Emit(OpCodes.Add); LdR(2); LdR(3); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
            LdR(4); LdR(5); il.Emit(OpCodes.Add); LdR(6); LdR(7); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
            // tail: for(; i<n; i++) res += a[i]
            EmitContiguousTail(il, locI, locRes, elemSize, ldT);
            il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Ret);

            // ---- SCALAR 8-accumulator block (stride != 1) --------------------------
            il.MarkLabel(LSCALAR);
            for (int k = 0; k < 8; k++)
            {
                int kk = k;
                PtrAtElems(() => il.Emit(OpCodes.Ldc_I8, (long)kk), applyStride: true); il.Emit(ldT); il.Emit(OpCodes.Stloc, r[k]);
            }
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locLim);
            il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Stloc, locI);
            var sL = il.DefineLabel(); var sE = il.DefineLabel();
            il.MarkLabel(sL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locLim); il.Emit(OpCodes.Bge, sE);
            for (int k = 0; k < 8; k++)
            {
                int kk = k;
                il.Emit(OpCodes.Ldloc, r[k]);
                PtrAtElems(() => { il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)kk); il.Emit(OpCodes.Add); }, applyStride: true);
                il.Emit(ldT); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, r[k]);
            }
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, sL); il.MarkLabel(sE);
            il.Emit(OpCodes.Ldloc, r[0]); il.Emit(OpCodes.Ldloc, r[1]); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, r[2]); il.Emit(OpCodes.Ldloc, r[3]); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, r[4]); il.Emit(OpCodes.Ldloc, r[5]); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, r[6]); il.Emit(OpCodes.Ldloc, r[7]); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
            // strided tail: for(; i<n; i++) res += a[i*stride]
            var stL = il.DefineLabel(); var stE = il.DefineLabel();
            il.MarkLabel(stL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Bge, stE);
            il.Emit(OpCodes.Ldloc, locRes);
            PtrAtElems(() => il.Emit(OpCodes.Ldloc, locI), applyStride: true); il.Emit(ldT); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, stL); il.MarkLabel(stE);
            il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Ret);

            // ---- REC: n2 = n/2; n2 -= n2%8; fold(a,n2,s) + fold(a+n2*s, n-n2, s) ---
            il.MarkLabel(LREC);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Div); il.Emit(OpCodes.Stloc, locN2);
            il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locN2);
            // left = fold(a, n2, stride)
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Call, dm);
            // right = fold(a + n2*stride*elemSize, n-n2, stride)
            PtrAtElems(() => il.Emit(OpCodes.Ldloc, locN2), applyStride: true);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Call, dm);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Ret);

            return dm;
        }

        // tail: for(; i<n; i++) res += a[i]  (contiguous, stride==1; n in arg1)
        private static void EmitContiguousTail(ILGenerator il, LocalBuilder locI, LocalBuilder locRes, int elemSize, OpCode ldT)
        {
            var tL = il.DefineLabel(); var tE = il.DefineLabel();
            il.MarkLabel(tL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Bge, tE);
            il.Emit(OpCodes.Ldloc, locRes);
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(ldT);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, tL); il.MarkLabel(tE);
        }

        /// <summary>
        /// Emit the NpyInnerLoopFunc kernel. PINNED (outStride==0): fold the whole stripe
        /// with <paramref name="fold"/> and accumulate into the seeded slot
        /// (<c>*out += fold(in, count, inS/sz)</c>). SLAB (outStride!=0): stream
        /// <c>out[c] += in[c]</c> (width-native vector add + scalar tail for the
        /// contiguous case, scalar strided otherwise).
        /// </summary>
        private static NpyInnerLoopFunc EmitPairwiseSumKernel(NPTypeCode tc, Type clrType, int elemSize, DynamicMethod fold)
        {
            Type pVV = typeof(void).MakePointerType().MakePointerType(); // void**
            Type pL = typeof(long).MakePointerType();                    // long*
            Type pV = typeof(void).MakePointerType();                    // void*
            var dm = new DynamicMethod($"PwSum_{tc}", typeof(void), new[] { pVV, pL, typeof(long), pV },
                typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            OpCode ldT = LdindFloat(elemSize), stT = StindFloat(elemSize);

            var locIn = il.DeclareLocal(pV);
            var locOut = il.DeclareLocal(pV);
            var locInS = il.DeclareLocal(typeof(long));
            var locOutS = il.DeclareLocal(typeof(long));
            var locI = il.DeclareLocal(typeof(long));

            // in = dataptrs[0]; out = dataptrs[1]; inS = strides[0]; outS = strides[1];
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, locIn);
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldc_I8, (long)IntPtr.Size); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, locOut);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, locInS);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, locOutS);

            var LSLAB = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locOutS); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Bne_Un, LSLAB);

            // PINNED: *out = *out + fold(in, count, inS/sz)
            il.Emit(OpCodes.Ldloc, locOut);                 // store address
            il.Emit(OpCodes.Ldloc, locOut); il.Emit(ldT);   // *out
            il.Emit(OpCodes.Ldloc, locIn);                  // a
            il.Emit(OpCodes.Ldarg_2);                       // n = count
            il.Emit(OpCodes.Ldloc, locInS); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Div); // stride = inS/sz
            il.Emit(OpCodes.Call, fold);
            il.Emit(OpCodes.Add);
            il.Emit(stT);
            il.Emit(OpCodes.Ret);

            // SLAB: out[c] += in[c]
            il.MarkLabel(LSLAB);
            var LSTRIDED = il.DefineLabel();
            // contiguous? inS==sz && outS==sz
            il.Emit(OpCodes.Ldloc, locInS); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Bne_Un, LSTRIDED);
            il.Emit(OpCodes.Ldloc, locOutS); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Bne_Un, LSTRIDED);

            int laneSlab = DirectILKernelGenerator.GetVectorCount(tc);
            int slabBits = DirectILKernelGenerator.VectorBits;
            MethodInfo mAddSlab = VectorMethodCache.BinaryX86(slabBits, "Add", clrType)
                                  ?? VectorMethodCache.Generic(slabBits, "Add", clrType, paramCount: 2);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            if (DirectILKernelGenerator.VectorBits >= 128 && laneSlab > 1)
            {
                // push (base + (i + offElems)*sz)
                void PushPtr(LocalBuilder bas, int offElems)
                {
                    il.Emit(OpCodes.Ldloc, bas); il.Emit(OpCodes.Ldloc, locI);
                    if (offElems != 0) { il.Emit(OpCodes.Ldc_I8, (long)offElems); il.Emit(OpCodes.Add); }
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                }
                // out[i+off] += in[i+off]  (one vector; EmitVectorStore wants [V, ptr])
                void Step(int offElems)
                {
                    PushPtr(locOut, offElems); DirectILKernelGenerator.EmitVectorLoad(il, tc);
                    PushPtr(locIn, offElems); DirectILKernelGenerator.EmitVectorLoad(il, tc);
                    il.Emit(OpCodes.Call, mAddSlab);
                    PushPtr(locOut, offElems); DirectILKernelGenerator.EmitVectorStore(il, tc);
                }
                // 4x-unrolled streaming add (matches the generic kernel's unroll factor)
                var u4L = il.DefineLabel(); var u4E = il.DefineLabel();
                il.MarkLabel(u4L);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)laneSlab * 4); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, u4E);
                Step(0); Step(laneSlab); Step(laneSlab * 2); Step(laneSlab * 3);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)laneSlab * 4); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, u4L); il.MarkLabel(u4E);
                // 1x remainder
                var u1L = il.DefineLabel(); var u1E = il.DefineLabel();
                il.MarkLabel(u1L);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)laneSlab); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, u1E);
                Step(0);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)laneSlab); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, u1L); il.MarkLabel(u1E);
            }
            // scalar tail: for(; i<count; i++) *(out+i*sz) += *(in+i*sz)
            var tcL = il.DefineLabel(); var tcE = il.DefineLabel();
            il.MarkLabel(tcL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, tcE);
            il.Emit(OpCodes.Ldloc, locOut); il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Dup); il.Emit(ldT);
            il.Emit(OpCodes.Ldloc, locIn); il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(ldT);
            il.Emit(OpCodes.Add); il.Emit(stT);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, tcL); il.MarkLabel(tcE);
            il.Emit(OpCodes.Ret);

            // SLAB strided: for(k=0;k<count;k++) *(out+k*outS) += *(in+k*inS)
            il.MarkLabel(LSTRIDED);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            var ksL = il.DefineLabel(); var ksE = il.DefineLabel();
            il.MarkLabel(ksL);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, ksE);
            il.Emit(OpCodes.Ldloc, locOut); il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locOutS); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Dup); il.Emit(ldT);
            il.Emit(OpCodes.Ldloc, locIn); il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locInS); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(ldT);
            il.Emit(OpCodes.Add); il.Emit(stT);
            il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, ksL); il.MarkLabel(ksE);
            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<NpyInnerLoopFunc>();
        }
    }
}
