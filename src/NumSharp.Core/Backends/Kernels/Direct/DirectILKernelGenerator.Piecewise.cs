using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using NumSharp.Utilities;

// =============================================================================
// DirectILKernelGenerator.Piecewise.cs — IL-emitted FUSED np.piecewise kernel
//                                        (the SCALAR-funclist fast path)
// =============================================================================
//
// MOTIVATION:
//   numpy.piecewise is a composition: zeros_like(x) then one boolean-mask
//   assignment `y[cond_k] = scalar_k` per condition, in FORWARD order so the LAST
//   true condition wins. In NumSharp each `y[cond] = scalar` runs
//   BooleanMaskSet — a mask popcount, a value materialize, and a masked scatter,
//   each allocating — so the small-N common case (the signum `[-1, 1]` pattern)
//   was allocation-bound and slower than NumPy (one boolean assignment measured
//   ~12 us at N=1000, vs a fused select kernel's ~2.8 us for the WHOLE op).
//
//   This kernel fuses the entire scalar-funclist piecewise into ONE pass: for
//   each output element it seeds an accumulator with the SEED (the default
//   function where one was given, else 0 — piecewise's zeros_like default) and
//   walks the conditions FORWARD, ConditionalSelect-ing each condition's scalar in
//   where its mask is true. The largest-index true condition writes last and
//   therefore wins — identical last-match semantics to NumPy's forward assignment
//   loop, but each condition is read once and the result is written exactly once,
//   with no per-condition allocation and without materialising the ~any(condlist)
//   default mask.
//
//   piecewise is, elementwise, the same np.where chain as np.select, so this
//   reuses np.select's SIMD building blocks (EmitInlineMaskCreation, the
//   4x-unrolled ConditionalSelect loop). The one structural difference from the
//   select kernel: the choices are SCALARS (each broadcast once into a vector,
//   like np.where's scalar fast path) rather than full-size arrays, and the chain
//   runs FORWARD (last-match) rather than reverse (first-match).
//
// KERNEL:
//   PiecewiseScalarKernel(bool** conds, void* funcVals, void* result, long count)
//     result[i] = funcVals[k]  for the LARGEST k in [0,n) with conds[k][i] != 0
//                 funcVals[n]   (the SEED) if no condition is true
//
//   `conds` is `n` boolean base pointers — the n REAL conditions (the default is
//   NOT appended as a condition; it becomes the seed instead, so the kernel never
//   needs the ~any(condlist) "otherwise" mask). `funcVals` is `n + 1` scalars
//   already cast to the result dtype: funcVals[0..n-1] are the per-condition
//   scalars, funcVals[n] is the seed — the default function where one was given,
//   else 0 (piecewise's zeros_like default). `result` is C-contiguous of `count`
//   elements. Because the kernel writes EVERY element (seeded from funcVals[n]),
//   the caller allocates the result UNINITIALIZED (no zeros_like fill needed).
//
//   The kernel is generated per (dtype, n) where n is the CONDITION count: n is
//   baked into the IL so the condition chain fully unrolls and each scalar (and
//   the seed) is broadcast once into a Vector.
//
// SCOPE:
//   Emitted only for the contiguous, all-C#-scalar-funclist, SIMD-eligible-dtype
//   case (see np.piecewise's TryPiecewiseScalarFused gate). Callable funcs,
//   NDArray funcs, non-contiguous / broadcast conditions, and the
//   bool/char/Half/Decimal/Complex dtypes stay on the zeros_like + BooleanMaskSet
//   composition (which already outruns NumPy at 100K/10M).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused np.piecewise (scalar-funclist) kernel. <paramref name="conds"/> is <c>n</c> boolean
    ///     base pointers (the real conditions), <paramref name="funcVals"/> is <c>n + 1</c> scalar
    ///     values of the result dtype (funcVals[0..n-1] the per-condition scalars, funcVals[n] the seed
    ///     — the default, or 0), and <paramref name="result"/> is C-contiguous of <paramref name="count"/>
    ///     elements. result[i] is funcVals[k] for the LARGEST k in [0,n) whose cond is true, else the seed.
    /// </summary>
    public unsafe delegate void PiecewiseScalarKernel(bool** conds, void* funcVals, void* result, long count);

    public static partial class DirectILKernelGenerator
    {
        #region Cache

        private readonly struct PiecewiseKernelKey : IEquatable<PiecewiseKernelKey>
        {
            public readonly NPTypeCode Dtype;
            public readonly int N;

            public PiecewiseKernelKey(NPTypeCode dtype, int n)
            {
                Dtype = dtype;
                N = n;
            }

            public bool Equals(PiecewiseKernelKey other) => Dtype == other.Dtype && N == other.N;
            public override bool Equals(object obj) => obj is PiecewiseKernelKey k && Equals(k);
            public override int GetHashCode() => HashCode.Combine((int)Dtype, N);
        }

        private static readonly ConcurrentDictionary<PiecewiseKernelKey, PiecewiseScalarKernel> _piecewiseKernelCache = new();

        #endregion

        #region Public API

        /// <summary>
        ///     True when the fused piecewise scalar kernel supports <paramref name="dtype"/> — the
        ///     1/2/4/8-byte numeric dtypes (shared with the select kernel). Bool/Char/Half/Decimal/
        ///     Complex stay on the composition.
        /// </summary>
        public static bool PiecewiseScalarKernelSupportsDtype(NPTypeCode dtype) => SelectElemType(dtype) != null;

        /// <summary>
        ///     Get or generate a fused piecewise scalar kernel for (<paramref name="dtype"/>,
        ///     <paramref name="n"/> conditions). Returns null if IL generation is disabled or
        ///     the dtype is not SIMD-eligible for this kernel (the caller then uses the composition).
        /// </summary>
        public static PiecewiseScalarKernel GetPiecewiseScalarKernel(NPTypeCode dtype, int n)
        {
            if (!Enabled)
                return null;

            var key = new PiecewiseKernelKey(dtype, n);
            if (_piecewiseKernelCache.TryGetValue(key, out var cached))
                return cached;

            try
            {
                var elemType = SelectElemType(dtype);
                if (elemType == null)
                    return null;

                var kernel = GeneratePiecewiseKernelIL(elemType, dtype, n);
                if (kernel == null)
                    return null;

                return _piecewiseKernelCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPiecewiseScalarKernel({dtype},{n}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region IL Emission

        private static PiecewiseScalarKernel GeneratePiecewiseKernelIL(Type elemType, NPTypeCode typeCode, int n)
        {
            int elementSize = InfoOf.GetSize(typeCode);
            var (emitSimd, useV256) = SelectSimdMode(elementSize);

            var dm = new DynamicMethod(
                name: $"IL_Piecewise_{elemType.Name}_n{n}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool**), typeof(void*), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var byteP = typeof(byte*);

            // Hoist the n condition base pointers into locals: conds[k] = *(conds + k).
            var locConds = new LocalBuilder[n];
            for (int k = 0; k < n; k++)
            {
                locConds[k] = il.DeclareLocal(byteP);
                EmitLoadPointerFromArray(il, argIndex: 0, k);   // reused from Select.cs
                il.Emit(OpCodes.Stloc, locConds[k]);
            }

            var locI = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (emitSimd)
            {
                int simdBits = useV256 ? 256 : 128;
                var vecT = VectorMethodCache.V(simdBits, elemType);

                // Broadcast each scalar funcVals[k] into a vector ONCE (np.where's scalar-hoist trick),
                // and the SEED (funcVals[n] — the default, or 0 when none) into its own vector.
                var locSeedVec = il.DeclareLocal(vecT);
                EmitLoadFuncVal(il, argIndex: 1, n, elementSize, typeCode);   // funcVals[n] = seed
                il.Emit(OpCodes.Call, VectorMethodCache.CreateBroadcast(simdBits, elemType));
                il.Emit(OpCodes.Stloc, locSeedVec);

                var funcVecs = new LocalBuilder[n];
                for (int k = 0; k < n; k++)
                {
                    funcVecs[k] = il.DeclareLocal(vecT);
                    EmitLoadFuncVal(il, argIndex: 1, k, elementSize, typeCode);   // funcVals[k]
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateBroadcast(simdBits, elemType));
                    il.Emit(OpCodes.Stloc, funcVecs[k]);
                }

                // Four independent accumulators so the 4x-unrolled bodies stay dependency-free.
                var accs = new LocalBuilder[4];
                for (int u = 0; u < 4; u++) accs[u] = il.DeclareLocal(vecT);

                EmitPiecewiseSimdLoop(il, useV256, elemType, n, locConds, funcVecs, locSeedVec, accs, locI, elementSize);
            }

            // Scalar tail — also the whole loop on non-SIMD platforms.
            EmitPiecewiseScalarTail(il, typeCode, n, locConds, locI, elementSize);

            il.Emit(OpCodes.Ret);
            return (PiecewiseScalarKernel)dm.CreateDelegate(typeof(PiecewiseScalarKernel));
        }

        /// <summary>Push <c>*(funcVals + k * elementSize)</c> — the k-th scalar value, as its element type.</summary>
        private static void EmitLoadFuncVal(ILGenerator il, int argIndex, int k, int elementSize, NPTypeCode typeCode)
        {
            il.Emit(OpCodes.Ldarg_1);                       // funcVals
            if (k != 0)
            {
                il.Emit(OpCodes.Ldc_I8, (long)(k * elementSize));
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            EmitLoadIndirect(il, typeCode);
        }

        private static void EmitPiecewiseSimdLoop(
            ILGenerator il, bool useV256, Type elemType, int n,
            LocalBuilder[] locConds, LocalBuilder[] funcVecs, LocalBuilder locSeedVec,
            LocalBuilder[] accs, LocalBuilder locI, long elementSize)
        {
            long vectorCount = useV256 ? (32 / elementSize) : (16 / elementSize);
            const long unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            var lblUnrollHead = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblVecHead = il.DefineLabel();
            var lblVecEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep ; vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // 4x unrolled SIMD loop
            il.MarkLabel(lblUnrollHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (long u = 0; u < unrollFactor; u++)
                EmitPiecewiseSimdBody(il, useV256, elemType, n, locConds, funcVecs, locSeedVec, accs[u], locI, elementSize, vectorCount * u);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnrollHead);
            il.MarkLabel(lblUnrollEnd);

            // 1-vector remainder loop
            il.MarkLabel(lblVecHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVecEnd);

            EmitPiecewiseSimdBody(il, useV256, elemType, n, locConds, funcVecs, locSeedVec, accs[0], locI, elementSize, 0);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblVecHead);
            il.MarkLabel(lblVecEnd);
        }

        /// <summary>
        ///     One SIMD lane-group: acc = 0; for k=0..n-1: acc = ConditionalSelect(mask_k, funcVec_k, acc);
        ///     store acc. The FORWARD chain makes the LARGEST-index true condition win (last-match), the
        ///     opposite of the select kernel's reverse chain. <paramref name="laneOffset"/> shifts the base
        ///     index by that many ELEMENTS so the unrolled bodies address distinct vectors.
        /// </summary>
        private static void EmitPiecewiseSimdBody(
            ILGenerator il, bool useV256, Type elemType, int n,
            LocalBuilder[] locConds, LocalBuilder[] funcVecs, LocalBuilder locSeedVec,
            LocalBuilder acc, LocalBuilder locI, long elementSize, long laneOffset)
        {
            int simdBits = useV256 ? 256 : 128;
            var storeM = VectorMethodCache.Store(simdBits, elemType);
            var selectM = VectorMethodCache.ConditionalSelect(simdBits, elemType);

            // seed acc = seed (funcVals[n])
            il.Emit(OpCodes.Ldloc, locSeedVec);
            il.Emit(OpCodes.Stloc, acc);

            // forward chain — first condition first, so the LAST true condition wins
            for (int k = 0; k < n; k++)
            {
                // mask = expand(conds[k] + (i + laneOffset))   [bool is 1 byte]
                il.Emit(OpCodes.Ldloc, locConds[k]);
                il.Emit(OpCodes.Ldloc, locI);
                if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                if (useV256) EmitInlineMaskCreationV256(il, (int)elementSize);
                else EmitInlineMaskCreationV128(il, (int)elementSize);

                // funcVec_k
                il.Emit(OpCodes.Ldloc, funcVecs[k]);

                // acc -> ConditionalSelect(mask, funcVec_k, acc)
                il.Emit(OpCodes.Ldloc, acc);
                il.Emit(OpCodes.Call, selectM);
                il.Emit(OpCodes.Stloc, acc);
            }

            // result[i(+laneOffset)] = acc
            il.Emit(OpCodes.Ldloc, acc);
            il.Emit(OpCodes.Ldarg_2);                       // result
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeM);
        }

        private static void EmitPiecewiseScalarTail(
            ILGenerator il, NPTypeCode typeCode, int n, LocalBuilder[] locConds, LocalBuilder locI, long elementSize)
        {
            var elemType = SelectElemType(typeCode);
            var locR = il.DeclareLocal(elemType);
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblEnd);

            // r = seed (funcVals[n] — the default, or 0 when none)
            EmitLoadFuncVal(il, argIndex: 1, n, (int)elementSize, typeCode);
            il.Emit(OpCodes.Stloc, locR);

            // for k=0..n-1: if (conds[k][i]) r = funcVals[k]   (forward => last true wins)
            for (int k = 0; k < n; k++)
            {
                var lblSkip = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, locConds[k]);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblSkip);

                EmitLoadFuncVal(il, argIndex: 1, k, (int)elementSize, typeCode);
                il.Emit(OpCodes.Stloc, locR);

                il.MarkLabel(lblSkip);
            }

            // result[i] = r
            il.Emit(OpCodes.Ldarg_2);                       // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locR);
            EmitStoreIndirect(il, typeCode);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);
            il.MarkLabel(lblEnd);
        }

        #endregion
    }
}
