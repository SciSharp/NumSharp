using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// DirectILKernelGenerator.Select.cs - IL-emitted FUSED np.select kernels
// =============================================================================
//
// MOTIVATION:
//   NumPy's numpy.select is a composition: np.full(default) followed by one
//   masked np.copyto per condition, in REVERSE order so the first matching
//   condition wins. That is (n+1) full passes over the whole result — every
//   masked copyto re-reads AND re-writes the result to preserve its masked-off
//   lanes, so the result buffer alone sees (2n+1) touches. For 8-byte dtypes at
//   scale (int64/float64) the op is memory-bandwidth bound and that redundant
//   result traffic is the whole cost.
//
//   This kernel fuses the entire select into ONE pass: for each output element
//   it seeds an accumulator with the default, then walks the conditions from
//   last to first, ConditionalSelect-ing each choice in where its condition is
//   true. The smallest-index true condition writes last and therefore wins —
//   identical first-match semantics to NumPy's reverse copyto, but each
//   condition and choice is read once and the result is written exactly once.
//
//   np.select is, elementwise, a chain of np.where — so this reuses np.where's
//   exact SIMD building blocks: EmitInlineMaskCreation (bool -> lane mask) and
//   Vector.ConditionalSelect, 4x-unrolled with four independent accumulators.
//
// KERNEL:
//   SelectFusedKernel(bool** conds, void** choices, void* defPtr, void* result, long count)
//     result[i] = choice_k[i]  for the smallest k with conds[k][i] != 0
//                 default[i]    if no condition is true
//
//   The kernel is generated per (dtype, n, defaultIsScalar) — n and the
//   default's scalar-ness are baked into the IL so the condition chain fully
//   unrolls with constant offsets and the scalar default is broadcast once
//   into a Vector via CreateBroadcast (like np.where's scalar fast path).
//
// SCOPE:
//   Emitted only for the contiguous, no-cast, full-size-array-choice case (see
//   np.select's TrySelectFused gate) — the case NumPy leaves slow. Every other
//   shape (broadcast/strided/cast/scalar-choice) stays on the copyto
//   composition, which already outruns NumPy comfortably.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused np.select kernel. <paramref name="conds"/> is <c>n</c> boolean base pointers,
    ///     <paramref name="choices"/> is <c>n</c> choice base pointers (all the result dtype,
    ///     C-contiguous, full result size). <paramref name="defPtr"/> is the default — a single
    ///     element when the kernel was generated for a scalar default, otherwise a full-size
    ///     array. <paramref name="result"/> is C-contiguous of <paramref name="count"/> elements.
    /// </summary>
    public unsafe delegate void SelectFusedKernel(bool** conds, void** choices, void* defPtr, void* result, long count);

    public static partial class DirectILKernelGenerator
    {
        #region Cache

        private readonly struct SelectKernelKey : IEquatable<SelectKernelKey>
        {
            public readonly NPTypeCode Dtype;
            public readonly int N;
            public readonly bool DefScalar;

            public SelectKernelKey(NPTypeCode dtype, int n, bool defScalar)
            {
                Dtype = dtype;
                N = n;
                DefScalar = defScalar;
            }

            public bool Equals(SelectKernelKey other) => Dtype == other.Dtype && N == other.N && DefScalar == other.DefScalar;
            public override bool Equals(object obj) => obj is SelectKernelKey k && Equals(k);
            public override int GetHashCode() => HashCode.Combine((int)Dtype, N, DefScalar);
        }

        private static readonly ConcurrentDictionary<SelectKernelKey, SelectFusedKernel> _selectKernelCache = new();

        #endregion

        #region Public API

        /// <summary>
        ///     Get or generate a fused select kernel for (<paramref name="dtype"/>, <paramref name="n"/>
        ///     conditions, <paramref name="defScalar"/>). Returns null if IL generation is disabled or
        ///     the dtype is not SIMD-eligible for this kernel (the caller then uses the composition).
        /// </summary>
        public static SelectFusedKernel GetSelectKernel(NPTypeCode dtype, int n, bool defScalar)
        {
            if (!Enabled)
                return null;

            var key = new SelectKernelKey(dtype, n, defScalar);
            if (_selectKernelCache.TryGetValue(key, out var cached))
                return cached;

            try
            {
                var elemType = SelectElemType(dtype);
                if (elemType == null)
                    return null;

                var kernel = GenerateSelectKernelIL(elemType, dtype, n, defScalar);
                if (kernel == null)
                    return null;

                return _selectKernelCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetSelectKernel({dtype},{n},{defScalar}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     True when the fused select kernel supports <paramref name="dtype"/> — the 1/2/4/8-byte
        ///     numeric dtypes. Everything else stays on the copyto composition.
        /// </summary>
        public static bool SelectKernelSupportsDtype(NPTypeCode dtype) => SelectElemType(dtype) != null;

        /// <summary>
        ///     The 1/2/4/8-byte numeric CLR types this kernel vectorises. Bool/Char/Half/Decimal/
        ///     Complex are deliberately absent — they stay on the copyto composition.
        /// </summary>
        private static Type SelectElemType(NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Byte => typeof(byte),
            NPTypeCode.SByte => typeof(sbyte),
            NPTypeCode.Int16 => typeof(short),
            NPTypeCode.UInt16 => typeof(ushort),
            NPTypeCode.Int32 => typeof(int),
            NPTypeCode.UInt32 => typeof(uint),
            NPTypeCode.Int64 => typeof(long),
            NPTypeCode.UInt64 => typeof(ulong),
            NPTypeCode.Single => typeof(float),
            NPTypeCode.Double => typeof(double),
            _ => null,
        };

        #endregion

        #region SIMD eligibility (mirrors the np.where scalar kernels)

        // V256 needs Avx2 for the byte-lane sign-extend in EmitInlineMaskCreation (dtypes > 1 byte);
        // V128 needs Sse41. 1-byte dtypes skip the x86 requirement.
        private static (bool emitSimd, bool useV256) SelectSimdMode(int elementSize)
        {
            bool canSimd = elementSize == 1 || elementSize == 2 || elementSize == 4 || elementSize == 8;
            bool needsX86 = elementSize > 1;
            bool useV256 = VectorBits >= 256 && (!needsX86 || Avx2.IsSupported);
            bool useV128 = !useV256 && VectorBits >= 128 && (!needsX86 || Sse41.IsSupported);
            return (canSimd && (useV256 || useV128), useV256);
        }

        #endregion

        #region IL Emission

        private static SelectFusedKernel GenerateSelectKernelIL(Type elemType, NPTypeCode typeCode, int n, bool defScalar)
        {
            int elementSize = InfoOf.GetSize(typeCode);
            var (emitSimd, useV256) = SelectSimdMode(elementSize);

            var dm = new DynamicMethod(
                name: $"IL_Select_{elemType.Name}_n{n}_{(defScalar ? "s" : "a")}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool**), typeof(void**), typeof(void*), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var byteP = typeof(byte*);

            // Hoist the condition and choice base pointers into locals once (pointer-prologue
            // hoisting): conds[k] = *(conds + k), choices[k] = *(choices + k).
            var locConds = new LocalBuilder[n];
            var locChoices = new LocalBuilder[n];
            for (int k = 0; k < n; k++)
            {
                locConds[k] = il.DeclareLocal(byteP);
                EmitLoadPointerFromArray(il, argIndex: 0, k);
                il.Emit(OpCodes.Stloc, locConds[k]);
            }
            for (int k = 0; k < n; k++)
            {
                locChoices[k] = il.DeclareLocal(byteP);
                EmitLoadPointerFromArray(il, argIndex: 1, k);
                il.Emit(OpCodes.Stloc, locChoices[k]);
            }

            // Full-size array default: hoist its base pointer. Scalar default: read once below.
            LocalBuilder locDef = null;
            if (!defScalar)
            {
                locDef = il.DeclareLocal(byteP);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Stloc, locDef);
            }

            var locI = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (emitSimd)
            {
                int simdBits = useV256 ? 256 : 128;
                var vecT = VectorMethodCache.V(simdBits, elemType);

                // Scalar default → broadcast once into a Vector (np.where's scalar-hoist trick).
                LocalBuilder locDefVec = null;
                if (defScalar)
                {
                    locDefVec = il.DeclareLocal(vecT);
                    il.Emit(OpCodes.Ldarg_2);
                    EmitLoadIndirect(il, typeCode);
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateBroadcast(simdBits, elemType));
                    il.Emit(OpCodes.Stloc, locDefVec);
                }

                // Four independent accumulators so the 4x-unrolled bodies stay dependency-free
                // and the CPU overlaps them (the select chain within one body is serial).
                var accs = new LocalBuilder[4];
                for (int u = 0; u < 4; u++) accs[u] = il.DeclareLocal(vecT);

                EmitSelectSimdLoop(il, useV256, elemType, n, defScalar, locConds, locChoices, locDef, locDefVec, accs, locI, elementSize);
            }

            // Scalar tail — also the whole loop on non-SIMD platforms.
            EmitSelectScalarTail(il, typeCode, n, defScalar, locConds, locChoices, locDef, locI, elementSize);

            il.Emit(OpCodes.Ret);
            return (SelectFusedKernel)dm.CreateDelegate(typeof(SelectFusedKernel));
        }

        /// <summary>Push <c>*(argIndex_ptr_array + k)</c> — the k-th base pointer.</summary>
        private static void EmitLoadPointerFromArray(ILGenerator il, int argIndex, int k)
        {
            il.Emit(argIndex == 0 ? OpCodes.Ldarg_0 : OpCodes.Ldarg_1);
            if (k != 0)
            {
                il.Emit(OpCodes.Ldc_I8, (long)(k * IntPtr.Size));
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldind_I);
        }

        private static void EmitSelectSimdLoop(
            ILGenerator il, bool useV256, Type elemType, int n, bool defScalar,
            LocalBuilder[] locConds, LocalBuilder[] locChoices, LocalBuilder locDef, LocalBuilder locDefVec,
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
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // 4x unrolled SIMD loop
            il.MarkLabel(lblUnrollHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (long u = 0; u < unrollFactor; u++)
                EmitSelectSimdBody(il, useV256, elemType, n, defScalar, locConds, locChoices, locDef, locDefVec, accs[u], locI, elementSize, vectorCount * u);

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

            EmitSelectSimdBody(il, useV256, elemType, n, defScalar, locConds, locChoices, locDef, locDefVec, accs[0], locI, elementSize, 0);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblVecHead);
            il.MarkLabel(lblVecEnd);
        }

        /// <summary>
        ///     One SIMD lane-group: acc = default; for k=n-1..0: acc = ConditionalSelect(mask_k, choice_k, acc);
        ///     store acc. <paramref name="laneOffset"/> shifts the base index by that many ELEMENTS so the
        ///     unrolled bodies address distinct vectors without advancing <c>locI</c> between them.
        /// </summary>
        private static void EmitSelectSimdBody(
            ILGenerator il, bool useV256, Type elemType, int n, bool defScalar,
            LocalBuilder[] locConds, LocalBuilder[] locChoices, LocalBuilder locDef, LocalBuilder locDefVec,
            LocalBuilder acc, LocalBuilder locI, long elementSize, long laneOffset)
        {
            int simdBits = useV256 ? 256 : 128;
            var loadM = VectorMethodCache.Load(simdBits, elemType);
            var storeM = VectorMethodCache.Store(simdBits, elemType);
            var selectM = VectorMethodCache.ConditionalSelect(simdBits, elemType);

            // seed acc = default
            if (defScalar)
            {
                il.Emit(OpCodes.Ldloc, locDefVec);
            }
            else
            {
                EmitElementAddress(il, locDef, locI, laneOffset, elementSize);
                il.Emit(OpCodes.Call, loadM);
            }
            il.Emit(OpCodes.Stloc, acc);

            // reverse chain — last condition first so the FIRST true condition wins
            for (int k = n - 1; k >= 0; k--)
            {
                // mask = expand(conds[k] + (i + laneOffset))   [bool is 1 byte]
                il.Emit(OpCodes.Ldloc, locConds[k]);
                il.Emit(OpCodes.Ldloc, locI);
                if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                if (useV256) EmitInlineMaskCreationV256(il, (int)elementSize);
                else EmitInlineMaskCreationV128(il, (int)elementSize);

                // choice_k vector
                EmitElementAddress(il, locChoices[k], locI, laneOffset, elementSize);
                il.Emit(OpCodes.Call, loadM);

                // acc  ->  ConditionalSelect(mask, choice_k, acc)
                il.Emit(OpCodes.Ldloc, acc);
                il.Emit(OpCodes.Call, selectM);
                il.Emit(OpCodes.Stloc, acc);
            }

            // result[i(+laneOffset)] = acc
            il.Emit(OpCodes.Ldloc, acc);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeM);
        }

        /// <summary>Push <c>base + (i + laneOffset) * elementSize</c> as a native pointer.</summary>
        private static void EmitElementAddress(ILGenerator il, LocalBuilder locBase, LocalBuilder locI, long laneOffset, long elementSize)
        {
            il.Emit(OpCodes.Ldloc, locBase);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        private static void EmitSelectScalarTail(
            ILGenerator il, NPTypeCode typeCode, int n, bool defScalar,
            LocalBuilder[] locConds, LocalBuilder[] locChoices, LocalBuilder locDef, LocalBuilder locI, long elementSize)
        {
            var elemType = SelectElemType(typeCode);
            var locR = il.DeclareLocal(elemType);
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblEnd);

            // r = default
            if (defScalar)
            {
                il.Emit(OpCodes.Ldarg_2);
                EmitLoadIndirect(il, typeCode);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locDef);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, typeCode);
            }
            il.Emit(OpCodes.Stloc, locR);

            // for k=n-1..0: if (conds[k][i]) r = choices[k][i]
            for (int k = n - 1; k >= 0; k--)
            {
                var lblSkip = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, locConds[k]);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblSkip);

                il.Emit(OpCodes.Ldloc, locChoices[k]);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, typeCode);
                il.Emit(OpCodes.Stloc, locR);

                il.MarkLabel(lblSkip);
            }

            // result[i] = r
            il.Emit(OpCodes.Ldarg_3);
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
