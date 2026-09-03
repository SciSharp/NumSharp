using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.InnerLoop2D.cs — 2-D coalesced-block element-wise kernel
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A (rows, w) iteration that is CONTIGUOUS on the inner axis and STRIDED on the
// outer axis (the classic `m[:, :w]` column slice, `A op col`, `view op scalar`,
// a broadcast-row operand) cannot coalesce to 1-D, so the ordinary Tier-3B route
// drives it under EXTERNAL_LOOP: the iterator calls the per-chunk kernel ONCE PER
// ROW, advancing the odometer (NDIter.ExternalLoopNext) between calls. For a
// narrow w that pays, per row:
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
// DISCOVERY.md §7 angle 2; docs/NDITER_2D_BLOCK_KERNEL.md).
//
// THE 2-D CONTRACT
// ----------------
//   void(void** dataptrs, long* innerByteStrides, long innerCount,
//        long* outerByteStrides, long outerCount)
//
// The kernel loops the OUTER axis itself: it runs the inner-mode dispatch and the
// loop-bound setup ONCE, then for each of `outerCount` rows it executes the
// 4x-unrolled SIMD + remainder + tail inner loop over `innerCount` elements
// (addressed as ptr + i*elemSize, exactly like the 1-D EmitSimdContigLoop) and
// advances each operand pointer by its own `outerByteStrides[op]`. So neither the
// odometer nor the prologue runs per row.
//
// INNER MODES (the second-generation contract, 2026-09-03)
// --------------------------------------------------------
// Every INPUT operand's inner byte stride is either elemSize (C — contiguous row)
// or 0 (S — a value broadcast along the row: `add(A, col)`, `multiply(view, 2.0)`,
// `x - mean[:, None]`); the output's is always elemSize. The kernel dispatches on
// the (mode…) pattern once per call — unary {C, S}, binary {CC, SC, CS} — and each
// pattern owns a full 2-D loop: an S operand is loaded once per row, broadcast
// with Vector.Create, and re-used by every vector op of that row (the L3-d shape
// the per-chunk kernel emits per chunk, here hoisted per row). The first-generation
// kernel required C for every operand, so those shapes fell back to the per-row
// route at ~7x the cost of the row-broadcast case the kernel already handled.
//
// SUB-VECTOR ROWS — the masked tail
// ---------------------------------
// The row remainder `innerCount % lanes` (and the WHOLE row when innerCount <
// lanes: xyz triples, `m[:, :2]`, RGB channels) used to run a scalar loop, one
// element per compare-and-branch; for a compute op such as sqrt that is scalar-
// throughput-bound and lost to NumPy (0.4-0.5x at w=2/3), because NumPy's inner
// loops finish their tail with a masked vector (npyv_load_tillz / npyv_store_till
// = vmaskmovpd/vmaskmovps on AVX2). This kernel does the same for 32/64-bit lanes
// on an AVX2 host with a 256-bit vector width: the lane mask is built ONCE per
// call (GreaterThan(Create(tail), {0,1,2,…})) and the tail of every row is one
// MaskLoad per C input → the same vectorBody → one MaskStore. Masked-off lanes
// are never touched in memory (architecturally no fault), so a row end at the
// end of a buffer is safe. Their register values are garbage the vectorBody
// computes on and MaskStore discards; for INTEGER lanes they are filled with 1
// after the load (a software Vector.Divide on a 0 lane would throw), float
// lanes stay 0 (no exception class exists). Hosts without the mask ops (V128,
// V512, non-x86) and 1/2-byte lanes keep the scalar tail.
//
// SCALAR-ONLY BLOCK
// -----------------
// Ops WITHOUT a vector body (the f64 transcendental family, power, mod, mixed-
// dtype fused converts, Half/Decimal/Complex) get the same outer-loop treatment
// around a scalar inner loop — constant-stride addressing when every operand's
// inner stride is its own element size, runtime strides otherwise (which also
// covers an S operand). The first-generation kernel required a vector body, so
// those ops paid the per-row route (exp on 4-wide rows: +43% over contiguous).
//
// SCOPE / GATING (enforced by NDIterRef.TryExecute2DElementwise before compiling)
// ------------------------------------------------------------------------------
//   • unbuffered, no where= mask, EXTERNAL_LOOP, NDim >= 2
//   • output inner axis element-contiguous; every input inner axis contiguous OR
//     broadcast (element stride 1 or 0)
//   • the trailing outer axes that every operand walks contiguously fold into one
//     (outerCount, outerStride); any leading axes left are walked by the caller
//     with a per-BLOCK odometer (a doubly-strided x[::2, :, :w] costs one kernel
//     call per plane, not per row)
// A buffered cast, a masked ufunc, or a strided (gather) inner axis keeps the
// per-chunk ForEach route unchanged.
//
// The emitted bodies are the SAME scalarBody/vectorBody emit delegates the 1-D
// kernel uses, so the arithmetic is byte-identical to the ForEach path; only the
// loop structure differs, and element-wise ops carry no cross-element state.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// A whole-2-D-block element-wise kernel: it loops the outer (strided) axis
    /// itself, running an inner SIMD loop over <paramref name="innerCount"/>
    /// elements per row and advancing each operand pointer by
    /// <paramref name="outerByteStrides"/>[op]. One call covers the whole
    /// coalesced 2-D block.
    /// </summary>
    /// <param name="dataptrs">One byte-pointer per operand (inputs then output), at the block start.</param>
    /// <param name="innerByteStrides">Per-operand inner-axis stride in BYTES: the element size (a contiguous row) or 0 (an input broadcast along the row). The output's is always the element size.</param>
    /// <param name="innerCount">Elements per row (the inner axis length).</param>
    /// <param name="outerByteStrides">Per-operand outer-axis stride in BYTES (may be 0 for a broadcast operand, or negative).</param>
    /// <param name="outerCount">Number of rows (the outer axis length).</param>
    public unsafe delegate void ND2DElementwiseKernel(
        void** dataptrs, long* innerByteStrides, long innerCount, long* outerByteStrides, long outerCount);

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
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            if (_innerLoop2DCache.TryGetValue(key, out var cached))
                return cached;

            var kernel = Generate2DElementwiseKernel(operandTypes, scalarBody, vectorBody, key.ToCacheKey());
            _innerLoop2DCache.TryAdd(key, kernel);
            return kernel;
        }

        #endregion

        #region Masked-tail support (AVX2 vmaskmov)

        /// <summary>Reflection handles for one dtype's masked tail emission.</summary>
        internal readonly struct MaskSupport
        {
            /// <summary><c>Avx{2}.MaskLoad(T*, Vector256&lt;T&gt; mask)</c>.</summary>
            public readonly MethodInfo MaskLoad;
            /// <summary><c>Avx{2}.MaskStore(T*, Vector256&lt;T&gt; mask, Vector256&lt;T&gt; source)</c>.</summary>
            public readonly MethodInfo MaskStore;
            /// <summary><c>Vector256.Create(M value)</c> broadcast for the mask lane type M (int or long).</summary>
            public readonly MethodInfo MaskBroadcast;
            /// <summary><c>Vector256.Create(M e0, …, M e_{lanes-1})</c> — the lane-index vector.</summary>
            public readonly MethodInfo MaskElements;
            /// <summary><c>Vector256.GreaterThan(Vector256&lt;M&gt;, Vector256&lt;M&gt;)</c>.</summary>
            public readonly MethodInfo MaskGreaterThan;
            /// <summary><c>Vector256.As&lt;M,T&gt;</c>, or null when the lane type IS the mask type.</summary>
            public readonly MethodInfo? MaskAs;
            /// <summary>Integer lanes only: <c>Vector256&lt;T&gt;.One</c> getter (null for float lanes).</summary>
            public readonly MethodInfo? One;
            /// <summary>Integer lanes only: <c>Vector256.AndNot(V, V)</c> (null for float lanes).</summary>
            public readonly MethodInfo? AndNot;
            /// <summary>Integer lanes only: <c>Vector256&lt;T&gt;.op_BitwiseOr</c> (null for float lanes).</summary>
            public readonly MethodInfo? Or;
            /// <summary>Lanes per vector (8 for 32-bit lanes, 4 for 64-bit).</summary>
            public readonly int Lanes;
            /// <summary>True for 32-bit lanes (mask built as int), false for 64-bit (long).</summary>
            public readonly bool Is32;

            public MaskSupport(MethodInfo maskLoad, MethodInfo maskStore, MethodInfo maskBroadcast, MethodInfo maskElements,
                MethodInfo maskGreaterThan, MethodInfo? maskAs, MethodInfo? one, MethodInfo? andNot, MethodInfo? or, int lanes, bool is32)
            {
                MaskLoad = maskLoad; MaskStore = maskStore; MaskBroadcast = maskBroadcast; MaskElements = maskElements;
                MaskGreaterThan = maskGreaterThan; MaskAs = maskAs; One = one; AndNot = andNot; Or = or; Lanes = lanes; Is32 = is32;
            }
        }

        /// <summary>
        /// Whether <paramref name="dtype"/> can finish a row with an AVX2 masked vector on this
        /// machine: the baked SIMD width must be 256 (the mask ops are Vector256) and the lane
        /// must be a 32/64-bit type — the same set NumPy's <c>npyv_load_tillz</c>/<c>store_till</c>
        /// cover. Other dtypes/widths keep the scalar tail.
        /// </summary>
        internal static bool TryGetMaskSupport(NPTypeCode dtype, out MaskSupport support)
        {
            support = default;
            if (VectorBits != 256 || !Avx2.IsSupported)
                return false;

            Type clr;
            bool integer;
            switch (dtype)
            {
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    clr = GetClrType(dtype);
                    integer = true;
                    break;
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                    clr = GetClrType(dtype);
                    integer = false;
                    break;
                default:
                    return false;
            }

            bool is32 = GetTypeSize(dtype) == 4;
            int lanes = is32 ? 8 : 4;
            Type maskElem = is32 ? typeof(int) : typeof(long);
            Type owner = integer ? typeof(Avx2) : typeof(Avx);
            Type vecT = typeof(Vector256<>).MakeGenericType(clr);

            MethodInfo maskLoad = owner.GetMethod("MaskLoad", new[] { clr.MakePointerType(), vecT })
                ?? throw new InvalidOperationException($"{owner.Name}.MaskLoad({clr.Name}*) not found");
            MethodInfo maskStore = owner.GetMethod("MaskStore", new[] { clr.MakePointerType(), vecT, vecT })
                ?? throw new InvalidOperationException($"{owner.Name}.MaskStore({clr.Name}*) not found");

            MethodInfo bcast = VectorMethodCache.CreateBroadcast(256, maskElem);
            MethodInfo elems = VectorMethodCache.CreateElements(256, maskElem);
            MethodInfo gt = VectorMethodCache.GreaterThan(256, maskElem);
            MethodInfo? asM = clr == maskElem ? null : VectorMethodCache.As(256, maskElem, clr);

            MethodInfo? one = null, andNot = null, or = null;
            if (integer)
            {
                one = VectorMethodCache.One(256, clr);
                andNot = VectorMethodCache.Generic(256, "AndNot", clr, paramCount: 2);
                or = VectorMethodCache.Operator(256, clr, "op_BitwiseOr");
            }

            support = new MaskSupport(maskLoad, maskStore, bcast, elems, gt, asM, one, andNot, or, lanes, is32);
            return true;
        }

        #endregion

        #region Generation

        private static void EmitLdarg(ILGenerator il, int index)
        {
            switch (index)
            {
                case 0: il.Emit(OpCodes.Ldarg_0); break;
                case 1: il.Emit(OpCodes.Ldarg_1); break;
                case 2: il.Emit(OpCodes.Ldarg_2); break;
                case 3: il.Emit(OpCodes.Ldarg_3); break;
                default: il.Emit(OpCodes.Ldarg_S, (byte)index); break;
            }
        }

        private const int Arg2DDataPtrs = 0, Arg2DInnerStrides = 1, Arg2DInnerCount = 2, Arg2DOuterStrides = 3, Arg2DOuterCount = 4;

        private static ND2DElementwiseKernel Generate2DElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            var dm = new DynamicMethod(
                name: $"ND2DLoop_{Sanitize(cacheKey)}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            var ptrLocals = new LocalBuilder[nOp];
            var innerStrideLocals = new LocalBuilder[nOp];
            var outerStrideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                innerStrideLocals[op] = il.DeclareLocal(typeof(long));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }

            // ---- Prologue (ONCE per call): ptr[op] = dataptrs[op]; inner/outer[op] = strides[op]. ----
            for (int op = 0; op < nOp; op++)
            {
                EmitLdarg(il, Arg2DDataPtrs);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * IntPtr.Size);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }
            EmitLoadStrideArray(il, Arg2DInnerStrides, nOp, innerStrideLocals);
            EmitLoadStrideArray(il, Arg2DOuterStrides, nOp, outerStrideLocals);

            var lblScalar = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            bool simdPossible = vectorBody != null && CanSimdAllOperands(operandTypes);
            if (simdPossible)
            {
                int elemSize = GetTypeSize(operandTypes[0]);   // all operands same dtype here

                // The output row must be contiguous for the vector store.
                il.Emit(OpCodes.Ldloc, innerStrideLocals[nIn]);
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                il.Emit(OpCodes.Bne_Un, lblScalar);

                if (nIn == 1)
                {
                    // Unary: input contiguous (C) or broadcast along the row (S).
                    var lblC = il.DefineLabel();
                    var lblS = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Beq, lblC);
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Beq, lblS);
                    il.Emit(OpCodes.Br, lblScalar);

                    il.MarkLabel(lblC);
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new[] { false }, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);

                    il.MarkLabel(lblS);
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new[] { true }, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);
                }
                else if (nIn == 2)
                {
                    // Binary: (e,e) → CC, (0,e) → SC, (e,0) → CS; (0,0) and anything else → scalar.
                    var lblCC = il.DefineLabel();
                    var lblSC = il.DefineLabel();
                    var lblCS = il.DefineLabel();
                    var lblLhsIsZero = il.DefineLabel();

                    il.Emit(OpCodes.Ldloc, innerStrideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Beq, lblLhsIsZero);

                    il.Emit(OpCodes.Ldloc, innerStrideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblScalar);

                    il.Emit(OpCodes.Ldloc, innerStrideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Beq, lblCC);
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Beq, lblCS);
                    il.Emit(OpCodes.Br, lblScalar);

                    il.MarkLabel(lblLhsIsZero);
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblScalar);
                    il.Emit(OpCodes.Br, lblSC);

                    il.MarkLabel(lblCC);
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new[] { false, false }, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);

                    il.MarkLabel(lblSC);
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new[] { true, false }, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);

                    il.MarkLabel(lblCS);
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new[] { false, true }, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);
                }
                else
                {
                    // Ternary and wider: every input contiguous, else scalar.
                    for (int op = 0; op < nIn; op++)
                    {
                        il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                        il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                        il.Emit(OpCodes.Bne_Un, lblScalar);
                    }
                    Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new bool[nIn], vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);
                }
            }

            // ---- Scalar 2-D fallback (always present). ----
            // Constant-stride addressing when every operand's inner stride is its own element
            // size (the JIT folds the multiply to a shift), runtime strides otherwise — which is
            // what serves a broadcast (stride-0) input on the scalar path.
            il.MarkLabel(lblScalar);
            var lblStrided = il.DefineLabel();
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, (long)GetTypeSize(operandTypes[op]));
                il.Emit(OpCodes.Bne_Un, lblStrided);
            }
            Emit2DScalarBlock(il, operandTypes, ptrLocals, innerStrideLocals, outerStrideLocals, contig: true, scalarBody);
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblStrided);
            Emit2DScalarBlock(il, operandTypes, ptrLocals, innerStrideLocals, outerStrideLocals, contig: false, scalarBody);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ND2DElementwiseKernel>();
        }

        /// <summary>locals[op] = ((long*)arg)[op] for each operand.</summary>
        private static void EmitLoadStrideArray(ILGenerator il, int argIndex, int nOp, LocalBuilder[] locals)
        {
            for (int op = 0; op < nOp; op++)
            {
                EmitLdarg(il, argIndex);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * sizeof(long));
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, locals[op]);
            }
        }

        /// <summary>
        /// Emit: for each operand, ptr[op] += outerByteStride[op] (the per-row advance).
        /// </summary>
        private static void EmitAdvanceRow(ILGenerator il, LocalBuilder[] ptrLocals, LocalBuilder[] outerStrideLocals)
        {
            for (int op = 0; op < ptrLocals.Length; op++)
            {
                il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                il.Emit(OpCodes.Ldloc, outerStrideLocals[op]);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }
        }

        /// <summary>
        /// Emit one inner-mode pattern's 2-D loop: <paramref name="isScalar"/>[op] marks an input
        /// that is broadcast along the row (loaded and Vector.Create'd ONCE per row), every other
        /// input and the output are contiguous rows addressed as <c>ptr + i*elemSize</c>. The
        /// inner-loop bound computations (unrollEnd, vectorEnd, the tail mask) are HOISTED above
        /// the outer loop because innerCount is the same for every row, and the ROW SHAPE is
        /// decided once per call, not once per row:
        /// <list type="bullet">
        ///   <item><b>generic</b> — the 4x-unrolled SIMD + 1-vector remainder + tail inner loop
        ///   (the masked vector when the host has it, the scalar loop otherwise), mirroring
        ///   <see cref="EmitSimdContigLoop"/>'s inner shape so the two routes produce identical
        ///   results;</item>
        ///   <item><b>one-vector rows</b> (<c>innerCount == lanes</c>: an (N,4) f64, (N,8) f32,
        ///   (N,32) u8 block) — a single full vector op per row and NO inner loop, so the row costs
        ///   its loads/op/store plus the pointer bump; the generic shape paid three loop compares
        ///   and an index update per row for the same work (measured 2.03 vs 1.59 ns/row on 1M);</item>
        ///   <item><b>sub-vector rows</b> (<c>innerCount &lt; lanes</c>, masked hosts) — a single
        ///   masked vector op per row: xyz triples, pairs, RGB, all with no inner loop.</item>
        /// </list>
        /// The inner loop never mutates the base pointers; after a row the outer loop advances each
        /// by its own byte stride.
        /// </summary>
        private static void Emit2DSimdBlock(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] outerStrideLocals,
            bool[] isScalar,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long vectorCount = GetVectorCount(outType);
            long unrollStep = vectorCount * 4;
            var laneClr = GetSimdLaneType(outType);
            var vecType = VectorMethodCache.V(VectorBits, laneClr);

            bool masked = TryGetMaskSupport(outType, out var ms) && ms.Lanes == vectorCount;

            var ctx = new Simd2DContext
            {
                OperandTypes = operandTypes,
                PtrLocals = ptrLocals,
                OuterStrideLocals = outerStrideLocals,
                IsScalar = isScalar,
                VectorBody = vectorBody,
                ScalarBody = scalarBody,
                ElemSize = elemSize,
                VectorCount = vectorCount,
                Masked = masked,
                Mask = ms,
                LocUnrollEnd = il.DeclareLocal(typeof(long)),   // innerCount - unrollStep (hoisted)
                LocVectorEnd = il.DeclareLocal(typeof(long)),   // innerCount - vectorCount (hoisted)
                LocI = il.DeclareLocal(typeof(long)),           // inner element index
                LocOuter = il.DeclareLocal(typeof(long)),       // remaining rows
                LocVScalar = new LocalBuilder?[nIn],
                LocScalarVal = new LocalBuilder?[nIn],
            };
            if (masked)
            {
                ctx.LocTail = il.DeclareLocal(typeof(long));
                ctx.LocMask = il.DeclareLocal(vecType);
                ctx.LocOutVec = il.DeclareLocal(vecType);
                if (ms.One != null) ctx.LocFill = il.DeclareLocal(vecType);
            }
            for (int op = 0; op < nIn; op++)
            {
                if (!isScalar[op]) continue;
                ctx.LocVScalar[op] = il.DeclareLocal(vecType);
                ctx.LocScalarVal[op] = il.DeclareLocal(GetClrType(operandTypes[op]));
            }

            // ---- Hoisted bounds ----
            // unrollEnd = innerCount - unrollStep
            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, ctx.LocUnrollEnd);

            // vectorEnd = innerCount - vectorCount
            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, ctx.LocVectorEnd);

            if (masked)
            {
                // tail = innerCount & (lanes - 1)   (lanes is a power of two)
                EmitLdarg(il, Arg2DInnerCount);
                il.Emit(OpCodes.Ldc_I8, vectorCount - 1);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, ctx.LocTail!);

                // mask = GreaterThan(Create(tail), {0, 1, …, lanes-1})  — all-ones where lane < tail
                il.Emit(OpCodes.Ldloc, ctx.LocTail!);
                if (ms.Is32) il.Emit(OpCodes.Conv_I4);
                il.EmitCall(OpCodes.Call, ms.MaskBroadcast, null);
                for (int k = 0; k < ms.Lanes; k++)
                {
                    if (ms.Is32) il.Emit(OpCodes.Ldc_I4, k);
                    else il.Emit(OpCodes.Ldc_I8, (long)k);
                }
                il.EmitCall(OpCodes.Call, ms.MaskElements, null);
                il.EmitCall(OpCodes.Call, ms.MaskGreaterThan, null);
                if (ms.MaskAs != null)
                    il.EmitCall(OpCodes.Call, ms.MaskAs, null);
                il.Emit(OpCodes.Stloc, ctx.LocMask!);

                if (ctx.LocFill != null)
                {
                    // fill = AndNot(One, mask) — a 1 in every masked-OFF lane, 0 elsewhere.
                    il.EmitCall(OpCodes.Call, ms.One!, null);
                    il.Emit(OpCodes.Ldloc, ctx.LocMask!);
                    il.EmitCall(OpCodes.Call, ms.AndNot!, null);
                    il.Emit(OpCodes.Stloc, ctx.LocFill);
                }
            }

            // ---- Per-CALL row-shape dispatch ----
            var lblShortFull = il.DefineLabel();
            var lblShortMasked = il.DefineLabel();
            var lblBlockEnd = il.DefineLabel();

            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Beq, lblShortFull);
            if (masked)
            {
                EmitLdarg(il, Arg2DInnerCount);
                il.Emit(OpCodes.Ldc_I8, vectorCount);
                il.Emit(OpCodes.Blt, lblShortMasked);
            }

            // ===== GENERIC ROWS: 4x-unrolled SIMD + 1-vector remainder + tail =====
            Emit2DOuterLoop(il, ctx, il2 =>
            {
                var lblUnroll = il2.DefineLabel();
                var lblUnrollEnd = il2.DefineLabel();
                var lblRem = il2.DefineLabel();
                var lblRemEnd = il2.DefineLabel();
                var lblTailEnd = il2.DefineLabel();

                // i = 0
                il2.Emit(OpCodes.Ldc_I8, 0L);
                il2.Emit(OpCodes.Stloc, ctx.LocI);

                // --- 4x UNROLLED SIMD ---
                il2.MarkLabel(lblUnroll);
                il2.Emit(OpCodes.Ldloc, ctx.LocI);
                il2.Emit(OpCodes.Ldloc, ctx.LocUnrollEnd);
                il2.Emit(OpCodes.Bgt, lblUnrollEnd);

                for (int u = 0; u < 4; u++)
                    Emit2DVectorGroup(il2, ctx, ctx.LocI, u * vectorCount, masked: false);

                il2.Emit(OpCodes.Ldloc, ctx.LocI);
                il2.Emit(OpCodes.Ldc_I8, unrollStep);
                il2.Emit(OpCodes.Add);
                il2.Emit(OpCodes.Stloc, ctx.LocI);
                il2.Emit(OpCodes.Br, lblUnroll);
                il2.MarkLabel(lblUnrollEnd);

                // --- REMAINDER SIMD (1 vector at a time) ---
                il2.MarkLabel(lblRem);
                il2.Emit(OpCodes.Ldloc, ctx.LocI);
                il2.Emit(OpCodes.Ldloc, ctx.LocVectorEnd);
                il2.Emit(OpCodes.Bgt, lblRemEnd);

                Emit2DVectorGroup(il2, ctx, ctx.LocI, 0, masked: false);

                il2.Emit(OpCodes.Ldloc, ctx.LocI);
                il2.Emit(OpCodes.Ldc_I8, vectorCount);
                il2.Emit(OpCodes.Add);
                il2.Emit(OpCodes.Stloc, ctx.LocI);
                il2.Emit(OpCodes.Br, lblRem);
                il2.MarkLabel(lblRemEnd);

                if (masked)
                {
                    // --- MASKED TAIL: one masked vector for the 1..lanes-1 leftover elements ---
                    il2.Emit(OpCodes.Ldloc, ctx.LocTail!);
                    il2.Emit(OpCodes.Ldc_I8, 0L);
                    il2.Emit(OpCodes.Beq, lblTailEnd);
                    Emit2DVectorGroup(il2, ctx, ctx.LocI, 0, masked: true);
                }
                else
                {
                    // --- SCALAR TAIL (contiguous; broadcast inputs use the row's scalar) ---
                    var lblTail = il2.DefineLabel();
                    var locOutVal = il2.DeclareLocal(GetClrType(outType));

                    il2.MarkLabel(lblTail);
                    il2.Emit(OpCodes.Ldloc, ctx.LocI);
                    EmitLdarg(il2, Arg2DInnerCount);
                    il2.Emit(OpCodes.Bge, lblTailEnd);

                    for (int op = 0; op < nIn; op++)
                    {
                        if (isScalar[op])
                        {
                            il2.Emit(OpCodes.Ldloc, ctx.LocScalarVal[op]!);
                            continue;
                        }
                        EmitAddrIPlusOffset(il2, ptrLocals[op], ctx.LocI, 0, GetTypeSize(operandTypes[op]));
                        EmitLoadIndirect(il2, operandTypes[op]);
                    }
                    scalarBody(il2);
                    il2.Emit(OpCodes.Stloc, locOutVal);
                    EmitAddrIPlusOffset(il2, ptrLocals[nIn], ctx.LocI, 0, elemSize);
                    il2.Emit(OpCodes.Ldloc, locOutVal);
                    EmitStoreIndirect(il2, outType);

                    il2.Emit(OpCodes.Ldloc, ctx.LocI);
                    il2.Emit(OpCodes.Ldc_I8, 1L);
                    il2.Emit(OpCodes.Add);
                    il2.Emit(OpCodes.Stloc, ctx.LocI);
                    il2.Emit(OpCodes.Br, lblTail);
                }
                il2.MarkLabel(lblTailEnd);
            });
            il.Emit(OpCodes.Br, lblBlockEnd);

            // ===== ONE-VECTOR ROWS: innerCount == lanes — a single full vector per row, no inner loop =====
            il.MarkLabel(lblShortFull);
            Emit2DOuterLoop(il, ctx, il2 => Emit2DVectorGroup(il2, ctx, null, 0, masked: false));
            il.Emit(OpCodes.Br, lblBlockEnd);

            // ===== SUB-VECTOR ROWS: innerCount < lanes — a single masked vector per row =====
            // (tail == innerCount here, so the hoisted mask covers exactly the row; an empty row
            // yields an all-false mask and touches nothing.)
            if (masked)
            {
                il.MarkLabel(lblShortMasked);
                Emit2DOuterLoop(il, ctx, il2 => Emit2DVectorGroup(il2, ctx, null, 0, masked: true));
            }

            il.MarkLabel(lblBlockEnd);
        }

        /// <summary>Everything one <see cref="Emit2DSimdBlock"/> instantiation shares between its row shapes.</summary>
        private sealed class Simd2DContext
        {
            public NPTypeCode[] OperandTypes = null!;
            public LocalBuilder[] PtrLocals = null!;
            public LocalBuilder[] OuterStrideLocals = null!;
            public bool[] IsScalar = null!;
            public Action<ILGenerator> VectorBody = null!;
            public Action<ILGenerator> ScalarBody = null!;
            public int ElemSize;
            public long VectorCount;
            public bool Masked;
            public MaskSupport Mask;
            public LocalBuilder LocUnrollEnd = null!, LocVectorEnd = null!, LocI = null!, LocOuter = null!;
            public LocalBuilder? LocTail, LocMask, LocFill, LocOutVec;
            public LocalBuilder?[] LocVScalar = null!;
            public LocalBuilder?[] LocScalarVal = null!;
        }

        /// <summary>
        /// The outer (per-row) loop: <c>for (outer = outerCount; outer &gt; 0; outer--)</c> — load
        /// and broadcast each S input's scalar for this row, run <paramref name="rowBody"/>,
        /// advance every operand pointer by its outer byte stride.
        /// </summary>
        private static void Emit2DOuterLoop(ILGenerator il, Simd2DContext ctx, Action<ILGenerator> rowBody)
        {
            int nIn = ctx.OperandTypes.Length - 1;
            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();

            // locOuter = outerCount
            EmitLdarg(il, Arg2DOuterCount);
            il.Emit(OpCodes.Stloc, ctx.LocOuter);

            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, ctx.LocOuter);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblOuterEnd);

            // Broadcast inputs: load the row's scalar once, keep it for the scalar tail, broadcast it.
            for (int op = 0; op < nIn; op++)
            {
                if (!ctx.IsScalar[op]) continue;
                il.Emit(OpCodes.Ldloc, ctx.PtrLocals[op]);
                EmitLoadIndirect(il, ctx.OperandTypes[op]);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Stloc, ctx.LocScalarVal[op]!);
                EmitVectorCreate(il, ctx.OperandTypes[op]);
                il.Emit(OpCodes.Stloc, ctx.LocVScalar[op]!);
            }

            rowBody(il);

            // --- advance each operand pointer to the next row ---
            EmitAdvanceRow(il, ctx.PtrLocals, ctx.OuterStrideLocals);

            // locOuter--
            il.Emit(OpCodes.Ldloc, ctx.LocOuter);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, ctx.LocOuter);
            il.Emit(OpCodes.Br, lblOuter);
            il.MarkLabel(lblOuterEnd);
        }

        /// <summary>
        /// One vector group at element position <c>(locI ?? 0) + offset</c>: load every C input
        /// (a full vector, or a masked one when <paramref name="masked"/>), push every S input's
        /// pre-broadcast vector, run the vector body, store the result (full or masked). With
        /// <paramref name="locI"/> null the row start is the address — the short-row shapes
        /// keep no index at all.
        /// </summary>
        private static void Emit2DVectorGroup(ILGenerator il, Simd2DContext ctx, LocalBuilder? locI, long offset, bool masked)
        {
            int nIn = ctx.OperandTypes.Length - 1;
            NPTypeCode outType = ctx.OperandTypes[nIn];

            for (int op = 0; op < nIn; op++)
            {
                if (ctx.IsScalar[op])
                {
                    il.Emit(OpCodes.Ldloc, ctx.LocVScalar[op]!);
                    continue;
                }
                Emit2DRowAddr(il, ctx.PtrLocals[op], locI, offset, ctx.ElemSize);
                if (masked)
                {
                    il.Emit(OpCodes.Ldloc, ctx.LocMask!);
                    il.EmitCall(OpCodes.Call, ctx.Mask.MaskLoad, null);
                    if (ctx.LocFill != null)
                    {
                        il.Emit(OpCodes.Ldloc, ctx.LocFill);
                        il.EmitCall(OpCodes.Call, ctx.Mask.Or!, null);
                    }
                }
                else
                {
                    EmitVectorLoad(il, ctx.OperandTypes[op]);
                }
            }

            ctx.VectorBody(il);

            if (masked)
            {
                il.Emit(OpCodes.Stloc, ctx.LocOutVec!);
                Emit2DRowAddr(il, ctx.PtrLocals[nIn], locI, offset, ctx.ElemSize);
                il.Emit(OpCodes.Ldloc, ctx.LocMask!);
                il.Emit(OpCodes.Ldloc, ctx.LocOutVec!);
                il.EmitCall(OpCodes.Call, ctx.Mask.MaskStore, null);
            }
            else
            {
                Emit2DRowAddr(il, ctx.PtrLocals[nIn], locI, offset, ctx.ElemSize);
                EmitVectorStore(il, outType);
            }
        }

        /// <summary>Push <c>basePtr + ((locI ?? 0) + offset) * elemSize</c>.</summary>
        private static void Emit2DRowAddr(ILGenerator il, LocalBuilder basePtr, LocalBuilder? locI, long offset, int elemSize)
        {
            if (locI != null)
            {
                EmitAddrIPlusOffset(il, basePtr, locI, offset, elemSize);
                return;
            }
            il.Emit(OpCodes.Ldloc, basePtr);
            if (offset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset * elemSize);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>
        /// Emit the outer (per-row) loop around a scalar inner loop: the per-row route's
        /// scalar body without its per-row driver and prologue. <paramref name="contig"/>
        /// addresses each operand as <c>ptr + i*elemSize</c> (its own element size, a JIT
        /// constant); otherwise as <c>ptr + i*innerByteStride</c>, which serves a broadcast
        /// (stride-0) input and any other inner stride. Reuses <see cref="EmitScalarElement"/>,
        /// so the loads, the scalar body and the store are exactly the per-chunk kernel's.
        /// </summary>
        private static void Emit2DScalarBlock(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals,
            bool contig,
            Action<ILGenerator> scalarBody)
        {
            var locI = il.DeclareLocal(typeof(long));
            var locOuter = il.DeclareLocal(typeof(long));

            EmitLdarg(il, Arg2DOuterCount);
            il.Emit(OpCodes.Stloc, locOuter);

            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblOuterEnd);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);
            il.Emit(OpCodes.Ldloc, locI);
            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Bge, lblLoopEnd);

            EmitScalarElement(il, operandTypes, ptrLocals, contig ? null : innerStrideLocals, locI, contig, scalarBody);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            EmitAdvanceRow(il, ptrLocals, outerStrideLocals);

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
