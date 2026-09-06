using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.InnerLoop2D.Masked.cs — the where= form of the 2-D block kernel
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A ufunc with where= over narrow strided rows (`np.add(m[:, :4], n[:, :4], out=o,
// where=mask)`) used to be excluded from the 2-D block route: the masked ForEach
// driver ran per ROW — the odometer advance, the mask-run scan over a handful of
// bytes (below any vector width, so scalar), and one per-chunk kernel call per run
// with its own SIMD-viability prologue — for a few elements of work. Measured on
// 1M float64 as (rows, 3/4) views: 0.4–0.9x NumPy where the unmasked rows run
// 2–8x (docs/NDITER_PERF_DISCOVERY.md §7 lever 3; NumPy pays the same per-row
// structure, it is just cheaper at it).
//
// THE MASKED CONTRACT
// -------------------
// Same delegate as ND2DElementwiseKernel — (dataptrs, innerByteStrides, innerCount,
// outerByteStrides, outerCount) — with ONE extra trailing operand: the mask, a byte
// array read as "non-zero = write". The kernel loops the outer axis itself and picks
// the mask mode ONCE per call from the mask's inner byte stride:
//   • 1  — a mask byte per element (the ordinary where= array): every vector group
//          builds its lane mask from the mask bytes (pmovzx bytes → 32/64-bit lanes,
//          compare against zero) and runs the SAME masked load / vector body /
//          masked store the unmasked kernel uses for its sub-vector tails, so a
//          masked-off element is never read for its lane's sake nor written. The
//          row tail (innerCount % lanes, or the whole row when it is shorter than a
//          vector) ANDs that with the hoisted tail mask. Integer masked-off lanes are
//          filled with 1 (a software divide on a 0 lane would throw), as the tail
//          path already does.
//   • 0  — one mask byte per row (a (rows,1) broadcast mask): the row is gated as a
//          whole and runs the plain full-vector loop + masked tail.
//   • any other stride, or a dtype/op without vector support — the scalar 2-D block
//          with a per-element (or per-row) mask test.
// The mask bytes of a full group are one 8-byte (32-bit lanes) or 4-byte (64-bit
// lanes) load; a TAIL group assembles its word from exactly `tail` bytes, so the
// kernel never reads past the end of the mask array.
//
// SCOPE / GATING (NDIterRef.Is2DMaskedElementwiseShape / TryExecute2DMasked)
// --------------------------------------------------------------------------
//   • unbuffered, EXTERNAL_LOOP, NDim >= 2, the ARRAYMASK operand trailing
//   • output inner axis element-contiguous; every input inner axis contiguous or
//     broadcast; the mask inner axis contiguous (1 byte) or broadcast (0)
//   • trailing mutually-contiguous outer axes fold into one block; leading axes
//     are walked per block by the dispatcher
//
// The emitted bodies are the SAME scalarBody/vectorBody the per-chunk kernel uses,
// so a masked element's value is byte-identical to the ForEach masked driver's.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Cache for the where= 2-D block kernels, keyed by the ufunc's string cache key (the masked
        /// routes carry that key; it is unique per (op, dtypes)). Separate from the unmasked cache:
        /// the same key names a kernel with one more operand and a different loop structure.
        /// </summary>
        internal static readonly ConcurrentDictionary<string, ND2DElementwiseKernel> _innerLoop2DMaskedCache = new();

        /// <summary>
        /// Packed-key front cache for the masked 2-D block kernels — the production ufunc routes
        /// identify their kernel by <see cref="InnerLoopKernelKey"/>, so a hit builds no string.
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, ND2DElementwiseKernel> _innerLoop2DMaskedKeyCache = new();

        /// <summary>
        /// Serve the masked 2-D block kernel for <paramref name="cacheKey"/> (the data operand
        /// types; the mask is the implicit trailing operand), else compile and register it.
        /// </summary>
        internal static ND2DElementwiseKernel Compile2DMaskedElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            if (_innerLoop2DMaskedCache.TryGetValue(cacheKey, out var cached))
                return cached;

            var kernel = Generate2DMaskedElementwiseKernel(operandTypes, scalarBody, vectorBody, cacheKey);
            _innerLoop2DMaskedCache.TryAdd(cacheKey, kernel);
            return kernel;
        }

        /// <summary>
        /// Packed-key form of <see cref="Compile2DMaskedElementwiseKernel(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/>:
        /// a hit costs one dictionary probe; a miss compiles under the equivalent string key
        /// (<see cref="InnerLoopKernelKey.ToCacheKey"/>) and registers both.
        /// </summary>
        internal static ND2DElementwiseKernel Compile2DMaskedElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            if (_innerLoop2DMaskedKeyCache.TryGetValue(key, out var cached))
                return cached;

            var kernel = Compile2DMaskedElementwiseKernel(operandTypes, scalarBody, vectorBody, key.ToCacheKey());
            _innerLoop2DMaskedKeyCache.TryAdd(key, kernel);
            return kernel;
        }

        /// <summary>Reflection handles for turning mask BYTES into a 256-bit lane mask.</summary>
        private readonly struct MaskWordSupport
        {
            /// <summary>The word holding one group's mask bytes: long (8 × 32-bit lanes) or int (4 × 64-bit lanes).</summary>
            public readonly Type WordType;
            /// <summary><c>Vector128.CreateScalarUnsafe(word)</c>.</summary>
            public readonly MethodInfo CreateScalarUnsafe;
            /// <summary><c>Vector128.As&lt;word, byte&gt;</c>.</summary>
            public readonly MethodInfo AsByte;
            /// <summary><c>Avx2.ConvertToVector256Int32/Int64(Vector128&lt;byte&gt;)</c> — zero-extending widen.</summary>
            public readonly MethodInfo Widen;
            /// <summary><c>Vector256&lt;M&gt;.Zero</c> getter for the mask lane type.</summary>
            public readonly MethodInfo ZeroM;
            /// <summary><c>Vector256&lt;M&gt;.op_BitwiseAnd</c>.</summary>
            public readonly MethodInfo AndM;

            public MaskWordSupport(Type wordType, MethodInfo createScalarUnsafe, MethodInfo asByte, MethodInfo widen, MethodInfo zeroM, MethodInfo andM)
            {
                WordType = wordType; CreateScalarUnsafe = createScalarUnsafe; AsByte = asByte; Widen = widen; ZeroM = zeroM; AndM = andM;
            }
        }

        private static bool TryGetMaskWordSupport(in MaskSupport ms, out MaskWordSupport support)
        {
            support = default;
            if (!Avx2.IsSupported)
                return false;
            Type word = ms.Is32 ? typeof(long) : typeof(int);
            Type maskElem = ms.Is32 ? typeof(int) : typeof(long);
            var create = typeof(Vector128).GetMethod("CreateScalarUnsafe", new[] { word });
            var widen = typeof(Avx2).GetMethod(ms.Is32 ? "ConvertToVector256Int32" : "ConvertToVector256Int64", new[] { typeof(Vector128<byte>) });
            if (create == null || widen == null)
                return false;
            support = new MaskWordSupport(
                word,
                create,
                VectorMethodCache.As(128, word, typeof(byte)),
                widen,
                VectorMethodCache.Zero(256, maskElem),
                VectorMethodCache.Operator(256, maskElem, "op_BitwiseAnd"));
            return true;
        }

        /// <summary>Everything one masked 2-D block instantiation shares between its emitters.</summary>
        private sealed class Masked2DContext
        {
            public NPTypeCode[] OperandTypes = null!;      // data operands (inputs then output)
            public LocalBuilder[] PtrLocals = null!;       // nOp + 1 (mask last)
            public LocalBuilder[] InnerStrideLocals = null!;
            public LocalBuilder[] OuterStrideLocals = null!;
            public bool[] IsScalar = null!;
            public Action<ILGenerator> VectorBody = null!;
            public Action<ILGenerator> ScalarBody = null!;
            public int ElemSize;
            public long Lanes;
            public MaskSupport Mask;
            public MaskWordSupport Word;
            public LocalBuilder LocI = null!, LocOuter = null!, LocVectorEnd = null!, LocTail = null!;
            public LocalBuilder LocTailMaskM = null!;      // Vector256<M>: lane < tail
            public LocalBuilder LocTailMaskT = null!;      // the same as Vector256<T>
            public LocalBuilder LocDynM = null!, LocDynT = null!, LocOutVec = null!;
            public LocalBuilder? LocFill;                   // integer lanes: 1 in every masked-off lane of the CURRENT mask
            public LocalBuilder LocWord = null!, LocK = null!;
            public LocalBuilder?[] LocVScalar = null!;
            public LocalBuilder?[] LocScalarVal = null!;
            public int MaskIndex => OperandTypes.Length;
        }

        private static ND2DElementwiseKernel Generate2DMaskedElementwiseKernel(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            var dm = new DynamicMethod(
                name: $"ND2DMaskedLoop_{Sanitize(cacheKey)}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            int nAll = nOp + 1;                                   // + the mask
            int maskIdx = nOp;
            var ptrLocals = new LocalBuilder[nAll];
            var innerStrideLocals = new LocalBuilder[nAll];
            var outerStrideLocals = new LocalBuilder[nAll];
            for (int op = 0; op < nAll; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                innerStrideLocals[op] = il.DeclareLocal(typeof(long));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }

            // ---- Prologue (ONCE per call) ----
            for (int op = 0; op < nAll; op++)
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
            EmitLoadStrideArray(il, Arg2DInnerStrides, nAll, innerStrideLocals);
            EmitLoadStrideArray(il, Arg2DOuterStrides, nAll, outerStrideLocals);

            var lblScalar = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            NPTypeCode outType = operandTypes[nIn];
            MaskSupport ms = default;
            MaskWordSupport mw = default;
            bool simdPossible = vectorBody != null
                                && CanSimdAllOperands(operandTypes)
                                && TryGetMaskSupport(outType, out ms)
                                && ms.Lanes == GetVectorCount(outType)
                                && TryGetMaskWordSupport(ms, out mw);
            if (simdPossible)
            {
                int elemSize = GetTypeSize(outType);

                // The mask inner stride must be a byte per element (1) or per row (0); the output
                // row contiguous; every input contiguous or broadcast — else the scalar block.
                var lblByteMask = il.DefineLabel();
                var lblRowGate = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, innerStrideLocals[nIn]);
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                il.Emit(OpCodes.Bne_Un, lblScalar);
                for (int op = 0; op < nIn; op++)
                {
                    var lblOk = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Beq, lblOk);
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Bne_Un, lblScalar);
                    il.MarkLabel(lblOk);
                }
                il.Emit(OpCodes.Ldloc, innerStrideLocals[maskIdx]);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Beq, lblByteMask);
                il.Emit(OpCodes.Ldloc, innerStrideLocals[maskIdx]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblRowGate);
                il.Emit(OpCodes.Br, lblScalar);

                // Which inputs are broadcast along the row is a runtime property of the strides;
                // the emitter needs it at generation time, so every C/S pattern gets its own
                // block, selected by a per-call dispatch (unary {C,S}, binary {CC,SC,CS}; wider
                // ops require all-C, as the unmasked kernel does).
                var patterns = ScalarPatterns(nIn);

                il.MarkLabel(lblByteMask);
                EmitMaskedPatternDispatch(il, operandTypes, ptrLocals, innerStrideLocals, outerStrideLocals, patterns, elemSize, ms, mw, vectorBody!, scalarBody, byteMask: true, lblScalar, lblEnd);

                il.MarkLabel(lblRowGate);
                EmitMaskedPatternDispatch(il, operandTypes, ptrLocals, innerStrideLocals, outerStrideLocals, patterns, elemSize, ms, mw, vectorBody!, scalarBody, byteMask: false, lblScalar, lblEnd);
            }

            // ---- Scalar masked 2-D block (always present): per-element mask test with the mask's
            // runtime stride (0 gates the row, 1 is the byte mask, anything else a strided mask).
            il.MarkLabel(lblScalar);
            Emit2DMaskedScalarBlock(il, operandTypes, ptrLocals, innerStrideLocals, outerStrideLocals, scalarBody);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ND2DElementwiseKernel>();
        }

        /// <summary>The C/S input patterns a masked SIMD block is generated for.</summary>
        private static bool[][] ScalarPatterns(int nIn)
        {
            if (nIn == 1) return new[] { new[] { false }, new[] { true } };
            if (nIn == 2) return new[] { new[] { false, false }, new[] { true, false }, new[] { false, true } };
            return new[] { new bool[nIn] };
        }

        /// <summary>
        /// Per-call dispatch over the C/S patterns: compare each input's inner stride against
        /// the pattern (elemSize = C, 0 = S) and jump to that pattern's masked block; no match
        /// → <paramref name="lblScalar"/>.
        /// </summary>
        private static void EmitMaskedPatternDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals, bool[][] patterns, int elemSize, MaskSupport ms, MaskWordSupport mw,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, bool byteMask, Label lblScalar, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;
            var labels = new Label[patterns.Length];
            for (int p = 0; p < patterns.Length; p++)
            {
                labels[p] = il.DefineLabel();
                var lblNext = il.DefineLabel();
                for (int op = 0; op < nIn; op++)
                {
                    il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, patterns[p][op] ? 0L : (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblNext);
                }
                il.Emit(OpCodes.Br, labels[p]);
                il.MarkLabel(lblNext);
            }
            il.Emit(OpCodes.Br, lblScalar);

            for (int p = 0; p < patterns.Length; p++)
            {
                il.MarkLabel(labels[p]);
                Emit2DMaskedSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, patterns[p], elemSize, ms, mw, vectorBody, scalarBody, byteMask);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One masked C/S pattern's 2-D loop: per row, a full-vector loop plus a masked tail
        /// (<paramref name="byteMask"/>: every group's mask comes from the mask bytes; else the
        /// row is gated by its single mask byte and runs full vectors + the static tail mask).
        /// Bounds and the static tail mask are hoisted above the outer loop.
        /// </summary>
        private static void Emit2DMaskedSimdBlock(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] outerStrideLocals,
            bool[] isScalar, int elemSize, MaskSupport ms, MaskWordSupport mw,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, bool byteMask)
        {
            int nIn = operandTypes.Length - 1;
            NPTypeCode outType = operandTypes[nIn];
            long lanes = ms.Lanes;
            Type maskElem = ms.Is32 ? typeof(int) : typeof(long);
            var vecM = typeof(Vector256<>).MakeGenericType(maskElem);
            var vecT = VectorMethodCache.V(256, GetSimdLaneType(outType));

            var ctx = new Masked2DContext
            {
                OperandTypes = operandTypes,
                PtrLocals = ptrLocals,
                OuterStrideLocals = outerStrideLocals,
                IsScalar = isScalar,
                VectorBody = vectorBody,
                ScalarBody = scalarBody,
                ElemSize = elemSize,
                Lanes = lanes,
                Mask = ms,
                Word = mw,
                LocI = il.DeclareLocal(typeof(long)),
                LocOuter = il.DeclareLocal(typeof(long)),
                LocVectorEnd = il.DeclareLocal(typeof(long)),
                LocTail = il.DeclareLocal(typeof(long)),
                LocTailMaskM = il.DeclareLocal(vecM),
                LocTailMaskT = il.DeclareLocal(vecT),
                LocDynM = il.DeclareLocal(vecM),
                LocDynT = il.DeclareLocal(vecT),
                LocOutVec = il.DeclareLocal(vecT),
                LocFill = ms.One != null ? il.DeclareLocal(vecT) : null,
                LocWord = il.DeclareLocal(typeof(long)),
                LocK = il.DeclareLocal(typeof(long)),
                LocVScalar = new LocalBuilder?[nIn],
                LocScalarVal = new LocalBuilder?[nIn],
            };
            for (int op = 0; op < nIn; op++)
            {
                if (!isScalar[op]) continue;
                ctx.LocVScalar[op] = il.DeclareLocal(vecT);
                ctx.LocScalarVal[op] = il.DeclareLocal(GetClrType(operandTypes[op]));
            }

            // ---- Hoisted: vectorEnd = innerCount - lanes; tail = innerCount & (lanes - 1) ----
            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Ldc_I8, lanes);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, ctx.LocVectorEnd);

            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Ldc_I8, lanes - 1);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, ctx.LocTail);

            // tailMaskM = GreaterThan(Create(tail), {0..lanes-1}); tailMaskT = As<M,T>(tailMaskM)
            il.Emit(OpCodes.Ldloc, ctx.LocTail);
            if (ms.Is32) il.Emit(OpCodes.Conv_I4);
            il.EmitCall(OpCodes.Call, ms.MaskBroadcast, null);
            for (int k = 0; k < ms.Lanes; k++)
            {
                if (ms.Is32) il.Emit(OpCodes.Ldc_I4, k);
                else il.Emit(OpCodes.Ldc_I8, (long)k);
            }
            il.EmitCall(OpCodes.Call, ms.MaskElements, null);
            il.EmitCall(OpCodes.Call, ms.MaskGreaterThan, null);
            il.Emit(OpCodes.Stloc, ctx.LocTailMaskM);
            il.Emit(OpCodes.Ldloc, ctx.LocTailMaskM);
            if (ms.MaskAs != null)
                il.EmitCall(OpCodes.Call, ms.MaskAs, null);
            il.Emit(OpCodes.Stloc, ctx.LocTailMaskT);

            if (!byteMask && ctx.LocFill != null)
            {
                // Row-gated rows use the static tail mask for every row: hoist its fill too.
                // fill = AndNot(One, tailMaskT)
                il.EmitCall(OpCodes.Call, ms.One!, null);
                il.Emit(OpCodes.Ldloc, ctx.LocTailMaskT);
                il.EmitCall(OpCodes.Call, ms.AndNot!, null);
                il.Emit(OpCodes.Stloc, ctx.LocFill);
            }

            // ---- Outer loop ----
            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblAdvance = il.DefineLabel();

            EmitLdarg(il, Arg2DOuterCount);
            il.Emit(OpCodes.Stloc, ctx.LocOuter);

            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, ctx.LocOuter);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblOuterEnd);

            if (!byteMask)
            {
                // Row gate: the row's single mask byte.
                il.Emit(OpCodes.Ldloc, ptrLocals[ctx.MaskIndex]);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblAdvance);
            }

            // Broadcast inputs: load the row's scalar once and broadcast it.
            for (int op = 0; op < nIn; op++)
            {
                if (!isScalar[op]) continue;
                il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                EmitLoadIndirect(il, operandTypes[op]);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Stloc, ctx.LocScalarVal[op]!);
                EmitVectorCreate(il, operandTypes[op]);
                il.Emit(OpCodes.Stloc, ctx.LocVScalar[op]!);
            }

            // i = 0; full-vector loop
            var lblVec = il.DefineLabel();
            var lblTail = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, ctx.LocI);

            il.MarkLabel(lblVec);
            il.Emit(OpCodes.Ldloc, ctx.LocI);
            il.Emit(OpCodes.Ldloc, ctx.LocVectorEnd);
            il.Emit(OpCodes.Bgt, lblTail);

            if (byteMask)
            {
                EmitMaskedDynamicMask(il, ctx, tailGroup: false);
                EmitMaskedVectorGroup(il, ctx, ctx.LocDynT, ctx.LocFill);
            }
            else
            {
                EmitMaskedVectorGroup(il, ctx, null, null);
            }

            il.Emit(OpCodes.Ldloc, ctx.LocI);
            il.Emit(OpCodes.Ldc_I8, lanes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, ctx.LocI);
            il.Emit(OpCodes.Br, lblVec);

            // tail: 1..lanes-1 leftover elements (or the whole row when innerCount < lanes)
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, ctx.LocTail);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Beq, lblAdvance);

            if (byteMask)
            {
                EmitMaskedDynamicMask(il, ctx, tailGroup: true);
                EmitMaskedVectorGroup(il, ctx, ctx.LocDynT, ctx.LocFill);
            }
            else
            {
                EmitMaskedVectorGroup(il, ctx, ctx.LocTailMaskT, ctx.LocFill);   // fill hoisted above
            }

            // advance every operand (mask included) to the next row
            il.MarkLabel(lblAdvance);
            EmitAdvanceRow(il, ptrLocals, outerStrideLocals);

            il.Emit(OpCodes.Ldloc, ctx.LocOuter);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, ctx.LocOuter);
            il.Emit(OpCodes.Br, lblOuter);
            il.MarkLabel(lblOuterEnd);
        }

        /// <summary>
        /// Build the current group's lane mask from the mask bytes at <c>maskPtr + i</c> into
        /// <c>LocDynT</c> (and <c>LocFill</c> for integer lanes): a full group reads one word;
        /// a <paramref name="tailGroup"/> assembles exactly <c>tail</c> bytes, then ANDs the
        /// static tail mask, so no byte past the row end is read and no lane past it is touched.
        /// </summary>
        private static void EmitMaskedDynamicMask(ILGenerator il, Masked2DContext ctx, bool tailGroup)
        {
            var ms = ctx.Mask;
            var mw = ctx.Word;
            var maskPtr = ctx.PtrLocals[ctx.MaskIndex];

            if (!tailGroup)
            {
                // word = *(long|int*)(maskPtr + i)
                il.Emit(OpCodes.Ldloc, maskPtr);
                il.Emit(OpCodes.Ldloc, ctx.LocI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(mw.WordType == typeof(long) ? OpCodes.Ldind_I8 : OpCodes.Ldind_I4);
            }
            else
            {
                // word = 0; for (k = 0; k < tail; k++) word |= (long)maskPtr[i + k] << (8*k)
                var lblK = il.DefineLabel();
                var lblKEnd = il.DefineLabel();
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, ctx.LocWord);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, ctx.LocK);
                il.MarkLabel(lblK);
                il.Emit(OpCodes.Ldloc, ctx.LocK);
                il.Emit(OpCodes.Ldloc, ctx.LocTail);
                il.Emit(OpCodes.Bge, lblKEnd);
                il.Emit(OpCodes.Ldloc, ctx.LocWord);
                il.Emit(OpCodes.Ldloc, maskPtr);
                il.Emit(OpCodes.Ldloc, ctx.LocI);
                il.Emit(OpCodes.Ldloc, ctx.LocK);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Conv_U8);
                il.Emit(OpCodes.Ldloc, ctx.LocK);
                il.Emit(OpCodes.Conv_I4);
                il.Emit(OpCodes.Ldc_I4_3);
                il.Emit(OpCodes.Shl);
                il.Emit(OpCodes.Shl);
                il.Emit(OpCodes.Or);
                il.Emit(OpCodes.Stloc, ctx.LocWord);
                il.Emit(OpCodes.Ldloc, ctx.LocK);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, ctx.LocK);
                il.Emit(OpCodes.Br, lblK);
                il.MarkLabel(lblKEnd);
                il.Emit(OpCodes.Ldloc, ctx.LocWord);
                if (mw.WordType == typeof(int))
                    il.Emit(OpCodes.Conv_I4);
            }

            // dynM = GreaterThan(Widen(As<byte>(CreateScalarUnsafe(word))), Zero)
            il.EmitCall(OpCodes.Call, mw.CreateScalarUnsafe, null);
            il.EmitCall(OpCodes.Call, mw.AsByte, null);
            il.EmitCall(OpCodes.Call, mw.Widen, null);
            il.EmitCall(OpCodes.Call, mw.ZeroM, null);
            il.EmitCall(OpCodes.Call, ms.MaskGreaterThan, null);
            if (tailGroup)
            {
                il.Emit(OpCodes.Ldloc, ctx.LocTailMaskM);
                il.EmitCall(OpCodes.Call, mw.AndM, null);
            }
            il.Emit(OpCodes.Stloc, ctx.LocDynM);

            // dynT = As<M,T>(dynM)
            il.Emit(OpCodes.Ldloc, ctx.LocDynM);
            if (ms.MaskAs != null)
                il.EmitCall(OpCodes.Call, ms.MaskAs, null);
            il.Emit(OpCodes.Stloc, ctx.LocDynT);

            if (ctx.LocFill != null)
            {
                // fill = AndNot(One, dynT)
                il.EmitCall(OpCodes.Call, ms.One!, null);
                il.Emit(OpCodes.Ldloc, ctx.LocDynT);
                il.EmitCall(OpCodes.Call, ms.AndNot!, null);
                il.Emit(OpCodes.Stloc, ctx.LocFill);
            }
        }

        /// <summary>
        /// One vector group at element <c>i</c>: C inputs loaded in full (<paramref name="maskLocal"/>
        /// null) or through <c>MaskLoad</c> with the given lane mask (integer lanes OR-ed with the
        /// masked-off fill), S inputs pushed from their row broadcast, the vector body, then a full
        /// or masked store of the result.
        /// </summary>
        private static void EmitMaskedVectorGroup(ILGenerator il, Masked2DContext ctx, LocalBuilder? maskLocal, LocalBuilder? fillLocal)
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
                EmitAddrIPlusOffset(il, ctx.PtrLocals[op], ctx.LocI, 0, ctx.ElemSize);
                if (maskLocal != null)
                {
                    il.Emit(OpCodes.Ldloc, maskLocal);
                    il.EmitCall(OpCodes.Call, ctx.Mask.MaskLoad, null);
                    if (fillLocal != null)
                    {
                        il.Emit(OpCodes.Ldloc, fillLocal);
                        il.EmitCall(OpCodes.Call, ctx.Mask.Or!, null);
                    }
                }
                else
                {
                    EmitVectorLoad(il, ctx.OperandTypes[op]);
                }
            }

            ctx.VectorBody(il);

            if (maskLocal != null)
            {
                il.Emit(OpCodes.Stloc, ctx.LocOutVec);
                EmitAddrIPlusOffset(il, ctx.PtrLocals[nIn], ctx.LocI, 0, ctx.ElemSize);
                il.Emit(OpCodes.Ldloc, maskLocal);
                il.Emit(OpCodes.Ldloc, ctx.LocOutVec);
                il.EmitCall(OpCodes.Call, ctx.Mask.MaskStore, null);
            }
            else
            {
                EmitAddrIPlusOffset(il, ctx.PtrLocals[nIn], ctx.LocI, 0, ctx.ElemSize);
                EmitVectorStore(il, outType);
            }
        }

        /// <summary>
        /// The masked scalar 2-D block: per row, a mask stride of 0 gates the whole row on its
        /// one byte; otherwise every element tests <c>mask[i * maskStride]</c> before the
        /// per-chunk kernel's own scalar element (runtime-stride addressing for every operand,
        /// which serves broadcast inputs and any inner stride alike).
        /// </summary>
        private static void Emit2DMaskedScalarBlock(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals, Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int maskIdx = nOp;
            var dataPtrs = new LocalBuilder[nOp];
            var dataStrides = new LocalBuilder[nOp];
            Array.Copy(ptrLocals, dataPtrs, nOp);
            Array.Copy(innerStrideLocals, dataStrides, nOp);

            var locI = il.DeclareLocal(typeof(long));
            var locOuter = il.DeclareLocal(typeof(long));

            EmitLdarg(il, Arg2DOuterCount);
            il.Emit(OpCodes.Stloc, locOuter);

            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblAdvance = il.DefineLabel();
            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblNext = il.DefineLabel();
            var lblPerElement = il.DefineLabel();

            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblOuterEnd);

            // Row gate when the mask stride is 0.
            il.Emit(OpCodes.Ldloc, innerStrideLocals[maskIdx]);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bne_Un, lblPerElement);
            il.Emit(OpCodes.Ldloc, ptrLocals[maskIdx]);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblAdvance);
            il.MarkLabel(lblPerElement);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);
            il.Emit(OpCodes.Ldloc, locI);
            EmitLdarg(il, Arg2DInnerCount);
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // if (mask[i * maskStride] == 0) continue;   (a stride-0 mask re-reads its row byte: already true)
            EmitAddrIStrided(il, ptrLocals[maskIdx], locI, innerStrideLocals[maskIdx]);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblNext);

            EmitScalarElement(il, operandTypes, dataPtrs, dataStrides, locI, contig: false, scalarBody);

            il.MarkLabel(lblNext);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            il.MarkLabel(lblAdvance);
            EmitAdvanceRow(il, ptrLocals, outerStrideLocals);

            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuter);
            il.Emit(OpCodes.Br, lblOuter);
            il.MarkLabel(lblOuterEnd);
        }
    }
}
