using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.Clip — IL-generated clip kernels
// =============================================================================
//
// Single entry point for ALL clip operations. The engine just builds an
// (NPTypeCode, ClipMode, ClipBoundsKind) tuple, hands raw pointers to
// `Clip(...)`, and DirectILKernelGenerator picks (or generates and caches) a
// DynamicMethod that runs the entire loop.
//
// Width-adaptive: the SIMD section is emitted against
// `GetVectorContainerType()`, which resolves at startup to
// `Vector512` / `Vector256` / `Vector128` based on hardware capability.
// Loop body is `Vector.Max(src, lo) ; Vector.Min(., hi)`.
//
// Semantics: result = Min(Max(src, lo), hi) — matches NumPy.
//   - BothBounds: both clamps applied (in that order, so min > max gives max).
//   - MinOnly:    only Max(src, lo).
//   - MaxOnly:    only Min(src, hi).
//
// Bounds kind:
//   - Scalar: `lo`, `hi` point to one element of the dtype. Broadcast into
//             a register vector once, before the SIMD loop.
//   - Array:  `lo`, `hi` point to `size`-element arrays of the dtype. Loaded
//             per vector iteration.
//
// The same kernel handles in-place (src == dst) and fused copy+clip
// (src != dst) — the pointers are independent.
//
// Non-SIMD dtypes (Char, Decimal, Half, Complex) get a pure scalar IL loop.
// Complex / Half delegate the per-element clamp to small static helpers
// that implement NaN-aware semantics; the loop itself is IL.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Public API

        public enum ClipMode : byte
        {
            BothBounds = 0,
            MinOnly    = 1,
            MaxOnly    = 2,
        }

        public enum ClipBoundsKind : byte
        {
            Scalar = 0,
            Array  = 1,
        }

        /// <summary>
        /// Universal clip kernel signature: read from <paramref name="src"/>,
        /// clamp to <paramref name="lo"/> / <paramref name="hi"/>, write to
        /// <paramref name="dst"/>. <paramref name="size"/> is element count.
        /// Bound pointers are interpreted per the mode the kernel was
        /// generated for (scalar = single value, array = `size` values).
        /// Bound pointers for unused sides are ignored by the generated IL.
        /// </summary>
        public unsafe delegate void ClipKernel(void* src, void* dst, long size, void* lo, void* hi);

        internal static readonly ConcurrentDictionary<int, ClipKernel> _clipKernelCache = new();

        /// <summary>
        /// Run a clip operation. Picks (and on first call, IL-generates) the
        /// appropriate DynamicMethod for the (dtype, mode, kind) tuple and
        /// invokes it with the supplied pointers.
        /// </summary>
        public static unsafe void Clip(
            NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind,
            void* src, void* dst, long size, void* lo, void* hi)
        {
            int key = ((int)dtype << 16) | ((int)mode << 8) | (int)kind;
            var kernel = _clipKernelCache.GetOrAdd(key, _ => Generate(dtype, mode, kind));
            kernel(src, dst, size, lo, hi);
        }

        #endregion

        #region IL Generation

        private static ClipKernel Generate(NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind)
        {
            var dm = new DynamicMethod(
                name: $"Clip_{dtype}_{mode}_{kind}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(void*), typeof(long), typeof(void*), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            int sz = GetTypeSize(dtype);

            // Locals shared between SIMD loop and scalar tail
            var locI = il.DeclareLocal(typeof(long));

            // For scalar bounds: hoist the scalar value(s) into locals so both
            // the SIMD broadcast vector and the scalar tail can re-use them.
            var clrType = GetClrType(dtype);
            LocalBuilder locLoVal = (mode != ClipMode.MaxOnly && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(clrType) : null;
            LocalBuilder locHiVal = (mode != ClipMode.MinOnly && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(clrType) : null;

            if (locLoVal != null)
            {
                il.Emit(OpCodes.Ldarg_3);                  // lo
                EmitLoadIndirect(il, dtype);
                il.Emit(OpCodes.Stloc, locLoVal);
            }
            if (locHiVal != null)
            {
                il.Emit(OpCodes.Ldarg_S, (byte)4);         // hi
                EmitLoadIndirect(il, dtype);
                il.Emit(OpCodes.Stloc, locHiVal);
            }

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // SIMD section — only for hardware-supported types (skipped for
            // Char / Decimal / Half / Complex, which have no Vector<T> Min/Max).
            if (CanUseSimd(dtype) && SupportsVectorMinMax(dtype))
                EmitSimdLoop(il, dtype, mode, kind, sz, locI, locLoVal, locHiVal);

            // Scalar loop covers the tail after SIMD and the entire range for
            // non-SIMD dtypes.
            EmitScalarLoop(il, dtype, mode, kind, sz, locI, locLoVal, locHiVal);

            il.Emit(OpCodes.Ret);
            return (ClipKernel)dm.CreateDelegate(typeof(ClipKernel));
        }

        // Vector<T>.Min/Max exist for: byte, sbyte, short, ushort, int, uint,
        // long, ulong (since .NET 8), float, double. They don't exist for
        // Char (unsigned 16-bit but no overload) — we route Char through the
        // scalar IL loop.
        private static bool SupportsVectorMinMax(NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or
            NPTypeCode.Int16 or NPTypeCode.UInt16 or
            NPTypeCode.Int32 or NPTypeCode.UInt32 or
            NPTypeCode.Int64 or NPTypeCode.UInt64 or
            NPTypeCode.Single or NPTypeCode.Double => true,
            _ => false
        };

        // Emits the address `argN + byteOffset` (as native int) onto the stack.
        // argIdx: 0=src, 1=dst, 3=lo, 4=hi (arg2 is the long size).
        // Callers pre-compute `byteOffset = i * sz` once per iteration into a
        // local, then pass that local in — avoids recomputing the same i*sz
        // multiplication for the 2–4 pointers touched each iteration.
        private static void EmitOffsetAddrFromLocal(ILGenerator il, int argIdx, LocalBuilder locByteOff)
        {
            switch (argIdx)
            {
                case 0: il.Emit(OpCodes.Ldarg_0); break;
                case 1: il.Emit(OpCodes.Ldarg_1); break;
                case 3: il.Emit(OpCodes.Ldarg_3); break;
                case 4: il.Emit(OpCodes.Ldarg_S, (byte)4); break;
                default: throw new ArgumentException($"Unexpected argIdx {argIdx}");
            }
            il.Emit(OpCodes.Ldloc, locByteOff);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Conv_I);
        }

        private const int ClipUnroll = 4;

        private static void EmitSimdLoop(
            ILGenerator il, NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind, int sz,
            LocalBuilder locI, LocalBuilder locLoVal, LocalBuilder locHiVal)
        {
            var vectorType = VectorMethodCache.V(VectorBits, GetClrType(dtype));
            int vectorCount = GetVectorCount(dtype);
            bool needLo = mode != ClipMode.MaxOnly;
            bool needHi = mode != ClipMode.MinOnly;

            // Hoist Vector.Create(scalarValue) outside the loop for scalar bounds.
            LocalBuilder locLoVec = (needLo && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(vectorType) : null;
            LocalBuilder locHiVec = (needHi && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(vectorType) : null;
            if (locLoVec != null)
            {
                il.Emit(OpCodes.Ldloc, locLoVal);
                EmitVectorCreate(il, dtype);
                il.Emit(OpCodes.Stloc, locLoVec);
            }
            if (locHiVec != null)
            {
                il.Emit(OpCodes.Ldloc, locHiVal);
                EmitVectorCreate(il, dtype);
                il.Emit(OpCodes.Stloc, locHiVec);
            }

            long vcBytes = (long)vectorCount * sz;

            // ── ClipUnroll×-unrolled body ──────────────────────────────────────
            // clip has NO loop-carried dependency, so unrolling cuts the per-vector
            // loop overhead (the i*sz multiply + bounds branch) and feeds the two
            // min/max ALU ports more independent work. Each iteration processes
            // ClipUnroll vectors at byteOff0 + k*vcBytes (one multiply, constant
            // adds). Measured f64 100K clip(out=) 0.68x -> ~0.95x vs NumPy.
            var bo = new LocalBuilder[ClipUnroll];
            for (int k = 0; k < ClipUnroll; k++) bo[k] = il.DeclareLocal(typeof(long));

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)(ClipUnroll * vectorCount));
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            var lblUnroll = il.DefineLabel();
            var lblAfterUnroll = il.DefineLabel();
            il.MarkLabel(lblUnroll);
            // if (i > unrollEnd) break  (note: skipped entirely when size < ClipUnroll*vc)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblAfterUnroll);

            // bo[0] = i * sz ; bo[k] = bo[0] + k*vcBytes
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            for (int k = 1; k < ClipUnroll; k++)
            {
                il.Emit(OpCodes.Ldloc, bo[0]);
                il.Emit(OpCodes.Ldc_I8, k * vcBytes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, bo[k]);
            }
            for (int k = 0; k < ClipUnroll; k++)
                EmitClipVectorBody(il, dtype, kind, bo[k], needLo, needHi, locLoVec, locHiVec);

            // i += ClipUnroll * vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)(ClipUnroll * vectorCount));
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblAfterUnroll);

            // ── Single-vector remainder ────────────────────────────────────────
            // vecEnd = size - vectorCount
            var locVecEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);
            // if (i > vecEnd) break
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblEnd);

            // byteOff = i * sz
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            EmitClipVectorBody(il, dtype, kind, bo[0], needLo, needHi, locLoVec, locHiVec);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        // Emits ONE vector clip step: load src at byteOff, clamp to lo/hi (NaN-propagating
        // float min/max via EmitVectorMinOrMax), store at dst+byteOff. Stack-neutral. Shared
        // by the unrolled body and the single-vector remainder so both stay bit-identical.
        private static void EmitClipVectorBody(
            ILGenerator il, NPTypeCode dtype, ClipBoundsKind kind, LocalBuilder locByteOff,
            bool needLo, bool needHi, LocalBuilder locLoVec, LocalBuilder locHiVec)
        {
            EmitOffsetAddrFromLocal(il, 0, locByteOff);
            EmitVectorLoad(il, dtype);                                         // [vec]

            if (needLo)
            {
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locLoVec);
                else
                {
                    EmitOffsetAddrFromLocal(il, 3, locByteOff);
                    EmitVectorLoad(il, dtype);
                }
                EmitVectorMinOrMax(il, isMax: true, dtype, propagateNaN: true);    // vec = Max(vec, lo)
            }

            if (needHi)
            {
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locHiVec);
                else
                {
                    EmitOffsetAddrFromLocal(il, 4, locByteOff);
                    EmitVectorLoad(il, dtype);
                }
                EmitVectorMinOrMax(il, isMax: false, dtype, propagateNaN: true);   // vec = Min(vec, hi)
            }

            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            EmitVectorStore(il, dtype);                                        // []
        }

        private static void EmitScalarLoop(
            ILGenerator il, NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind, int sz,
            LocalBuilder locI, LocalBuilder locLoVal, LocalBuilder locHiVal)
        {
            var clrType = GetClrType(dtype);
            bool needLo = mode != ClipMode.MaxOnly;
            bool needHi = mode != ClipMode.MinOnly;

            // Reusable temp for the in-flight value
            var locVal = il.DeclareLocal(clrType);
            var locByteOff = il.DeclareLocal(typeof(long));

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.MarkLabel(lblLoop);
            // if (i >= size) break
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblEnd);

            // byteOff = i * sz
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locByteOff);

            // val = *(T*)(src + byteOff)
            EmitOffsetAddrFromLocal(il, 0, locByteOff);
            EmitLoadIndirect(il, dtype);
            il.Emit(OpCodes.Stloc, locVal);

            if (needLo)
            {
                il.Emit(OpCodes.Ldloc, locVal);
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locLoVal);
                else
                {
                    EmitOffsetAddrFromLocal(il, 3, locByteOff);
                    EmitLoadIndirect(il, dtype);
                }
                EmitScalarClamp(il, dtype, isMax: true);      // val = Max(val, lo)
                il.Emit(OpCodes.Stloc, locVal);
            }

            if (needHi)
            {
                il.Emit(OpCodes.Ldloc, locVal);
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locHiVal);
                else
                {
                    EmitOffsetAddrFromLocal(il, 4, locByteOff);
                    EmitLoadIndirect(il, dtype);
                }
                EmitScalarClamp(il, dtype, isMax: false);     // val = Min(val, hi)
                il.Emit(OpCodes.Stloc, locVal);
            }

            // *(T*)(dst + byteOff) = val
            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            il.Emit(OpCodes.Ldloc, locVal);
            EmitStoreIndirect(il, dtype);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        // Emits scalar Max/Min consuming two T values on the stack, producing
        // one T. Uses Math.Max/Min where available (covers byte..ulong, float,
        // double, decimal), falls back to small static helpers for Half /
        // Complex (NaN/lex semantics) and to a branch-based emit for Char.
        //
        // nanIgnore selects the float/double/half/complex NaN policy:
        //   false (default) — NaN-PROPAGATING (np.clip / np.maximum / np.minimum):
        //                     a NaN operand wins, first-operand-NaN preferred.
        //   true            — NaN-IGNORING (np.fmax / np.fmin): the non-NaN operand
        //                     wins; both-NaN returns the second. Integer / decimal /
        //                     char have no NaN domain, so the policy is irrelevant there.
        private static void EmitScalarClamp(ILGenerator il, NPTypeCode dtype, bool isMax, bool nanIgnore = false)
        {
            var clrType = GetClrType(dtype);

            if (dtype == NPTypeCode.Half)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(HalfMaxNum) : nameof(HalfMinNum))
                    : (isMax ? nameof(HalfMaxNaN) : nameof(HalfMinNaN))), null);
                return;
            }

            if (dtype == NPTypeCode.Complex)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(ComplexMaxNum) : nameof(ComplexMinNum))
                    : (isMax ? nameof(ComplexMaxNaN) : nameof(ComplexMinNaN))), null);
                return;
            }

            // float32/float64: NaN-propagating, signed-zero-tie-to-second (matches the SIMD body
            // and NumPy). Math.Max would resolve the +0/-0 tie to +0 and diverge in the scalar tail.
            if (dtype == NPTypeCode.Single)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(FloatMaxNum) : nameof(FloatMinNum))
                    : (isMax ? nameof(FloatMaxNaN) : nameof(FloatMinNaN))), null);
                return;
            }
            if (dtype == NPTypeCode.Double)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(DoubleMaxNum) : nameof(DoubleMinNum))
                    : (isMax ? nameof(DoubleMaxNaN) : nameof(DoubleMinNaN))), null);
                return;
            }

            // Math.Max(T, T) / Math.Min(T, T) for all sized numerics + decimal.
            // ScalarMethodCache.Get throws on missing — fall back to the manual select below
            // for the one type that's not covered (Char has no Math.Max overload).
            MethodInfo mathMethod = null;
            try
            {
                mathMethod = ScalarMethodCache.Get(typeof(Math),
                    isMax ? nameof(Math.Max) : nameof(Math.Min), clrType, clrType);
            }
            catch (MissingMethodException) { /* fall through to manual select */ }

            if (mathMethod != null)
            {
                il.EmitCall(OpCodes.Call, mathMethod, null);
                return;
            }

            // Char has no Math.Max overload — emit a manual select.
            // Stack: [a, b]. Result Max: a if a>=b else b ; Min: a if a<=b else b.
            var locA = il.DeclareLocal(clrType);
            var locB = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Stloc, locB);
            il.Emit(OpCodes.Stloc, locA);
            var lblPickB = il.DefineLabel();
            var lblDone  = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Ldloc, locB);
            // for Max: jump to PickB when a < b; else fall through and push a
            // for Min: jump to PickB when a > b; else fall through and push a
            il.Emit(isMax ? OpCodes.Blt : OpCodes.Bgt, lblPickB);
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblPickB);
            il.Emit(OpCodes.Ldloc, locB);
            il.MarkLabel(lblDone);
        }

        #endregion

        #region Per-Element Helpers for Non-SIMD Types (called from generated IL)

        // Half: comparison ops work natively in .NET 7+, but Math.Max(Half,Half)
        // doesn't exist as a single-precision-aware overload. NumPy semantics:
        // NaN propagates — if either operand is NaN, result is NaN.
        //
        // Signed-zero tie-break: NumPy's float16 maximum/minimum return the FIRST
        // operand when the two compare equal (so maximum(+0,-0)=+0, maximum(-0,+0)=-0,
        // and likewise for minimum). `a` here is the first operand (clip's src), `b`
        // the second (the bound), so the `>=` / `<=` tie goes to `a` — matching NumPy.
        // (+0 == -0 numerically, so this only changes the result's sign bit on a zero
        // tie; every non-zero equal pair is bit-identical and unaffected.)
        //
        // These per-element helpers run once per element inside the IL kernel's inner loop,
        // so they carry AggressiveInlining (inline into the kernel where the JIT can) plus
        // AggressiveOptimization (full tier-1 codegen from the first call when they're
        // invoked standalone via the kernel's Call, skipping the tier-0 hot-path penalty).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMaxNaN(Half a, Half b)
        {
            if (Half.IsNaN(a) || Half.IsNaN(b)) return Half.NaN;
            return a >= b ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMinNaN(Half a, Half b)
        {
            if (Half.IsNaN(a) || Half.IsNaN(b)) return Half.NaN;
            return a <= b ? a : b;
        }

        // Half NaN-IGNORING max/min (np.fmax / np.fmin): the non-NaN operand wins.
        // Tie convention follows HalfMaxNaN/HalfMinNaN (first operand on an equal tie).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMaxNum(Half a, Half b)
        {
            if (Half.IsNaN(b)) return a;   // b-first: both-NaN -> a (first), per the array ufunc loop
            if (Half.IsNaN(a)) return b;
            return a >= b ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMinNum(Half a, Half b)
        {
            if (Half.IsNaN(b)) return a;
            if (Half.IsNaN(a)) return b;
            return a <= b ? a : b;
        }

        // float32/float64 max/min matching the NaN-propagating SIMD path (EmitVectorMinOrMax)
        // and NumPy maximum/minimum/clip. NaN in the first operand propagates; otherwise the
        // STRICT comparison returns the SECOND operand on a tie — so the signed-zero tie resolves
        // like hardware MAXPS/MINPD (f32/f64 maximum(+0,-0) = -0), the OPPOSITE of float16 (which
        // ties to the first operand, hence the separate Half helpers above) and of Math.Max (+0).
        // Used by both the IL kernel's scalar tail (EmitScalarClamp) and the engine's ClipStrided,
        // so the contiguous and strided paths agree bit-for-bit on signed-zero / NaN.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMaxNaN(float a, float b) => float.IsNaN(a) ? a : (a > b ? a : b);
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMinNaN(float a, float b) => float.IsNaN(a) ? a : (a < b ? a : b);
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMaxNaN(double a, double b) => double.IsNaN(a) ? a : (a > b ? a : b);
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMinNaN(double a, double b) => double.IsNaN(a) ? a : (a < b ? a : b);

        // float32/float64 NaN-IGNORING max/min — np.fmax / np.fmin ARRAY semantics (probed
        // 2.4.2): a NaN operand is skipped and the OTHER operand returned. The NaN checks are
        // ordered B-FIRST so both-NaN returns the FIRST operand (a): fmax(nanA, nanB) -> nanA,
        // matching the ufunc loop (NumPy's SCALAR fmax() returns the second — the array loop
        // does not). For two finite operands the strict comparison returns the SECOND on a tie
        // — identical to maximum/minimum and the hardware MAXPS/MINPS the SIMD body uses, so
        // np.fmax([+0],[-0]) -> -0 like np.maximum. Used by EmitScalarClamp(nanIgnore:true) and
        // the strided fmax/fmin loop.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMaxNum(float a, float b) => float.IsNaN(b) ? a : (float.IsNaN(a) ? b : (a > b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMinNum(float a, float b) => float.IsNaN(b) ? a : (float.IsNaN(a) ? b : (a < b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMaxNum(double a, double b) => double.IsNaN(b) ? a : (double.IsNaN(a) ? b : (a > b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMinNum(double a, double b) => double.IsNaN(b) ? a : (double.IsNaN(a) ? b : (a < b ? a : b));

        // Complex: lex ordering on (real, imag). NaN propagation: if either
        // operand contains a NaN component, that operand wins (first-encountered
        // for the Max-then-Min sequence — matches NumPy clip output bit-for-bit).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool ComplexIsNaN(Complex z) => double.IsNaN(z.Real) || double.IsNaN(z.Imaginary);
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMaxNaN(Complex a, Complex b)
        {
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            if (a.Real > b.Real) return a;
            if (a.Real < b.Real) return b;
            return a.Imaginary > b.Imaginary ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMinNaN(Complex a, Complex b)
        {
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            if (a.Real > b.Real) return b;
            if (a.Real < b.Real) return a;
            return a.Imaginary > b.Imaginary ? b : a;
        }

        // Complex NaN-IGNORING max/min (np.fmax / np.fmin): the non-NaN operand wins; lex
        // ordering on (real, imag) otherwise. Mirrors ComplexMaxNaN/ComplexMinNaN with the
        // NaN branches returning the OTHER operand instead of the NaN one.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMaxNum(Complex a, Complex b)
        {
            if (ComplexIsNaN(b)) return a;   // b-first: both-NaN -> a (first)
            if (ComplexIsNaN(a)) return b;
            if (a.Real > b.Real) return a;
            if (a.Real < b.Real) return b;
            return a.Imaginary > b.Imaginary ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMinNum(Complex a, Complex b)
        {
            if (ComplexIsNaN(b)) return a;
            if (ComplexIsNaN(a)) return b;
            if (a.Real > b.Real) return b;
            if (a.Real < b.Real) return a;
            return a.Imaginary > b.Imaginary ? b : a;
        }

        #endregion
    }
}
