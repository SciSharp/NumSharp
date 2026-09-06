using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.NanToNum — IL-generated np.nan_to_num kernels
// =============================================================================
//
// A single fused whole-array pass that replaces, per element:
//     NaN  -> nan     +inf -> posinf     -inf -> neginf     (finite -> itself)
// where nan/posinf/neginf are runtime scalar fills (pointers to one value of the
// kernel dtype). This is the write-through analog of Clip: read src, transform,
// write dst — the pointers are independent so it serves both copy (src != dst)
// and in-place (src == dst).
//
// NumPy composes nan_to_num from isnan + isposinf + isneginf + three copyto(where=)
// passes (~12 array passes + several bool temporaries). The fused kernel does it
// in ONE read + ONE write with no intermediate allocation.
//
// Dtypes: Half / Single / Double only — the inexact-real dtypes np.nan_to_num
// actually replaces (int/bool/decimal are returned unchanged upstream). Complex
// is handled by the np.* layer running the Double kernel over the raw 2N-double
// buffer (real & imag components each get the same scalar fill — NumPy parity for
// scalar fills).
//
// SIMD (Single/Double): a branchless ConditionalSelect chain seeded with the
// source vector, overlaying the nan/posinf/neginf fills where the classification
// masks fire (masks all computed from the ORIGINAL vector, so overlay order is
// irrelevant — a lane is exactly one of NaN/+inf/-inf/finite). Half runs the
// scalar branch loop (no Vector<Half> arithmetic in the BCL), which is also the
// tail for Single/Double.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Universal nan_to_num kernel signature: read from <paramref name="src"/>,
        /// replace non-finite values with the scalar fills, write to
        /// <paramref name="dst"/>. <paramref name="size"/> is the element count.
        /// <paramref name="nan"/> / <paramref name="posinf"/> / <paramref name="neginf"/>
        /// each point to ONE value of the kernel dtype.
        /// </summary>
        public unsafe delegate void NanToNumKernel(
            void* src, void* dst, long size, void* nan, void* posinf, void* neginf);

        private static readonly ConcurrentDictionary<NPTypeCode, NanToNumKernel> _nanToNumKernelCache = new();

        /// <summary>
        /// Run a nan_to_num pass. Picks (and on first call IL-generates) the kernel
        /// for <paramref name="dtype"/> (Half/Single/Double) and invokes it.
        /// </summary>
        public static unsafe void NanToNum(
            NPTypeCode dtype, void* src, void* dst, long size, void* nan, void* posinf, void* neginf)
        {
            var kernel = _nanToNumKernelCache.GetOrAdd(dtype, static dt => GenerateNanToNum(dt));
            kernel(src, dst, size, nan, posinf, neginf);
        }

        private static NanToNumKernel GenerateNanToNum(NPTypeCode dtype)
        {
            var dm = new DynamicMethod(
                name: $"NanToNum_{dtype}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(void*), typeof(long), typeof(void*), typeof(void*), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            int sz = GetTypeSize(dtype);
            var clrType = GetClrType(dtype);

            // Hoist the three fill scalars into locals (shared by the SIMD broadcast
            // and the scalar tail) — loaded once, re-used every iteration.
            var locNanVal = il.DeclareLocal(clrType);
            var locPosVal = il.DeclareLocal(clrType);
            var locNegVal = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Ldarg_3); EmitLoadIndirect(il, dtype); il.Emit(OpCodes.Stloc, locNanVal);
            il.Emit(OpCodes.Ldarg_S, (byte)4); EmitLoadIndirect(il, dtype); il.Emit(OpCodes.Stloc, locPosVal);
            il.Emit(OpCodes.Ldarg_S, (byte)5); EmitLoadIndirect(il, dtype); il.Emit(OpCodes.Stloc, locNegVal);

            var locI = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // SIMD for Single/Double (CanUseSimd is false for Half).
            if (CanUseSimd(dtype))
                EmitNanToNumSimdLoop(il, dtype, sz, locI, locNanVal, locPosVal, locNegVal);

            // Scalar loop: tail after SIMD, and the whole range for Half.
            EmitNanToNumScalarLoop(il, dtype, sz, locI, locNanVal, locPosVal, locNegVal);

            il.Emit(OpCodes.Ret);
            return (NanToNumKernel)dm.CreateDelegate(typeof(NanToNumKernel));
        }

        private const int NanToNumUnroll = 4;

        private static void EmitNanToNumSimdLoop(
            ILGenerator il, NPTypeCode dtype, int sz, LocalBuilder locI,
            LocalBuilder locNanVal, LocalBuilder locPosVal, LocalBuilder locNegVal)
        {
            var clrType = GetClrType(dtype);
            var vectorType = VectorMethodCache.V(VectorBits, clrType);
            int vectorCount = GetVectorCount(dtype);
            long vcBytes = (long)vectorCount * sz;

            // Broadcast the fills + the ±inf comparison constants into registers once.
            var locNanVec = il.DeclareLocal(vectorType);
            var locPosVec = il.DeclareLocal(vectorType);
            var locNegVec = il.DeclareLocal(vectorType);
            var locPosInf = il.DeclareLocal(vectorType);
            var locNegInf = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Ldloc, locNanVal); EmitVectorCreate(il, dtype); il.Emit(OpCodes.Stloc, locNanVec);
            il.Emit(OpCodes.Ldloc, locPosVal); EmitVectorCreate(il, dtype); il.Emit(OpCodes.Stloc, locPosVec);
            il.Emit(OpCodes.Ldloc, locNegVal); EmitVectorCreate(il, dtype); il.Emit(OpCodes.Stloc, locNegVec);
            if (dtype == NPTypeCode.Single) il.Emit(OpCodes.Ldc_R4, float.PositiveInfinity); else il.Emit(OpCodes.Ldc_R8, double.PositiveInfinity);
            EmitVectorCreate(il, dtype); il.Emit(OpCodes.Stloc, locPosInf);
            if (dtype == NPTypeCode.Single) il.Emit(OpCodes.Ldc_R4, float.NegativeInfinity); else il.Emit(OpCodes.Ldc_R8, double.NegativeInfinity);
            EmitVectorCreate(il, dtype); il.Emit(OpCodes.Stloc, locNegInf);

            // ── NanToNumUnroll×-unrolled body (no loop-carried dependency) ──────
            var bo = new LocalBuilder[NanToNumUnroll];
            for (int k = 0; k < NanToNumUnroll; k++) bo[k] = il.DeclareLocal(typeof(long));

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)(NanToNumUnroll * vectorCount));
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            var lblUnroll = il.DefineLabel();
            var lblAfterUnroll = il.DefineLabel();
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblAfterUnroll);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            for (int k = 1; k < NanToNumUnroll; k++)
            {
                il.Emit(OpCodes.Ldloc, bo[0]);
                il.Emit(OpCodes.Ldc_I8, k * vcBytes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, bo[k]);
            }
            for (int k = 0; k < NanToNumUnroll; k++)
                EmitNanToNumVectorBody(il, dtype, vectorType, bo[k], locNanVec, locPosVec, locNegVec, locPosInf, locNegInf);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)(NanToNumUnroll * vectorCount));
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblAfterUnroll);

            // ── Single-vector remainder ────────────────────────────────────────
            var locVecEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblEnd);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            EmitNanToNumVectorBody(il, dtype, vectorType, bo[0], locNanVec, locPosVec, locNegVec, locPosInf, locNegInf);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        // ONE vector step: load v at byteOff; r = v; overlay nan/posinf/neginf fills
        // where the (original-v) masks fire; store r at dst+byteOff. Stack-neutral.
        private static void EmitNanToNumVectorBody(
            ILGenerator il, NPTypeCode dtype, Type vectorType, LocalBuilder locByteOff,
            LocalBuilder locNanVec, LocalBuilder locPosVec, LocalBuilder locNegVec,
            LocalBuilder locPosInf, LocalBuilder locNegInf)
        {
            var clrType = GetClrType(dtype);
            var locV = il.DeclareLocal(vectorType);
            var locR = il.DeclareLocal(vectorType);

            // v = src[byteOff]
            EmitOffsetAddrFromLocal(il, 0, locByteOff);
            EmitVectorLoad(il, dtype);
            il.Emit(OpCodes.Stloc, locV);

            // r = v
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Stloc, locR);

            // r = ConditionalSelect(~(v == v), nanVec, r)   — NaN lanes -> nan
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldloc, locV);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);
            il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(VectorBits, clrType), null);
            il.Emit(OpCodes.Ldloc, locNanVec);
            il.Emit(OpCodes.Ldloc, locR);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
            il.Emit(OpCodes.Stloc, locR);

            // r = ConditionalSelect(v == +inf, posVec, r)
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldloc, locPosInf);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);
            il.Emit(OpCodes.Ldloc, locPosVec);
            il.Emit(OpCodes.Ldloc, locR);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
            il.Emit(OpCodes.Stloc, locR);

            // r = ConditionalSelect(v == -inf, negVec, r)
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldloc, locNegInf);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);
            il.Emit(OpCodes.Ldloc, locNegVec);
            il.Emit(OpCodes.Ldloc, locR);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
            il.Emit(OpCodes.Stloc, locR);

            // dst[byteOff] = r   (EmitVectorStore stack: Vector, T*)
            il.Emit(OpCodes.Ldloc, locR);
            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            EmitVectorStore(il, dtype);
        }

        private static void EmitNanToNumScalarLoop(
            ILGenerator il, NPTypeCode dtype, int sz, LocalBuilder locI,
            LocalBuilder locNanVal, LocalBuilder locPosVal, LocalBuilder locNegVal)
        {
            var clrType = GetClrType(dtype);
            var isNan = ScalarMethodCache.Predicate(clrType, "IsNaN");
            var isPos = ScalarMethodCache.Predicate(clrType, "IsPositiveInfinity");
            var isNeg = ScalarMethodCache.Predicate(clrType, "IsNegativeInfinity");

            var locVal = il.DeclareLocal(clrType);
            var locR = il.DeclareLocal(clrType);
            var locByteOff = il.DeclareLocal(typeof(long));

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);
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

            // r = val (default)
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Stloc, locR);

            var lblNotNan = il.DefineLabel();
            var lblNotPos = il.DefineLabel();
            var lblStore = il.DefineLabel();

            // if IsNaN(val) r = nan
            il.Emit(OpCodes.Ldloc, locVal);
            il.EmitCall(OpCodes.Call, isNan, null);
            il.Emit(OpCodes.Brfalse, lblNotNan);
            il.Emit(OpCodes.Ldloc, locNanVal); il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Br, lblStore);

            // else if IsPositiveInfinity(val) r = posinf
            il.MarkLabel(lblNotNan);
            il.Emit(OpCodes.Ldloc, locVal);
            il.EmitCall(OpCodes.Call, isPos, null);
            il.Emit(OpCodes.Brfalse, lblNotPos);
            il.Emit(OpCodes.Ldloc, locPosVal); il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Br, lblStore);

            // else if IsNegativeInfinity(val) r = neginf
            il.MarkLabel(lblNotPos);
            il.Emit(OpCodes.Ldloc, locVal);
            il.EmitCall(OpCodes.Call, isNeg, null);
            il.Emit(OpCodes.Brfalse, lblStore);
            il.Emit(OpCodes.Ldloc, locNegVal); il.Emit(OpCodes.Stloc, locR);

            // *(T*)(dst + byteOff) = r
            il.MarkLabel(lblStore);
            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            il.Emit(OpCodes.Ldloc, locR);
            EmitStoreIndirect(il, dtype);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }
    }
}
